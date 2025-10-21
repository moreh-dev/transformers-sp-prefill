import math

import torch
import torch.distributed as dist
from yunchang.comm.all_to_all import SeqAllToAll4D
from yunchang.kernels import AttnType
from yunchang.ring.utils import RingComm, update_out_and_lse


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def torch_attention_with_sinks_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
    is_causal: bool,
    window_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, num_query_heads, head_size = query.shape
    _, _, num_kv_heads, _ = key.shape

    query_transposed = query.transpose(1, 2)
    key_transposed = key.transpose(1, 2)
    value_transposed = value.transpose(1, 2)

    num_queries_per_kv = num_query_heads // num_kv_heads
    key_transposed = repeat_kv(key_transposed, num_queries_per_kv)
    value_transposed = repeat_kv(value_transposed, num_queries_per_kv)

    attn_weights = torch.matmul(query_transposed, key_transposed.transpose(-2, -1)) * scale

    if is_causal:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool), diagonal=1)
        attn_weights.masked_fill_(causal_mask[None, None, :, :], float("-inf"))

    q_indices = torch.arange(seq_len, device=query.device)[:, None]
    k_indices = torch.arange(seq_len, device=query.device)[None, :]

    if window_size[0] != -1:
        past_mask = (q_indices - k_indices) >= window_size[0]
        attn_weights.masked_fill_(past_mask[None, None, :, :], float("-inf"))

    if window_size[1] != -1:
        future_mask = (k_indices - q_indices) > window_size[1]
        attn_weights.masked_fill_(future_mask[None, None, :, :], float("-inf"))

    row_max_val, _ = torch.max(attn_weights, dim=-1)  # shape: (B, Nq, S)

    # If max is -inf, it means the entire row is masked.
    is_inf_row = torch.isinf(row_max_val)  # shape: (B, Nq, S)

    all_masked = False
    if torch.all(is_inf_row).item():
        all_masked = True
        return None, None, all_masked

    expanded_sinks = sinks.view(1, -1, 1, 1).expand(batch_size, -1, seq_len, 1)
    combined_logits = torch.cat([attn_weights, expanded_sinks], dim=-1)

    m, _ = torch.max(combined_logits, dim=-1, keepdim=True)
    m = torch.where(torch.isinf(m), 0.0, m)

    p = torch.exp(combined_logits - m)
    l = torch.sum(p, dim=-1)
    lse = m.squeeze(-1) + torch.log(l + 1e-9)

    probs = p / (l.unsqueeze(-1) + 1e-9)
    scores = probs[..., :-1]
    scores = scores.to(query.dtype)

    output_transposed = torch.matmul(scores, value_transposed)
    output = output_transposed.transpose(1, 2).contiguous()

    final_lse = torch.where(is_inf_row, -torch.inf, lse)

    mask_for_output = is_inf_row.transpose(1, 2).unsqueeze(-1)
    final_output = torch.where(mask_for_output, 0.0, output)

    return final_output, final_lse, all_masked


def moreh_gpt_attention(
    module,
    query,
    key,
    value,
    sinks,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    is_kernel_bhsd: bool = True,
) -> torch.Tensor:
    assert module.use_pack_qkv is False, "Packed QKV is not supported in this attention implementation."
    assert module.attn_type == AttnType.TORCH

    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    query_layer = SeqAllToAll4D.apply(module.ulysses_pg, query, module.scatter_idx, module.gather_idx)
    key_layer = SeqAllToAll4D.apply(module.ulysses_pg, key, module.scatter_idx, module.gather_idx)
    value_layer = SeqAllToAll4D.apply(module.ulysses_pg, value, module.scatter_idx, module.gather_idx)

    key_layer = key_layer.contiguous()
    value_layer = value_layer.contiguous()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(query_layer.size(-1))
    comm = RingComm(module.ring_pg)

    out = None
    lse = None

    next_k, next_v = None, None

    original_window_size = window_size

    chunk_len = query_layer.shape[1]

    for step in range(comm.world_size):
        attn_done_local = torch.zeros(1, device=query.device, dtype=torch.int32)

        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(key_layer)
            next_v: torch.Tensor = comm.send_recv(value_layer)
            comm.commit()

        key, value = key_layer, value_layer

        if original_window_size[0] == -1:
            adjusted_left = -1
        else:
            adjusted_left = original_window_size[0] - step * chunk_len

        adjusted_right = original_window_size[1] if step == 0 else -1
        adjusted_window_size = (adjusted_left, adjusted_right)

        if not causal or step <= comm.rank:
            block_out, block_lse, all_masked = torch_attention_with_sinks_forward(
                query_layer,
                key,
                value,
                sinks,
                scale=softmax_scale,
                is_causal=causal and step == 0,
                window_size=adjusted_window_size,
            )

            if not all_masked:
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
                attn_done_local[0] = True

        dist.all_reduce(attn_done_local, op=dist.ReduceOp.SUM)

        if attn_done_local.item() == 0:
            print(f"rank {comm.rank} early terminating at step {step}")
            break

        if step + 1 != comm.world_size:
            comm.wait()
            key_layer = next_k
            value_layer = next_v

    out = out.to(query.dtype)
    output = SeqAllToAll4D.apply(module.ulysses_pg, out, module.gather_idx, module.scatter_idx)

    return output
