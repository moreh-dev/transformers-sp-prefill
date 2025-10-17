import math
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from xfuser.core.distributed import (
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from yunchang.comm.all_to_all import SeqAllToAll4D
from yunchang.kernels import AttnType
from yunchang.ring.utils import RingComm, update_out_and_lse


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def vanila_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    scaling: float,
    is_causal: bool = False,
    window_size: tuple = (-1, -1),
    dropout: float = 0.0,
):
    assert query.shape[1] % key.shape[1] == 0
    num_key_value_groups = query.shape[1] // key.shape[1]

    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    q_len, kv_len = query.shape[-2], key_states.shape[-2]

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    w_past, w_future = window_size
    if is_causal or w_past != -1 or w_future != -1:
        attention_mask = torch.zeros(q_len, kv_len, device=query.device)
        q_indices = torch.arange(q_len, device=query.device)[:, None]
        k_indices = torch.arange(kv_len, device=query.device)[None, :]

        if is_causal:
            causal_mask = k_indices > q_indices
            attention_mask.masked_fill_(causal_mask, float('-inf'))

        if w_past != -1:
            past_mask = k_indices < q_indices - w_past
            # In the original code, the condition was `q_indices - w_past + 1`.
            # Using just `q_indices - w_past` is more standard for sliding window.
            # I will keep the original logic for perfect replication.
            past_mask = k_indices < q_indices - w_past + 1
            attention_mask.masked_fill_(past_mask, float('-inf'))

        if w_future != -1:
            future_mask = k_indices > q_indices + w_future
            attention_mask.masked_fill_(future_mask, float('-inf'))

        expanded_mask = attention_mask[None, None, :, :].expand(query.shape[0], query.shape[1], -1, -1)
        attn_weights = attn_weights + expanded_mask

    sinks = sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], 1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    max_logits = combined_logits.max(dim=-1, keepdim=True).values
    stable_logits = combined_logits - max_logits

    probs = F.softmax(stable_logits, dim=-1, dtype=torch.float32).to(query.dtype)

    scores = probs[..., :-1]

    attn_output = torch.matmul(scores, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous() # Output: (B, S, H, D)

    lse = max_logits.squeeze(-1) + torch.log(torch.exp(stable_logits).sum(dim=-1)) # LSE: (B, H, S)

    return attn_output, lse

def torch_attention_with_sinks_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
    is_causal: bool,
    window_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation equivalent to the provided Triton kernel.

    Args:
        query (torch.Tensor): Query tensor (B, S, H, D)
        key (torch.Tensor): Key tensor (B, S, H_kv, D)
        value (torch.Tensor): Value tensor (B, S, H_kv, D)
        sinks (torch.Tensor): Sinks tensor (H,)
        scale (float): Attention score scaling factor
        is_causal (bool): Whether to apply a causal mask
        window_size (Tuple[int, int]): Window size for sliding window attention (past, future)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - output (torch.Tensor): Attention output (B, S, H, D)
            - lse (torch.Tensor): Log-Sum-Exp values (B, H, S)
    """
    batch_size, seq_len, num_query_heads, head_size = query.shape
    _, _, num_kv_heads, _ = key.shape

    # Transpose to (B, H, S, D) for computation
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Repeat Key/Value for Grouped-Query Attention (GQA)
    num_queries_per_kv = num_query_heads // num_kv_heads
    key = repeat_kv(key, num_queries_per_kv)
    value = repeat_kv(value, num_queries_per_kv)

    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply causal and sliding window masks
    if is_causal:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool), diagonal=1)
        attn_weights.masked_fill_(causal_mask[None, None, :, :], float("-inf"))

    q_indices = torch.arange(seq_len, device=query.device)[:, None]
    k_indices = torch.arange(seq_len, device=query.device)[None, :]

    if window_size[0] != -1:  # Past window
        past_mask = (q_indices - k_indices) >= window_size[0]
        attn_weights.masked_fill_(past_mask[None, None, :, :], float("-inf"))

    if window_size[1] != -1:  # Future window
        future_mask = (k_indices - q_indices) > window_size[1]
        attn_weights.masked_fill_(future_mask[None, None, :, :], float("-inf"))

    # Expand sinks and concatenate them to the attention logits
    expanded_sinks = sinks.view(1, -1, 1, 1).expand(batch_size, -1, seq_len, 1)
    combined_logits = torch.cat([attn_weights, expanded_sinks], dim=-1)

    # Calculate Log-Sum-Exp efficiently
    lse = torch.logsumexp(combined_logits, dim=-1)

    # Compute softmax over logits including the sink for stable probability calculation
    # Use float32 for softmax stability
    probs = F.softmax(combined_logits, dim=-1, dtype=torch.float32).to(query.dtype)

    # Exclude the last probability value which corresponds to the sink
    scores = probs[..., :-1]

    # Compute attention output
    output = torch.matmul(scores, value)

    # Transpose output back to (B, S, H, D)
    output = output.transpose(1, 2).contiguous()

    return output, lse

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
    ) -> torch.Tensor:
    assert module.use_pack_qkv is False, "Packed QKV is not supported in this attention implementation."
    assert module.attn_type == AttnType.TORCH

    query_layer = SeqAllToAll4D.apply(
        module.ulysses_pg, query, module.scatter_idx, module.gather_idx
    )
    key_layer = SeqAllToAll4D.apply(
        module.ulysses_pg, key, module.scatter_idx, module.gather_idx
    )
    value_layer = SeqAllToAll4D.apply(
        module.ulysses_pg, value, module.scatter_idx, module.gather_idx
    )

    key_layer = key_layer.contiguous()
    value_layer = value_layer.contiguous()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(query_layer.size(-1))
    comm = RingComm(module.ring_pg)

    out = None
    lse = None

    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(key_layer)
            next_v: torch.Tensor = comm.send_recv(value_layer)
            comm.commit()
            key, value = key_layer, value_layer
        if not causal or step <= comm.rank:
            block_out, block_lse = torch_attention_with_sinks_forward(
                query_layer,
                key,
                value,
                sinks,
                scale=softmax_scale,
                is_causal=causal and step == 0,
                window_size=window_size,
            )
            #block_out = block_out.transpose(1, 2)
            block_lse = block_lse.to(query_layer.dtype)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            key_layer = next_k
            value_layer = next_v

    out = out.to(query.dtype)
    output = SeqAllToAll4D.apply(
            module.ulysses_pg, out, module.gather_idx, module.scatter_idx)

    return output

class RingAttentionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()
        torch.cuda.set_device(cls.rank)

        ring_size = cls.world_size
        ulysses_size = cls.world_size // ring_size

        init_distributed_environment(rank=cls.rank, world_size=cls.world_size)
        initialize_model_parallel(
            sequence_parallel_degree=cls.world_size,
            ring_degree=ring_size,
            ulysses_degree=ulysses_size,
        )

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def test_ring_attention_vs_sdpa(self):
        batch_size = 1
        total_seq_len = 2048
        assert total_seq_len % self.world_size == 0, "Total sequence length must be divisible by world size."
        seq_len_per_device = total_seq_len // self.world_size
        num_heads = 64
        kv_num_heads = 8
        head_dim = 64
        hidden_dim = num_heads * head_dim
        dtype = torch.bfloat16
        device = f"cuda:{self.rank}"

        torch.manual_seed(42)

        q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False, device=device, dtype=dtype)
        k_proj = nn.Linear(hidden_dim, kv_num_heads * head_dim, bias=False, device=device, dtype=dtype)
        v_proj = nn.Linear(hidden_dim, kv_num_heads * head_dim, bias=False, device=device, dtype=dtype)
        o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False, device=device, dtype=dtype)

        if self.rank == 0:
            full_hidden_states = torch.randn(
                (batch_size, total_seq_len, hidden_dim), device=device, dtype=dtype
            )
            full_cos = torch.randn((batch_size, total_seq_len, head_dim), device=device, dtype=dtype)
            full_sin = torch.randn((batch_size, total_seq_len, head_dim), device=device, dtype=dtype)
        else:
            full_hidden_states = torch.empty((batch_size, total_seq_len, hidden_dim), device=device, dtype=dtype)
            full_cos = torch.empty((batch_size, total_seq_len, head_dim), device=device, dtype=dtype)
            full_sin = torch.empty((batch_size, total_seq_len, head_dim), device=device, dtype=dtype)

        dist.broadcast(full_hidden_states, src=0)
        dist.broadcast(full_cos, src=0)
        dist.broadcast(full_sin, src=0)

        local_hidden_states = full_hidden_states.chunk(self.world_size, dim=1)[self.rank].contiguous()
        local_cos = full_cos.chunk(self.world_size, dim=1)[self.rank].contiguous()
        local_sin = full_sin.chunk(self.world_size, dim=1)[self.rank].contiguous()

        local_query = q_proj(local_hidden_states).view(batch_size, seq_len_per_device, num_heads, head_dim)
        local_key = k_proj(local_hidden_states).view(batch_size, seq_len_per_device, kv_num_heads, head_dim)
        local_value = v_proj(local_hidden_states).view(batch_size, seq_len_per_device, kv_num_heads, head_dim)

        local_query = local_query.transpose(1, 2)
        local_key = local_key.transpose(1, 2)

        local_query, local_key = apply_rotary_pos_emb(local_query, local_key, local_cos, local_sin)

        local_query = local_query.transpose(1, 2)
        local_key = local_key.transpose(1, 2)

        sinks = torch.full((num_heads,), float('-inf'), device=device, dtype=dtype)
        sinks = torch.randn_like(sinks)
        is_causal = True
        window_size = (-1, -1)

        attn_class= xFuserLongContextAttention(attn_type=AttnType.TORCH)
        attn_output_local = moreh_gpt_attention(
            attn_class,
            local_query,
            local_key,
            local_value,
            sinks,
            dropout_p=0.0,
            causal=is_causal,
            window_size=window_size,
        )

        attn_output_local = attn_output_local.reshape(batch_size, seq_len_per_device, hidden_dim)
        output_local_xfuser = o_proj(attn_output_local)

        output_gathered_xfuser = get_sp_group().all_gather(output_local_xfuser, dim=1)

        full_query = q_proj(full_hidden_states).view(batch_size, total_seq_len, num_heads, head_dim).transpose(1, 2)
        full_key = k_proj(full_hidden_states).view(batch_size, total_seq_len, kv_num_heads, head_dim).transpose(1, 2)
        full_value = v_proj(full_hidden_states).view(batch_size, total_seq_len, kv_num_heads, head_dim).transpose(1, 2)

        full_query_rope, full_key_rope = apply_rotary_pos_emb(
            full_query, full_key, full_cos, full_sin
        )

        sdpa_output = vanila_attention(
            full_query_rope,
            full_key_rope,
            full_value,
            sinks=sinks,
            is_causal=is_causal,
            window_size=window_size,
            scaling=head_dim ** -0.5,
            dropout=0.0,
        )[0]

        sdpa_output = sdpa_output.transpose(1, 2).contiguous().view(batch_size, total_seq_len, hidden_dim)
        expected_output = o_proj(sdpa_output)

        if self.rank == 0:
            print("xFuser Ring Attention output shape:", output_gathered_xfuser.shape)
            print("PyTorch SDPA output shape:", expected_output.shape)

            # Define variables for clarity
            actual_output = output_gathered_xfuser
            expected_output = expected_output
            atol = 0.2
            rtol = 0.2

            # Perform the allclose check
            is_close = torch.allclose(actual_output, expected_output, atol=atol, rtol=rtol)

            if is_close:
                print("\n✅ Test Passed: xFuser Ring Attention is allclose to PyTorch SDPA.")
            else:
                # If the test fails, print detailed statistics
                print("\n❌ Test Failed: Outputs are not allclose. Analyzing differences...")

                # Calculate difference statistics
                diff = actual_output - expected_output
                abs_diff = torch.abs(diff)

                max_abs_diff = abs_diff.max().item()
                mean_abs_diff = abs_diff.mean().item()
                median_abs_diff = abs_diff.median().item()

                print("  - Absolute Difference Stats:")
                print(f"    - Max:    {max_abs_diff:.6f}")
                print(f"    - Mean:   {mean_abs_diff:.6f}")
                print(f"    - Median: {median_abs_diff:.6f}")

                # Calculate the rate of differing elements based on atol/rtol
                mismatched_elements = torch.sum(torch.abs(diff) > (atol + rtol * torch.abs(expected_output)))
                total_elements = expected_output.numel()
                mismatch_rate = (mismatched_elements.item() / total_elements) * 100 if total_elements > 0 else 0

                print(f"\n  - Mismatch Rate (atol={atol}, rtol={rtol}):")
                print(f"    - Mismatched Elements: {mismatched_elements.item()} / {total_elements}")
                print(f"    - Percentage: {mismatch_rate:.4f}%\n")

            # The assertion still ensures the test fails correctly
            self.assertTrue(
                is_close,
                "The outputs of xFuser Ring Attention and PyTorch SDPA do not match. See detailed analysis above."
            )
if __name__ == "__main__":
    unittest.main()
