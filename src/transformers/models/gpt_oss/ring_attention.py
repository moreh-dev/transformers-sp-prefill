import math

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import yunchang.comm.extract_local
from yunchang.comm.all_to_all import SeqAllToAll4D
from yunchang.globals import PROCESS_GROUP
from yunchang.kernels import AttnType
from yunchang.ring.utils import RingComm, update_out_and_lse


def zigzag_extract_local_patched(value, rank, world_size, rd, ud, dim=1, *args, **kwargs):
    """
    value is a tensor of shape (bs, seqlen, ...)
    """
    input_dim = value.dim()
    assert input_dim >= 2

    shape = list(value.shape)
    seqlen = shape[dim]

    value_chunks = value.chunk(2 * rd, dim=dim)

    r_rank = dist.get_rank(group=PROCESS_GROUP.RING_PG)
    u_rank = dist.get_rank(group=PROCESS_GROUP.ULYSSES_PG)

    assert dist.get_world_size(group=PROCESS_GROUP.RING_PG) == rd
    assert dist.get_world_size(group=PROCESS_GROUP.ULYSSES_PG) == ud

    local_value = torch.cat([value_chunks[r_rank], value_chunks[2 * rd - r_rank - 1]], dim=dim).chunk(ud, dim=dim)[
        u_rank
    ]

    new_shape = shape
    new_shape[dim] = seqlen // world_size
    return local_value.reshape(new_shape).contiguous()


def all_gather_zigzag(local_tensor, rd, ud, dim=1, *args, **kwargs):
    """
    Inverse of zigzag_extract_local_patched (All-Gather with Reordering).
    """

    ring_pg = PROCESS_GROUP.RING_PG
    ulysses_pg = PROCESS_GROUP.ULYSSES_PG
    r_rank = dist.get_rank(group=ring_pg)

    # --- 1. ulysses All-Gather ---

    # (B, H, S_local, D) -> (S_local, H, B, D)
    local_tensor_trans = local_tensor.transpose(0, dim).contiguous()

    # out shape (S_local*ud, H, B, D)
    shape_trans = list(local_tensor_trans.shape)
    shape_trans[0] *= ud
    concatenated_chunk_trans = torch.empty(shape_trans, dtype=local_tensor.dtype, device=local_tensor.device)

    dist.all_gather_into_tensor(concatenated_chunk_trans, local_tensor_trans, group=ulysses_pg)

    # (S_local*ud, H, B, D) -> (B, H, S_local*ud, D)
    concatenated_chunk = concatenated_chunk_trans.transpose(0, dim).contiguous()

    # --- 2. Chunk Split ---
    chunk_O_r, chunk_O_other = concatenated_chunk.chunk(2, dim=dim)

    chunk_O_r = chunk_O_r.contiguous()
    chunk_O_other = chunk_O_other.contiguous()

    # --- 3. Ring All-Gather ---

    # first half chunks
    chunk_O_r_trans = chunk_O_r.transpose(0, dim).contiguous()
    shape_r_trans = list(chunk_O_r_trans.shape)
    shape_r_trans[0] *= rd
    gathered_O_r_trans = torch.empty(shape_r_trans, dtype=local_tensor.dtype, device=local_tensor.device)
    dist.all_gather_into_tensor(gathered_O_r_trans, chunk_O_r_trans, group=ring_pg)
    gathered_O_r = gathered_O_r_trans.transpose(0, dim).contiguous()

    # second half chunks
    chunk_O_other_trans = chunk_O_other.transpose(0, dim).contiguous()
    shape_other_trans = list(chunk_O_other_trans.shape)
    shape_other_trans[0] *= rd
    gathered_O_other_trans = torch.empty(shape_other_trans, dtype=local_tensor.dtype, device=local_tensor.device)
    dist.all_gather_into_tensor(gathered_O_other_trans, chunk_O_other_trans, group=ring_pg)
    gathered_O_other = gathered_O_other_trans.transpose(0, dim).contiguous()

    # --- 4. Final Assembly ---
    other_chunks_list = gathered_O_other.chunk(rd, dim=dim)
    gathered_O_other_reversed = torch.cat(list(reversed(other_chunks_list)), dim=dim)

    global_tensor = torch.cat([gathered_O_r, gathered_O_other_reversed], dim=dim)

    return global_tensor.contiguous()


yunchang.comm.extract_local.EXTRACT_FUNC_DICT["zigzag"] = zigzag_extract_local_patched


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
    window_size: tuple[int, int] = (-1, -1),
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


@triton.jit
def kernel_attention_contiguous(
    output_ptr,
    lse_ptr,
    all_masked_ptr,  # NEW: Pointer to store the all_masked flag
    query_ptr,
    key_ptr,
    value_ptr,
    sinks_ptr,
    # Strides for (B, S, H, D) layout
    stride_out_batch,
    stride_out_seq,
    stride_out_head,
    stride_lse_batch,
    stride_lse_head,
    stride_lse_seq,
    # NEW: Strides for all_masked output (grid shape)
    stride_am_batch,
    stride_am_head,
    stride_q_batch,
    stride_q_seq,
    stride_q_head,
    stride_k_batch,
    stride_k_seq,
    stride_k_head,
    stride_v_batch,
    stride_v_seq,
    stride_v_head,
    num_query_heads,
    num_kv_heads,
    seq_len,
    head_size,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WINDOW_SIZE_PAST: tl.constexpr,
    WINDOW_SIZE_FUTURE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    start_m = tl.program_id(2) * BLOCK_M

    num_queries_per_kv = num_query_heads // num_kv_heads
    kv_head_idx = head_idx // num_queries_per_kv

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = query_ptr + (
        batch_idx * stride_q_batch + offs_m[:, None] * stride_q_seq + head_idx * stride_q_head + offs_d[None, :]
    )

    k_ptrs_base = key_ptr + (batch_idx * stride_k_batch + kv_head_idx * stride_k_head)
    v_ptrs_base = value_ptr + (batch_idx * stride_v_batch + kv_head_idx * stride_v_head)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    sink_val = tl.load(sinks_ptr + head_idx).to(tl.float32)
    m_i = tl.full([BLOCK_M], sink_val, tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)

    # NEW: Initialize all_masked flag to 1 (True).
    # It will be set to 0 if any valid attention score is found.
    all_masked_flag = 1

    q_mask = offs_m < seq_len
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    q = (q * scale).to(q.dtype)

    end_n = seq_len
    if IS_CAUSAL:
        end_n = start_m + BLOCK_M

    start_n = 0
    while start_n < end_n:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = k_ptrs_base + (offs_d[:, None] * 1 + offs_n[None, :] * stride_k_seq)
        k_mask = offs_n[None, :] < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        mask = q_mask[:, None] & (offs_n[None, :] < end_n)
        if IS_CAUSAL:
            mask = mask & (offs_m[:, None] >= offs_n[None, :])

        if WINDOW_SIZE_PAST != -1:
            past_mask = (offs_m[:, None] - offs_n[None, :]) >= WINDOW_SIZE_PAST
            qk = tl.where(past_mask, float("-inf"), qk)

        if WINDOW_SIZE_FUTURE != -1:
            future_mask = (offs_n[None, :] - offs_m[:, None]) > WINDOW_SIZE_FUTURE
            qk = tl.where(future_mask, float("-inf"), qk)

        qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_j = tl.sum(p, 1)

        # NEW: Check if any attention was computed in this block.
        # If the sum of probabilities `p` is greater than 0,
        # it means at least one logit was not -inf.
        if tl.sum(p) > 0.0:
            all_masked_flag = 0

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]

        v_ptrs = v_ptrs_base + (offs_n[:, None] * stride_v_seq + offs_d[None, :])
        v_mask = offs_n[:, None] < seq_len
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_j
        m_i = m_ij

        start_n += BLOCK_N

    # NEW: Store the all_masked flag for the current program instance.
    pid_m_block = tl.program_id(2)
    all_masked_out_ptr = all_masked_ptr + batch_idx * stride_am_batch + head_idx * stride_am_head + pid_m_block
    tl.store(all_masked_out_ptr, all_masked_flag)

    lse = m_i + tl.log(l_i)
    lse_ptrs = lse_ptr + (batch_idx * stride_lse_batch + head_idx * stride_lse_head + offs_m)
    tl.store(lse_ptrs, lse, mask=q_mask)

    acc = acc / l_i[:, None]
    out_ptrs = output_ptr + (
        batch_idx * stride_out_batch + offs_m[:, None] * stride_out_seq + head_idx * stride_out_head + offs_d[None, :]
    )
    tl.store(out_ptrs, acc, mask=q_mask[:, None])


# --- MODIFIED Python Wrapper for Triton Kernel ---
def triton_attention_forward(query, key, value, sinks, scale, is_causal: bool, window_size: tuple):
    # Input shapes: (B, S, H, D)
    batch_size, seq_len, num_query_heads, head_size = query.shape
    _, _, num_kv_heads, _ = key.shape

    output = torch.empty_like(query)
    # LSE shape: (B, H, S)
    lse_output = torch.empty((batch_size, num_query_heads, seq_len), dtype=torch.float32, device=query.device)

    BLOCK_M = 16
    BLOCK_N = 64

    grid = (batch_size, num_query_heads, triton.cdiv(seq_len, BLOCK_M))

    # Create a tensor to store the all_masked flag for each program in the grid.
    all_masked_output = torch.empty(grid, dtype=torch.int32, device=query.device)

    kernel_attention_contiguous[grid](
        output,
        lse_output,
        all_masked_output,
        query,
        key,
        value,
        sinks,
        # Strides for Output (B, S, H, D)
        output.stride(0),
        output.stride(1),
        output.stride(2),
        # Strides for LSE (B, H, S)
        lse_output.stride(0),
        lse_output.stride(1),
        lse_output.stride(2),
        # Strides for all_masked (matches grid shape)
        all_masked_output.stride(0),
        all_masked_output.stride(1),
        # Strides for Q (B, S, H, D)
        query.stride(0),
        query.stride(1),
        query.stride(2),
        # Strides for K (B, S, H_kv, D)
        key.stride(0),
        key.stride(1),
        key.stride(2),
        # Strides for V (B, S, H_kv, D)
        value.stride(0),
        value.stride(1),
        value.stride(2),
        num_query_heads,
        num_kv_heads,
        seq_len,
        head_size,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=head_size,
        IS_CAUSAL=is_causal,
        WINDOW_SIZE_PAST=window_size[0],
        WINDOW_SIZE_FUTURE=window_size[1],
    )

    # MODIFIED: Return the all_masked_output tensor as a boolean tensor.
    # The int32 tensor (1 for masked, 0 for not) is converted to bool (True for masked, False for not).
    all_masked_scalar = torch.all(all_masked_output.to(torch.bool)).item()

    return output, lse_output, all_masked_scalar


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

    if dist.get_world_size(module.ulysses_pg) > 1:
        query_layer = SeqAllToAll4D.apply(module.ulysses_pg, query, module.scatter_idx, module.gather_idx)
        key_layer = SeqAllToAll4D.apply(module.ulysses_pg, key, module.scatter_idx, module.gather_idx)
        value_layer = SeqAllToAll4D.apply(module.ulysses_pg, value, module.scatter_idx, module.gather_idx)
    else:
        query_layer = query
        key_layer = key
        value_layer = value

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(query_layer.size(-1))
    comm = RingComm(module.ring_pg)

    out = None
    lse = None

    next_k, next_v = None, None

    original_window_size = window_size

    chunk_len = query_layer.shape[1]

    for step in range(comm.world_size):
        if window_size[0] != -1 and step > 1:
            break
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
            block_out, block_lse, all_masked = triton_attention_forward(
                query_layer,
                key,
                value,
                sinks,
                scale=softmax_scale,
                is_causal=causal and step == 0,
                window_size=adjusted_window_size,
            )

            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            key_layer = next_k
            value_layer = next_v

    out = out.to(query.dtype)
    if dist.get_world_size(module.ulysses_pg) > 1:
        output = SeqAllToAll4D.apply(module.ulysses_pg, out, module.gather_idx, module.scatter_idx)
    else:
        output = out

    return output


def moreh_gpt_attention_balanced(
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
    assert window_size == (-1, -1), "Balanced Ring Attention currently only supports full attention (no windowing)."

    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    if dist.get_world_size(module.ulysses_pg) > 1:
        query_layer = SeqAllToAll4D.apply(module.ulysses_pg, query, module.scatter_idx, module.gather_idx)
        key_layer = SeqAllToAll4D.apply(module.ulysses_pg, key, module.scatter_idx, module.gather_idx)
        value_layer = SeqAllToAll4D.apply(module.ulysses_pg, value, module.scatter_idx, module.gather_idx)
    else:
        query_layer = query
        key_layer = key
        value_layer = value

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(query_layer.size(-1))
    comm = RingComm(module.ring_pg)

    assert causal, "Balanced Ring Attention requires causal=True"
    block_seq_len = query_layer.shape[1] // 2
    query1 = query_layer[:, block_seq_len:]

    out = None
    lse = None

    next_k, next_v = None, None

    if comm.rank == 3:
        breakpoint()
    else:
        while True:
            pass
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(key_layer)
            next_v: torch.Tensor = comm.send_recv(value_layer)
            comm.commit()

        key, value = key_layer, value_layer

        if step == 0:
            block_out, block_lse, _ = triton_attention_forward(
                query_layer,
                key,
                value,
                sinks,
                scale=softmax_scale,
                is_causal=causal,
                window_size=window_size,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        elif step <= comm.rank:
            key0 = key[:, :block_seq_len]
            value0 = value[:, :block_seq_len]

            if key0.shape[1] > 0:
                block_out, block_lse, _ = triton_attention_forward(
                    query_layer,
                    key0,
                    value0,
                    sinks,
                    scale=softmax_scale,
                    is_causal=False,
                    window_size=window_size,
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        else:
            block_out, block_lse, _ = triton_attention_forward(
                query1,
                key,
                value,
                sinks,
                scale=softmax_scale,
                is_causal=False,
                window_size=window_size,
            )
            out, lse = update_out_and_lse(
                out,
                lse,
                block_out,
                block_lse,
                slice_=(slice(None), slice(block_seq_len, None)),
            )

        if step + 1 != comm.world_size:
            comm.wait()
            key_layer = next_k
            value_layer = next_v

    out = out.to(query.dtype)
    if dist.get_world_size(module.ulysses_pg) > 1:
        output = SeqAllToAll4D.apply(module.ulysses_pg, out, module.gather_idx, module.scatter_idx)
    else:
        output = out

    return output
