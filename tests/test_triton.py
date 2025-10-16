import unittest

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# --- Eager PyTorch implementation (Unchanged) ---

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
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

# --- MODIFIED Triton Kernel ---

@triton.jit
def kernel_attention_contiguous(
    output_ptr, lse_ptr, query_ptr, key_ptr, value_ptr, sinks_ptr,
    # Strides for (B, S, H, D) layout
    stride_out_batch, stride_out_seq, stride_out_head,
    stride_lse_batch, stride_lse_head, stride_lse_seq,
    stride_q_batch, stride_q_seq, stride_q_head,
    stride_k_batch, stride_k_seq, stride_k_head,
    stride_v_batch, stride_v_seq, stride_v_head,
    num_query_heads, num_kv_heads, seq_len, head_size,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr, WINDOW_SIZE_PAST: tl.constexpr, WINDOW_SIZE_FUTURE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    num_queries_per_kv = num_query_heads // num_kv_heads
    kv_head_idx = head_idx // num_queries_per_kv

    start_m = tl.program_id(2) * BLOCK_M

    offs_m = start_m + tl.arange(0, BLOCK_M) # Query sequence offsets
    offs_d = tl.arange(0, BLOCK_DMODEL)     # Head dimension offsets

    # Pointers for Q, K, V based on (B, S, H, D) layout
    q_ptrs = query_ptr + (batch_idx * stride_q_batch +
                          offs_m[:, None] * stride_q_seq +
                          head_idx * stride_q_head +
                          offs_d[None, :])

    k_ptrs_base = key_ptr + (batch_idx * stride_k_batch + kv_head_idx * stride_k_head)
    v_ptrs_base = value_ptr + (batch_idx * stride_v_batch + kv_head_idx * stride_v_head)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    sink_val = tl.load(sinks_ptr + head_idx).to(tl.float32)
    m_i = tl.full([BLOCK_M], sink_val, tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32) # exp(sink_val - sink_val) = 1

    q_mask = offs_m < seq_len

    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    q = (q * scale).to(q.dtype)

    end_n = seq_len
    if IS_CAUSAL:
        end_n = start_m + BLOCK_M

    start_n = 0
    while start_n < end_n:
        offs_n = start_n + tl.arange(0, BLOCK_N) # Key/Value sequence offsets

        # Load K.T for the dot product
        k_ptrs = k_ptrs_base + (offs_d[:, None] * 1 + offs_n[None, :] * stride_k_seq)

        k_mask = offs_n[None, :] < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        # Apply masking
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

        # Online Softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_j = tl.sum(p, 1)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]

        # Load V
        v_ptrs = v_ptrs_base + (offs_n[:, None] * stride_v_seq + offs_d[None, :])
        v_mask = offs_n[:, None] < seq_len
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_j
        m_i = m_ij

        start_n += BLOCK_N

    # Store LSE (Log-Sum-Exp)
    lse = m_i + tl.log(l_i)
    lse_ptrs = lse_ptr + (batch_idx * stride_lse_batch + head_idx * stride_lse_head + offs_m)
    tl.store(lse_ptrs, lse, mask=q_mask)

    # Store Attention Output
    acc = acc / l_i[:, None]
    out_ptrs = output_ptr + (batch_idx * stride_out_batch +
                            offs_m[:, None] * stride_out_seq +
                            head_idx * stride_out_head +
                            offs_d[None, :])
    tl.store(out_ptrs, acc, mask=q_mask[:, None])

# --- MODIFIED Python Wrapper for Triton Kernel ---
def triton_attention_forward(query, key, value, sinks, scale, is_causal: bool, window_size: tuple):
    # Input shapes: (B, S, H, D)
    batch_size, seq_len, num_query_heads, head_size = query.shape
    _, _, num_kv_heads, _ = key.shape

    output = torch.empty_like(query)
    # LSE shape: (B, H, S) as it's computed per head and per query token
    lse_output = torch.empty((batch_size, num_query_heads, seq_len), dtype=torch.float32, device=query.device)

    BLOCK_M = 16
    BLOCK_N = 64

    grid = (batch_size, num_query_heads, triton.cdiv(seq_len, BLOCK_M))

    kernel_attention_contiguous[grid](
        output, lse_output, query, key, value, sinks,
        # Strides for (B, S, H, D)
        output.stride(0), output.stride(1), output.stride(2),
        # Strides for LSE (B, H, S)
        lse_output.stride(0), lse_output.stride(1), lse_output.stride(2),
        # Strides for Q (B, S, H, D)
        query.stride(0), query.stride(1), query.stride(2),
        # Strides for K (B, S, H_kv, D)
        key.stride(0), key.stride(1), key.stride(2),
        # Strides for V (B, S, H_kv, D)
        value.stride(0), value.stride(1), value.stride(2),
        num_query_heads, num_kv_heads, seq_len, head_size,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_size,
        IS_CAUSAL=is_causal,
        WINDOW_SIZE_PAST=window_size[0], WINDOW_SIZE_FUTURE=window_size[1],
    )
    # Output is already (B, S, H, D), no transpose needed
    return output, lse_output

# --- MODIFIED Unit Test ---

class TestAttentionEquality(unittest.TestCase):
    def test_attention_kernel(self):
        BATCH_SIZE = 2
        SEQ_LEN = 128
        NUM_QUERY_HEADS = 16
        NUM_KV_HEADS = 4
        HEAD_SIZE = 64
        DTYPE = torch.bfloat16
        DEVICE = 'cuda'

        torch.manual_seed(0)
        # Eager implementation expects (B, H, S, D)
        q = torch.randn((BATCH_SIZE, NUM_QUERY_HEADS, SEQ_LEN, HEAD_SIZE), dtype=DTYPE, device=DEVICE)
        k = torch.randn((BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_SIZE), dtype=DTYPE, device=DEVICE)
        v = torch.randn((BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_SIZE), dtype=DTYPE, device=DEVICE)
        sinks = torch.randn(NUM_QUERY_HEADS, dtype=torch.float32, device=DEVICE)

        scale = 1.0 / (HEAD_SIZE ** 0.5)

        cos = torch.randn((SEQ_LEN, HEAD_SIZE), dtype=DTYPE, device=DEVICE)
        sin = torch.randn((SEQ_LEN, HEAD_SIZE), dtype=DTYPE, device=DEVICE)

        test_cases = [
            {"name": "Causal", "is_causal": True, "window_size": (-1, -1)},
            {"name": "Full", "is_causal": False, "window_size": (-1, -1)},
            {"name": "Sliding Window Causal", "is_causal": True, "window_size": (64, -1)},
            {"name": "Sliding Window Both Dirs", "is_causal": False, "window_size": (32, 32)},
        ]

        for params in test_cases:
            is_causal_test = params["is_causal"]
            window_size_test = params["window_size"]
            case_name = params["name"]

            with self.subTest(case=case_name):
                q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)

                # Eager output is (B, S, H, D)
                eager_output, eager_lse = eager_attention_forward(
                    query=q_rope, key=k_rope, value=v, sinks=sinks,
                    scaling=scale, is_causal=is_causal_test, window_size=window_size_test,
                )

                # Transpose inputs for Triton kernel to (B, S, H, D)
                q_triton = q_rope.transpose(1, 2).contiguous()
                k_triton = k_rope.transpose(1, 2).contiguous()
                v_triton = v.transpose(1, 2).contiguous()

                # Triton output is (B, S, H, D)
                triton_output, triton_lse = triton_attention_forward(
                    q_triton, k_triton, v_triton, sinks, scale,
                    is_causal=is_causal_test, window_size=window_size_test,
                )

                # --- 1. Attention Output Comparison ---
                self.assertTrue(eager_output.shape == triton_output.shape,
                                f"Shape mismatch: Eager={eager_output.shape}, Triton={triton_output.shape}")

                atol_attn, rtol_attn = 1e-1, 1e-2 # Tolerances for bfloat16
                self.assertTrue(
                    torch.allclose(eager_output, triton_output, atol=atol_attn, rtol=rtol_attn),
                    f"Attn Output Test failed for {case_name}. Max diff: {torch.max(torch.abs(eager_output - triton_output))}"
                )
                print(f"✅ Attn Output Test Passed for {case_name}")

                # --- 2. LSE Comparison ---
                self.assertTrue(eager_lse.shape == triton_lse.shape,
                                f"LSE Shape mismatch: Eager={eager_lse.shape}, Triton={triton_lse.shape}")

                atol_lse, rtol_lse = 1e-1, 1e-2
                self.assertTrue(
                    torch.allclose(eager_lse, triton_lse, atol=atol_lse, rtol=rtol_lse),
                    f"LSE Test failed for {case_name}. Max diff: {torch.max(torch.abs(eager_lse - triton_lse))}"
                )
                print(f"✅ LSE Test Passed for {case_name}\n")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
