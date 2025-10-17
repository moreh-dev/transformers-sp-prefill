import unittest

import torch
import torch.nn.functional as F


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



def torch_attention_with_sinks_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
    is_causal: bool,
    window_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, num_query_heads, head_size = query.shape
    _, _, num_kv_heads, _ = key.shape

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    num_queries_per_kv = num_query_heads // num_kv_heads
    key = repeat_kv(key, num_queries_per_kv)
    value = repeat_kv(value, num_queries_per_kv)

    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    mask = torch.full((seq_len, seq_len), float("-inf"), device=query.device)

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

    expanded_sinks = sinks.view(1, -1, 1, 1).expand(batch_size, -1, seq_len, 1)
    combined_logits = torch.cat([attn_weights, expanded_sinks], dim=-1)

    m, _ = torch.max(combined_logits, dim=-1, keepdim=True)
    m = torch.where(torch.isinf(m), 0.0, m)

    p = torch.exp(combined_logits - m)
    l = torch.sum(p, dim=-1)
    lse = m.squeeze(-1) + torch.log(l)

    probs = p / l.unsqueeze(-1)

    scores = probs[..., :-1]

    scores = scores.to(query.dtype)

    output = torch.matmul(scores, value)

    output = output.transpose(1, 2).contiguous()

    return output, lse

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


class TestAttentionEquality(unittest.TestCase):
    def test_attention_kernel(self):
        BATCH_SIZE = 1
        SEQ_LEN = 2048
        NUM_QUERY_HEADS = 64
        NUM_KV_HEADS = 8
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
            {"name": "Sliding Window Both Dirs", "is_causal": True, "window_size": (128, 128)},
        ]

        for params in test_cases:
            is_causal_test = params["is_causal"]
            window_size_test = params["window_size"]
            case_name = params["name"]

            with self.subTest(case=case_name):
                q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)

                # Transpose inputs for Triton kernel to (B, S, H, D)
                q_triton = q_rope.transpose(1, 2).contiguous()
                k_triton = k_rope.transpose(1, 2).contiguous()
                v_triton = v.transpose(1, 2).contiguous()

                # Eager output is (B, S, H, D)
                eager_output, eager_lse = torch_attention_with_sinks_forward(
                    query=q_triton, key=k_triton, value=v_triton, sinks=sinks,
                    scale=scale, is_causal=is_causal_test, window_size=window_size_test,
                )

                # Triton output is (B, S, H, D)
                triton_output, triton_lse = vanila_attention(
                    q_rope, k_rope, v, sinks, scaling=scale,
                    is_causal=is_causal_test, window_size=window_size_test,
                )

                # --- 1. Attention Output Comparison ---
                self.assertTrue(eager_output.shape == triton_output.shape,
                                f"Shape mismatch: Eager={eager_output.shape}, Triton={triton_output.shape}")

                atol_attn, rtol_attn = 1e-2, 1e-2 # Tolerances for bfloat16
                self.assertTrue(
                    torch.allclose(eager_output, triton_output, atol=atol_attn, rtol=rtol_attn),
                    f"Attn Output Test failed for {case_name}. Max diff: {torch.max(torch.abs(eager_output - triton_output))}"
                )
                print(f"✅ Attn Output Test Passed for {case_name}")

                # --- 2. LSE Comparison ---
                self.assertTrue(eager_lse.shape == triton_lse.shape,
                                f"LSE Shape mismatch: Eager={eager_lse.shape}, Triton={triton_lse.shape}")

                atol_lse, rtol_lse = 1e-2, 1e-2
                self.assertTrue(
                    torch.allclose(eager_lse, triton_lse, atol=atol_lse, rtol=rtol_lse),
                    f"LSE Test failed for {case_name}. Max diff: {torch.max(torch.abs(eager_lse - triton_lse))}"
                )
                print(f"✅ LSE Test Passed for {case_name}\n")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
