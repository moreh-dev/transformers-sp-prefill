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

def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
):
    assert query.shape[1] % key.shape[1] == 0, "Number of attention heads must be divisible by number of key/value heads."
    num_key_value_groups = query.shape[1] // key.shape[1]

    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    sinks = sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights

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


        attn_output_local = xFuserLongContextAttention()(
            None,
            local_query,
            local_key,
            local_value,
            dropout_p=0.0,
            causal=False,
            #window_size=(128, 128),
            window_size=(-1, -1),
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

        sdpa_output = eager_attention_forward(
            full_query_rope,
            full_key_rope,
            full_value,
            #sinks=torch.zeros((num_heads), device=device, dtype=dtype),
            sinks = torch.full((num_heads,), float('-inf'), device=device, dtype=dtype),
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

            if False:
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
