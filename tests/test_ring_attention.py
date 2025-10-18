import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
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
    #sinks: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
):
    assert query.shape[1] % key.shape[1] == 0, "Number of attention heads must be divisible by number of key/value heads."
    num_key_value_groups = query.shape[1] // key.shape[1]

    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    probs = F.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)

    attn_weights = nn.functional.dropout(probs, p=dropout, training=False)
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
        q_total = torch.load('q.pt')
        k_total = torch.load('k.pt')
        v_total = torch.load('v.pt')
        
        #batch_size, total_seq_len, num_heads, head_dim = q_total.shape
        batch_size, num_heads, total_seq_len, head_dim = q_total.shape
        seq_dim = 2

        rank = get_sequence_parallel_rank()
        q_total = q_total.to(torch.device(rank))
        k_total = k_total.to(torch.device(rank))
        v_total = v_total.to(torch.device(rank))

        q_total_trans = q_total.transpose(1, 2).contiguous()
        k_total_trans = k_total.transpose(1, 2).contiguous()
        v_total_trans = v_total.transpose(1, 2).contiguous()


        q = q_total.chunk(self.world_size, dim=seq_dim)[self.rank].contiguous()
        k = k_total.chunk(self.world_size, dim=seq_dim)[self.rank].contiguous()
        v = v_total.chunk(self.world_size, dim=seq_dim)[self.rank].contiguous()
        
        q.trans = q.transpose(1, 2).contiguous()
        k.trans = k.transpose(1, 2).contiguous()
        v.trans = v.transpose(1, 2).contiguous()


        causal = True
        window_size = (-1, -1)
        softmax_scale=head_dim ** -0.5

        attn_output_local = xFuserLongContextAttention()(
            None,
            q.trans,
            k.trans,
            v.trans,
            dropout_p=0.0,
            causal=causal,
            softmax_scale=softmax_scale,
            window_size=window_size,
        )

        attn_output = get_sp_group().all_gather(attn_output_local, dim=1)

        from flash_attn import flash_attn_func
        sdpa_output = flash_attn_func(q_total_trans, k_total_trans, v_total_trans,
                                      causal=causal,
                                      dropout_p=0.0,
                                      softmax_scale=softmax_scale)

        actual_output = attn_output.view(-1, total_seq_len, num_heads, head_dim)
        expected_output = sdpa_output.view(-1, total_seq_len, num_heads, head_dim)
        if self.rank == 0:
            atol = 1e-2
            rtol = 1e-2

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
