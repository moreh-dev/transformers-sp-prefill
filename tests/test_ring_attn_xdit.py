import unittest

import torch
import torch.distributed as dist
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from yunchang.kernels import AttnType

from src.transformers.models.gpt_oss.ring_attention import moreh_gpt_attention, torch_attention_with_sinks_forward


class RingAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(2)

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
        rank = get_sequence_parallel_rank()
        device = torch.device(rank)
        batch_size = 1
        num_heads = 64
        kv_num_heads = 8
        total_seq_len = 2048
        head_dim = 64
        dtype = torch.bfloat16

        q_total = torch.randn(batch_size, num_heads, total_seq_len, head_dim, device=device, dtype=dtype)
        k_total = torch.randn(batch_size, kv_num_heads, total_seq_len, head_dim, device=device, dtype=dtype)
        v_total = torch.randn(batch_size, kv_num_heads, total_seq_len, head_dim, device=device, dtype=dtype)

        dist.broadcast(q_total, src=0)
        dist.broadcast(k_total, src=0)
        dist.broadcast(v_total, src=0)

        seq_dim = 2

        q_total_trans = q_total.transpose(1, 2).contiguous()
        k_total_trans = k_total.transpose(1, 2).contiguous()
        v_total_trans = v_total.transpose(1, 2).contiguous()

        q = q_total.chunk(self.world_size, dim=seq_dim)[self.rank].contiguous()
        k = k_total.chunk(self.world_size, dim=seq_dim)[self.rank].contiguous()
        v = v_total.chunk(self.world_size, dim=seq_dim)[self.rank].contiguous()

        q_trans = q.transpose(1, 2).contiguous()
        k_trans = k.transpose(1, 2).contiguous()
        v_trans = v.transpose(1, 2).contiguous()

        causal = True
        window_size = (128, -1)
        softmax_scale = head_dim**-0.5
        sinks = torch.full((num_heads,), float("-inf"), device=device, dtype=dtype)
        sinks = torch.randn_like(sinks)

        vanila_output = torch_attention_with_sinks_forward(
            q_total_trans,
            k_total_trans,
            v_total_trans,
            sinks,
            scale=softmax_scale,
            is_causal=causal,
            window_size=window_size,
        )[0]

        sdpa_output = vanila_output

        attn_class = xFuserLongContextAttention(attn_type=AttnType.TORCH)
        attn_output_local = moreh_gpt_attention(
            attn_class,
            q,
            k,
            v,
            sinks,
            dropout_p=0.0,
            causal=causal,
            window_size=window_size,
        )

        attn_output = get_sp_group().all_gather(attn_output_local, dim=1)

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
                "The outputs of xFuser Ring Attention and PyTorch SDPA do not match. [See] detailed analysis above.",
            )


if __name__ == "__main__":
    unittest.main()
