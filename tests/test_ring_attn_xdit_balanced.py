import os
import unittest

import torch
import torch.distributed as dist
import yunchang.comm.extract_local
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from yunchang.kernels import AttnType

from src.transformers.models.gpt_oss.ring_attention import (
    all_gather_zigzag,
    moreh_gpt_attention_balanced,
    torch_attention_with_sinks_forward,
)


class RingAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(2)

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()
        torch.cuda.set_device(cls.rank)

        cls.ring_size = cls.world_size
        cls.ulysses_size = cls.world_size // cls.ring_size

        init_distributed_environment(rank=cls.rank, world_size=cls.world_size)
        initialize_model_parallel(
            sequence_parallel_degree=cls.world_size,
            ring_degree=cls.ring_size,
            ulysses_degree=cls.ulysses_size,
        )
        cls.sp_size = get_sequence_parallel_world_size()
        cls.sp_rank = get_sequence_parallel_rank()

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def test_ring_attention_vs_sdpa(self):
        rank = int(os.environ["LOCAL_RANK"])
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

        q_total_trans = q_total.transpose(1, 2).contiguous()
        k_total_trans = k_total.transpose(1, 2).contiguous()
        v_total_trans = v_total.transpose(1, 2).contiguous()

        seq_dim = 2

        q = q_total.chunk(self.world_size, dim=seq_dim)[rank].contiguous()
        k = k_total.chunk(self.world_size, dim=seq_dim)[rank].contiguous()
        v = v_total.chunk(self.world_size, dim=seq_dim)[rank].contiguous()

        q_reordered = (
            yunchang.comm.extract_local.EXTRACT_FUNC_DICT["zigzag"](
                q_total, rank, world_size=self.sp_size, rd=self.ring_size, ud=self.ulysses_size, dim=seq_dim
            )
            .detach()
            .clone()
        )
        k_reordered = (
            yunchang.comm.extract_local.EXTRACT_FUNC_DICT["zigzag"](
                k_total, rank, world_size=self.sp_size, rd=self.ring_size, ud=self.ulysses_size, dim=seq_dim
            )
            .detach()
            .clone()
        )
        v_reordered = (
            yunchang.comm.extract_local.EXTRACT_FUNC_DICT["zigzag"](
                v_total, rank, world_size=self.sp_size, rd=self.ring_size, ud=self.ulysses_size, dim=seq_dim
            )
            .detach()
            .clone()
        )

        assert q.shape == q_reordered.shape
        assert k.shape == k_reordered.shape
        assert v.shape == v_reordered.shape

        q_g = get_sp_group().all_gather(q, dim=seq_dim)
        k_g = get_sp_group().all_gather(k, dim=seq_dim)
        v_g = get_sp_group().all_gather(v, dim=seq_dim)

        q_reordered_g = all_gather_zigzag(q_reordered, self.ring_size, self.ulysses_size, dim=seq_dim)
        k_reordered_g = all_gather_zigzag(k_reordered, self.ring_size, self.ulysses_size, dim=seq_dim)
        v_reordered_g = all_gather_zigzag(v_reordered, self.ring_size, self.ulysses_size, dim=seq_dim)

        assert torch.allclose(q_g, q_reordered_g)
        assert torch.allclose(k_g, k_reordered_g)
        assert torch.allclose(v_g, v_reordered_g)

        causal = True
        softmax_scale = head_dim**-0.5
        sinks = torch.full((num_heads,), float("-inf"), device=device, dtype=dtype)
        sinks = torch.randn_like(sinks)

        window_size = (128, -1)

        vanila_output = torch_attention_with_sinks_forward(
            q_total_trans,
            k_total_trans,
            v_total_trans,
            sinks,
            scale=softmax_scale,
            window_size=window_size,
            is_causal=causal,
        )[0]

        sdpa_output = vanila_output

        attn_class = xFuserLongContextAttention(attn_type=AttnType.TORCH)
        attn_output_local = moreh_gpt_attention_balanced(
            attn_class,
            q_reordered,
            k_reordered,
            v_reordered,
            sinks,
            dropout_p=0.0,
            window_size=window_size,
            causal=causal,
        )

        attn_output = all_gather_zigzag(attn_output_local, self.ring_size, self.ulysses_size, dim=1)

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
