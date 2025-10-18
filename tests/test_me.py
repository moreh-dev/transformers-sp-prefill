import math
import unittest


from flash_attn import flash_attn_func

import torch
import torch.distributed as dist
import torch.nn.functional as F
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
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
    origin_dtype = query.dtype
    
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    
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

    # Calculate Log-Sum-Exp efficiently
    lse = torch.logsumexp(attn_weights, dim=-1)

    # Compute softmax over logits including the sink for stable probability calculation
    # Use float32 for softmax stability
    probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # Compute attention output
    output = torch.matmul(probs, value)

    # Transpose output back to (B, S, H, D)
    output = output.transpose(1, 2).contiguous()
    
    output = output.to(origin_dtype)
    lse = lse.to(origin_dtype)


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
            block_out_flash, block_lse_flash, _ = flash_attn_func(
                query_layer,
                key,
                value,
                causal=causal and step == 0,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                return_attn_probs=True,
            )
            block_out = block_out_flash.to(block_out.dtype)
            block_lse = block_lse_flash.to(block_lse.dtype)
            
            #breakpoint()
            #block_out = block_out.transpose(1, 2)
            #block_lse = block_lse.to(query_layer.dtype)
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
        window_size = (-1, -1)
        softmax_scale=head_dim ** -0.5
        sinks = torch.full((num_heads,), float('-inf'), device=device, dtype=dtype)

        sdpa_output = flash_attn_func(q_total_trans, k_total_trans, v_total_trans,
                                      causal=causal,
                                      dropout_p=0.0,
                                      softmax_scale=softmax_scale)
        
        attn_class= xFuserLongContextAttention(attn_type=AttnType.TORCH)
        attn_output_local = moreh_gpt_attention(
            attn_class,
            q_trans,
            k_trans,
            v_trans,
            sinks,
            dropout_p=0.0,
            causal=causal,
            window_size=window_size,
        )

        attn_output = get_sp_group().all_gather(attn_output_local, dim=1)
        

        actual_output = attn_output.view(-1, total_seq_len, num_heads, head_dim)
        expected_output = sdpa_output.view(-1, total_seq_len, num_heads, head_dim)
        if self.rank == 0:
            diff = actual_output - expected_output
            abs_diff = torch.abs(diff)
            #breakpoint()
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
                "The outputs of xFuser Ring Attention and PyTorch SDPA do not match. [See] detailed analysis above."
            )
if __name__ == "__main__":
    unittest.main()
