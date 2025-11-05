import os
import unittest

import torch
import torch.distributed as dist


PROCESS_GROUP = None


class MockProcessGroup:
    def __init__(self):
        self.RING_PG = None
        self.ULYSSES_PG = None


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


def all_gather_zigzag_inverse(local_tensor, rd, ud, dim=1, *args, **kwargs):
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


class TestZigzagInverse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            os.environ["LOCAL_RANK"] = "0"

        dist.init_process_group("nccl")

        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()

        local_rank = int(os.environ["LOCAL_RANK"])
        cls.device = f"cuda:{local_rank}"
        torch.cuda.set_device(cls.device)

        # only tested ring only
        cls.rd = cls.world_size
        cls.ud = 1

        r_rank = cls.rank // cls.ud
        u_rank = cls.rank % cls.ud

        ring_ranks = [i * cls.ud + u_rank for i in range(cls.rd)]
        ulysses_ranks = [r_rank * cls.ud + i for i in range(cls.ud)]

        cls.ring_pg = dist.new_group(ring_ranks)
        cls.ulysses_pg = dist.new_group(ulysses_ranks)

        global PROCESS_GROUP
        PROCESS_GROUP = MockProcessGroup()
        PROCESS_GROUP.RING_PG = cls.ring_pg
        PROCESS_GROUP.ULYSSES_PG = cls.ulysses_pg

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def test_extract_and_inverse_all_gather(self):
        seq_dim = 2  # (bs, num_heads, seq_len, head_dim)
        bs, num_heads, seq_len_total, head_dim = 1, 64, 2048, 64

        assert seq_len_total % (2 * self.rd) == 0
        assert seq_len_total % self.world_size == 0

        torch.manual_seed(22)
        if self.rank == 0:
            global_tensor = torch.randn(
                (bs, num_heads, seq_len_total, head_dim), device=self.device, dtype=torch.float32
            )
        else:
            global_tensor = torch.empty(
                (bs, num_heads, seq_len_total, head_dim), device=self.device, dtype=torch.float32
            )

        dist.broadcast(global_tensor, src=0)

        local_tensor = zigzag_extract_local_patched(
            global_tensor, self.rank, self.world_size, self.rd, self.ud, dim=seq_dim
        )

        expected_local_seq_len = seq_len_total // self.world_size
        expected_shape = (bs, num_heads, expected_local_seq_len, head_dim)
        self.assertEqual(local_tensor.shape, expected_shape)

        restored_global_tensor = all_gather_zigzag_inverse(local_tensor, self.rd, self.ud, dim=seq_dim)

        self.assertEqual(global_tensor.shape, restored_global_tensor.shape)

        self.assertTrue(
            torch.allclose(global_tensor, restored_global_tensor, atol=1e-6),
            "Zigzag Extract and All-Gather Inverse test failed!",
        )


if __name__ == "__main__":
    unittest.main()
