import argparse
import os
import time

import torch
import torch.distributed as dist
import yunchang.comm.extract_local
from xfuser.core.distributed import (
    get_ring_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    init_distributed_environment,
    initialize_model_parallel,
)

from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_distributed():
    """Initializes the distributed environment."""
    if dist.is_initialized():
        return

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    world_size = dist.get_world_size()
    ring_size = world_size
    ulysses_size = world_size // ring_size

    init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())

    initialize_model_parallel(
        sequence_parallel_degree=dist.get_world_size(),
        ring_degree=ring_size,
        ulysses_degree=ulysses_size,
    )


def main(layout: str = "zigzag"):
    torch.manual_seed(2)

    setup_distributed()
    rank = dist.get_rank()
    device = f"cuda:{rank}"
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"--- Starting benchmark on {world_size} GPUs ---")

    model_id = "/root/.cache/huggingface/hub/gpt-oss-120b/"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    isl = 1024 * 2
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, isl)).to(device)
    input_ids = torch.arange(input_ids.numel()).view(input_ids.size()).to(device)

    dist.broadcast(input_ids, src=0)

    print(f"layout: {layout}, rank: {rank}, input_ids shape: {input_ids.shape}")
    os.environ["LAYOUT"] = layout
    extract_func = yunchang.comm.extract_local.EXTRACT_FUNC_DICT.get(layout)
    if extract_func is None:
        raise ValueError(
            f"Unknown layout '{layout}', available: {list(yunchang.comm.extract_local.EXTRACT_FUNC_DICT.keys())}"
        )

    local_input_ids = (
        extract_func(
            input_ids,
            get_sequence_parallel_rank(),
            world_size=get_sequence_parallel_world_size(),
            rd=get_ring_parallel_world_size(),
            ud=get_ulysses_parallel_world_size(),
            dim=1,
        )
        .detach()
        .clone()
    )

    if rank == 0:
        print(input_ids)
        print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device,
    )
    model.eval()

    model = torch.compile(model)

    if isl % world_size != 0:
        if rank == 0:
            print(f"Error: Input sequence length {isl} is not divisible by world_size {world_size}.")
        return

    osl = 1  # Example fixed output length

    # --- Warm-up Run ---
    if rank == 0:
        print(f"  -> Running warm-up for output length {osl}...")

    # All processes participate in generation
    _ = model.generate(
        local_input_ids,
        max_new_tokens=osl,
        do_sample=False,
    )
    # Use a barrier to synchronize all processes before starting measurement
    dist.barrier()

    if rank == 0:
        print("  -> Warm-up complete.", flush=True)

    # --- Measurement Run ---
    if rank == 0:
        print("  -> Running measurement...")

    # Synchronize before timing
    torch.cuda.synchronize()
    dist.barrier()

    start_time = time.time()

    out = model.generate(local_input_ids, max_new_tokens=osl, do_sample=False)

    # Synchronize after timing
    torch.cuda.synchronize()
    dist.barrier()

    end_time = time.time()

    # 4. Collect and print results only on the main process
    if rank == 0:
        elapsed_time = end_time - start_time
        tokens_per_second = (osl) / elapsed_time

        print("\n--- Benchmark Summary ---")
        print(f"Input Length:  {isl}")
        print(f"Output Length: {osl}")
        print(f"World Size:    {world_size} GPUs")
        print(f"Total Time:    {elapsed_time:.4f} seconds")
        print(f"Throughput:    {tokens_per_second:.2f} tokens/sec")
        print(f"output_ids: {out}, shape: {out.shape}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=str, default="basic", help="layout type: basic, zigzag, etc.")
    args = parser.parse_args()
    main(layout=args.layout)
