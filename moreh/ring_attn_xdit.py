import os
import time

import torch
import torch.distributed as dist
from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_distributed():
    """Initializes the distributed environment."""
    if dist.is_initialized():
        return

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]) % 8)

    world_size = dist.get_world_size()
    ring_size = world_size
    ulysses_size = world_size // ring_size

    init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())

    initialize_model_parallel(
        sequence_parallel_degree=dist.get_world_size(),
        ring_degree=ring_size,
        ulysses_degree=ulysses_size,
    )


def main():
    # 1. Set up the distributed environment
    torch.manual_seed(2)

    setup_distributed()
    rank = dist.get_rank()
    device = f"cuda:{rank % 8}"
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"--- Starting benchmark on {world_size} GPUs ---")

    model_id = "/root/.cache/huggingface/hub/gpt-oss-120b/"

    # Each process loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Load the model onto the specific GPU for each process
    # The model code itself MUST have the Ring Attention logic implemented.
    # We are NOT using device_map here.

    isl = 1024 * 8
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, isl)).to(device)
    dist.broadcast(input_ids, src=0)

    local_input_ids = torch.empty(1, isl // world_size, dtype=input_ids.dtype).to(device)

    if rank == 0:
        print(input_ids)
        print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device,
    )
    model.eval()

    # num_layers = len(model.model.layers)
    # model.model.layers = model.model.layers[:num_layers//4]
    # del(model.model.layers[num_layers//4:])
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()

    # torch.compile can be added here if desired, but test without it first.
    model = torch.compile(model)

    # Input sequence length must be divisible by world_size for Ring Attention
    if isl % world_size != 0:
        if rank == 0:
            print(f"Error: Input sequence length {isl} is not divisible by world_size {world_size}.")
        return

    if rank == 0:
        dist.scatter(local_input_ids, list(input_ids.chunk(world_size, 1)), src=0)
    else:
        dist.scatter(local_input_ids, None, src=0)

    # The output sequence length for the benchmark
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

    _ = model.generate(local_input_ids, max_new_tokens=osl, do_sample=False)

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
        print(f"output_ids: {_}", flush=True)


if __name__ == "__main__":
    main()
