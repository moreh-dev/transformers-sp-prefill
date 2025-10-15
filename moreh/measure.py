import argparse
import os
import time
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoTokenizer, AutoModelForCausalLM


def init_distributed(sp_size, pp_size):
    """Initialize distributed environment and create device mesh."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Distributed mode
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        if world_size != sp_size * pp_size:
            raise ValueError(
                f"World size ({world_size}) must equal SP size ({sp_size}) * PP size ({pp_size})"
            )
        torch.cuda.set_device(local_rank)
        mesh = torch.arange(world_size).reshape(sp_size, pp_size)
        device_mesh = DeviceMesh(device_type="cuda", mesh=mesh, mesh_dim_names=("sp", "pp"))
        device = torch.device(f"cuda:{local_rank}")
        print(f"Rank {rank}/{world_size} | Local rank: {local_rank} | Device: {device}")
        print(f"Device mesh created: {device_mesh}")
        return device, device_mesh
    else:
        # Single device mode
        if sp_size != 1 or pp_size != 1:
            raise ValueError(
                "Single device mode requires sp_size=1 and pp_size=1. Use torchrun for distributed execution."
            )
        device = torch.device("cuda:0")
        print(f"Running in single device mode on {device}")
        return device, None


def load_model(model_id, device):
    """Load tokenizer and model."""
    print(f"Loading model from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device,
    )

    model = torch.compile(model)
    print("Model loaded and compiled successfully.")

    return tokenizer, model


def measure_performance(model, input_ids, output_sequence_lengths):
    """Measure generation performance for different output sequence lengths."""
    results = {}

    for osl in output_sequence_lengths:
        print(f"\n[Output Sequence Length: {osl}]")

        # Warm-up
        print("  -> Running warm-up...")
        _ = model.generate(
            input_ids,
            max_new_tokens=osl,
            do_sample=False,
        )
        torch.cuda.synchronize()
        print("  -> Warm-up complete.")

        # Measurement
        print("  -> Running measurement...")
        torch.cuda.synchronize()
        start_time = time.time()

        outputs = model.generate(
            input_ids,
            max_new_tokens=osl,
            do_sample=False
        )

        torch.cuda.synchronize()
        end_time = time.time()

        elapsed_time = end_time - start_time
        results[osl] = elapsed_time
        print(f"  -> Generation took: {elapsed_time:.4f} seconds")

    return results


def print_summary(results):
    """Print benchmark summary."""
    print("\n--- Benchmark Summary ---")
    for osl, t in results.items():
        print(f"Tokens: {osl:<4} | Time: {t:.4f} sec")


def main():
    parser = argparse.ArgumentParser(description="Measure LLM generation performance")
    parser.add_argument(
        "--model",
        type=str,
        default="/root/.cache/huggingface/hub/gpt-oss-120b/",
        help="Model ID or path to load"
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=2048,
        help="Input sequence length"
    )
    parser.add_argument(
        "--output-lengths",
        type=int,
        nargs="+",
        default=[1],
        help="Output sequence lengths to test (e.g., 1 2 4 8 16)"
    )
    parser.add_argument(
        "--sp-size",
        type=int,
        default=1,
        help="Sequence parallel size"
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help="Pipeline parallel size"
    )

    args = parser.parse_args()

    # Initialize distributed environment
    device, device_mesh = init_distributed(args.sp_size, args.pp_size)

    # Load model
    tokenizer, model = load_model(args.model, device)

    # Generate random input
    print(f"\nGenerating random input with length {args.input_length}...")
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, args.input_length)).to(device)

    # Measure performance
    results = measure_performance(model, input_ids, args.output_lengths)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
