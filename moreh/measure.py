import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_id):
    """Load tokenizer and model."""
    print(f"Loading model from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map='cuda:0',
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

    args = parser.parse_args()

    # Load model
    tokenizer, model = load_model(args.model)

    # Generate random input
    print(f"\nGenerating random input with length {args.input_length}...")
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, args.input_length)).to('cuda:0')

    # Measure performance
    results = measure_performance(model, input_ids, args.output_lengths)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
