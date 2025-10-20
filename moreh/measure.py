import argparse
import gc
import logging
import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import _ScheduleForwardOnly
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
        logger.info(f"Rank {rank}/{world_size} | Local rank: {local_rank} | Device: {device}")
        logger.info(f"Device mesh created: {device_mesh}")
        return device, device_mesh
    else:
        # Single device mode
        if sp_size != 1 or pp_size != 1:
            raise ValueError(
                "Single device mode requires sp_size=1 and pp_size=1. Use torchrun for distributed execution."
            )
        device = torch.device("cuda:0")
        logger.info(f"Running in single device mode on {device}")
        return device, None


def load_model(model_id, device, device_map=None):
    """Load tokenizer and model."""
    logger.info(f"Loading model from {model_id}...")

    # Use provided device_map, or default to single device
    if device_map is None:
        device_map = device

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device_map,
    )

    return tokenizer, model


def get_num_layers(model):
    """Get the number of transformer layers from the model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return len(model.model.layers)
    else:
        raise ValueError("Unable to determine number of layers from model")


def generate_module_names_for_stage(num_layers, pp_size, stage_idx):
    """
    Generate module names for a specific pipeline stage.

    Args:
        num_layers: Total number of transformer layers
        pp_size: Pipeline parallel size
        stage_idx: Index of the current stage

    Returns:
        Tuple of (stage_modules, unused_modules):
            - stage_modules: List of module names needed for this stage
            - unused_modules: List of module names not needed for this stage
    """
    assert num_layers % pp_size == 0, f"num_layers ({num_layers}) must be divisible by pp_size ({pp_size})"

    layers_per_stage = num_layers // pp_size
    start_layer = stage_idx * layers_per_stage

    stage_modules = []
    unused_modules = []

    # All stages must have rotary embedding for position_embeddings
    stage_modules.append('model.rotary_emb')

    # Embedding tokens
    if stage_idx == 0:
        stage_modules.append('model.embed_tokens')
    else:
        unused_modules.append('model.embed_tokens')

    # Add transformer layers for this stage and mark others as unused
    for layer_idx in range(num_layers):
        if start_layer <= layer_idx < start_layer + layers_per_stage:
            stage_modules.append(f'model.layers.{layer_idx}')
        else:
            unused_modules.append(f'model.layers.{layer_idx}')

    # Last stage includes norm and lm_head
    if stage_idx == pp_size - 1:
        stage_modules.append('model.norm')
        stage_modules.append('lm_head')
    else:
        unused_modules.append('model.norm')
        unused_modules.append('lm_head')

    return stage_modules, unused_modules


def build_stage_from_modules(
    model: nn.Module,
    stage_idx: int,
    num_stages: int,
    module_names: list[str],
    device: torch.device,
    pp_group,
) -> tuple[PipelineStage, nn.Module]:
    """
    Build a pipeline stage from a whole model by keeping only specified modules.

    Note: This function modifies the model in-place by setting unused modules to None.
    For production use with large models, this approach avoids the memory overhead of deepcopy.

    Args:
        model: The complete model
        stage_idx: Index of this stage
        num_stages: Total number of stages
        module_names: List of module names to keep in this stage
        device: Device to place the model on
        pp_group: Process group for pipeline parallelism

    Returns:
        Tuple of (PipelineStage, model_chunk)
    """

    class IdentityModule(nn.Module):
        """Module that returns its first input unchanged (identity function)."""
        def forward(self, x, *args, **kwargs):
            return x

    modules_to_keep = set(module_names)

    # Handle model.layers (the transformer layers)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers_to_keep = set()
        for name in modules_to_keep:
            if name.startswith('model.layers.'):
                parts = name.split('.')
                if len(parts) >= 3 and parts[2].isdigit():
                    layers_to_keep.add(int(parts[2]))

        if layers_to_keep:
            # Create new ModuleList with only the layers we need
            original_layer_count = len(model.model.layers)
            new_layers = nn.ModuleList([
                layer for i, layer in enumerate(model.model.layers)
                if i in layers_to_keep
            ])
            model.model.layers = new_layers
        else:
            # No layers needed for this stage
            original_layer_count = len(model.model.layers)
            model.model.layers = nn.ModuleList()

        # Handle other model.* modules (embed_tokens, norm, rotary_emb, etc.)
        for module_name in list(model.model._modules.keys()):
            if module_name == 'layers':
                continue
            full_name = f'model.{module_name}'
            if full_name not in modules_to_keep:
                setattr(model.model, module_name, IdentityModule())

    # Handle top-level modules (like lm_head)
    for module_name in list(model._modules.keys()):
        if module_name not in modules_to_keep and not any(name.startswith(f"{module_name}.") for name in modules_to_keep):
            setattr(model, module_name, IdentityModule())

    # Create PipelineStage
    stage = PipelineStage(
        model,
        stage_idx,
        num_stages,
        device,
        group=pp_group,
    )

    return stage, model


def apply_pipeline_parallel(model, device_mesh, device, stage_modules):
    """
    Apply pipeline parallelism to the model.

    Args:
        model: The model to parallelize
        device_mesh: DeviceMesh with 'pp' dimension
        device: Device to place model on
        stage_modules: Modules names in this stage

    Returns:
        Tuple of (pp_schedule, model_parts, has_first_stage, has_last_stage)
    """
    pp_mesh = device_mesh["pp"]
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()

    num_layers = get_num_layers(model)

    stage_idx = pp_rank

    stage, model_chunk = build_stage_from_modules(
        model,
        stage_idx,
        pp_size,
        stage_modules,
        device,
        pp_mesh.get_group(),
    )

    pp_schedule = _ScheduleForwardOnly(
        stage,
        n_microbatches=pp_size,
    )

    has_first_stage = (stage_idx == 0)
    has_last_stage = (stage_idx == pp_size - 1)

    # Return single stage in a list for consistency with torchtitan API
    return pp_schedule, [model_chunk], has_first_stage, has_last_stage


def measure_performance(model, input_ids, output_sequence_lengths):
    """Measure generation performance for different output sequence lengths."""
    results = {}

    for osl in output_sequence_lengths:
        logger.info(f"Output Sequence Length: {osl}")

        # Warm-up
        logger.info("Running warm-up...")
        _ = model.generate(
            input_ids,
            max_new_tokens=osl,
            do_sample=False,
        )
        torch.cuda.synchronize()
        logger.info("Warm-up complete.")

        # Measurement
        logger.info("Running measurement...")
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
        logger.info(f"Generation took: {elapsed_time:.4f} seconds")

    return results


def print_summary(results):
    """Print benchmark summary."""
    logger.info("--- Benchmark Summary ---")
    for osl, t in results.items():
        logger.info(f"Tokens: {osl:<4} | Time: {t:.4f} sec")


def main():
    torch.manual_seed(2025)
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

    if args.pp_size > 1:
        config = AutoConfig.from_pretrained(args.model)
        num_layers = config.num_hidden_layers

        # Get pp_rank
        pp_mesh = device_mesh["pp"]
        pp_rank = pp_mesh.get_local_rank()

        # Generate module names for this stage
        stage_modules, unused_modules = generate_module_names_for_stage(
            num_layers, args.pp_size, pp_rank
        )

        # Create device map: stage modules on device, unused on meta
        device_map = {**{x: device for x in stage_modules}, **{y: 'meta' for y in unused_modules}}

        # Load model with device_map
        tokenizer, model = load_model(args.model, device, device_map=device_map)

        (
            pp_schedule,
            model_parts,
            has_first_stage,
            has_last_stage,
        ) = apply_pipeline_parallel(
            model,
            device_mesh,
            device,
            stage_modules,
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

        with torch.autograd.grad_mode.inference_mode():
            if has_first_stage:
                input_ids = torch.randint(0, tokenizer.vocab_size, (args.pp_size, args.input_length)).to(device)
                output = pp_schedule.step(input_ids)
            else:
                output = pp_schedule.step()
        logger.info(f'output ({type(output)}): {output}')

    else:
        # Single device mode
        tokenizer, model = load_model(args.model, device)

        model = torch.compile(model)
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, args.input_length)).to(device)
        results = measure_performance(model, input_ids, args.output_lengths)
        print_summary(results)


if __name__ == "__main__":
    main()
