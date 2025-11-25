import os
import sys
import torch
import torch.distributed as dist
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM, set_tp_group, set_pp_group

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_size", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--input_seq_len", type=int, default=128)
    args = parser.parse_args()

    # Init distributed
    if 'RANK' in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    else:
        # Local test mode or single process
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    pp_size = args.pipeline_size
    tp_size = args.tensor_parallel_size
    
    if world_size != pp_size * tp_size:
        # If running single process for testing, adjust args or warn
        if world_size == 1:
            print("Running in single process mode (ignoring PP/TP size mismatch for testing logic)")
            pp_size = 1
            tp_size = 1
        else:
            raise ValueError(f"World size {world_size} != PP {pp_size} * TP {tp_size}")

    # Create groups
    # TP groups: Consecutive ranks
    for i in range(pp_size):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        if len(ranks) > 1:
            group = dist.new_group(ranks)
            if rank in ranks:
                set_tp_group(group)
        else:
            if rank in ranks:
                set_tp_group(None) # None implies world or self? My code uses None -> 1.

    # PP groups: Strided ranks
    for i in range(tp_size):
        ranks = list(range(i, world_size, tp_size))
        if len(ranks) > 1:
            group = dist.new_group(ranks)
            if rank in ranks:
                set_pp_group(group)
        else:
            if rank in ranks:
                set_pp_group(None)

    # Calculate ranks
    pp_rank = rank // tp_size
    tp_rank = rank % tp_size
    
    print(f"Rank {rank}: PP Rank {pp_rank}, TP Rank {tp_rank}")

    # Config
    config = DeepseekV3Config()
    config.pipeline_size = pp_size
    config.pipeline_rank = pp_rank
    config.tensor_parallel_size = tp_size
    
    # Use small model for testing
    config.num_hidden_layers = 4
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.vocab_size = 1000
    config.n_routed_experts = 4
    config.num_experts_per_tok = 2
    config.n_shared_experts = 1
    
    # Disable some features to simplify
    config.q_lora_rank = None 
    config.kv_lora_rank = 64
    config.qk_rope_head_dim = 32
    config.qk_nope_head_dim = 32
    config.v_head_dim = 64

    model = DeepseekV3ForCausalLM(config)
    
    # Input
    input_ids = torch.randint(0, config.vocab_size, (1, args.input_seq_len))
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        model.to(device)
        input_ids = input_ids.to(device)

    # Run
    print(f"Rank {rank}: Running forward...")
    with torch.no_grad():
        output = model(input_ids)

    if pp_rank == pp_size - 1:
        if output.logits is not None:
            print(f"Rank {rank}: Output shape {output.logits.shape}")
        else:
            print(f"Rank {rank}: Output logits is None (Unexpected for last stage)")
    else:
        print(f"Rank {rank}: Intermediate Stage Completed")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
