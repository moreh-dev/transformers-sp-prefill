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
    parser.add_argument("--sequence_parallel_size", type=int, default=1)
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
    sp_size = args.sequence_parallel_size
    
    if world_size != pp_size * sp_size:
        # If running single process for testing, adjust args or warn
        if world_size == 1:
            print("Running in single process mode (ignoring PP/SP size mismatch for testing logic)")
            pp_size = 1
            sp_size = 1
        else:
            raise ValueError(f"World size {world_size} != PP {pp_size} * SP {sp_size}")

    # Create groups
    # SP groups: Consecutive ranks (similar to TP)
    for i in range(pp_size):
        ranks = list(range(i * sp_size, (i + 1) * sp_size))
        if len(ranks) > 1:
            group = dist.new_group(ranks)
            if rank in ranks:
                set_sp_group(group)
        else:
            if rank in ranks:
                set_sp_group(None)

    # PP groups: Strided ranks
    for i in range(sp_size):
        ranks = list(range(i, world_size, sp_size))
        if len(ranks) > 1:
            group = dist.new_group(ranks)
            if rank in ranks:
                set_pp_group(group)
        else:
            if rank in ranks:
                set_pp_group(None)

    # Calculate ranks
    pp_rank = rank // sp_size
    sp_rank = rank % sp_size
    
    print(f"Rank {rank}: PP Rank {pp_rank}, SP Rank {sp_rank}")

    # Config
    config = DeepseekV3Config(
        pipeline_size=pp_size,
        pipeline_rank=pp_rank,
        sequence_parallel_size=sp_size,
        num_hidden_layers=4,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=1000,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        q_lora_rank=None,
        kv_lora_rank=64,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=64,
    )

    model = DeepseekV3ForCausalLM(config)
    
    # Input
    # Generate full input on all ranks (or just rank 0 and broadcast, but random with same seed is easier if seed set)
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (1, args.input_seq_len))
    
    # Split input_ids for SP
    if sp_size > 1:
        seq_len = args.input_seq_len
        sp_seq_len = seq_len // sp_size
        start_idx = sp_rank * sp_seq_len
        end_idx = start_idx + sp_seq_len
        input_ids = input_ids[:, start_idx:end_idx]
        print(f"Rank {rank}: Processing sequence chunk {start_idx}:{end_idx}")
    
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
