import argparse
import os
import time

import torch
import torch.distributed as dist
import yunchang.comm.extract_local
from xfuser.core.distributed import (
    get_ring_parallel_world_size,
    get_sp_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    init_distributed_environment,
    initialize_model_parallel,
)

from transformers.models.gpt_oss.ring_attention import all_gather_zigzag

from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import torch

import os
import torch
# torch.distributed 및 사용자 정의 함수 (get_sp_group 등)가 
# 이 파일의 스코프 내에 정의되어 있거나 import 되었다고 가정합니다.
# 
# 예: 
# from your_parallel_utils import get_sp_group, all_gather_zigzag, get_sequence_parallel_rank

import os
import torch
# torch.distributed 및 사용자 정의 함수 (get_sp_group 등)가 
# 이 파일의 스코프 내에 정의되어 있거나 import 되었다고 가정합니다.
# 
# 예: 
# from your_parallel_utils import get_sp_group, all_gather_zigzag, get_sequence_parallel_rank

def save_tensor(tensor, path, seq_dim=1):
    
    # --- 1. 공통 설정 로직 ---
    os.makedirs(os.path.dirname(path), exist_ok=True)
    layout = os.environ.get("LAYOUT", None)
    assert layout in ["basic", "zigzag"], f"Unknown LAYOUT '{layout}'"
    zigzag = layout == "zigzag"

    sp_group = get_sp_group()

    # --- 2. DynamicCache 여부 확인 ---
    is_dynamic_cache = (
        hasattr(tensor, 'layers') and 
        isinstance(tensor.layers, (list, tuple))
    )

    if is_dynamic_cache and len(tensor.layers) > 0:
        first_layer_cache = tensor.layers[0]
        is_dynamic_cache = (
            hasattr(first_layer_cache, 'keys') and 
            hasattr(first_layer_cache, 'values')
        )
    elif is_dynamic_cache: 
        print(f"Warning: DynamicCache at '{path}' has 0 layers. Nothing to save.")
        return 
    else:
        is_dynamic_cache = False 

    # --- 3. 분기 처리 ---

    # SP가 없는 경우, 텐서/캐시를 그대로 저장
    if sp_group.world_size == 0:
        torch.save(tensor, path)
        return

    # --- SP가 활성화된 경우 (world_size > 0) ---

    if is_dynamic_cache:
        # --- 3-A. DynamicCache일 경우 ---
        
        # (모든 랭크) 각 레이어의 K, V 텐서를 all-gather
        gathered_layers_data = []
        for layer_cache in tensor.layers:
            k_tensor = layer_cache.keys
            v_tensor = layer_cache.values
            
            gathered_k = None
            if k_tensor is not None:
                if zigzag:
                    gathered_k = all_gather_zigzag(k_tensor.contiguous(), dim=seq_dim)
                else:
                    gathered_k = sp_group.all_gather(k_tensor.contiguous(), dim=seq_dim)

            gathered_v = None
            if v_tensor is not None:
                if zigzag:
                    gathered_v = all_gather_zigzag(v_tensor.contiguous(), dim=seq_dim)
                else:
                    gathered_v = sp_group.all_gather(v_tensor.contiguous(), dim=seq_dim)
            
            gathered_layers_data.append((gathered_k, gathered_v))

        # (수정됨) SP 랭크 0에서만 '기존 캐시 객체'를 수정하고 저장
        if get_sequence_parallel_rank() == 0:
            
            # 'tensor'는 랭크 0의 (아직 sharded된) 원본 캐시 객체입니다.
            # 'gathered_layers_data'는 all-gather된 (K, V) 튜플 리스트입니다.
            assert len(tensor.layers) == len(gathered_layers_data), \
                "Cache layer count mismatch after gather"

            # 원본 캐시 객체(tensor)의 각 레이어에 
            # all-gather된 텐서를 '재할당(reassign)'합니다.
            for i, layer_cache in enumerate(tensor.layers):
                gathered_k, gathered_v = gathered_layers_data[i]
                
                # 랭크 0의 원본 객체(tensor) 내부 값을 덮어씁니다.
                layer_cache.keys = gathered_k 
                layer_cache.values = gathered_v

            # 이제 'tensor' 객체는 랭크 0에서
            # all-gather된 전체 텐서를 포함하게 되었습니다.
            
            # '하나의' (수정된) 캐시 객체를 저장합니다.
            torch.save(tensor, path)

    else:
        # --- 3-B. 일반 텐서일 경우 (기존 로직) ---
        if zigzag:
            gathered_tensor = all_gather_zigzag(tensor.contiguous(), dim=seq_dim)
        else:
            gathered_tensor = sp_group.all_gather(tensor.contiguous(), dim=seq_dim)

        if get_sequence_parallel_rank() == 0:
            torch.save(gathered_tensor, path)
                   
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
        print(f"--- Starting KV cache generation on {world_size} GPUs ---")

    model_id = "/root/.cache/huggingface/hub/gpt-oss-120b/"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    isl = 1024 * 2  # Target input sequence length

    # 2. Use a string prompt and pad to isl
    prompt_text = "Who won the World Series in 2020?"
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_ids = None
    attention_mask = None

    if rank == 0:
        unpadded_inputs = tokenizer(prompt_text, return_tensors="pt").input_ids
        current_length = unpadded_inputs.shape[1]
        
        assert current_length <= isl, \
            f"Prompt length ({current_length}) exceeds target isl ({isl})"

        pad_length = isl - current_length
        if pad_length > 0:
            padding_tensor = torch.full(
                (1, pad_length), tokenizer.pad_token_id, dtype=torch.long
            )
            input_ids = torch.cat([padding_tensor, unpadded_inputs], dim=1).to(device)
        else:
            input_ids = unpadded_inputs.to(device)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    else:
        # Allocate space on other ranks
        input_ids = torch.empty((1, isl), dtype=torch.long, device=device)
        attention_mask = torch.empty((1, isl), dtype=torch.long, device=device)

    # Broadcast the prepared input_ids and attention_mask to all ranks
    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)

    print(f"layout: {layout}, rank: {rank}, input_ids shape: {input_ids.shape}")
    os.environ["LAYOUT"] = layout
    extract_func = yunchang.comm.extract_local.EXTRACT_FUNC_DICT.get(layout)
    if extract_func is None:
        raise ValueError(
            f"Unknown layout '{layout}', available: {list(yunchang.comm.extract_local.EXTRACT_FUNC_DICT.keys())}"
        )

    # Extract local (sharded) tensors for this rank
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
    
    local_attention_mask = (
        extract_func(
            attention_mask,
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
        print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device,
    )
    model.eval()

    #model = torch.compile(model)

    if isl % world_size != 0:
        if rank == 0:
            print(f"Error: Input sequence length {isl} is not divisible by world_size {world_size}.")
        return

    # 1. Warm-up run removed.

    # 3. Run forward pass and save KV cache
    if rank == 0:
        print("  -> Running forward pass to generate KV cache...")

    # Synchronize before forward pass
    torch.cuda.synchronize()
    dist.barrier()
    
    past_key_values = None
    with torch.no_grad():
        outputs = model(
            input_ids=local_input_ids,
            attention_mask=local_attention_mask,
            use_cache=True
        )
        past_key_values = outputs.past_key_values

    # Synchronize after forward pass
    torch.cuda.synchronize()
    dist.barrier()

    if past_key_values is not None:
        sp_size = get_sequence_parallel_world_size()
        save_path = f"half_fa_zigzag_{sp_size}/kv_cache.pt" if os.environ["LAYOUT"] == "zigzag" else f"half_fa_basic_{sp_size}/kv_cache.pt"
        save_tensor(past_key_values, save_path, seq_dim=2)
        print(f"Rank {rank}: Saved KV cache to {save_path}", flush=True)
    else:
        print(f"Rank {rank}: No KV cache returned from model.", flush=True)

    if rank == 0:
        print("\n--- KV cache generation complete ---")
        print(f"Input Length: {isl}")
        print(f"World Size:   {world_size} GPUs")
        print(f"Layout:       {layout}")
        print(f'output: {outputs}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=str, default="basic", help="layout type: basic, zigzag, etc.")
    args = parser.parse_args()
    main(layout=args.layout)