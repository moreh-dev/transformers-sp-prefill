from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/root/.cache/huggingface/hub/gpt-oss-120b/"
#kv_cache_path = "/root/.cache/transformers-sp-prefill/past_key_values/single_triton/kv_cache_rank_0.pt"
#kv_cache_path = "pkv.pt"
#kv_cache_path = "/root/.cache/transformers-sp-prefill/zigzag_8/kv_cache.pt"
#kv_cache_path = "/root/.cache/transformers-sp-prefill/basic_8/kv_cache.pt"
#kv_cache_path = "/root/.cache/transformers-sp-prefill/basic_1/kv_cache.pt"
kv_cache_path = "/root/.cache/transformers-sp-prefill/half_fa_zigzag_8/kv_cache.pt"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto', device_map='cuda:0')
model.eval()

# KV cache 로드
past_key_values = torch.load(kv_cache_path, weights_only=False)

# "The"를 토크나이즈
first_token = "The"
input_ids = tokenizer.encode(first_token, return_tensors='pt').to('cuda:0')

generated_tokens = input_ids.tolist()[0]
max_new_tokens = 100

with torch.no_grad():
    for _ in range(max_new_tokens):
        # Forward pass with past_key_values
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )
        
        # 다음 토큰 예측
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        
        # EOS 토큰이면 종료
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        # 생성된 토큰 추가
        generated_tokens.append(next_token.item())
        
        # 다음 iteration을 위한 준비
        input_ids = next_token.unsqueeze(0)
        past_key_values = outputs.past_key_values

# 디코딩
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(generated_text)
