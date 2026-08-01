import torch
from transformers import AutoModelForCausalLM
path = '/public/home/u43077/lzh/models/Qwen3-8B'
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='cuda')
ids = torch.ones((1, 4096), dtype=torch.long, device='cuda')
with torch.inference_mode():
    for _ in range(2):
        model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push('prefill_4096')
    out = model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    past = out.past_key_values
    x = torch.ones((1, 1), dtype=torch.long, device='cuda')
    torch.cuda.nvtx.range_push('decode_128')
    for _ in range(128):
        out = model(input_ids=x, past_key_values=past, use_cache=True)
        past = out.past_key_values
        x = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
