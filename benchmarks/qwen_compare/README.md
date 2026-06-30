# Qwen3-0.6B Inference Comparison

Model path:

```bash
/home/undefined/Desktop/ai/.model_cache/huggingface/Qwen3-0.6B
```

Run Transformers:

```bash
cd /home/undefined/Desktop/ai
source scripts/use_vllm.sh
python benchmarks/qwen_compare/run_transformers.py
```

Run vLLM:

```bash
cd /home/undefined/Desktop/ai
source scripts/use_vllm.sh
python benchmarks/qwen_compare/run_vllm.py
```

`scripts/use_vllm.sh` disables the FlashInfer sampler path by default with
`VLLM_USE_FLASHINFER_SAMPLER=0`.

Observed generation-only result for 3 prompts x 128 tokens: about 690 output tok/s.

Run SGLang:

```bash
cd /home/undefined/Desktop/ai
source scripts/use_sglang.sh
python benchmarks/qwen_compare/run_sglang.py
```

Current local status: SGLang runs after installing `nvidia-cuda-nvcc`,
`nvidia-cuda-cccl`, and adding CUDA wheel compatibility links for `lib64` and
`libcudart.so`.

Observed generation-only result for 3 prompts x 128 tokens: about 282 output tok/s.

Run Nano-vLLM:

```bash
cd /home/undefined/Desktop/ai
source scripts/use_vllm.sh
python benchmarks/qwen_compare/run_nanovllm.py
```

Current local status: Nano-vLLM runs with a local PyTorch attention fallback when
`flash-attn` is not installed. This is slower than real FlashAttention but keeps
the scheduler, KV cache, and block table path usable for learning.

Observed generation-only result for 3 prompts x 128 tokens: about 115 output tok/s.

Transformers baseline was run with the same 3 prompts and 128 generated tokens each.
Observed generation-only result: about 277 output tok/s.
