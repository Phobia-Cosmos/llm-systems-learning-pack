# Qwen3-0.6B Inference Comparison

Model path:

```bash
/home/undefined/Desktop/ai/.model_cache/huggingface/Qwen3-0.6B
```

This benchmark uses 3 prompts and `max_new_tokens=128`, so each successful run
produces 384 output tokens. The reported speed is generation-only output tokens
per second for this local test, not a production serving benchmark.

## Environment

- GPU: NVIDIA GeForce RTX 4070 SUPER, 12GB.
- Driver: 580.159.03.
- System CUDA Toolkit: `/usr/local/cuda-13.0`, nvcc 13.0.88.
- vLLM env: `.venv-vllm`, torch 2.11.0+cu130, vLLM 0.23.0.
- SGLang env: `.venv-sglang`, torch 2.9.1+cu130, SGLang 0.5.9,
  flashinfer-python 0.6.3, flash-attn 2.8.3.

## Full Runs

Run native Transformers baseline:

```bash
cd /home/undefined/Desktop/ai
source scripts/use_vllm.sh
python benchmarks/qwen_compare/run_transformers.py
```

Run vLLM full path:

```bash
cd /home/undefined/Desktop/ai
source scripts/use_vllm.sh
VLLM_USE_FLASHINFER_SAMPLER=1 python benchmarks/qwen_compare/run_vllm_full.py
```

The vLLM log confirms both `Using FlashInfer for top-p & top-k sampling` and
`Using FlashAttention version 2`.

Run SGLang full path:

```bash
cd /home/undefined/Desktop/ai
source scripts/use_sglang.sh
python benchmarks/qwen_compare/run_sglang_full.py
```

This uses FlashInfer attention, FlashInfer sampling, and CUDA graph. The
SGLang environment script now sets:

```bash
CUDA_HOME=/usr/local/cuda-13.0
FLASHINFER_NVCC=/usr/local/cuda-13.0/bin/nvcc
FLASHINFER_WORKSPACE_BASE=/home/undefined/Disk/cache/flashinfer-system-cuda-release
```

Run nano-vLLM full path:

```bash
cd /home/undefined/Desktop/ai
source scripts/use_sglang.sh
python benchmarks/qwen_compare/run_nanovllm_flash.py
```

nano-vLLM is run from the SGLang env here because that env has the matching
prebuilt `flash-attn` wheel. This path uses FlashAttention and CUDA graph.

## Final Results From 2026-07-01

| Path | Full configuration used | Log | tok/s |
| --- | --- | --- | ---: |
| Transformers | Native Hugging Face generate | `logs/2026-07-01/transformers_baseline_final.log` | 239.67 |
| vLLM | FlashAttention v2 + FlashInfer sampler | `logs/2026-07-01/vllm_full_final.log` | 725.34 |
| SGLang | FlashInfer attention + FlashInfer sampler + CUDA graph | `logs/2026-07-01/sglang_full_fixed.log` | 497.69 |
| nano-vLLM | FlashAttention + CUDA graph | `logs/2026-07-01/nanovllm_flash_full_final.log` | 406.45 |

Additional controls:

| Path | Configuration | Log | tok/s |
| --- | --- | --- | ---: |
| SGLang | Triton attention + PyTorch sampler + no CUDA graph | `logs/2026-07-01/sglang_stable_triton_pytorch.log` | 253.52 |
| SGLang | FlashInfer attention + FlashInfer sampler + no CUDA graph | `logs/2026-07-01/sglang_flashinfer_no_cudagraph_system_cuda_release_cached.log` | 286.24 |
| nano-vLLM | PyTorch fallback attention + eager mode | `logs/2026-07-01/nanovllm_fallback.log` | 101.09 |

## Important Failure And Fix

The first SGLang full attempt failed because FlashInfer reused an old JIT cache
whose `build.ninja` hard-coded the venv CUDA wheel path:

```text
/home/undefined/Desktop/ai/.venv-sglang/lib/python3.12/site-packages/nvidia/cu13
```

That path mixed CUDA compiler/header packages and produced:

```text
CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths
```

The fix was to use a clean FlashInfer workspace on `/home/undefined/Disk` and
force FlashInfer JIT to compile with the system CUDA Toolkit 13.0 nvcc. After
that, SGLang full FlashInfer attention + sampler + CUDA graph ran successfully.

## Variant Script

`run_sglang_variant.py` can isolate SGLang backend choices:

```bash
source scripts/use_sglang.sh
python benchmarks/qwen_compare/run_sglang_variant.py \
  --attention-backend flashinfer \
  --sampling-backend flashinfer

python benchmarks/qwen_compare/run_sglang_variant.py \
  --attention-backend flashinfer \
  --sampling-backend flashinfer \
  --disable-cuda-graph

python benchmarks/qwen_compare/run_sglang_variant.py \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph
```

## Interpretation

- vLLM is fastest in this small local batch because its production scheduler,
  KV cache, CUDA kernels, and FlashInfer sampler are all active.
- SGLang becomes much faster once FlashInfer attention/sampling and CUDA graph
  are enabled with a clean system-CUDA JIT path.
- nano-vLLM is useful for learning the vLLM-style engine design. With real
  FlashAttention it is much faster than its fallback path, but it still has less
  production optimization than vLLM.
- Native Transformers is the baseline. It is simple and reliable, but it lacks
  the serving-engine features that matter for batching, KV-cache management,
  request scheduling, and high concurrency.
