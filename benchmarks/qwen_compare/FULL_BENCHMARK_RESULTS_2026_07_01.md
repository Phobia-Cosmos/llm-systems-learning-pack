# Full Inference Benchmark Results - 2026-07-01

## Goal

Compare four local inference paths on the same Qwen3-0.6B model:

1. Native Hugging Face Transformers.
2. vLLM full path.
3. SGLang full path.
4. nano-vLLM full path.

The run uses 3 prompts x 128 generated tokens = 384 output tokens per engine.
All numbers are single-machine local generation throughput on an RTX 4070 SUPER.
They are useful for learning and relative comparison, not a formal serving
benchmark.

## Environment Snapshot

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12282 MiB |
| Driver | 580.159.03 |
| System CUDA Toolkit | `/usr/local/cuda-13.0` |
| nvcc | CUDA 13.0, V13.0.88 |
| Model | `.model_cache/huggingface/Qwen3-0.6B` |
| vLLM env | torch 2.11.0+cu130, vLLM 0.23.0, flashinfer-python 0.6.12 |
| SGLang env | torch 2.9.1+cu130, SGLang 0.5.9, sgl-kernel 0.3.21, flashinfer-python 0.6.3, flash-attn 2.8.3 |

`flash_attn_varlen_func` and `flash_attn_with_kvcache` both import correctly in
`.venv-sglang`.

## Final Comparison

| Rank | Engine | Configuration | Elapsed s | Output tokens | tok/s | Log |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | vLLM | FlashAttention v2 + FlashInfer sampler | 0.529 | 384 | 725.34 | `benchmarks/qwen_compare/logs/2026-07-01/vllm_full_final.log` |
| 2 | SGLang | FlashInfer attention + FlashInfer sampler + CUDA graph | 0.772 | 384 | 497.69 | `benchmarks/qwen_compare/logs/2026-07-01/sglang_full_fixed.log` |
| 3 | nano-vLLM | FlashAttention + CUDA graph | 0.945 | 384 | 406.45 | `benchmarks/qwen_compare/logs/2026-07-01/nanovllm_flash_full_final.log` |
| 4 | Transformers | Native Hugging Face generate | 1.602 | 384 | 239.67 | `benchmarks/qwen_compare/logs/2026-07-01/transformers_baseline_final.log` |

## Controls

| Engine | Configuration | Elapsed s | tok/s | Log |
| --- | --- | ---: | ---: | --- |
| SGLang | FlashInfer attention + FlashInfer sampler, no CUDA graph | 1.342 | 286.24 | `benchmarks/qwen_compare/logs/2026-07-01/sglang_flashinfer_no_cudagraph_system_cuda_release_cached.log` |
| SGLang | Triton attention + PyTorch sampler, no CUDA graph | 1.515 | 253.52 | `benchmarks/qwen_compare/logs/2026-07-01/sglang_stable_triton_pytorch.log` |
| nano-vLLM | Fallback PyTorch attention + eager mode | 3.799 | 101.09 | `benchmarks/qwen_compare/logs/2026-07-01/nanovllm_fallback.log` |

## Commands

Transformers:

```bash
source scripts/use_vllm.sh
python benchmarks/qwen_compare/run_transformers.py
```

vLLM full:

```bash
source scripts/use_vllm.sh
VLLM_USE_FLASHINFER_SAMPLER=1 python benchmarks/qwen_compare/run_vllm_full.py
```

SGLang full:

```bash
source scripts/use_sglang.sh
python benchmarks/qwen_compare/run_sglang_full.py
```

nano-vLLM full:

```bash
source scripts/use_sglang.sh
python benchmarks/qwen_compare/run_nanovllm_flash.py
```

## FlashAttention Install Result

`flash-attn` is installed successfully in `.venv-sglang` from the matching
prebuilt wheel:

```text
flash_attn-2.8.3+cu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

The vLLM env still does not install the standalone `flash_attn` Python package,
because there is no matching prebuilt wheel for its current matrix:

```text
Python 3.12 + torch 2.11.0+cu130 + Linux x86_64
```

vLLM does not need that Python package for this benchmark; its own wheel uses
FlashAttention v2 internally, confirmed by the vLLM log.

## SGLang FlashInfer JIT Issue

Initial full SGLang failed in:

```text
benchmarks/qwen_compare/logs/2026-07-01/sglang_full.log
benchmarks/qwen_compare/logs/2026-07-01/sglang_flashinfer_no_cudagraph.log
```

The error was:

```text
CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths
```

Root cause: FlashInfer reused an old JIT cache whose `build.ninja` hard-coded the
venv CUDA wheel path. That pulled CUDA 13.3 compiler/header packages into a torch
2.9.1+cu130 runtime and caused the mismatch.

Fix applied in `scripts/use_sglang.sh`:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export FLASHINFER_NVCC=/usr/local/cuda-13.0/bin/nvcc
export FLASHINFER_WORKSPACE_BASE=/home/undefined/Disk/cache/flashinfer-system-cuda-release
```

With a clean release JIT cache, the full SGLang configuration runs. The first
release run includes JIT compilation cost (`25.45 tok/s`), while cached steady
runs are much faster.

## Takeaways

- Full vLLM is currently the fastest local path in this test.
- Full SGLang works after fixing the FlashInfer JIT compiler/cache path, and CUDA
  graph roughly doubles this small-batch result versus no CUDA graph.
- nano-vLLM only becomes representative after installing real FlashAttention; its
  fallback attention path is mainly for code learning.
- Native Transformers remains the easiest baseline but does not exercise the
  serving-engine optimizations that vLLM/SGLang/nano-vLLM are designed to teach.
