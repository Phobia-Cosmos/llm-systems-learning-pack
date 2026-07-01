# CUDA / FlashAttention / Inference Environment Notes

Date: 2026-07-01

## Current Local State

- OS: Ubuntu 24.04.4 LTS.
- GPU: NVIDIA GeForce RTX 4070 SUPER, 12GB.
- Driver: 580.159.03. `nvidia-smi` reports CUDA API compatibility up to 13.0.
- vLLM env:
  - Path: `/home/undefined/Desktop/ai/.venv-vllm`
  - Actual storage: `/home/undefined/Disk/ai-storage/.venv-vllm`
  - `torch==2.11.0+cu130`, `vllm==0.23.0`, `flashinfer-python==0.6.12`
  - `flash_attn` is not installed.
- SGLang env:
  - Path: `/home/undefined/Desktop/ai/.venv-sglang`
  - Actual storage: `/home/undefined/Disk/ai-storage/.venv-sglang`
  - `torch==2.9.1+cu130`, `sglang==0.5.9`, `sgl-kernel==0.3.21`, `flashinfer-python==0.6.3`
  - `flash_attn==2.8.3` is installed from a matching prebuilt CUDA 13 / torch 2.9 wheel.
  - `flash_attn_varlen_func` and `flash_attn_with_kvcache` import successfully.
- Model cache:
  - Path: `/home/undefined/Desktop/ai/.model_cache`
  - Actual storage: `/home/undefined/Disk/ai-storage/.model_cache`
- Disk after migration and system CUDA install:
  - `/`: 21G available.
  - `/home/undefined/Disk`: 71G available.

## CUDA Wheel vs System CUDA Toolkit

There are three different CUDA layers:

1. NVIDIA driver
   - Kernel/user driver that lets programs talk to the GPU.
   - Checked with `nvidia-smi`.
   - Your current driver works and reports CUDA 13.0 compatibility.

2. Python CUDA wheels
   - Packages installed inside a virtualenv, for example `torch==...+cu130`, `nvidia-cuda-runtime`, `nvidia-cuda-nvcc`, `nvidia-cuda-cccl`.
   - They make Python packages self-contained. PyTorch and vLLM can often run without `/usr/local/cuda`.
   - They can be mixed accidentally. That is what happened here: PyTorch is built for CUDA 13.0, while the vLLM venv had nvcc/header packages around CUDA 13.2/13.3 during FlashAttention compilation attempts.

3. System CUDA Toolkit
   - Usually installed under `/usr/local/cuda-X.Y`.
   - Contains `nvcc`, CUDA headers, runtime libraries, nvrtc, cuBLAS/cuFFT/cuSPARSE, profiling/debug tools, etc.
   - Best for compiling CUDA extensions, writing CUDA kernels, and avoiding a fragile pile of per-venv CUDA compiler/header packages.

NVIDIA's Linux guide treats package-manager installs, runfile installs, conda installs, and pip wheels as different installation methods. For Ubuntu network repository installation, NVIDIA documents installing `cuda-keyring`, running `apt update`, then installing CUDA Toolkit packages. Source: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

## Does Each Project Need A Different CUDA?

No. They do not conceptually need different CUDA installations.

The target should be:

- One system driver: already installed, 580.159.03.
- One system CUDA Toolkit for local compiling: recommended `cuda-toolkit-13-0` for this machine because the driver reports CUDA 13.0 and PyTorch wheels are `cu130`.
- Separate Python virtualenvs for dependency isolation:
  - vLLM can keep its own env.
  - SGLang can keep its own env.
  - nano-vLLM can either live in the vLLM env for learning, or get its own cleaner env if we want to solve `flash-attn` properly without perturbing vLLM.

The important rule is not "one CUDA per framework"; it is "do not mix incompatible compiler, header, runtime, and PyTorch build versions inside the same compilation path."

## Why vLLM Runs Without System nvcc, But SGLang Needed nvcc

vLLM ran because the installed wheel already includes or depends on prebuilt CUDA kernels and runtime packages. For normal inference, it can load compiled shared libraries and call CUDA through PyTorch/runtime libraries; it does not need to compile kernels locally each time.

SGLang needed more local CUDA compiler pieces because its runtime path triggered JIT/build steps for kernels from `sgl-kernel`/Triton/CUDA support packages. The earlier SGLang failures were caused by missing compiler headers (`nv/target`) and missing link-time runtime names like `libcudart.so`, not by the model itself.

So:

- Inference with prebuilt wheels: often no `nvcc`.
- Building/JIT-compiling CUDA extensions: needs compatible `nvcc`, headers, and runtime libraries.

## What flash-attn Needs

FlashAttention's upstream README says:

- Install command: `pip install flash-attn --no-build-isolation`.
- CUDA support requirement: CUDA 12.0 and above.
- It recommends NVIDIA's PyTorch container because it already has the required tools.
- It warns that `ninja` can consume too much RAM with many parallel jobs and suggests `MAX_JOBS` to limit compilation.
- FlashAttention-2 supports Ampere/Ada/Hopper GPUs and fp16/bf16. RTX 4070 SUPER is Ada, so the GPU is suitable.

Sources:

- FlashAttention README: https://github.com/Dao-AILab/flash-attention
- FlashAttention paper: https://arxiv.org/abs/2205.14135

## Why flash-attn Failed Here

Observed logs:

- `benchmarks/qwen_compare/logs/2026-06-30/flash_attn_install_vllm.log`
- `benchmarks/qwen_compare/logs/2026-06-30/flash_attn_install_vllm_after_cccl132.log`
- `benchmarks/qwen_compare/logs/2026-06-30/flash_attn_install_vllm_after_cuda132_align.log`
- `benchmarks/qwen_compare/logs/2026-06-30/flash_attn_install_vllm_maxjobs1.log`

Important lines:

```text
Guessing wheel URL:
https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu13torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
Precompiled wheel not found. Building from source...
```

That means no prebuilt wheel exists for this exact combination:

- Python 3.12
- Torch 2.11
- CUDA 13
- Linux x86_64

The first source build failed with:

```text
CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths
```

At that moment, the vLLM env mixed CUDA compiler/header/runtime pieces from different minor versions:

- PyTorch: CUDA 13.0 build.
- `nvidia-cuda-nvcc`: 13.2.
- `nvidia-cuda-cccl`: originally 13.3.
- `nvidia-cuda-crt`: originally 13.3.

After aligning several packages to the 13.2 series, the header mismatch moved forward, but source compilation later hit:

```text
Killed
HTTP Error 404: Not Found
```

The 404 was from the missing prebuilt wheel. The `Killed` happened during nvcc compilation of large FlashAttention kernels. On this machine, root disk had become very tight during build; we later freed space by moving venvs and caches to `/home/undefined/Disk`.

The most likely root cause is a fragile unsupported build matrix: `flash-attn==2.8.3.post1` did not provide a wheel for `cu13 + torch2.11 + cp312`, and building it locally with the pip CUDA wheel stack is heavy and sensitive to compiler/header/runtime consistency.

## How To Solve flash-attn Cleanly

Recommended options, from most stable to most experimental:

1. Use a supported container.
   - NVIDIA PyTorch container or a known vLLM/SGLang CUDA container.
   - Best for avoiding host/compiler mismatch.

2. Create a nano-vLLM-specific environment with a common FlashAttention matrix.
   - Python 3.10 or 3.11.
   - PyTorch CUDA 12.4/12.6/12.8, depending on available wheels.
   - Then install `flash-attn --no-build-isolation`.
   - This avoids perturbing the working vLLM/SGLang envs.

3. Install system CUDA Toolkit 13.0 and retry source build.
   - Use `/usr/local/cuda-13.0` as `CUDA_HOME`.
   - Keep compiler, headers, and runtime from the same system toolkit.
   - Still not guaranteed because upstream may not fully support `torch2.11 + cu13 + cp312` wheels.

4. Keep nano-vLLM fallback attention for learning.
   - It preserves the scheduler/KV-cache/block-table learning path.
   - It is not representative of production nano-vLLM performance because attention is the core hot path.

## Native CUDA Toolkit Install Result

CUDA Toolkit 13.0 was installed from NVIDIA's Ubuntu 24.04 apt repository.

Installed key packages include:

```text
cuda-toolkit-13-0 13.0.3-1
cuda-nvcc-13-0    13.0.88-1
cuda-cccl-13-0    13.0.85-1
cuda-cudart-13-0  13.0.96-1
```

Verified paths:

```text
/usr/local/cuda -> /etc/alternatives/cuda
/usr/local/cuda-13.0
/usr/local/cuda-13.0/include
/usr/local/cuda-13.0/lib64
```

Verified compiler:

```text
/usr/local/cuda/bin/nvcc --version
Cuda compilation tools, release 13.0, V13.0.88
```

The environment scripts now prefer this system toolkit:

```bash
source scripts/use_vllm.sh
source scripts/use_sglang.sh
```

Both scripts set `CUDA_HOME=/usr/local/cuda-13.0` when it exists. They fall back to venv CUDA wheel paths only if the system toolkit is absent.

Global shell configuration was added:

```text
/etc/profile.d/cuda-13-0.sh
```

New login shells can find:

```text
/usr/local/cuda-13.0/bin/nvcc
Cuda compilation tools, release 13.0, V13.0.88
```

Avoid installing the plain `cuda` meta-package unless you intentionally want driver changes. For this machine, the installed driver was kept unchanged.

## uv Strategy

`uv` is already installed at:

```bash
/home/undefined/.local/bin/uv
```

You do not need to delete existing virtualenvs to use uv.

Use uv against an existing venv:

```bash
uv pip list --python .venv-vllm/bin/python
uv pip install --python .venv-vllm/bin/python <package>
uv pip install --python .venv-sglang/bin/python <package>
```

For a clean reproducible rebuild later:

```bash
uv venv .venv-nanovllm-fa --python 3.11
uv pip install --python .venv-nanovllm-fa/bin/python torch torchvision --index-url <matching-pytorch-cuda-index>
uv pip install --python .venv-nanovllm-fa/bin/python flash-attn --no-build-isolation
uv pip install --python .venv-nanovllm-fa/bin/python -e projects/nano-vllm --no-deps
```

Do not delete the current `.venv-vllm` and `.venv-sglang` yet. They are working, and deleting them would lose the current comparison baseline.

## Space Migration Done

Moved to `/home/undefined/Disk` and kept original paths as symlinks:

```text
/home/undefined/Desktop/ai/.model_cache  -> /home/undefined/Disk/ai-storage/.model_cache
/home/undefined/Desktop/ai/.venv-vllm    -> /home/undefined/Disk/ai-storage/.venv-vllm
/home/undefined/Desktop/ai/.venv-sglang  -> /home/undefined/Disk/ai-storage/.venv-sglang
/home/undefined/.cache/pip               -> /home/undefined/Disk/cache/pip
```

Also removed failed FlashAttention build leftovers from `/tmp`.

The vLLM and SGLang environments were verified after migration and system CUDA install:

```text
vLLM env: CUDA_HOME=/usr/local/cuda-13.0, torch 2.11.0+cu130, vllm 0.23.0, flashinfer present, flash_attn missing
SGLang env: CUDA_HOME=/usr/local/cuda-13.0, torch 2.9.1+cu130, sglang 0.5.9, sgl-kernel OK, flashinfer present, flash_attn 2.8.3 installed
```


## FlashAttention Install Result On 2026-07-01

`flash-attn` is now installed successfully in the SGLang environment:

```text
.venv-sglang
flash-attn 2.8.3
wheel: flash_attn-2.8.3+cu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Verification:

```text
torch 2.9.1+cu130 cuda 13.0
flash_attn 2.8.3
flashinfer 0.6.3
sglang 0.5.9
flash_attn_varlen_func True
flash_attn_with_kvcache True
```

The vLLM environment does not install the standalone `flash_attn` package. Its
current matrix is `Python 3.12 + torch 2.11.0+cu130`; no matching upstream
prebuilt flash-attn wheel was found, and local source builds are expensive and
fragile. vLLM still uses FlashAttention internally from its own wheel, confirmed
by the benchmark log line `Using FlashAttention version 2`.

## FlashInfer JIT Cache Fix For SGLang

The earlier SGLang full failure was not caused by the absence of system CUDA. It
was caused by an old FlashInfer JIT cache. The cached `build.ninja` had hard-coded
this venv CUDA wheel path:

```text
/home/undefined/Desktop/ai/.venv-sglang/lib/python3.12/site-packages/nvidia/cu13
```

That path mixed CUDA 13.3 compiler/header packages with torch 2.9.1+cu130 and
produced:

```text
CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths
```

`scripts/use_sglang.sh` now defaults to the system CUDA Toolkit and a clean Disk
cache:

```bash
CUDA_HOME=/usr/local/cuda-13.0
FLASHINFER_NVCC=/usr/local/cuda-13.0/bin/nvcc
FLASHINFER_WORKSPACE_BASE=/home/undefined/Disk/cache/flashinfer-system-cuda-release
```

After this change, SGLang full FlashInfer attention + FlashInfer sampling + CUDA
graph runs successfully.

Important shell note: a bare shell may not have `nvcc` in `PATH`. Either run
`source scripts/use_sglang.sh`, `source scripts/use_vllm.sh`, or call:

```bash
/usr/local/cuda-13.0/bin/nvcc --version
```

## Final Benchmark Pointer

The full four-way comparison is documented in:

```text
benchmarks/qwen_compare/FULL_BENCHMARK_RESULTS_2026_07_01.md
```

Final generation-only results for Qwen3-0.6B, 3 prompts x 128 tokens:

```text
vLLM full:         725.34 output tok/s
SGLang full:       497.69 output tok/s
nano-vLLM full:    406.45 output tok/s
Transformers:      239.67 output tok/s
```
