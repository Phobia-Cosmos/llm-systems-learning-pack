#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/home/undefined/Desktop/ai/.model_cache/huggingface}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/home/undefined/Desktop/ai/.model_cache/modelscope}"

export SGLANG_VENV="/home/undefined/Desktop/ai/.venv-sglang"
export CUDA_HOME="${CUDA_HOME:-$SGLANG_VENV/lib/python3.12/site-packages/nvidia/cu13}"
if [ -d "$CUDA_HOME/lib" ] && [ ! -e "$CUDA_HOME/lib64" ]; then
  ln -s lib "$CUDA_HOME/lib64"
fi
if [ -e "$CUDA_HOME/lib/libcudart.so.13" ] && [ ! -e "$CUDA_HOME/lib/libcudart.so" ]; then
  ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$SGLANG_VENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$SGLANG_VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$SGLANG_VENV/lib/python3.12/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"

source /home/undefined/Desktop/ai/.venv-sglang/bin/activate
