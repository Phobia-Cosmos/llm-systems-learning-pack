#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/home/undefined/Desktop/ai/.model_cache/huggingface}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

export AI_VLLM_VENV="/home/undefined/Desktop/ai/.venv-vllm"
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -d /usr/local/cuda-13.0 ]; then
    export CUDA_HOME="/usr/local/cuda-13.0"
  else
    export CUDA_HOME="$AI_VLLM_VENV/lib/python3.12/site-packages/nvidia/cu13"
  fi
fi
if [ -d "$CUDA_HOME/lib64" ]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
elif [ -d "$CUDA_HOME/lib" ]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
export PATH="$CUDA_HOME/bin:$PATH"

source /home/undefined/Desktop/ai/.venv-vllm/bin/activate
