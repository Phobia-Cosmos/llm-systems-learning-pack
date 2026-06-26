#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/home/undefined/Desktop/ai/.model_cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/home/undefined/Desktop/ai/.model_cache/modelscope}"

source /home/undefined/Desktop/ai/.venv-sglang/bin/activate
