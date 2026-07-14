#!/usr/bin/env bash
set -euo pipefail

export AI_DISK_ROOT="${AI_DISK_ROOT:-/home/undefined/Disk/ai-storage}"
export AI_DISK_VENV="${AI_DISK_VENV:-${AI_DISK_ROOT}/.venv-sglang}"
export HF_HOME="${HF_HOME:-${AI_DISK_ROOT}/.model_cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${AI_DISK_ROOT}/.model_cache/modelscope}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${AI_DISK_ROOT}/.uv-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/home/undefined/Disk/cache/pip}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/home/undefined/Disk/cache/torch_extensions}"

deactivate 2>/dev/null || true
export VIRTUAL_ENV="${AI_DISK_VENV}"
export PATH="${AI_DISK_VENV}/bin:${PATH}"

echo "Activated shared Disk AI environment: ${AI_DISK_VENV}"
python - <<'PY'
import sys
print(sys.executable)
PY
