#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPERATOR_DIR="$ROOT_DIR/05_python_operators"
CPU_PYTHON="${CPU_PYTHON:-/home/undefined/Disk/python-envs/ai-core-py312/bin/python}"
CUDA_PYTHON="${CUDA_PYTHON:-/home/undefined/Disk/python-envs/vllm/bin/python}"

"$CPU_PYTHON" -m unittest discover -v -s "$OPERATOR_DIR" -p 'test_operators.py'
"$CUDA_PYTHON" -m unittest discover -v -s "$OPERATOR_DIR" -p 'test_operators.py'
"$CUDA_PYTHON" "$OPERATOR_DIR/benchmark.py" \
  --device cuda \
  --dtype float32 \
  --profile smoke \
  --variants reference,teaching \
  --output "$ROOT_DIR/results/python_operators/gpu_smoke"
"$CUDA_PYTHON" "$OPERATOR_DIR/benchmark.py" \
  --operators fused_bias_silu \
  --device cuda \
  --dtype float16 \
  --profile llm \
  --variants reference,teaching,compiled,triton \
  --warmup 10 \
  --repeats 20 \
  --inner 20 \
  --output "$ROOT_DIR/results/python_operators/fused_bias_silu_llm"
