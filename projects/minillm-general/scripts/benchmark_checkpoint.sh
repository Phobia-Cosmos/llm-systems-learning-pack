#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${EVAL_PYTHON:-/public/home/u43077/lzh/python-envs/minillm-eval-py311/bin/python}"
DATASETS="${EVAL_DATASETS:-/public/home/u43077/lzh/datasets/evals}"
CHECKPOINT="${1:?Usage: benchmark_checkpoint.sh CHECKPOINT [OUTPUT_JSON]}"
CHECKPOINT="$(readlink -f "$CHECKPOINT")"
STEP="$(basename "$CHECKPOINT" .pt)"
OUTPUT="${2:-/public/home/u43077/lzh/outputs/minillm-general/benchmarks/trajectory/${STEP}.json}"

mkdir -p "$(dirname "$OUTPUT")"
cd "$ROOT"
"$PY" benchmark_mc.py \
  --backend minillm \
  --model "$CHECKPOINT" \
  --datasets-dir "$DATASETS" \
  --limit-per-task "${EVAL_LIMIT_PER_TASK:-500}" \
  --batch-size 24 \
  --score-content \
  --output "$OUTPUT"

STEP_NUMBER="${STEP#step-}"
if [[ "${RUN_EXPANDED_EVAL:-1}" == "1" \
  && "$(basename "$(dirname "$OUTPUT")")" == "trajectory" \
  && $((10#$STEP_NUMBER)) -ge 61036 ]]; then
  EXPANDED_OUTPUT="$(dirname "$(dirname "$OUTPUT")")/trajectory-expanded/$(basename "$OUTPUT")"
  mkdir -p "$(dirname "$EXPANDED_OUTPUT")"
  if [[ ! -f "$EXPANDED_OUTPUT" ]]; then
    EVAL_LIMIT_PER_TASK=2000 RUN_EXPANDED_EVAL=0 \
      "$0" "$CHECKPOINT" "$EXPANDED_OUTPUT"
  fi
fi
