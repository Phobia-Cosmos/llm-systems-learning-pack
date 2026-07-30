#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/public/home/u43077/lzh/python-envs/llm-py311/bin/python}"
DATASET="${DATASET_DIR:-/public/home/u43077/lzh/datasets/minillm-general/general-openbpe-32k-8k-v1}"
OUT="${OUT_DIR:-/public/home/u43077/lzh/outputs/minillm-general/new-model-v1/160m-openbpe-32k-8k}"
RESUME="${RESUME:?Set RESUME to the selected stable 4K checkpoint}"
MODE="${1:-capacity}"
MICRO_BATCH="${MICRO_BATCH:-4}"
ACCUMULATION="${ACCUMULATION:-1}"

[[ -f "$DATASET/manifest.json" ]] || { echo "dataset not found: $DATASET" >&2; exit 2; }
[[ -f "$RESUME" ]] || { echo "checkpoint not found: $RESUME" >&2; exit 2; }

START_STEP=$(
  "$PY" - "$RESUME" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(c["step"]))
PY
)

case "$MODE" in
  capacity)
    ADDITIONAL_STEPS="${ADDITIONAL_STEPS:-20}"
    EVAL_INTERVAL="$ADDITIONAL_STEPS"
    SAVE_INTERVAL="$ADDITIONAL_STEPS"
    EVAL_BATCHES=2
    WARMUP_STEPS=0
    WARMUP_START_RATIO=1
    ;;
  train)
    ADDITIONAL_TOKENS="${ADDITIONAL_TOKENS:-1000000000}"
    TOKENS_PER_STEP=$((8192 * MICRO_BATCH * ACCUMULATION))
    ADDITIONAL_STEPS=$(((ADDITIONAL_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP))
    EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-3052}"
    EVAL_BATCHES="${EVAL_BATCHES:-20}"
    WARMUP_STEPS="${WARMUP_STEPS:-200}"
    WARMUP_START_RATIO="${WARMUP_START_RATIO:-0.1}"
    ;;
  *) echo "usage: $0 capacity|train" >&2; exit 2 ;;
esac

END_STEP=$((START_STEP + ADDITIONAL_STEPS))
mkdir -p "$OUT"
cd "$ROOT"
"$PY" train_general.py \
  --dataset-dir "$DATASET" --out-dir "$OUT" --resume "$RESUME" \
  --allow-dataset-change --allow-context-extension \
  --n-layer 22 --n-head 12 --num-key-value-heads 4 \
  --n-embd 768 --intermediate-size 2048 \
  --block-size 8192 --sequence-length 8192 \
  --micro-batch-size "$MICRO_BATCH" \
  --gradient-accumulation-steps "$ACCUMULATION" \
  --batch-layout records \
  --learning-rate "${LEARNING_RATE:-3e-5}" \
  --warmup-start-lr-ratio "$WARMUP_START_RATIO" \
  --warmup-steps "$WARMUP_STEPS" \
  --schedule-start-step "$START_STEP" --schedule-end-step "$END_STEP" \
  --max-steps "$END_STEP" --eval-interval "$EVAL_INTERVAL" \
  --eval-batches "$EVAL_BATCHES" --save-interval "$SAVE_INTERVAL" \
  --keep-checkpoints 4 --log-interval 10 --compile
