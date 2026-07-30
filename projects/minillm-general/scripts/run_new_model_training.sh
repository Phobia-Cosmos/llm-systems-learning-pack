#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/public/home/u43077/lzh/python-envs/llm-py311/bin/python}"
OUT_ROOT="${OUT_ROOT:-/public/home/u43077/lzh/outputs/minillm-general/new-model-v1}"
MODE="${1:?Usage: run_new_model_training.sh proxy32|proxy48|160-capacity|160-train}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"

mkdir -p "$OUT_ROOT"
case "$MODE" in
  proxy32|proxy48)
    if [[ "$MODE" == "proxy32" ]]; then
      CANDIDATE="openbpe-32k"
    else
      CANDIDATE="openbpe-48k"
    fi
    DATASET="${DATASET_DIR:-/public/home/u43077/lzh/datasets/minillm-general/proxy-$CANDIDATE-v1}"
    OUT="$OUT_ROOT/$MODE"
    set -- \
      --dataset-dir "$DATASET" --out-dir "$OUT" \
      --n-layer 12 --n-head 12 --num-key-value-heads 4 \
      --n-embd 768 --intermediate-size 2048 \
      --block-size 2048 --sequence-length 1024 \
      --micro-batch-size 8 --gradient-accumulation-steps 4 \
      --batch-layout contiguous \
      --max-steps "${PROXY_STEPS:-2000}" --warmup-steps 100 \
      --eval-interval 250 --eval-batches 40 \
      --save-interval 500 --log-interval 10 --compile
    ;;
  160-capacity|160-train)
    DATASET="${DATASET_DIR:-/public/home/u43077/lzh/datasets/minillm-general/general-openbpe-32k-v1}"
    OUT="$OUT_ROOT/160m-openbpe-32k-4k"
    TARGET_STEP="${TARGET_STEP:-200}"
    MICRO_BATCH="${MICRO_BATCH:-8}"
    ACCUMULATION="${ACCUMULATION:-1}"
    if [[ "$MODE" == "160-capacity" ]]; then
      OUT="$OUT_ROOT/160m-capacity-m${MICRO_BATCH}-a${ACCUMULATION}"
      WARMUP_STEPS=10
      EVAL_INTERVAL="$TARGET_STEP"
      SAVE_INTERVAL="$TARGET_STEP"
      EVAL_BATCHES=4
    else
      WARMUP_STEPS="${WARMUP_STEPS:-2000}"
      EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
      SAVE_INTERVAL="${SAVE_INTERVAL:-3052}"
      EVAL_BATCHES="${EVAL_BATCHES:-20}"
    fi
    RESUME_ARGS=()
    if [[ "$MODE" == "160-train" && -f "$OUT/latest.pt" ]]; then
      RESUME_ARGS=(--resume auto)
    fi
    set -- \
      --dataset-dir "$DATASET" --out-dir "$OUT" \
      --n-layer 22 --n-head 12 --num-key-value-heads 4 \
      --n-embd 768 --intermediate-size 2048 \
      --block-size 8192 --sequence-length 4096 \
      --micro-batch-size "$MICRO_BATCH" \
      --gradient-accumulation-steps "$ACCUMULATION" \
      --batch-layout contiguous \
      --max-steps "$TARGET_STEP" \
      --schedule-end-step "${FINAL_SCHEDULE_STEP:-305176}" \
      --warmup-steps "$WARMUP_STEPS" \
      --eval-interval "$EVAL_INTERVAL" --eval-batches "$EVAL_BATCHES" \
      --save-interval "$SAVE_INTERVAL" --keep-checkpoints "${KEEP_CHECKPOINTS:-4}" \
      --log-interval 10 \
      --compile "${RESUME_ARGS[@]}"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac

if [[ ! -f "$DATASET/manifest.json" ]]; then
  echo "packed dataset not found: $DATASET" >&2
  exit 2
fi

LOG="$OUT_ROOT/$MODE.log"
if [[ "$MODE" == "160-capacity" ]]; then
  LOG="$OUT_ROOT/$MODE-m${MICRO_BATCH}-a${ACCUMULATION}.log"
fi
date '+%Y-%m-%d %H:%M:%S %z' | tee -a "$LOG"
nvidia-smi -L | tee -a "$LOG"
cd "$ROOT"
"$PY" train_general.py "$@" 2>&1 | tee -a "$LOG"
