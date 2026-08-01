#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/public/home/u43077/lzh/python-envs/llm-py311/bin/python}"
OUT_ROOT="${OUT_ROOT:-/public/home/u43077/lzh/outputs/minillm-general/new-model-v1}"
MODE="${1:?Usage: run_new_model_training.sh proxy32|proxy48|160-capacity|160-train}"

[[ -x "$PY" ]] || { echo "Python interpreter not found: $PY" >&2; exit 2; }
VISIBLE_GPUS="$($PY -c 'import torch; print(torch.cuda.device_count())')"
TRAIN_GPUS="${TRAIN_GPUS:-auto}"
if [[ "$TRAIN_GPUS" == "auto" ]]; then
  TRAIN_GPUS="$VISIBLE_GPUS"
fi
[[ "$TRAIN_GPUS" =~ ^[1-9][0-9]*$ ]] || {
  echo "TRAIN_GPUS must be auto or a positive integer" >&2
  exit 2
}
(( TRAIN_GPUS <= VISIBLE_GPUS )) || {
  echo "requested $TRAIN_GPUS GPUs, but only $VISIBLE_GPUS are visible" >&2
  exit 2
}

CPU_THREADS_PER_RANK=$(( $(nproc) / TRAIN_GPUS ))
(( CPU_THREADS_PER_RANK > 8 )) && CPU_THREADS_PER_RANK=8
(( CPU_THREADS_PER_RANK < 1 )) && CPU_THREADS_PER_RANK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS_PER_RANK}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS_PER_RANK}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS_PER_RANK}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

RUNNER=("$PY" "$ROOT/train_general.py")
if (( TRAIN_GPUS > 1 )); then
  RUNNER=(
    "$PY" -m torch.distributed.run
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
    --local_addr="$MASTER_ADDR"
    --nproc_per_node="$TRAIN_GPUS" "$ROOT/train_general.py"
  )
fi

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
    MICRO_BATCH="${MICRO_BATCH:-8}"
    (( MICRO_BATCH > 0 )) || { echo "MICRO_BATCH must be positive" >&2; exit 2; }
    ACCUMULATION="${ACCUMULATION:-$((32 / (MICRO_BATCH * TRAIN_GPUS)))}"
    (( ACCUMULATION > 0 && MICRO_BATCH * ACCUMULATION * TRAIN_GPUS == 32 )) || {
      echo "proxy configuration must preserve 32 samples/update" >&2
      exit 2
    }
    set -- \
      --dataset-dir "$DATASET" --out-dir "$OUT" \
      --n-layer 12 --n-head 12 --num-key-value-heads 4 \
      --n-embd 768 --intermediate-size 2048 \
      --block-size 2048 --sequence-length 1024 \
      --micro-batch-size "$MICRO_BATCH" \
      --gradient-accumulation-steps "$ACCUMULATION" \
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
    (( MICRO_BATCH > 0 && ACCUMULATION > 0 )) || {
      echo "MICRO_BATCH and ACCUMULATION must be positive" >&2
      exit 2
    }
    if (( 8 % TRAIN_GPUS != 0 )); then
      echo "TRAIN_GPUS must divide the fixed global validation batch of 8" >&2
      exit 2
    fi
    EVAL_MICRO_BATCH="${EVAL_MICRO_BATCH:-$((8 / TRAIN_GPUS))}"
    TOKENS_PER_UPDATE=$((4096 * MICRO_BATCH * ACCUMULATION * TRAIN_GPUS))
    REFERENCE_TOKENS_PER_STEP=32768
    INTERVAL_SCALE=$(((TOKENS_PER_UPDATE + REFERENCE_TOKENS_PER_STEP - 1) / REFERENCE_TOKENS_PER_STEP))
    if [[ "$MODE" == "160-capacity" ]]; then
      OUT="$OUT_ROOT/160m-capacity-m${MICRO_BATCH}-a${ACCUMULATION}"
      WARMUP_STEPS=10
      EVAL_INTERVAL="$TARGET_STEP"
      SAVE_INTERVAL="$TARGET_STEP"
      EVAL_BATCHES=4
    else
      WARMUP_STEPS="${WARMUP_STEPS:-2000}"
      EVAL_INTERVAL="${EVAL_INTERVAL:-$(((500 + INTERVAL_SCALE - 1) / INTERVAL_SCALE))}"
      SAVE_INTERVAL="${SAVE_INTERVAL:-$(((3052 + INTERVAL_SCALE - 1) / INTERVAL_SCALE))}"
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
      --schedule-reference-tokens-per-step "$REFERENCE_TOKENS_PER_STEP" \
      --warmup-steps "$WARMUP_STEPS" \
      --eval-interval "$EVAL_INTERVAL" --eval-batches "$EVAL_BATCHES" \
      --eval-micro-batch-size "$EVAL_MICRO_BATCH" \
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
echo "train_gpus=$TRAIN_GPUS micro_batch=$MICRO_BATCH accumulation=$ACCUMULATION eval_micro_batch=${EVAL_MICRO_BATCH:-$MICRO_BATCH} tokens_per_update=${TOKENS_PER_UPDATE:-n/a} omp_threads=$OMP_NUM_THREADS" | tee -a "$LOG"
cd "$ROOT"
"${RUNNER[@]}" "$@" 2>&1 | tee -a "$LOG"
