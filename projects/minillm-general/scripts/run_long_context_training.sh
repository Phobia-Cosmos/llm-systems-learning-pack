#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/public/home/u43077/lzh/python-envs/llm-py311/bin/python}"
DATASET="${DATASET_DIR:-/public/home/u43077/lzh/datasets/minillm-general/general-openbpe-32k-8k-v1}"
OUT_ROOT="${OUT_ROOT:-/public/home/u43077/lzh/outputs/minillm-general/new-model-v1}"
MODE="${1:-capacity}"
MICRO_BATCH="${MICRO_BATCH:-4}"
ACCUMULATION="${ACCUMULATION:-1}"

[[ -x "$PY" ]] || { echo "Python interpreter not found: $PY" >&2; exit 2; }
[[ -f "$DATASET/manifest.json" ]] || { echo "dataset not found: $DATASET" >&2; exit 2; }
(( MICRO_BATCH > 0 && ACCUMULATION > 0 )) || {
  echo "MICRO_BATCH and ACCUMULATION must be positive" >&2
  exit 2
}
(( MICRO_BATCH * ACCUMULATION == 4 )) || {
  echo "8K training must preserve 32,768 tokens/rank/update" >&2
  exit 2
}

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
(( 4 % TRAIN_GPUS == 0 )) || {
  echo "TRAIN_GPUS must divide the fixed global validation batch of 4" >&2
  exit 2
}
EVAL_MICRO_BATCH="${EVAL_MICRO_BATCH:-$((4 / TRAIN_GPUS))}"

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

case "$MODE" in
  capacity)
    OUT="${OUT_DIR:-$OUT_ROOT/160m-8k-capacity-g${TRAIN_GPUS}-m${MICRO_BATCH}-a${ACCUMULATION}}"
    RESUME_PATH="${RESUME:?Set RESUME to the selected stable 4K checkpoint}"
    [[ -f "$RESUME_PATH" ]] || {
      echo "checkpoint not found: $RESUME_PATH" >&2
      exit 2
    }
    ADDITIONAL_STEPS="${ADDITIONAL_STEPS:-20}"
    (( ADDITIONAL_STEPS > 0 )) || {
      echo "ADDITIONAL_STEPS must be positive" >&2
      exit 2
    }
    EVAL_INTERVAL="$ADDITIONAL_STEPS"
    SAVE_INTERVAL="$ADDITIONAL_STEPS"
    EVAL_BATCHES=2
    WARMUP_STEPS=0
    WARMUP_START_RATIO=1
    KEEP_CHECKPOINTS=1
    LOG_INTERVAL=1
    ;;
  train)
    OUT="${OUT_DIR:-$OUT_ROOT/160m-openbpe-32k-8k}"
    if [[ -f "$OUT/latest.pt" ]]; then
      RESUME_PATH="$OUT/latest.pt"
    else
      RESUME_PATH="${RESUME:?Set RESUME to the selected stable 4K checkpoint}"
    fi
    [[ -f "$RESUME_PATH" ]] || {
      echo "checkpoint not found: $RESUME_PATH" >&2
      exit 2
    }
    EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-3052}"
    EVAL_BATCHES="${EVAL_BATCHES:-20}"
    WARMUP_STEPS="${WARMUP_STEPS:-200}"
    WARMUP_START_RATIO="${WARMUP_START_RATIO:-0.1}"
    KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-4}"
    LOG_INTERVAL="${LOG_INTERVAL:-10}"
    ;;
  *) echo "usage: $0 capacity|train" >&2; exit 2 ;;
esac

CHECKPOINT_STATE_OUTPUT=$(
  "$PY" - "$RESUME_PATH" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
arguments = checkpoint.get("args", {})
print(int(checkpoint["step"]))
print(int(arguments.get("sequence_length", arguments.get("block_size", 0))))
print(int(arguments.get("schedule_start_step") or 0))
print(int(arguments.get("schedule_end_step") or checkpoint["step"]))
PY
)
mapfile -t CHECKPOINT_STATE <<<"$CHECKPOINT_STATE_OUTPUT"
if (( ${#CHECKPOINT_STATE[@]} != 4 )); then
  echo "could not read checkpoint schedule state from $RESUME_PATH" >&2
  exit 2
fi
START_STEP="${CHECKPOINT_STATE[0]}"
CHECKPOINT_SEQUENCE_LENGTH="${CHECKPOINT_STATE[1]}"
CHECKPOINT_SCHEDULE_START="${CHECKPOINT_STATE[2]}"
CHECKPOINT_SCHEDULE_END="${CHECKPOINT_STATE[3]}"
TOKENS_PER_RANK_STEP=$((8192 * MICRO_BATCH * ACCUMULATION))
TOKENS_PER_STEP=$((TOKENS_PER_RANK_STEP * TRAIN_GPUS))

if [[ "$MODE" == "capacity" ]]; then
  STAGE_START_STEP="$START_STEP"
  STAGE_END_STEP=$((START_STEP + ADDITIONAL_STEPS))
  END_STEP="$STAGE_END_STEP"
else
  if [[ -n "${STAGE_START_STEP:-}" ]]; then
    STAGE_START_STEP="$STAGE_START_STEP"
  elif (( CHECKPOINT_SEQUENCE_LENGTH == 8192 )); then
    STAGE_START_STEP="$CHECKPOINT_SCHEDULE_START"
  else
    STAGE_START_STEP="$START_STEP"
  fi

  if [[ -n "${STAGE_END_STEP:-}" ]]; then
    STAGE_END_STEP="$STAGE_END_STEP"
  elif (( CHECKPOINT_SEQUENCE_LENGTH == 8192 && CHECKPOINT_SCHEDULE_END > START_STEP )); then
    STAGE_END_STEP="$CHECKPOINT_SCHEDULE_END"
  else
    ADDITIONAL_TOKENS="${ADDITIONAL_TOKENS:-1000000000}"
    (( ADDITIONAL_TOKENS > 0 )) || {
      echo "ADDITIONAL_TOKENS must be positive" >&2
      exit 2
    }
    STAGE_END_STEP=$((STAGE_START_STEP + (ADDITIONAL_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP))
  fi
  END_STEP="${TARGET_STEP:-$STAGE_END_STEP}"
  (( STAGE_START_STEP <= START_STEP )) || {
    echo "STAGE_START_STEP must not exceed checkpoint step" >&2
    exit 2
  }
  (( START_STEP < END_STEP )) || {
    echo "checkpoint step $START_STEP has already reached target $END_STEP"
    exit 0
  }
  (( END_STEP <= STAGE_END_STEP )) || {
    echo "TARGET_STEP $END_STEP exceeds STAGE_END_STEP $STAGE_END_STEP" >&2
    exit 2
  }
fi

mkdir -p "$OUT"
LOG="$OUT_ROOT/8k-${MODE}-g${TRAIN_GPUS}-m${MICRO_BATCH}-a${ACCUMULATION}.log"
if [[ "$MODE" == "train" ]]; then
  LOG="$OUT_ROOT/8k-train.log"
fi
{
  date '+%Y-%m-%d %H:%M:%S %z'
  nvidia-smi -L
  echo "resume=$RESUME_PATH start_step=$START_STEP target_step=$END_STEP stage_start=$STAGE_START_STEP stage_end=$STAGE_END_STEP world_size=$TRAIN_GPUS tokens_per_rank_step=$TOKENS_PER_RANK_STEP tokens_per_step=$TOKENS_PER_STEP eval_micro_batch=$EVAL_MICRO_BATCH omp_threads=$OMP_NUM_THREADS"
} | tee -a "$LOG"

cd "$ROOT"
"${RUNNER[@]}" \
  --dataset-dir "$DATASET" --out-dir "$OUT" --resume "$RESUME_PATH" \
  --allow-dataset-change --allow-context-extension \
  --n-layer 22 --n-head 12 --num-key-value-heads 4 \
  --n-embd 768 --intermediate-size 2048 \
  --block-size 8192 --sequence-length 8192 \
  --micro-batch-size "$MICRO_BATCH" \
  --gradient-accumulation-steps "$ACCUMULATION" \
  --eval-micro-batch-size "$EVAL_MICRO_BATCH" \
  --batch-layout records \
  --learning-rate "${LEARNING_RATE:-3e-5}" \
  --warmup-start-lr-ratio "$WARMUP_START_RATIO" \
  --warmup-steps "$WARMUP_STEPS" \
  --schedule-start-step "$STAGE_START_STEP" --schedule-end-step "$STAGE_END_STEP" \
  --max-steps "$END_STEP" --eval-interval "$EVAL_INTERVAL" \
  --eval-batches "$EVAL_BATCHES" --save-interval "$SAVE_INTERVAL" \
  --keep-checkpoints "$KEEP_CHECKPOINTS" --log-interval "$LOG_INTERVAL" --compile \
  2>&1 | tee -a "$LOG"
