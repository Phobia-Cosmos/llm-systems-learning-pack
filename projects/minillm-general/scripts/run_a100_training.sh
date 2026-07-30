#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/public/home/u43077/lzh/python-envs/llm-py311/bin/python}"
DATASET="${DATASET_DIR:-/public/home/u43077/lzh/datasets/minillm-general/pretrain-mini-v1}"
OUT_BASE="${OUT_BASE:-/public/home/u43077/lzh/outputs/minillm-general}"
MODE="${1:-smoke}"
RUNNER=("$PY" "$ROOT/train_general.py")
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/public/home/u43077/lzh/outputs/minillm-general/90m-pretrain-mini/step-00020000.pt}"
TARGET_STEP="${TARGET_STEP:-60000}"
FINAL_SCHEDULE_STEP="${FINAL_SCHEDULE_STEP:-60000}"

# GPU training is not CPU-bound. Keep BLAS and Inductor compilation from
# creating more runnable workers than the Notebook cpuset can schedule.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"

if [[ ! -x "$PY" ]]; then
  echo "Python interpreter not found: $PY" >&2
  exit 2
fi
if ! "$PY" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
  echo "CUDA GPU is required. Run this script inside a GPU Notebook." >&2
  exit 3
fi

mkdir -p "$OUT_BASE"
date '+%Y-%m-%d %H:%M:%S %z' | tee "$OUT_BASE/${MODE}.started"
nvidia-smi | tee "$OUT_BASE/${MODE}.nvidia-smi.txt"
"$PY" -c 'import torch; print({"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)})' | tee "$OUT_BASE/${MODE}.runtime.txt"

cd "$ROOT"
case "$MODE" in
  smoke)
    OUT="$OUT_BASE/a100-smoke"
    set -- --dataset-dir "$DATASET" --out-dir "$OUT" --n-layer 4 --n-head 6 --num-key-value-heads 2 --n-embd 384 --intermediate-size 1024 --block-size 512 --micro-batch-size 4 --gradient-accumulation-steps 2 --max-steps 20 --warmup-steps 2 --log-interval 1 --eval-interval 10 --eval-batches 2 --save-interval 10
    ;;
  train)
    OUT="$OUT_BASE/90m-pretrain-mini"
    set -- --dataset-dir "$DATASET" --out-dir "$OUT" --max-steps 20000 --compile
    ;;
  resume)
    OUT="$OUT_BASE/90m-pretrain-mini"
    set -- --dataset-dir "$DATASET" --out-dir "$OUT" --max-steps 20000 --resume auto --compile
    ;;
  ddp4)
    OUT="$OUT_BASE/90m-pretrain-mini"
    RUNNER=("$PY" -m torch.distributed.run --standalone --nproc_per_node=4 "$ROOT/train_general.py")
    # Steps 1..1500 used one GPU (32,768 tokens/step). Four GPUs process
    # 131,072 tokens/step, so step 6125 preserves the original ~655M-token budget.
    set -- --dataset-dir "$DATASET" --out-dir "$OUT" --max-steps 6125 --resume auto --compile
    ;;
  continue-full-1gpu|continue-full-ddp4)
    DATASET="${DATASET_DIR:-/public/home/u43077/lzh/datasets/minillm-general/pretrain-full-v1}"
    OUT="$OUT_BASE/90m-pretrain-full-stage2"
    if [[ -f "$OUT/latest.pt" ]]; then
      RESUME_CHECKPOINT="$OUT/latest.pt"
    else
      RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$BASE_CHECKPOINT}"
    fi
    if [[ "$MODE" == "continue-full-ddp4" ]]; then
      RUNNER=("$PY" -m torch.distributed.run --standalone --nproc_per_node=4 "$ROOT/train_general.py")
      ACCUMULATION=1
    else
      ACCUMULATION=4
    fi
    set -- \
      --dataset-dir "$DATASET" \
      --out-dir "$OUT" \
      --resume "$RESUME_CHECKPOINT" \
      --max-steps "$TARGET_STEP" \
      --schedule-start-step 20000 \
      --schedule-end-step "$FINAL_SCHEDULE_STEP" \
      --learning-rate 3e-5 \
      --min-lr-ratio 0.1 \
      --warmup-steps 0 \
      --micro-batch-size 8 \
      --gradient-accumulation-steps "$ACCUMULATION" \
      --batch-layout contiguous \
      --eval-interval 500 \
      --save-interval 5000 \
      --keep-checkpoints 4 \
      --compile
    ;;
  *)
    echo "Usage: $0 {smoke|train|resume|ddp4|continue-full-1gpu|continue-full-ddp4}" >&2
    exit 2
    ;;
esac

if [[ ! -f "$DATASET/manifest.json" ]]; then
  echo "Packed dataset not found: $DATASET" >&2
  exit 2
fi

"${RUNNER[@]}" "$@" 2>&1 | tee -a "$OUT_BASE/${MODE}.log"
