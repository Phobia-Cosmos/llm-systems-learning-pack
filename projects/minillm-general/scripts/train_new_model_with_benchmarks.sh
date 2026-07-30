#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/public/home/u43077/lzh/outputs/minillm-general/new-model-v1}"
RUN_OUT="$OUT_ROOT/160m-openbpe-32k-4k"
MILESTONE_DIR="$RUN_OUT/milestones"
TRAJECTORY="$OUT_ROOT/benchmarks/trajectory"
EXPANDED_TRAJECTORY="$OUT_ROOT/benchmarks/trajectory-expanded"
FINAL_SCHEDULE_STEP="${FINAL_SCHEDULE_STEP:-305176}"
MILESTONES="${MILESTONES:-15259 30518 61036 99854 152588 249634 305176}"
STOP_FILE="$OUT_ROOT/STOP"
LOCK_FILE="$OUT_ROOT/controller.lock"
PID_FILE="$OUT_ROOT/controller.pid"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another new-model controller is already running" >&2
  exit 3
fi
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

mkdir -p "$TRAJECTORY" "$EXPANDED_TRAJECTORY" "$MILESTONE_DIR"
ln -sfn ../tokenizer.json "$MILESTONE_DIR/tokenizer.json"
for target_step in $MILESTONES; do
  if [[ -f "$STOP_FILE" ]]; then
    echo "stop requested by $STOP_FILE"
    exit 0
  fi
  if (( target_step > FINAL_SCHEDULE_STEP )); then
    echo "milestone $target_step exceeds final schedule $FINAL_SCHEDULE_STEP" >&2
    exit 2
  fi
  checkpoint="$RUN_OUT/step-$(printf '%08d' "$target_step").pt"
  pinned_checkpoint="$MILESTONE_DIR/step-$(printf '%08d' "$target_step").pt"
  benchmark="$TRAJECTORY/step-$(printf '%08d' "$target_step").json"
  if [[ -f "$pinned_checkpoint" ]]; then
    checkpoint="$pinned_checkpoint"
  elif [[ ! -f "$checkpoint" ]]; then
    TARGET_STEP="$target_step" FINAL_SCHEDULE_STEP="$FINAL_SCHEDULE_STEP" \
      "$ROOT/scripts/run_new_model_training.sh" 160-train
  fi
  if [[ ! -f "$pinned_checkpoint" ]]; then
    ln "$checkpoint" "$pinned_checkpoint"
  fi
  checkpoint="$pinned_checkpoint"
  if [[ ! -f "$benchmark" ]]; then
    "$ROOT/scripts/benchmark_checkpoint.sh" "$checkpoint" "$benchmark"
  fi
  if (( target_step >= 61036 )); then
    expanded_benchmark="$EXPANDED_TRAJECTORY/step-$(printf '%08d' "$target_step").json"
    if [[ ! -f "$expanded_benchmark" ]]; then
      EVAL_LIMIT_PER_TASK=2000 \
        "$ROOT/scripts/benchmark_checkpoint.sh" "$checkpoint" "$expanded_benchmark"
    fi
  fi
  /public/home/u43077/lzh/python-envs/llm-py311/bin/python \
    "$ROOT/scripts/summarize_training_trajectory.py" \
    --training-log "$OUT_ROOT/160-train.log" \
    --benchmark-dir "$TRAJECTORY" \
    --output-json "$OUT_ROOT/benchmarks/trajectory-summary.json" \
    --output-markdown "$OUT_ROOT/benchmarks/trajectory-summary.md"
done
