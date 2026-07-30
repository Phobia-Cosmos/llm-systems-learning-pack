#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${TRAIN_MODE:-continue-full-1gpu}"
OUT_BASE="${OUT_BASE:-/public/home/u43077/lzh/outputs/minillm-general}"
RUN_OUT="$OUT_BASE/90m-pretrain-full-stage2"
FINAL_SCHEDULE_STEP="${FINAL_SCHEDULE_STEP:-60000}"
MILESTONES="${MILESTONES:-25000 30000 35000 40000 45000 50000 55000 60000}"

for target_step in $MILESTONES; do
  if (( target_step > FINAL_SCHEDULE_STEP )); then
    echo "milestone $target_step exceeds final schedule step $FINAL_SCHEDULE_STEP" >&2
    exit 2
  fi
  checkpoint="$RUN_OUT/step-$(printf '%08d' "$target_step").pt"
  benchmark="$OUT_BASE/benchmarks/trajectory/step-$(printf '%08d' "$target_step").json"
  if [[ -f "$benchmark" ]]; then
    echo "already benchmarked: $benchmark"
    continue
  fi
  if [[ ! -f "$checkpoint" ]]; then
    TARGET_STEP="$target_step" FINAL_SCHEDULE_STEP="$FINAL_SCHEDULE_STEP" \
      "$ROOT/scripts/run_a100_training.sh" "$MODE"
  fi
  "$ROOT/scripts/benchmark_checkpoint.sh" "$checkpoint" "$benchmark"
done
