#!/usr/bin/env bash
set -uo pipefail

ROOT="${SERVER_ROOT:-/public/home/u43077/lzh}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/benchmarks/next-20260802}"
SCRIPTS="${SCRIPT_ROOT:-$RESULT_ROOT/scripts}"
BUDGET_SECONDS="${BUDGET_SECONDS:-43200}"
RESERVE_SECONDS="${RESERVE_SECONDS:-1800}"
SOAK_STARTUP_GRACE_SECONDS="${SOAK_STARTUP_GRACE_SECONDS:-1200}"
START_EPOCH=$(date +%s)
mkdir -p "$RESULT_ROOT/stages"
date '+started_at=%Y-%m-%d %H:%M:%S %z' >"$RESULT_ROOT/pipeline-started.txt"

run_stage() {
  local name=$1
  local limit=$2
  shift 2
  local status_file="$RESULT_ROOT/stages/$name.status"
  local log="$RESULT_ROOT/stages/$name.log"
  if grep -q '^status=success$' "$status_file" 2>/dev/null; then
    return 0
  fi
  date '+started_at=%Y-%m-%d %H:%M:%S %z' >"$status_file"
  if timeout --signal=TERM --kill-after=120 "$limit" "$@" >"$log" 2>&1; then
    {
      echo 'status=success'
      date '+finished_at=%Y-%m-%d %H:%M:%S %z'
    } >>"$status_file"
  else
    code=$?
    {
      echo 'status=failed'
      echo "exit_code=$code"
      date '+finished_at=%Y-%m-%d %H:%M:%S %z'
    } >>"$status_file"
  fi
}

"$ROOT/scripts/mount-serving-runtime.sh" >"$RESULT_ROOT/mount-runtime.log" 2>&1
nvidia-smi -L >"$RESULT_ROOT/gpus.txt"
nvidia-smi topo -m >"$RESULT_ROOT/topology.txt"

run_stage ddp-scaling 5400 env \
  RESULT_DIR="$RESULT_ROOT/ddp-scaling" MEASURE_STEPS=30 \
  bash "$SCRIPTS/run_minillm_ddp_scaling.sh"

run_stage sglang-slim-profile 2400 env \
  RESULT_ROOT="$RESULT_ROOT" SCRIPT_ROOT="$SCRIPTS" \
  bash "$SCRIPTS/run_sglang_slim_profile.sh"

run_stage serving-ab 14400 env \
  RESULT_ROOT="$RESULT_ROOT" SCRIPT_ROOT="$SCRIPTS" \
  bash "$SCRIPTS/run_serving_ab_matrix.sh"

run_stage vllm-tp-dp 14400 env \
  RESULT_ROOT="$RESULT_ROOT" SCRIPT_ROOT="$SCRIPTS" \
  bash "$SCRIPTS/run_vllm_tp_dp_matrix.sh"

elapsed=$(( $(date +%s) - START_EPOCH ))
remaining=$(( BUDGET_SECONDS - RESERVE_SECONDS - elapsed ))
soak_duration=$(( remaining - SOAK_STARTUP_GRACE_SECONDS ))
if (( soak_duration >= 1800 )); then
  run_stage dp4-soak "$remaining" env \
    RESULT_ROOT="$RESULT_ROOT" SCRIPT_ROOT="$SCRIPTS" SOAK_SECONDS="$soak_duration" \
    bash "$SCRIPTS/run_vllm_dp_soak.sh"
else
  {
    echo 'status=skipped'
    echo "remaining_seconds=$remaining"
    echo "startup_grace_seconds=$SOAK_STARTUP_GRACE_SECONDS"
  } >"$RESULT_ROOT/stages/dp4-soak.status"
fi

date '+finished_at=%Y-%m-%d %H:%M:%S %z' >"$RESULT_ROOT/pipeline-finished.txt"
