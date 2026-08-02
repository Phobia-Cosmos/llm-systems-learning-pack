#!/usr/bin/env bash
set -euo pipefail

ROOT="${SERVER_ROOT:-/public/home/u43077/lzh}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/benchmarks/next-20260802}"
SCRIPTS="${SCRIPT_ROOT:-$RESULT_ROOT/scripts}"
DURATION="${SOAK_SECONDS:-3600}"
RESULT="$RESULT_ROOT/vllm-dp4-soak"
mkdir -p "$RESULT"

stop_all() {
  for port in 18500 18501 18502 18503; do
    mapfile -t pids < <(pgrep -f "vllm serve .*--port $port" || true)
    ((${#pids[@]} == 0)) || kill -TERM "${pids[@]}" 2>/dev/null || true
  done
}
trap stop_all EXIT

urls=()
for index in 0 1 2 3; do
  port=$((18500 + index))
  setsid env CUDA_VISIBLE_DEVICES="$index" MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.85 \
    "$ROOT/scripts/serve-vllm.sh" Qwen3-8B "$port" >"$RESULT/server-$index.log" 2>&1 < /dev/null &
  ready=0
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 5
  done
  (( ready == 1 ))
  urls+=("http://127.0.0.1:$port")
done

nvidia-smi dmon -s pucvmet -d 2 -o DT >"$RESULT/nvidia-smi-dmon.log" 2>&1 &
monitor_pid=$!
/opt/venvs/vllm/bin/python "$SCRIPTS/serving_soak.py" \
  --urls "${urls[@]}" --duration "$DURATION" --concurrency 128 \
  --output "$RESULT/results.json" >"$RESULT/progress.log" 2>&1
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
date '+completed_at=%Y-%m-%d %H:%M:%S %z' >"$RESULT/complete.txt"
