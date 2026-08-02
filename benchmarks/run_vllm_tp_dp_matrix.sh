#!/usr/bin/env bash
set -euo pipefail

ROOT="${SERVER_ROOT:-/public/home/u43077/lzh}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/benchmarks/next-20260802}"
SCRIPTS="${SCRIPT_ROOT:-$RESULT_ROOT/scripts}"
RESULT="$RESULT_ROOT/vllm-tp-dp"
mkdir -p "$RESULT"

stop_port() {
  local port=$1
  mapfile -t pids < <(pgrep -f "vllm serve .*--port $port" || true)
  ((${#pids[@]} == 0)) || kill -TERM "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    pgrep -f "vllm serve .*--port $port" >/dev/null || break
    sleep 1
  done
}

cleanup() {
  stop_port 18300
  for port in 18400 18401 18402 18403; do stop_port "$port"; done
}
trap cleanup EXIT

wait_health() {
  local port=$1
  for _ in $(seq 1 120); do
    curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

run_tp() {
  local tp=$1
  local gpus=$2
  local out="$RESULT/qwen14b-tp$tp"
  [[ -s "$out/complete.txt" || -s "$out/failed.txt" ]] && return 0
  mkdir -p "$out"

  setsid env CUDA_VISIBLE_DEVICES="$gpus" TP_SIZE="$tp" MAX_MODEL_LEN=8192 \
    GPU_MEMORY_UTILIZATION=0.90 "$ROOT/scripts/serve-vllm.sh" Qwen3-14B 18300 \
    >"$out/server.log" 2>&1 < /dev/null &
  if ! wait_health 18300; then
    date '+failed_at=%Y-%m-%d %H:%M:%S %z reason=startup' >"$out/failed.txt"
    stop_port 18300
    return 0
  fi

  nvidia-smi dmon -s pucvmet -d 1 -o DT >"$out/nvidia-smi-dmon.log" 2>&1 &
  monitor_pid=$!
  status=0
  /opt/venvs/vllm/bin/python "$SCRIPTS/streaming_ttft_tpot.py" \
    --output "$out/results.json" --engines vllm \
    --model-path "$ROOT/models/Qwen3-14B" --served-model-name Qwen3-14B \
    --vllm-urls http://127.0.0.1:18300 \
    --cases 512:128 4096:128 7680:32 4096:512 \
    --concurrencies 1 16 32 64 --prompt-modes unique \
    --request-multiplier 1 --min-requests 4 --repeats 2 \
    >"$out/workload.log" 2>&1 || status=$?
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  if (( status == 0 )); then
    date '+completed_at=%Y-%m-%d %H:%M:%S %z' >"$out/complete.txt"
  else
    date '+failed_at=%Y-%m-%d %H:%M:%S %z reason=workload' >"$out/failed.txt"
  fi
  stop_port 18300
}

run_tp 1 0
run_tp 2 0,1
run_tp 4 0,1,2,3

urls=()
for index in 0 1 2 3; do
  port=$((18400 + index))
  replica_out="$RESULT/qwen8b-replica-$index"
  mkdir -p "$replica_out"
  setsid env CUDA_VISIBLE_DEVICES="$index" MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.85 \
    "$ROOT/scripts/serve-vllm.sh" Qwen3-8B "$port" >"$replica_out/server.log" 2>&1 < /dev/null &
  if ! wait_health "$port"; then
    date '+failed_at=%Y-%m-%d %H:%M:%S %z reason=startup' >"$replica_out/failed.txt"
    cleanup
    exit 1
  fi
  urls+=("http://127.0.0.1:$port")

  replicas=$((index + 1))
  if (( replicas == 1 || replicas == 2 || replicas == 4 )); then
    out="$RESULT/qwen8b-dp$replicas"
    if [[ ! -s "$out/complete.txt" ]]; then
      mkdir -p "$out"
      nvidia-smi dmon -s pucvmet -d 1 -o DT >"$out/nvidia-smi-dmon.log" 2>&1 &
      monitor_pid=$!
      /opt/venvs/vllm/bin/python "$SCRIPTS/streaming_ttft_tpot.py" \
        --output "$out/results.json" --engines vllm --vllm-urls "${urls[@]}" \
        --cases 512:128 4096:32 --concurrencies 16 32 64 128 \
        --prompt-modes unique --request-multiplier 1 --min-requests 16 --repeats 3 \
        >"$out/workload.log" 2>&1
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
      date '+completed_at=%Y-%m-%d %H:%M:%S %z' >"$out/complete.txt"
    fi
  fi
done

cleanup
date '+completed_at=%Y-%m-%d %H:%M:%S %z' >"$RESULT/complete.txt"
