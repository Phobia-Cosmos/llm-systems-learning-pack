#!/usr/bin/env bash
set -euo pipefail

ROOT="${SERVER_ROOT:-/public/home/u43077/lzh}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/benchmarks/next-20260802}"
SCRIPTS="${SCRIPT_ROOT:-$RESULT_ROOT/scripts}"
RESULT="$RESULT_ROOT/serving-ab"
PORT=18200
ACTIVE_ENGINE=""
mkdir -p "$RESULT"

stop_service() {
  local pattern
  if [[ "$ACTIVE_ENGINE" == vllm ]]; then
    pattern='vllm serve .*--port 18200'
  else
    pattern='sglang.launch_server .*--port 18200'
  fi
  mapfile -t pids < <(pgrep -f "$pattern" || true)
  ((${#pids[@]} == 0)) || kill -TERM "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    pgrep -f "$pattern" >/dev/null || break
    sleep 1
  done
  ACTIVE_ENGINE=""
}
trap '[[ -z "$ACTIVE_ENGINE" ]] || stop_service' EXIT

run_config() {
  local engine=$1
  local name=$2
  local extra=$3
  local out="$RESULT/$name"
  [[ -s "$out/complete.txt" ]] && return 0
  mkdir -p "$out"
  ACTIVE_ENGINE="$engine"

  if [[ "$engine" == vllm ]]; then
    setsid env CUDA_VISIBLE_DEVICES=0 MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.85 \
      EXTRA_ARGS="$extra" "$ROOT/scripts/serve-vllm.sh" Qwen3-8B "$PORT" \
      >"$out/server.log" 2>&1 < /dev/null &
  else
    setsid env CUDA_VISIBLE_DEVICES=0 MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.85 \
      EXTRA_ARGS="$extra" "$ROOT/scripts/serve-sglang.sh" Qwen3-8B "$PORT" \
      >"$out/server.log" 2>&1 < /dev/null &
  fi

  for _ in $(seq 1 90); do
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 5
  done
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null
  nvidia-smi dmon -s pucvmet -d 1 -o DT >"$out/nvidia-smi-dmon.log" 2>&1 &
  monitor_pid=$!

  urls=("http://127.0.0.1:$PORT")
  /opt/venvs/vllm/bin/python "$SCRIPTS/streaming_ttft_tpot.py" \
    --output "$out/results.json" --engines "$engine" \
    --cases 512:128 4096:128 7680:32 512:512 4096:512 \
    --concurrencies 1 16 32 64 --prompt-modes unique shared \
    --request-multiplier 1 --min-requests 4 --repeats 2 \
    --vllm-urls "${urls[@]}" --sglang-urls "${urls[@]}" \
    >"$out/workload.log" 2>&1
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  date '+completed_at=%Y-%m-%d %H:%M:%S %z' >"$out/complete.txt"
  stop_service
}

run_config vllm vllm-default ''
run_config vllm vllm-prefix-cache '--enable-prefix-caching'
run_config vllm vllm-eager '--enforce-eager'
run_config sglang sglang-default ''
run_config sglang sglang-no-radix '--disable-radix-cache'
run_config sglang sglang-chunk-2048 '--chunked-prefill-size 2048'
run_config sglang sglang-eager '--disable-cuda-graph'
date '+completed_at=%Y-%m-%d %H:%M:%S %z' >"$RESULT/complete.txt"
