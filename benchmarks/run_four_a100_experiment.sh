#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/public/home/u43077/lzh}"
RESULT_ROOT="${RESULT_ROOT:-$SERVER_ROOT/benchmarks/next-20260801}"
SCRIPT_ROOT="$RESULT_ROOT/scripts"
STREAM="$SCRIPT_ROOT/streaming_ttft_tpot.py"
SUMMARIZE="$SCRIPT_ROOT/summarize_torch_trace.py"
DDP="$SCRIPT_ROOT/run_minillm_ddp_scaling.sh"
PY=/opt/venvs/vllm/bin/python
VLLM_PORT=18000
SGLANG_PORT=18001
ACTIVE_ENGINE=""

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/torch-profiler/vllm" "$RESULT_ROOT/torch-profiler/sglang"

health() {
  curl -fsS "http://127.0.0.1:$1/health" >/dev/null 2>&1
}

wait_health() {
  local port=$1
  local attempts=${2:-72}
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    health "$port" && return 0
    sleep 5
  done
  return 1
}

stop_engine() {
  local engine=$1
  local pattern
  if [[ "$engine" == vllm ]]; then
    pattern='vllm serve .*--port 18000'
  else
    pattern='sglang.launch_server .*--port 18001'
  fi
  mapfile -t pids < <(pgrep -f "$pattern" || true)
  ((${#pids[@]} == 0)) || kill -TERM "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    mapfile -t pids < <(pgrep -f "$pattern" || true)
    ((${#pids[@]} == 0)) && break
    sleep 1
  done
  ACTIVE_ENGINE=""
}

cleanup() {
  if [[ "$ACTIVE_ENGINE" == both ]]; then
    stop_engine vllm
    stop_engine sglang
  elif [[ -n "$ACTIVE_ENGINE" ]]; then
    stop_engine "$ACTIVE_ENGINE"
  fi
}
trap cleanup EXIT

start_engine() {
  local engine=$1
  if [[ "$engine" == vllm ]]; then
    if ! health "$VLLM_PORT"; then
      setsid env CUDA_VISIBLE_DEVICES=1 MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.85 \
        VLLM_TORCH_PROFILER_DIR="$RESULT_ROOT/torch-profiler/vllm" \
        "$SERVER_ROOT/scripts/serve-vllm.sh" Qwen3-8B "$VLLM_PORT" \
        >"$RESULT_ROOT/logs/auto-vllm-server.log" 2>&1 < /dev/null &
    fi
    wait_health "$VLLM_PORT"
  else
    if ! health "$SGLANG_PORT"; then
      setsid env CUDA_VISIBLE_DEVICES=2 MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.85 \
        SGLANG_TORCH_PROFILER_DIR="$RESULT_ROOT/torch-profiler/sglang" \
        "$SERVER_ROOT/scripts/serve-sglang.sh" Qwen3-8B "$SGLANG_PORT" \
        >"$RESULT_ROOT/logs/auto-sglang-server.log" 2>&1 < /dev/null &
    fi
    wait_health "$SGLANG_PORT"
  fi
  ACTIVE_ENGINE="$engine"
}

profile_engine() {
  local engine=$1
  local port=$2
  local profile_dir="$RESULT_ROOT/torch-profiler/$engine"
  local summary="$RESULT_ROOT/torch-profiler/$engine-summary.json"
  [[ -s "$summary" ]] && return 0

  start_engine "$engine"
  curl -fsS -X POST "http://127.0.0.1:$port/start_profile" \
    >"$RESULT_ROOT/logs/$engine-start-profile.log" 2>&1 || true
  "$PY" "$STREAM" \
    --output "$RESULT_ROOT/torch-profiler/$engine-workload.json" \
    --engines "$engine" --cases 4096:128 --concurrencies 16 --repeats 3 \
    >"$RESULT_ROOT/logs/$engine-profile-workload.log" 2>&1
  curl -fsS -X POST "http://127.0.0.1:$port/stop_profile" \
    >"$RESULT_ROOT/logs/$engine-stop-profile.log" 2>&1

  trace="$(find "$profile_dir" -maxdepth 1 -type f -name '*.pt.trace.json.gz' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
  [[ -n "$trace" && -s "$trace" ]]
  "$PY" "$SUMMARIZE" "$trace" --output "$summary"
  stop_engine "$engine"
}

"$SERVER_ROOT/scripts/mount-serving-runtime.sh"

if [[ ! -s "$RESULT_ROOT/streaming-ttft-tpot.json" ]]; then
  start_engine vllm
  start_engine sglang
  ACTIVE_ENGINE="both"
  "$PY" "$STREAM" --output "$RESULT_ROOT/streaming-ttft-tpot.json" \
    >"$RESULT_ROOT/streaming-ttft-tpot.log" 2>&1
  stop_engine vllm
  stop_engine sglang
fi

profile_engine vllm "$VLLM_PORT"
profile_engine sglang "$SGLANG_PORT"

stop_engine vllm
stop_engine sglang
bash "$DDP"

date '+completed_at=%Y-%m-%d %H:%M:%S %z' >"$RESULT_ROOT/experiment-complete.txt"
