#!/usr/bin/env bash
set -euo pipefail

ROOT="${SERVER_ROOT:-/public/home/u43077/lzh}"
RESULT="${RESULT_ROOT:-$ROOT/benchmarks/next-20260802}/sglang-slim-profile"
SCRIPTS="${SCRIPT_ROOT:-$ROOT/benchmarks/next-20260802/scripts}"
PORT=18001
mkdir -p "$RESULT/trace"

cleanup() {
  mapfile -t pids < <(pgrep -f 'sglang.launch_server .*--port 18001' || true)
  ((${#pids[@]} == 0)) || kill -TERM "${pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT

setsid env CUDA_VISIBLE_DEVICES=0 MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.85 \
  SGLANG_TORCH_PROFILER_DIR="$RESULT/trace" \
  "$ROOT/scripts/serve-sglang.sh" Qwen3-8B "$PORT" >"$RESULT/server.log" 2>&1 < /dev/null &

for _ in $(seq 1 90); do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 5
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

curl -fsS -X POST -H 'Content-Type: application/json' \
  -d '{"activities":["GPU"],"with_stack":false,"record_shapes":false}' \
  "http://127.0.0.1:$PORT/start_profile" >"$RESULT/start-profile.log"

/opt/venvs/vllm/bin/python "$SCRIPTS/streaming_ttft_tpot.py" \
  --output "$RESULT/workload.json" --engines sglang \
  --sglang-urls "http://127.0.0.1:$PORT" \
  --cases 4096:32 --concurrencies 8 --prompt-modes unique \
  --request-multiplier 1 --min-requests 8 --repeats 1 \
  >"$RESULT/workload.log" 2>&1

curl -fsS -X POST "http://127.0.0.1:$PORT/stop_profile" >"$RESULT/stop-profile.log"
trace="$(find "$RESULT/trace" -maxdepth 1 -type f -name '*.trace.json.gz' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
[[ -n "$trace" && -s "$trace" ]]
/opt/venvs/vllm/bin/python "$SCRIPTS/summarize_torch_trace.py" "$trace" --output "$RESULT/summary.json"
date '+completed_at=%Y-%m-%d %H:%M:%S %z' >"$RESULT/complete.txt"
