#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/public/home/u43077/lzh/python-envs/llm-py311/bin/python}"
OUT_ROOT="${OUT_ROOT:-/public/home/u43077/lzh/outputs/minillm-general/new-model-v1}"
PLAN="${STAGE_PLAN:-$OUT_ROOT/8k-stage-plan.json}"
RUN_OUT="${LONG_OUT_DIR:-$OUT_ROOT/160m-openbpe-32k-8k}"
MILESTONE_DIR="$RUN_OUT/milestones"
LONG_EVAL_DIR="$OUT_ROOT/benchmarks/long-context-trajectory"
CAPABILITY_DIR="$OUT_ROOT/benchmarks/long-context-capability"
REGRESSION_DATASET="${REGRESSION_DATASET_DIR:-/public/home/u43077/lzh/datasets/minillm-general/general-openbpe-32k-v1}"
BENCHMARK_COMMAND="${BENCHMARK_COMMAND:-$ROOT/scripts/benchmark_checkpoint.sh}"
MAX_ATTEMPTS="${EVAL_MAX_ATTEMPTS:-3}"
RETRY_SECONDS="${EVAL_RETRY_SECONDS:-30}"
STOP_FILE="$OUT_ROOT/STOP-8K"
LOCK_FILE="$OUT_ROOT/controller-8k.lock"
PID_FILE="$OUT_ROOT/controller-8k.pid"
EVENT_LOG="$OUT_ROOT/benchmarks/long-context-events.log"

[[ -x "$PY" ]] || { echo "Python interpreter not found: $PY" >&2; exit 2; }
[[ -f "$PLAN" ]] || { echo "stage plan not found: $PLAN" >&2; exit 2; }
[[ -f "$REGRESSION_DATASET/manifest.json" ]] || {
  echo "4K regression dataset not found: $REGRESSION_DATASET" >&2
  exit 2
}
[[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || {
  echo "EVAL_MAX_ATTEMPTS must be positive" >&2
  exit 2
}
[[ "$RETRY_SECONDS" =~ ^[0-9]+$ ]] || {
  echo "EVAL_RETRY_SECONDS must be non-negative" >&2
  exit 2
}

mapfile -t PLAN_FIELDS < <("$PY" - "$PLAN" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve()
plan = json.loads(path.read_text(encoding="utf-8"))
if plan.get("schema_version") != 2:
    raise SystemExit("unsupported 8K stage plan schema")
required = ("base_checkpoint", "dataset", "capacity", "milestones")
if any(key not in plan for key in required):
    raise SystemExit("8K stage plan is incomplete")
base = pathlib.Path(plan["base_checkpoint"]["path"])
dataset = pathlib.Path(plan["dataset"]["path"])
if not base.is_file() or not (dataset / "manifest.json").is_file():
    raise SystemExit("8K stage plan references missing artifacts")
selected = plan["capacity"]["selected"]
print(base)
print(dataset)
print(int(selected["micro_batch_size"]))
print(int(selected["gradient_accumulation_steps"]))
print(int(plan["world_size"]))
print(int(plan["stage_start_step"]))
print(int(plan["stage_end_step"]))
print(plan["dataset"]["manifest_sha256"])
PY
)
(( ${#PLAN_FIELDS[@]} == 8 )) || { echo "could not read 8K stage plan" >&2; exit 2; }
BASE_CHECKPOINT="${PLAN_FIELDS[0]}"
DATASET="${PLAN_FIELDS[1]}"
MICRO_BATCH="${PLAN_FIELDS[2]}"
ACCUMULATION="${PLAN_FIELDS[3]}"
TRAIN_GPUS="${PLAN_FIELDS[4]}"
STAGE_START_STEP="${PLAN_FIELDS[5]}"
STAGE_END_STEP="${PLAN_FIELDS[6]}"
LONG_MANIFEST_SHA256="${PLAN_FIELDS[7]}"

capability_valid() {
  local output="$1"
  [[ -s "$output" ]] || return 1
  "$PY" - "$output" >/dev/null 2>&1 <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
required = {"ceval", "cmmlu", "arc_easy", "arc_challenge", "hellaswag"}
tasks = payload.get("tasks") if isinstance(payload, dict) else None
raise SystemExit(0 if isinstance(tasks, dict) and required.issubset(tasks) else 1)
PY
}

long_eval_valid() {
  local output="$1"
  [[ -s "$output" ]] || return 1
  "$PY" - "$output" >/dev/null 2>&1 <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
required = {"checkpoint", "fixed_record_test", "regression_loss", "passkey_retrieval", "cache_parity"}
valid = isinstance(payload, dict) and required.issubset(payload)
valid = valid and payload["fixed_record_test"].get("position_coverage", {}).get("complete") is True
raise SystemExit(0 if valid else 1)
PY
}

record_event() {
  local status="$1" kind="$2" step="$3" output="$4" detail="$5"
  printf '%s status=%s kind=%s step=%s output=%q %s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$status" "$kind" "$step" "$output" "$detail" \
    >> "$EVENT_LOG"
}

record_failure() {
  local output="$1" kind="$2" step="$3" status="$4"
  local failure="${output}.failed" temporary="${output}.failed.tmp.$$"
  {
    printf 'status=failed\nkind=%s\nstep=%s\nlast_status=%s\n' "$kind" "$step" "$status"
    printf 'failed_at_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } > "$temporary"
  mv "$temporary" "$failure"
  record_event failed "$kind" "$step" "$output" "attempts=$MAX_ATTEMPTS last_status=$status"
}

run_capability() {
  local step="$1" checkpoint="$2" output="$3" attempt status temporary
  capability_valid "$output" && return 0
  [[ ! -e "$output" ]] || mv "$output" "${output}.invalid-$(date -u '+%Y%m%dT%H%M%SZ')-$$"
  status=1
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    temporary="${output}.attempt-${BASHPID}-${attempt}.tmp"
    rm -f "$temporary"
    status=0
    RUN_EXPANDED_EVAL=0 EVAL_LIMIT_PER_TASK="${CAPABILITY_LIMIT_PER_TASK:-2000}" \
      "$BENCHMARK_COMMAND" "$checkpoint" "$temporary" || status=$?
    if (( status == 0 )) && capability_valid "$temporary"; then
      mv "$temporary" "$output"
      rm -f "${output}.failed"
      return 0
    fi
    rm -f "$temporary"
    (( attempt == MAX_ATTEMPTS || RETRY_SECONDS == 0 )) || sleep "$RETRY_SECONDS"
  done
  record_failure "$output" capability "$step" "$status"
  return "$status"
}

run_long_eval() {
  local step="$1" checkpoint="$2" output="$3" attempt status temporary
  long_eval_valid "$output" && return 0
  [[ ! -e "$output" ]] || mv "$output" "${output}.invalid-$(date -u '+%Y%m%dT%H%M%SZ')-$$"
  status=1
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    temporary="${output}.attempt-${BASHPID}-${attempt}.tmp"
    rm -f "$temporary"
    status=0
    "$PY" "$ROOT/scripts/evaluate_long_context.py" \
      --checkpoint "$checkpoint" \
      --tokenizer "$DATASET/tokenizer.json" \
      --long-dataset-dir "$DATASET" \
      --expected-long-manifest-sha256 "$LONG_MANIFEST_SHA256" \
      --test-records "${LONG_TEST_RECORDS:-16}" \
      --test-batch-size "${LONG_TEST_BATCH_SIZE:-1}" \
      --regression-dataset-dir "$REGRESSION_DATASET" \
      --regression-sequence-length 4096 \
      --regression-records "${REGRESSION_RECORDS:-16}" \
      --passkey-lengths 2048,4096,8192 \
      --parity-lengths 512,4096,8192 \
      --output "$temporary" || status=$?
    if (( status == 0 )) && long_eval_valid "$temporary"; then
      mv "$temporary" "$output"
      rm -f "${output}.failed"
      return 0
    fi
    rm -f "$temporary"
    (( attempt == MAX_ATTEMPTS || RETRY_SECONDS == 0 )) || sleep "$RETRY_SECONDS"
  done
  record_failure "$output" long-context "$step" "$status"
  return "$status"
}

summarize() {
  "$PY" "$ROOT/scripts/summarize_long_context_trajectory.py" \
    --plan "$PLAN" \
    --long-eval-dir "$LONG_EVAL_DIR" \
    --capability-dir "$CAPABILITY_DIR" \
    --output-json "$OUT_ROOT/benchmarks/long-context-summary.json" \
    --output-markdown "$OUT_ROOT/benchmarks/long-context-summary.md"
}

exec 9>"$LOCK_FILE"
flock -n 9 || { echo "another 8K controller is already running" >&2; exit 3; }
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT
mkdir -p "$RUN_OUT" "$MILESTONE_DIR" "$LONG_EVAL_DIR" "$CAPABILITY_DIR"
ln -sfn "$DATASET/tokenizer.json" "$MILESTONE_DIR/tokenizer.json"

while IFS=$'\t' read -r requested_tokens target_step actual_tokens; do
  [[ -n "$target_step" ]] || continue
  if [[ -f "$STOP_FILE" ]]; then
    echo "stop requested by $STOP_FILE"
    exit 0
  fi
  checkpoint="$BASE_CHECKPOINT"
  if (( requested_tokens > 0 )); then
    pinned="$MILESTONE_DIR/step-$(printf '%08d' "$target_step").pt"
    checkpoint="$RUN_OUT/step-$(printf '%08d' "$target_step").pt"
    if [[ -f "$pinned" ]]; then
      checkpoint="$pinned"
    elif [[ ! -f "$checkpoint" ]]; then
      RESUME="$BASE_CHECKPOINT" DATASET_DIR="$DATASET" OUT_DIR="$RUN_OUT" \
        MICRO_BATCH="$MICRO_BATCH" ACCUMULATION="$ACCUMULATION" TRAIN_GPUS="$TRAIN_GPUS" \
        STAGE_START_STEP="$STAGE_START_STEP" STAGE_END_STEP="$STAGE_END_STEP" \
        TARGET_STEP="$target_step" \
        "$ROOT/scripts/run_long_context_training.sh" train
    fi
    if [[ ! -f "$pinned" ]]; then
      ln "$checkpoint" "$pinned"
    fi
    checkpoint="$pinned"
  fi

  long_output="$LONG_EVAL_DIR/step-$(printf '%08d' "$target_step").json"
  capability_output="$CAPABILITY_DIR/step-$(printf '%08d' "$target_step").json"
  run_long_eval "$target_step" "$checkpoint" "$long_output" || true
  run_capability "$target_step" "$checkpoint" "$capability_output" || true
  summarize || record_event failed summary "$target_step" \
    "$OUT_ROOT/benchmarks/long-context-summary.json" "requested_additional_tokens=$requested_tokens actual_additional_tokens=$actual_tokens"
done < <("$PY" - "$PLAN" <<'PY'
import json
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for milestone in plan["milestones"]:
    print(
        int(milestone["requested_additional_tokens"]),
        int(milestone["target_step"]),
        int(milestone["actual_additional_tokens"]),
        sep="\t",
    )
PY
)
