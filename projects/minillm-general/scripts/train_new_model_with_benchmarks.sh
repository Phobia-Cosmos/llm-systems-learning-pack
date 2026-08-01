#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/public/home/u43077/lzh/outputs/minillm-general/new-model-v1}"
RUN_OUT="$OUT_ROOT/160m-openbpe-32k-4k"
MILESTONE_DIR="$RUN_OUT/milestones"
TRAJECTORY="$OUT_ROOT/benchmarks/trajectory"
EXPANDED_TRAJECTORY="$OUT_ROOT/benchmarks/trajectory-expanded"
FINAL_SCHEDULE_STEP="${FINAL_SCHEDULE_STEP:-305176}"
REFERENCE_TOKENS_PER_STEP="${REFERENCE_TOKENS_PER_STEP:-32768}"
MILESTONES="${MILESTONES:-}"
MILESTONE_TOKENS="${MILESTONE_TOKENS:-500006912 1000013824 2000027648 3272015872 5000003584 8180006912 10000007168}"
BENCHMARK_COMMAND="${BENCHMARK_COMMAND:-$ROOT/scripts/benchmark_checkpoint.sh}"
BENCHMARK_MAX_ATTEMPTS="${BENCHMARK_MAX_ATTEMPTS:-3}"
BENCHMARK_RETRY_SECONDS="${BENCHMARK_RETRY_SECONDS:-30}"
BASE_EVAL_LIMIT_PER_TASK="${BASE_EVAL_LIMIT_PER_TASK:-${EVAL_LIMIT_PER_TASK:-500}}"
EXPANDED_EVAL_LIMIT_PER_TASK="${EXPANDED_EVAL_LIMIT_PER_TASK:-2000}"
CONTROLLER_PYTHON="${CONTROLLER_PYTHON:-/public/home/u43077/lzh/python-envs/llm-py311/bin/python}"
SUMMARY_COMMAND="${SUMMARY_COMMAND:-}"
STOP_FILE="$OUT_ROOT/STOP"
LOCK_FILE="$OUT_ROOT/controller.lock"
PID_FILE="$OUT_ROOT/controller.pid"
BENCHMARK_EVENT_LOG="$OUT_ROOT/benchmarks/benchmark-events.log"

if [[ ! "$BENCHMARK_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "BENCHMARK_MAX_ATTEMPTS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$BENCHMARK_RETRY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "BENCHMARK_RETRY_SECONDS must be a non-negative integer" >&2
  exit 2
fi

benchmark_output_is_valid() {
  local output="$1"
  [[ -s "$output" ]] || return 1
  "$CONTROLLER_PYTHON" -c \
    'import json, pathlib, sys; value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")); required = {"ceval", "cmmlu", "arc_easy", "arc_challenge", "hellaswag"}; tasks = value.get("tasks") if isinstance(value, dict) else None; valid = isinstance(tasks, dict) and required.issubset(tasks) and all(isinstance(tasks[name], dict) and isinstance(tasks[name].get("metrics"), dict) and tasks[name]["metrics"] for name in required); raise SystemExit(0 if valid else 1)' \
    "$output" >/dev/null 2>&1
}

record_benchmark_event() {
  local status="$1"
  local kind="$2"
  local target_step="$3"
  local output="$4"
  local detail="$5"
  printf '%s status=%s kind=%s step=%s output=%q %s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    "$status" "$kind" "$target_step" "$output" "$detail" \
    >> "$BENCHMARK_EVENT_LOG"
}

quarantine_invalid_benchmark() {
  local kind="$1"
  local target_step="$2"
  local output="$3"
  [[ -e "$output" ]] || return 0
  benchmark_output_is_valid "$output" && return 0

  local quarantined="${output}.invalid-$(date -u '+%Y%m%dT%H%M%SZ')-$$"
  mv "$output" "$quarantined"
  record_benchmark_event \
    "invalid_output_quarantined" "$kind" "$target_step" "$output" \
    "quarantined=$(printf '%q' "$quarantined")"
  echo "invalid benchmark output quarantined: $output -> $quarantined" >&2
}

write_benchmark_failure() {
  local failure_record="$1"
  local kind="$2"
  local target_step="$3"
  local checkpoint="$4"
  local output="$5"
  local attempts="$6"
  local last_status="$7"
  local last_reason="$8"
  local temporary="${failure_record}.tmp.$$"
  {
    printf 'status=failed\n'
    printf 'failed_at_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'kind=%s\n' "$kind"
    printf 'step=%s\n' "$target_step"
    printf 'checkpoint=%q\n' "$checkpoint"
    printf 'output=%q\n' "$output"
    printf 'attempts=%s\n' "$attempts"
    printf 'last_status=%s\n' "$last_status"
    printf 'last_reason=%s\n' "$last_reason"
  } > "$temporary"
  mv "$temporary" "$failure_record"
}

run_benchmark_with_retry() {
  local kind="$1"
  local target_step="$2"
  local checkpoint="$3"
  local output="$4"
  local limit_per_task="$5"
  local failure_record="${output}.failed"
  local attempt
  local command_status=0
  local last_status=1
  local last_reason="not_started"

  quarantine_invalid_benchmark "$kind" "$target_step" "$output"
  if benchmark_output_is_valid "$output"; then
    return 0
  fi

  for ((attempt = 1; attempt <= BENCHMARK_MAX_ATTEMPTS; attempt++)); do
    local attempt_output="${output}.attempt-${BASHPID}-${attempt}.tmp"
    rm -f "$attempt_output"
    echo "benchmark attempt $attempt/$BENCHMARK_MAX_ATTEMPTS: kind=$kind step=$target_step"

    command_status=0
    RUN_EXPANDED_EVAL=0 EVAL_LIMIT_PER_TASK="$limit_per_task" \
      "$BENCHMARK_COMMAND" "$checkpoint" "$attempt_output" \
      || command_status=$?

    if (( command_status == 0 )) && benchmark_output_is_valid "$attempt_output"; then
      mv "$attempt_output" "$output"
      if [[ -f "$failure_record" ]]; then
        record_benchmark_event \
          "recovered" "$kind" "$target_step" "$output" \
          "attempt=$attempt"
        rm -f "$failure_record"
      fi
      echo "benchmark succeeded: kind=$kind step=$target_step output=$output"
      return 0
    fi

    if (( command_status == 0 )); then
      last_status=65
      last_reason="command_succeeded_but_output_was_empty_or_invalid_json"
    else
      last_status="$command_status"
      last_reason="command_failed"
    fi
    rm -f "$attempt_output"
    echo \
      "benchmark attempt failed: kind=$kind step=$target_step attempt=$attempt/$BENCHMARK_MAX_ATTEMPTS status=$last_status reason=$last_reason" \
      >&2
    if (( attempt < BENCHMARK_MAX_ATTEMPTS && BENCHMARK_RETRY_SECONDS > 0 )); then
      sleep "$BENCHMARK_RETRY_SECONDS"
    fi
  done

  write_benchmark_failure \
    "$failure_record" "$kind" "$target_step" "$checkpoint" "$output" \
    "$BENCHMARK_MAX_ATTEMPTS" "$last_status" "$last_reason"
  record_benchmark_event \
    "failed" "$kind" "$target_step" "$output" \
    "attempts=$BENCHMARK_MAX_ATTEMPTS last_status=$last_status reason=$last_reason"
  echo \
    "benchmark exhausted retries; training controller will continue: kind=$kind step=$target_step failure_record=$failure_record" \
    >&2
  return "$last_status"
}

summarize_trajectory() {
  local command
  if [[ -n "$SUMMARY_COMMAND" ]]; then
    command=("$SUMMARY_COMMAND")
  else
    command=("$CONTROLLER_PYTHON" "$ROOT/scripts/summarize_training_trajectory.py")
  fi
  "${command[@]}" \
    --training-log "$OUT_ROOT/160-train.log" \
    --benchmark-dir "$TRAJECTORY" \
    --output-json "$OUT_ROOT/benchmarks/trajectory-summary.json" \
    --output-markdown "$OUT_ROOT/benchmarks/trajectory-summary.md"
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another new-model controller is already running" >&2
  exit 3
fi
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

mkdir -p "$TRAJECTORY" "$EXPANDED_TRAJECTORY" "$MILESTONE_DIR"
ln -sfn ../tokenizer.json "$MILESTONE_DIR/tokenizer.json"

process_benchmarks() {
  local target_step="$1"
  local checkpoint="$2"
  local expanded="$3"
  local benchmark="$TRAJECTORY/step-$(printf '%08d' "$target_step").json"
  if ! benchmark_output_is_valid "$benchmark"; then
    run_benchmark_with_retry \
      "trajectory" "$target_step" "$checkpoint" "$benchmark" \
      "$BASE_EVAL_LIMIT_PER_TASK" || true
  fi
  if [[ "$expanded" == "1" ]]; then
    local expanded_benchmark="$EXPANDED_TRAJECTORY/step-$(printf '%08d' "$target_step").json"
    if ! benchmark_output_is_valid "$expanded_benchmark"; then
      run_benchmark_with_retry \
        "trajectory-expanded" "$target_step" "$checkpoint" "$expanded_benchmark" \
        "$EXPANDED_EVAL_LIMIT_PER_TASK" || true
    fi
  fi
  summary_status=0
  summarize_trajectory || summary_status=$?
  if (( summary_status != 0 )); then
    record_benchmark_event \
      "summary_failed" "trajectory-summary" "$target_step" \
      "$OUT_ROOT/benchmarks/trajectory-summary.json" \
      "status=$summary_status"
    echo \
      "trajectory summary failed with status $summary_status; training controller will continue" \
      >&2
  fi
}

run_legacy_milestones() {
  local target_step checkpoint pinned_checkpoint expanded
  for target_step in $MILESTONES; do
    if [[ -f "$STOP_FILE" ]]; then
      echo "stop requested by $STOP_FILE"
      return 0
    fi
    if (( target_step > FINAL_SCHEDULE_STEP )); then
      echo "milestone $target_step exceeds final schedule $FINAL_SCHEDULE_STEP" >&2
      return 2
    fi
    checkpoint="$RUN_OUT/step-$(printf '%08d' "$target_step").pt"
    pinned_checkpoint="$MILESTONE_DIR/step-$(printf '%08d' "$target_step").pt"
    if [[ -f "$pinned_checkpoint" ]]; then
      checkpoint="$pinned_checkpoint"
    elif [[ ! -f "$checkpoint" ]]; then
      TARGET_STEP="$target_step" FINAL_SCHEDULE_STEP="$FINAL_SCHEDULE_STEP" \
        "$ROOT/scripts/run_new_model_training.sh" 160-train
    fi
    if [[ ! -f "$pinned_checkpoint" ]]; then
      ln "$checkpoint" "$pinned_checkpoint"
    fi
    expanded=0
    (( target_step >= 61036 )) && expanded=1
    process_benchmarks "$target_step" "$pinned_checkpoint" "$expanded"
  done
}

read_checkpoint_fields() {
  "$CONTROLLER_PYTHON" - "$1" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint["step"]))
print(int(checkpoint.get("tokens_processed", 0)))
PY
}

write_token_milestone_record() {
  local record="$1"
  local target_tokens="$2"
  local actual_step="$3"
  local actual_tokens="$4"
  local checkpoint="$5"
  local temporary="${record}.tmp.$$"
  "$CONTROLLER_PYTHON" - "$temporary" "$target_tokens" "$actual_step" "$actual_tokens" "$checkpoint" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "target_tokens": int(sys.argv[2]),
    "actual_step": int(sys.argv[3]),
    "actual_tokens": int(sys.argv[4]),
    "checkpoint": pathlib.Path(sys.argv[5]).name,
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  mv "$temporary" "$record"
}

read_token_milestone_record() {
  "$CONTROLLER_PYTHON" - "$1" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(payload["actual_step"]))
print(int(payload["actual_tokens"]))
print(payload["checkpoint"])
PY
}

run_token_milestones() {
  local target_tokens legacy_step legacy_checkpoint record
  local checkpoint pinned_checkpoint actual_step actual_tokens expanded
  local latest_fields latest_step latest_tokens visible_gpus train_gpus
  local micro_batch accumulation tokens_per_update additional_steps target_step
  [[ "$REFERENCE_TOKENS_PER_STEP" =~ ^[1-9][0-9]*$ ]] || {
    echo "REFERENCE_TOKENS_PER_STEP must be positive" >&2
    return 2
  }
  [[ -f "$RUN_OUT/latest.pt" ]] || {
    echo "token-driven controller requires $RUN_OUT/latest.pt" >&2
    return 2
  }

  for target_tokens in $MILESTONE_TOKENS; do
    if [[ -f "$STOP_FILE" ]]; then
      echo "stop requested by $STOP_FILE"
      return 0
    fi
    [[ "$target_tokens" =~ ^[1-9][0-9]*$ ]] || {
      echo "invalid milestone token count: $target_tokens" >&2
      return 2
    }
    legacy_step=$((target_tokens / REFERENCE_TOKENS_PER_STEP))
    legacy_checkpoint="$MILESTONE_DIR/step-$(printf '%08d' "$legacy_step").pt"
    record="$MILESTONE_DIR/target-tokens-$(printf '%012d' "$target_tokens").json"

    if [[ -f "$record" ]]; then
      mapfile -t milestone_fields < <(read_token_milestone_record "$record")
      actual_step="${milestone_fields[0]}"
      actual_tokens="${milestone_fields[1]}"
      pinned_checkpoint="$MILESTONE_DIR/${milestone_fields[2]}"
      [[ -f "$pinned_checkpoint" ]] || {
        echo "recorded milestone checkpoint is missing: $pinned_checkpoint" >&2
        return 2
      }
    elif (( target_tokens % REFERENCE_TOKENS_PER_STEP == 0 )) && [[ -f "$legacy_checkpoint" ]]; then
      actual_step="$legacy_step"
      actual_tokens="$target_tokens"
      pinned_checkpoint="$legacy_checkpoint"
    else
      mapfile -t latest_fields < <(read_checkpoint_fields "$RUN_OUT/latest.pt")
      latest_step="${latest_fields[0]}"
      latest_tokens="${latest_fields[1]}"
      if (( latest_tokens < target_tokens )); then
        visible_gpus="$($CONTROLLER_PYTHON -c 'import torch; print(torch.cuda.device_count())')"
        train_gpus="${TRAIN_GPUS:-$visible_gpus}"
        [[ "$train_gpus" =~ ^[1-9][0-9]*$ ]] && (( train_gpus <= visible_gpus )) || {
          echo "invalid TRAIN_GPUS=$train_gpus for $visible_gpus visible GPUs" >&2
          return 2
        }
        micro_batch="${MICRO_BATCH:-8}"
        accumulation="${ACCUMULATION:-1}"
        tokens_per_update=$((4096 * micro_batch * accumulation * train_gpus))
        additional_steps=$(((target_tokens - latest_tokens + tokens_per_update - 1) / tokens_per_update))
        target_step=$((latest_step + additional_steps))
        echo "token milestone: target=$target_tokens current=$latest_tokens world=$train_gpus tokens_per_update=$tokens_per_update target_step=$target_step"
        TARGET_STEP="$target_step" FINAL_SCHEDULE_STEP="$FINAL_SCHEDULE_STEP" \
          TRAIN_GPUS="$train_gpus" MICRO_BATCH="$micro_batch" ACCUMULATION="$accumulation" \
          "$ROOT/scripts/run_new_model_training.sh" 160-train
      fi

      mapfile -t latest_fields < <(read_checkpoint_fields "$RUN_OUT/latest.pt")
      actual_step="${latest_fields[0]}"
      actual_tokens="${latest_fields[1]}"
      (( actual_tokens >= target_tokens )) || {
        echo "training stopped before token milestone $target_tokens" >&2
        return 2
      }
      checkpoint="$RUN_OUT/step-$(printf '%08d' "$actual_step").pt"
      [[ -f "$checkpoint" ]] || checkpoint="$RUN_OUT/latest.pt"
      pinned_checkpoint="$MILESTONE_DIR/step-$(printf '%08d' "$actual_step").pt"
      if [[ ! -f "$pinned_checkpoint" ]]; then
        ln "$checkpoint" "$pinned_checkpoint"
      fi
      write_token_milestone_record \
        "$record" "$target_tokens" "$actual_step" "$actual_tokens" "$pinned_checkpoint"
    fi

    expanded=0
    (( target_tokens >= 2000027648 )) && expanded=1
    process_benchmarks "$actual_step" "$pinned_checkpoint" "$expanded"
  done
}

if [[ -n "$MILESTONES" ]]; then
  run_legacy_milestones
else
  run_token_milestones
fi
