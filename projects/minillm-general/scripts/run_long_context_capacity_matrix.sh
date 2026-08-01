#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/public/home/u43077/lzh/python-envs/llm-py311/bin/python}"
OUT_ROOT="${OUT_ROOT:-/public/home/u43077/lzh/outputs/minillm-general/new-model-v1}"
DATASET="${DATASET_DIR:-/public/home/u43077/lzh/datasets/minillm-general/general-openbpe-32k-8k-v1}"
RESUME="${RESUME:?Set RESUME to the selected stable 4K checkpoint}"
ADDITIONAL_STEPS="${ADDITIONAL_STEPS:-30}"
SPECS="${SPECS:-1:4 2:2 4:1}"

[[ -x "$PY" ]] || { echo "Python interpreter not found: $PY" >&2; exit 2; }
[[ -f "$RESUME" ]] || { echo "checkpoint not found: $RESUME" >&2; exit 2; }
[[ -f "$DATASET/manifest.json" ]] || { echo "dataset not found: $DATASET" >&2; exit 2; }
(( ADDITIONAL_STEPS >= 20 )) || {
  echo "ADDITIONAL_STEPS must be at least 20 for a meaningful compiled capacity scan" >&2
  exit 2
}
VISIBLE_GPUS="$($PY -c 'import torch; print(torch.cuda.device_count())')"
TRAIN_GPUS="${TRAIN_GPUS:-auto}"
if [[ "$TRAIN_GPUS" == "auto" ]]; then
  TRAIN_GPUS="$VISIBLE_GPUS"
fi
[[ "$TRAIN_GPUS" =~ ^[1-9][0-9]*$ ]] || {
  echo "TRAIN_GPUS must be auto or a positive integer" >&2
  exit 2
}
(( TRAIN_GPUS <= VISIBLE_GPUS )) || {
  echo "requested $TRAIN_GPUS GPUs, but only $VISIBLE_GPUS are visible" >&2
  exit 2
}
RESULT_DIR="$OUT_ROOT/8k-capacity-results-g${TRAIN_GPUS}"
mkdir -p "$RESULT_DIR"

for spec in $SPECS; do
  IFS=: read -r micro accumulation <<<"$spec"
  [[ "$micro" =~ ^[1-9][0-9]*$ && "$accumulation" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid capacity spec: $spec" >&2
    exit 2
  }
  if (( micro * accumulation != 4 )); then
    echo "capacity spec $spec does not preserve 32,768 tokens/rank/update" >&2
    exit 2
  fi

  output="$OUT_ROOT/160m-8k-capacity-g${TRAIN_GPUS}-m${micro}-a${accumulation}"
  result="$RESULT_DIR/m${micro}-a${accumulation}.json"
  log="$OUT_ROOT/8k-capacity-g${TRAIN_GPUS}-m${micro}-a${accumulation}.log"
  if [[ -f "$result" ]]; then
    echo "already measured: $result"
    continue
  fi
  if [[ -e "$output" ]]; then
    echo "capacity output exists without a result; inspect it before retrying: $output" >&2
    exit 2
  fi

  status=0
  OUT_DIR="$output" OUT_ROOT="$OUT_ROOT" DATASET_DIR="$DATASET" \
    RESUME="$RESUME" MICRO_BATCH="$micro" ACCUMULATION="$accumulation" \
    TRAIN_GPUS="$TRAIN_GPUS" ADDITIONAL_STEPS="$ADDITIONAL_STEPS" \
    "$ROOT/scripts/run_long_context_training.sh" capacity || status=$?

  "$PY" - "$result" "$log" "$output" "$micro" "$accumulation" "$TRAIN_GPUS" "$status" <<'PY'
import json
import os
import statistics
import sys
from pathlib import Path

result_path, log_path, output_dir = map(Path, sys.argv[1:4])
micro_batch, accumulation, world_size, status = map(int, sys.argv[4:8])
records = []
if log_path.is_file():
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "tokens_per_second" in value:
            records.append(value)
steady = records[5:] if len(records) > 5 else records
payload = {
    "schema_version": 2,
    "status": "success" if status == 0 else "failed",
    "exit_code": status,
    "micro_batch_size": micro_batch,
    "gradient_accumulation_steps": accumulation,
    "world_size": world_size,
    "tokens_per_rank_optimizer_step": 8192 * micro_batch * accumulation,
    "tokens_per_optimizer_step": 8192 * micro_batch * accumulation * world_size,
    "measured_steps": len(records),
    "steady_steps": len(steady),
    "median_tokens_per_second": (
        statistics.median(float(row["tokens_per_second"]) for row in steady)
        if steady
        else None
    ),
    "peak_gpu_memory_gib": (
        max(float(row["gpu_memory_gib"]) for row in records) if records else None
    ),
    "last_train_loss": float(records[-1]["train_loss"]) if records else None,
    "output_dir": str(output_dir),
    "log": str(log_path),
}
result_path.parent.mkdir(parents=True, exist_ok=True)
temporary = result_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, result_path)
PY
done

"$PY" - "$RESULT_DIR" "$RESULT_DIR/summary.json" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
results = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(root.glob("m*-a*.json"))
]
successful = [
    result
    for result in results
    if result["status"] == "success"
    and result["steady_steps"] >= 10
    and result["median_tokens_per_second"] is not None
]
recommended = (
    max(successful, key=lambda result: result["median_tokens_per_second"])
    if successful
    else None
)
payload = {
    "schema_version": 2,
    "world_size": int(results[0]["world_size"]) if results else None,
    "results": results,
    "recommended": (
        {
            "micro_batch_size": recommended["micro_batch_size"],
            "gradient_accumulation_steps": recommended["gradient_accumulation_steps"],
            "world_size": recommended["world_size"],
            "tokens_per_optimizer_step": recommended["tokens_per_optimizer_step"],
            "median_tokens_per_second": recommended["median_tokens_per_second"],
            "peak_gpu_memory_gib": recommended["peak_gpu_memory_gib"],
        }
        if recommended is not None
        else None
    ),
}
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
