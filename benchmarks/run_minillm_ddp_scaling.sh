#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/public/home/u43077/lzh}"
PROJECT="${PROJECT:-$SERVER_ROOT/ai/projects/minillm-general}"
PY="${PYTHON:-$SERVER_ROOT/python-envs/llm-py311/bin/python}"
DATASET="${DATASET_DIR:-$SERVER_ROOT/datasets/minillm-general/general-openbpe-32k-v1}"
CHECKPOINT="${CHECKPOINT:-$SERVER_ROOT/outputs/minillm-general/new-model-v1/160m-openbpe-32k-8k/latest.pt}"
RESULT_DIR="${RESULT_DIR:-$SERVER_ROOT/benchmarks/next-20260801/ddp-scaling}"
MEASURE_STEPS="${MEASURE_STEPS:-30}"

[[ -x "$PY" && -f "$PROJECT/train_general.py" ]]
[[ -f "$DATASET/manifest.json" && -f "$CHECKPOINT" ]]
(( MEASURE_STEPS >= 20 ))
mkdir -p "$RESULT_DIR"

START_STEP="$($PY - "$CHECKPOINT" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint["step"]))
PY
)"
TARGET_STEP=$((START_STEP + MEASURE_STEPS))

for world_size in 1 2 4; do
  result="$RESULT_DIR/world-${world_size}.json"
  [[ -s "$result" ]] && continue

  micro_batch=8
  accumulation=$((4 / world_size))
  eval_micro_batch=$((8 / world_size))
  log="$RESULT_DIR/world-${world_size}.log"
  scratch="/tmp/lzh-ddp-scaling-w${world_size}-$$"
  mkdir -p "$scratch"

  gpu_list="0"
  if (( world_size == 2 )); then
    gpu_list="0,1"
  elif (( world_size == 4 )); then
    gpu_list="0,1,2,3"
  fi

  threads_per_rank=$(( $(nproc) / world_size ))
  (( threads_per_rank > 8 )) && threads_per_rank=8
  export OMP_NUM_THREADS="$threads_per_rank"
  export MKL_NUM_THREADS="$threads_per_rank"
  export OPENBLAS_NUM_THREADS="$threads_per_rank"
  export TORCHINDUCTOR_COMPILE_THREADS=4
  export CUDA_VISIBLE_DEVICES="$gpu_list"

  runner=("$PY" "$PROJECT/train_general.py")
  if (( world_size > 1 )); then
    runner=(
      "$PY" -m torch.distributed.run
      --master_addr=127.0.0.1 --master_port="$((29600 + world_size))"
      --local_addr=127.0.0.1 --nproc_per_node="$world_size"
      "$PROJECT/train_general.py"
    )
  fi

  status=0
  {
    date '+%Y-%m-%d %H:%M:%S %z'
    echo "world_size=$world_size micro_batch=$micro_batch accumulation=$accumulation global_tokens_per_update=131072"
    "${runner[@]}" \
      --dataset-dir "$DATASET" --out-dir "$scratch" --resume "$CHECKPOINT" \
      --allow-dataset-change \
      --n-layer 22 --n-head 12 --num-key-value-heads 4 \
      --n-embd 768 --intermediate-size 2048 \
      --block-size 8192 --sequence-length 4096 \
      --micro-batch-size "$micro_batch" \
      --gradient-accumulation-steps "$accumulation" \
      --eval-micro-batch-size "$eval_micro_batch" \
      --batch-layout contiguous \
      --learning-rate 3e-5 --min-lr-ratio 1 \
      --warmup-steps 0 --schedule-start-step "$START_STEP" \
      --schedule-end-step "$TARGET_STEP" --max-steps "$TARGET_STEP" \
      --eval-interval "$TARGET_STEP" --eval-batches 1 \
      --save-interval "$TARGET_STEP" --keep-checkpoints 1 \
      --log-interval 1 --compile
  } >"$log" 2>&1 || status=$?

  "$PY" - "$log" "$result" "$world_size" "$micro_batch" "$accumulation" "$status" <<'PY'
import json
import statistics
import sys
from pathlib import Path

log_path, result_path = map(Path, sys.argv[1:3])
world_size, micro_batch, accumulation, status = map(int, sys.argv[3:7])
records = []
for line in log_path.read_text(encoding="utf-8").splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(row, dict) and "tokens_per_second" in row:
        records.append(row)
steady = records[5:]
payload = {
    "schema_version": 1,
    "status": "success" if status == 0 and len(steady) >= 15 else "failed",
    "exit_code": status,
    "world_size": world_size,
    "micro_batch_size": micro_batch,
    "gradient_accumulation_steps": accumulation,
    "global_tokens_per_update": 131072,
    "measured_steps": len(records),
    "steady_steps": len(steady),
    "median_tokens_per_second": (
        statistics.median(float(row["tokens_per_second"]) for row in steady)
        if steady else None
    ),
    "peak_gpu_memory_gib": (
        max(float(row["gpu_memory_gib"]) for row in records) if records else None
    ),
    "log": str(log_path),
}
result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
if payload["status"] != "success":
    raise SystemExit(1)
PY

  case "$scratch" in
    /tmp/lzh-ddp-scaling-*) rm -rf "$scratch" ;;
    *) echo "refusing to remove unexpected scratch path: $scratch" >&2; exit 2 ;;
  esac
done

"$PY" - "$RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads((root / f"world-{world}.json").read_text()) for world in (1, 2, 4)]
baseline = rows[0]["median_tokens_per_second"]
for row in rows:
    row["speedup_vs_one_gpu"] = row["median_tokens_per_second"] / baseline
    row["scaling_efficiency"] = row["speedup_vs_one_gpu"] / row["world_size"]
payload = {"schema_version": 1, "kind": "strong_scaling", "results": rows}
(root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
