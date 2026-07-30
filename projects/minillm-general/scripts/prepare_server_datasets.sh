#!/usr/bin/env bash
set -euo pipefail

ROOT="${SERVER_ROOT:-/public/home/u43077/lzh}"
PY="${EVAL_PYTHON:-$ROOT/python-envs/minillm-eval-py311/bin/python}"
PROJECT="$ROOT/ai/projects/minillm-general"
MINIMIND_SOURCE="$ROOT/datasets/minimind/pretrain_t2t.jsonl"
MINIMIND_EXPECTED_BYTES="${MINIMIND_EXPECTED_BYTES:-8275074893}"
MINIMIND_OUTPUT="$ROOT/datasets/minillm-general/pretrain-full-v1"
FINEWEB_OUTPUT="$ROOT/datasets/minillm-general/pretrain-fineweb-starter-v1"
TOKENIZER="$ROOT/outputs/minillm-general/90m-pretrain-mini/tokenizer.json"

while true; do
  source_bytes="$(stat -c %s "$MINIMIND_SOURCE" 2>/dev/null || printf 0)"
  fineweb_ready=0
  [[ -f "$FINEWEB_OUTPUT/manifest.json" ]] && fineweb_ready=1
  if [[ "$source_bytes" -eq "$MINIMIND_EXPECTED_BYTES" && "$fineweb_ready" -eq 1 ]]; then
    break
  fi
  if [[ "$fineweb_ready" -eq 0 ]] && ! pgrep -af "[p]repare_packed_dataset.py.*pretrain-fineweb-starter-v1" >/dev/null; then
    echo "FineWeb packing stopped without a manifest: $FINEWEB_OUTPUT" >&2
    exit 1
  fi
  echo "waiting: minimind_bytes=$source_bytes/$MINIMIND_EXPECTED_BYTES fineweb_ready=$fineweb_ready"
  sleep 30
done

if [[ -f "$MINIMIND_OUTPUT/manifest.json" ]]; then
  echo "already prepared: $MINIMIND_OUTPUT"
  exit 0
fi
if [[ -e "$MINIMIND_OUTPUT" ]]; then
  echo "output exists without a manifest: $MINIMIND_OUTPUT" >&2
  exit 1
fi

cd "$PROJECT"
"$PY" scripts/prepare_packed_dataset.py \
  --input "$MINIMIND_SOURCE" \
  --output-dir "$MINIMIND_OUTPUT" \
  --tokenizer "$TOKENIZER" \
  --validation-fraction 0.01 \
  --test-fraction 0.01 \
  --min-chars 32 \
  --batch-documents 512
