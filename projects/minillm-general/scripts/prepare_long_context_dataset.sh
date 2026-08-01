#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/public/home/u43077/lzh/python-envs/minillm-eval-py311/bin/python}"
CORPUS="${CORPUS_DIR:-/public/home/u43077/lzh/datasets/minillm-general/general-corpus-v1}"
OUTPUT="${OUTPUT_DIR:-/public/home/u43077/lzh/datasets/minillm-general/general-openbpe-32k-8k-v1}"
TOKENIZER="${TOKENIZER:-/public/home/u43077/lzh/datasets/minillm-general/general-openbpe-32k-v1/tokenizer.json}"
TARGETS="${TARGETS:-$ROOT/configs/long_context_targets.server.json}"

if [[ -e "$OUTPUT" ]]; then
  echo "long-context dataset already exists: $OUTPUT" >&2
  exit 2
fi

"$PY" "$ROOT/scripts/prepare_long_context_dataset.py" \
  --train-input "$CORPUS/train.jsonl" \
  --holdout-input "$CORPUS/validation.jsonl" \
  --output-dir "$OUTPUT" \
  --tokenizer "$TOKENIZER" \
  --targets "$TARGETS" \
  --batch-documents 128 \
  --drop-cross-role-duplicates
