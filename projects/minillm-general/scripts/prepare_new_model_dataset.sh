#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/public/home/u43077/lzh/python-envs/minillm-eval-py311/bin/python}"
CORPUS="${CORPUS_DIR:-/public/home/u43077/lzh/datasets/minillm-general/general-corpus-v1}"
OUTPUT="${OUTPUT_DIR:-/public/home/u43077/lzh/datasets/minillm-general/general-openbpe-32k-v1}"
CONFIG="${CORPUS_CONFIG:-$ROOT/configs/general_corpus_sources.server.json}"
TOKENIZER="${TOKENIZER:-/public/home/u43077/lzh/outputs/minillm-general/tokenizers/open-v1/openbpe-32k/tokenizer.json}"

if [[ ! -f "$CORPUS/manifest.json" ]]; then
  "$PY" "$ROOT/scripts/prepare_tokenizer_lab_corpus.py" \
    --config "$CONFIG" \
    --output-dir "$CORPUS" \
    --allow-shortfall
fi

if [[ ! -f "$OUTPUT/manifest.json" ]]; then
  "$PY" "$ROOT/scripts/prepare_packed_dataset.py" \
    --input "$CORPUS/train.jsonl" \
    --input "$CORPUS/validation.jsonl" \
    --output-dir "$OUTPUT" \
    --tokenizer "$TOKENIZER" \
    --validation-fraction 0.005 \
    --test-fraction 0.005 \
    --min-chars 64 \
    --batch-documents 512
fi
