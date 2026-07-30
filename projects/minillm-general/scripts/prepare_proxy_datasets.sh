#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/public/home/u43077/lzh/python-envs/llm-py311/bin/python}"
CORPUS="${PROXY_CORPUS:-/public/home/u43077/lzh/datasets/tokenizer-lab/corpus-open-v1/train.jsonl}"
TOKENIZER_ROOT="${TOKENIZER_ROOT:-/public/home/u43077/lzh/outputs/minillm-general/tokenizers/open-v1}"
DATA_ROOT="${DATA_ROOT:-/public/home/u43077/lzh/datasets/minillm-general}"

for candidate in openbpe-32k openbpe-48k; do
  tokenizer="$TOKENIZER_ROOT/$candidate/tokenizer.json"
  output="$DATA_ROOT/proxy-$candidate-v1"
  if [[ -f "$output/manifest.json" ]]; then
    echo "already prepared: $output"
    continue
  fi
  "$PY" "$ROOT/scripts/prepare_packed_dataset.py" \
    --input "$CORPUS" \
    --output-dir "$output" \
    --tokenizer "$tokenizer" \
    --validation-fraction 0.01 \
    --test-fraction 0.01 \
    --min-chars 32 \
    --batch-documents 512
done
