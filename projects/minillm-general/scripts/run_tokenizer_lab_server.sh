#!/usr/bin/env bash
set -euo pipefail

export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=true

ROOT="${SERVER_ROOT:-/public/home/u43077/lzh}"
PY="${TOKENIZER_PYTHON:-$ROOT/python-envs/minillm-eval-py311/bin/python}"
PROJECT="$ROOT/ai/projects/minillm-general"
SOURCE_ROOT="$ROOT/datasets/tokenizer-lab/sources"
CORPUS_ROOT="$ROOT/datasets/tokenizer-lab/corpus-open-v1"
CANDIDATE_ROOT="$ROOT/outputs/minillm-general/tokenizers/open-v1"
EVALUATION_ROOT="$ROOT/benchmarks/tokenizer-lab/open-v1"
SOURCE_CONFIG="$PROJECT/configs/tokenizer_lab_sources.server.json"
CURRENT_TOKENIZER="$ROOT/outputs/minillm-general/90m-pretrain-mini/tokenizer.json"
QWEN_TOKENIZER="$ROOT/models/Qwen3-0.6B/tokenizer.json"

cd "$PROJECT"
mkdir -p "$SOURCE_ROOT" "$CANDIDATE_ROOT" "$EVALUATION_ROOT"

"$PY" scripts/download_tokenizer_lab_datasets.py \
  --output-dir "$SOURCE_ROOT"

if [[ ! -f "$CORPUS_ROOT/manifest.json" ]]; then
  if [[ -e "$CORPUS_ROOT" ]]; then
    echo "corpus output exists without a manifest: $CORPUS_ROOT" >&2
    exit 1
  fi
  "$PY" scripts/prepare_tokenizer_lab_corpus.py \
    --config "$SOURCE_CONFIG" \
    --output-dir "$CORPUS_ROOT"
fi

"$PY" scripts/train_tokenizer_candidates.py \
  --corpus-dir "$CORPUS_ROOT" \
  --output-dir "$CANDIDATE_ROOT" \
  --candidate openbpe-32k:32768 \
  --candidate openbpe-48k:49152

tokenizer_args=(
  --tokenizer "openbpe-32k=$CANDIDATE_ROOT/openbpe-32k/tokenizer.json"
  --tokenizer "openbpe-48k=$CANDIDATE_ROOT/openbpe-48k/tokenizer.json"
)
if [[ -f "$CURRENT_TOKENIZER" ]]; then
  tokenizer_args+=(--tokenizer "minillm-current-16k=$CURRENT_TOKENIZER")
fi
if [[ -f "$QWEN_TOKENIZER" ]]; then
  tokenizer_args+=(--tokenizer "qwen3-151k=$QWEN_TOKENIZER")
fi

"$PY" scripts/evaluate_tokenizer_candidates.py \
  --validation "$CORPUS_ROOT/validation.jsonl" \
  --output-dir "$EVALUATION_ROOT" \
  --max-documents-per-domain 2000 \
  "${tokenizer_args[@]}"

echo "tokenizer lab complete"
echo "candidates: $CANDIDATE_ROOT"
echo "evaluation: $EVALUATION_ROOT/evaluation.md"
