#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LIST="$ROOT/paper_list.tsv"
OUT="$ROOT/PAPER_INDEX.md"

{
  echo "# Paper Index"
  echo
  echo "Generated from \`paper_list.tsv\`. Status is based on files under \`papers/\`."
  echo
  current=""
  tail -n +2 "$LIST" | while IFS=$'\t' read -r category filename url title; do
    [ -z "${category:-}" ] && continue
    if [ "$category" != "$current" ]; then
      current="$category"
      echo
      echo "## $category"
      echo
    fi
    path="papers/$category/$filename"
    if [ -s "$ROOT/$path" ]; then
      status="downloaded"
      link="[$filename]($path)"
    else
      status="missing"
      link="\`$filename\`"
    fi
    echo "- [$status] $link - $title"
    echo "  Source: $url"
  done
} > "$OUT"

echo "Wrote $OUT"
