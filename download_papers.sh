#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LIST="$ROOT/paper_list.tsv"
PAPERS="$ROOT/papers"
LOG="$ROOT/download_failures.log"

: > "$LOG"

tail -n +2 "$LIST" | while IFS=$'\t' read -r category filename url title; do
  [ -z "$category" ] && continue
  mkdir -p "$PAPERS/$category"
  dest="$PAPERS/$category/$filename"
  if [ -s "$dest" ]; then
    echo "SKIP $category/$filename"
    continue
  fi
  tmp="$dest.part"
  echo "GET  $category/$filename"
  if curl --http1.1 -L --fail --silent --show-error --retry 5 --retry-delay 2 --connect-timeout 20 --speed-time 120 --speed-limit 1024 --max-time 360 -C - -o "$tmp" "$url"; then
    mv "$tmp" "$dest"
  else
    rm -f "$tmp"
    printf "%s\t%s\t%s\t%s\n" "$category" "$filename" "$url" "$title" >> "$LOG"
    echo "FAIL $category/$filename"
  fi
done

if [ -s "$LOG" ]; then
  echo "Some downloads failed. See $LOG"
  exit 1
fi

echo "All downloads completed."
