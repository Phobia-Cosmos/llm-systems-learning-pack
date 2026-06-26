#!/usr/bin/env bash
set -euo pipefail

repo="/home/undefined/Desktop/ai"
mode="${1:-}"
message="${1:-sync: workspace update}"

cd "$repo"

echo "[git-sync] repo: $repo"
echo "[git-sync] status before sync:"
git status --short

if [[ "$mode" == "--dry-run" ]]; then
  echo "[git-sync] dry run: no add, commit, pull, or push was executed"
  exit 0
fi

git add -A

if git diff --cached --quiet; then
  echo "[git-sync] no staged changes to commit"
else
  echo "[git-sync] committing staged changes"
  git commit -m "$message"
fi

echo "[git-sync] pulling remote changes with rebase"
git pull --rebase --autostash

echo "[git-sync] pushing to remote"
git push

echo "[git-sync] done"
