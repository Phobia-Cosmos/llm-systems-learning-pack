#!/usr/bin/env bash
set -euo pipefail

repos=(
  "/home/undefined/Desktop/ai"
  "/home/undefined/Desktop/bci"
  "/home/undefined/Desktop/IPhone"
)

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
message="${*:-sync: workspace update}"

for repo in "${repos[@]}"; do
  echo
  echo "[git-sync] repo: $repo"

  if [[ ! -d "$repo/.git" ]] || [[ "$(git -C "$repo" rev-parse --show-toplevel 2>/dev/null)" != "$repo" ]]; then
    echo "[git-sync] error: top-level repository not found: $repo" >&2
    exit 1
  fi

  echo "[git-sync] status before sync:"
  git -C "$repo" status --short

  if [[ "$dry_run" == true ]]; then
    echo "[git-sync] dry run: no add, commit, pull, or push was executed"
    continue
  fi

  git -C "$repo" add -A

  if git -C "$repo" diff --cached --quiet; then
    echo "[git-sync] no staged changes to commit"
  else
    echo "[git-sync] committing staged changes"
    git -C "$repo" commit -m "$message"
  fi

  echo "[git-sync] pulling remote changes with rebase"
  git -C "$repo" pull --rebase --autostash

  echo "[git-sync] pushing to remote"
  git -C "$repo" push
done

echo
echo "[git-sync] done"
