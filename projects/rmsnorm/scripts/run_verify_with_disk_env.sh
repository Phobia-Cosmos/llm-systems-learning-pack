#!/usr/bin/env bash
set -euo pipefail

AI_ROOT="${AI_ROOT:-/home/undefined/Desktop/ai}"
source "${AI_ROOT}/use_disk_ai_env.sh"

cd "${AI_ROOT}"
exec python projects/rmsnorm/experiments/verify_rmsnorm_paper.py "$@"
