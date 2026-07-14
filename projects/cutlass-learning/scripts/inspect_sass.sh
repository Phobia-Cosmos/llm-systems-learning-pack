#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root}/build}"
binary="${build_dir}/bin/cutlass_tensorop"

if [[ ! -x "${binary}" ]]; then
  "${root}/scripts/build.sh"
fi

echo "Tensor Core instructions found in cutlass_tensorop:"
cuobjdump --dump-sass "${binary}" | rg 'HMMA|MMA' | sort -u | head -n 30
