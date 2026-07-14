#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root}/build}"
binary="${build_dir}/bin/vector_add"

if [[ ! -x "${binary}" ]]; then
  "${root}/scripts/build.sh"
fi

echo "128-bit global-memory instructions found in vector_add:"
cuobjdump --dump-sass "${binary}" \
  | rg 'LDG[^;]*\.128|STG[^;]*\.128' \
  | sort -u

