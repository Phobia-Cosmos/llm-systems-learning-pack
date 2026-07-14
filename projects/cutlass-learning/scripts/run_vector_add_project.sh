#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root}/build}"

if [[ ! -x "${build_dir}/bin/vector_add" ]]; then
  "${root}/scripts/build.sh"
fi

echo "1/4 Boundary, block-size, and alignment CTest matrix"
ctest --test-dir "${build_dir}" -R '^vector_add_' --output-on-failure

echo
echo "2/4 Small working set: cache/launch-sensitive"
"${build_dir}/bin/vector_add" 262147 200

echo
echo "3/4 Working set larger than the RTX 4070 SUPER L2"
"${build_dir}/bin/vector_add" 16777219 50

echo
echo "4/4 Deliberately unaligned A/B/C: requested vector path must fall back"
"${build_dir}/bin/vector_add" 1000003 30 1 1 1
