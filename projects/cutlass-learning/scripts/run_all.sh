#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root}/build}"

if [[ ! -x "${build_dir}/bin/device_info" ]]; then
  "${root}/scripts/build.sh"
fi

ctest --test-dir "${build_dir}" --output-on-failure

echo
echo "Representative runs"
"${build_dir}/bin/device_info"
"${build_dir}/bin/vector_add" 16777219 50
"${build_dir}/bin/tiled_gemm" 257 259 263 30
"${build_dir}/bin/cutlass_sgemm" 513 509 257 30
"${build_dir}/bin/cutlass_tensorop" 520 504 264 30
"${build_dir}/bin/cute_layout"

