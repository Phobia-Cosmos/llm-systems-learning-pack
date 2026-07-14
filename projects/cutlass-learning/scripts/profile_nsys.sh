#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root}/build}"
profile_dir="${root}/profiles"

if [[ ! -x "${build_dir}/bin/cutlass_tensorop" ]]; then
  "${root}/scripts/build.sh"
fi

mkdir -p "${profile_dir}"
nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true \
  --output="${profile_dir}/cutlass_tensorop" \
  "${build_dir}/bin/cutlass_tensorop" 520 504 264 10

echo
echo "Report: ${profile_dir}/cutlass_tensorop.nsys-rep"
echo "Summary: nsys stats ${profile_dir}/cutlass_tensorop.nsys-rep"

