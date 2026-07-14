#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root}/build}"
results_dir="${root}/results"
output="${results_dir}/shape_sweep.txt"

if [[ ! -x "${build_dir}/bin/cutlass_tensorop" ]]; then
  "${root}/scripts/build.sh"
fi

mkdir -p "${results_dir}"
: > "${output}"

run_case() {
  echo | tee -a "${output}"
  echo "> $*" | tee -a "${output}"
  "$@" | tee -a "${output}"
}

run_case "${build_dir}/bin/cutlass_sgemm" 513 509 257 50
run_case "${build_dir}/bin/cutlass_sgemm" 1024 1024 1024 100
run_case "${build_dir}/bin/cutlass_sgemm" 32 4096 4096 50

run_case "${build_dir}/bin/cutlass_tensorop" 520 504 264 50
run_case "${build_dir}/bin/cutlass_tensorop" 1024 1024 1024 100
run_case "${build_dir}/bin/cutlass_tensorop" 32 4096 4096 50

echo
echo "Saved: ${output}"
