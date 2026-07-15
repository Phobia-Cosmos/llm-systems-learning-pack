#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root}/build}"
profile_dir="${root}/profiles/vector_add_advanced"
results_dir="${root}/results/vector_add_advanced"

if [[ ! -x "${build_dir}/bin/vector_add_advanced" ]]; then
  "${root}/scripts/build.sh"
fi

mkdir -p "${profile_dir}" "${results_dir}"
report="${profile_dir}/nvtx_timeline"

nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true \
  --output="${report}" \
  "${build_dir}/bin/vector_add_advanced" \
  --n 1048579 --iterations 3 --rounds 2 --warmup 1 --block 256 \
  --types all --variants all --offset 0 \
  --csv "${results_dir}/nsys_profile.csv"

nsys stats --force-export=true --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum \
  "${report}.nsys-rep" > "${profile_dir}/nvtx_timeline_stats.txt"

echo "Report: ${report}.nsys-rep"
echo "Stats:  ${profile_dir}/nvtx_timeline_stats.txt"
