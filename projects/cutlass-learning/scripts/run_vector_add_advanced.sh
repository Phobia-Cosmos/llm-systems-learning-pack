#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root}/build}"
results_dir="${root}/results/vector_add_advanced"
python="${PLOT_PYTHON:-/usr/bin/python3}"

if [[ ! -x "${build_dir}/bin/vector_add_advanced" ]]; then
  "${root}/scripts/build.sh"
fi

mkdir -p "${results_dir}"
n="${N:-16777219}"
iterations="${ITERATIONS:-20}"
rounds="${ROUNDS:-10}"
warmup="${WARMUP:-3}"
min_block="${MIN_BLOCK:-32}"
max_block="${MAX_BLOCK:-1024}"
step="${BLOCK_STEP:-32}"

aligned_csv="${results_dir}/aligned.csv"
unaligned_csv="${results_dir}/unaligned_offset1.csv"

echo "Aligned scalar/packed sweep"
"${build_dir}/bin/vector_add_advanced" \
  --n "${n}" --iterations "${iterations}" --rounds "${rounds}" \
  --warmup "${warmup}" --min-block "${min_block}" \
  --max-block "${max_block}" --step "${step}" --offset 0 \
  --csv "${aligned_csv}"

echo
echo "Offset=1 alignment-fallback sweep"
"${build_dir}/bin/vector_add_advanced" \
  --n "${n}" --iterations "${iterations}" --rounds "${rounds}" \
  --warmup "${warmup}" --min-block "${min_block}" \
  --max-block "${max_block}" --step "${step}" --offset 1 \
  --csv "${unaligned_csv}"

echo
echo "Generating PNG/SVG curves"
"${python}" "${root}/scripts/plot_vector_add_advanced.py" \
  "${aligned_csv}" --unaligned-csv "${unaligned_csv}" \
  --output-prefix "${results_dir}/vector_add"

echo "Results: ${results_dir}"

