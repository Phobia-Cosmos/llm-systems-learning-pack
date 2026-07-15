#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root}/build}"

if [[ ! -x "${build_dir}/bin/tiled_gemm" ||
      ! -x "${build_dir}/bin/vector_add" ||
      ! -x "${build_dir}/bin/vector_add_advanced" ]]; then
  "${root}/scripts/build.sh"
fi

echo "Vector Add: aligned float4 plus scalar tail"
compute-sanitizer --error-exitcode=1 --tool memcheck \
  "${build_dir}/bin/vector_add" 1000003 1 0 0 0 256

echo "Vector Add: A/B/C deliberately unaligned, scalar fallback"
compute-sanitizer --error-exitcode=1 --tool memcheck \
  "${build_dir}/bin/vector_add" 257 1 1 1 1 256

echo "Vector Add advanced: float4/half2/int4 tails and guards"
compute-sanitizer --error-exitcode=1 --tool memcheck \
  "${build_dir}/bin/vector_add_advanced" \
  --n 257 --iterations 1 --rounds 1 --warmup 0 --block 96 \
  --types all --variants all --offset 0 \
  --csv "${build_dir}/results/vector_add_advanced_sanitizer.csv"

echo "Vector Add advanced: packed paths deliberately misaligned"
compute-sanitizer --error-exitcode=1 --tool memcheck \
  "${build_dir}/bin/vector_add_advanced" \
  --n 257 --iterations 1 --rounds 1 --warmup 0 --block 96 \
  --types all --variants vector --offset 1 \
  --csv "${build_dir}/results/vector_add_advanced_sanitizer_unaligned.csv"

echo "Tiled GEMM: memory bounds"
compute-sanitizer --error-exitcode=1 --tool memcheck \
  "${build_dir}/bin/tiled_gemm" 129 131 127 1

echo "Tiled GEMM: shared-memory races"
compute-sanitizer --error-exitcode=1 --tool racecheck \
  "${build_dir}/bin/tiled_gemm" 129 131 127 1
