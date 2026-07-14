#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${PROFILER_BUILD_DIR:-${root}/build-profiler}"
profiler="${build_dir}/tools/profiler/cutlass_profiler"

if [[ ! -x "${profiler}" ]]; then
  "${root}/scripts/build_profiler.sh"
fi

echo "FP32 CUDA-core SGEMM"
"${profiler}" \
  --kernels=cutlass_simt_sgemm_128x128_8x2_nn_align1 \
  --m=512 --n=512 --k=512 --profiling-iterations=20 \
  --verification-enabled=true

echo
echo "FP16 Tensor-Core GEMM with FP32 accumulation"
"${profiler}" \
  --kernels=cutlass_tensorop_s1688gemm_f16_256x128_32x2_nt_align8 \
  --m=1024 --n=1024 --k=1024 --profiling-iterations=20 \
  --verification-enabled=true
