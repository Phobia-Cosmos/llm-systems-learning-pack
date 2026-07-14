#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${PROFILER_BUILD_DIR:-${root}/build-profiler}"
compute_capability="${CUDA_COMPUTE_CAPABILITY:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')}"
architecture="${CUDA_ARCH:-${compute_capability/./}}"
kernels="${PROFILER_KERNELS:-cutlass_simt_sgemm_128x128_8x2_nn_align1,cutlass_tensorop_s1688gemm_f16_256x128_32x2_nt_align8}"

"${root}/scripts/bootstrap.sh"

cmake -S "${root}/third_party/cutlass" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_NVCC_ARCHS="${architecture}" \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_ENABLE_TOOLS=ON \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON \
  -DCUTLASS_LIBRARY_KERNELS="${kernels}"

# CUTLASS's reference providers are template-heavy even when only two kernels
# are generated. Unity mode reduces translation-unit count; one job caps the
# memory peak on the 30 GiB workstation.
cmake --build "${build_dir}" --target cutlass_profiler --parallel "${JOBS:-1}"
echo "Profiler: ${build_dir}/tools/profiler/cutlass_profiler"
