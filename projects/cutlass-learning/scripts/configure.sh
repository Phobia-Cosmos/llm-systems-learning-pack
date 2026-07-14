#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root}/build}"
compute_capability="${CUDA_COMPUTE_CAPABILITY:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')}"
architecture="${CUDA_ARCH:-${compute_capability/./}}"

cmake -S "${root}" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_ARCHITECTURES="${architecture}" \
  -DCUTLASS_DIR="${root}/third_party/cutlass"

