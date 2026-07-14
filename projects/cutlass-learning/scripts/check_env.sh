#!/usr/bin/env bash
set -euo pipefail

required=(nvidia-smi nvcc cmake ninja git g++ rg)
missing=0
for command_name in "${required[@]}"; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "MISSING: ${command_name}"
    missing=1
  fi
done
if (( missing != 0 )); then
  exit 1
fi

echo "GPU"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total \
  --format=csv,noheader

echo
echo "CUDA compiler"
nvcc --version | tail -n 1

echo
echo "Build tools"
cmake --version | head -n 1
echo "ninja $(ninja --version)"
g++ --version | head -n 1

compute_capability="${CUDA_COMPUTE_CAPABILITY:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')}"
architecture="${compute_capability/./}"
echo
echo "Detected CMake CUDA architecture: ${architecture}"
if [[ "${architecture}" == "89" ]]; then
  echo "PASS: this RTX 4070 SUPER should be built only for sm_89."
else
  echo "NOTE: review architecture-specific exercises for this GPU."
fi

for profiler in ncu nsys compute-sanitizer cuobjdump nvdisasm; do
  if command -v "${profiler}" >/dev/null 2>&1; then
    echo "FOUND: ${profiler} -> $(command -v "${profiler}")"
  else
    echo "OPTIONAL MISSING: ${profiler}"
  fi
done
