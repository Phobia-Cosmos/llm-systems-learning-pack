#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root}/build}"
profile_dir="${root}/profiles"

if [[ ! -x "${build_dir}/bin/tiled_gemm" ]]; then
  "${root}/scripts/build.sh"
fi

if [[ ${EUID} -ne 0 ]] && \
   rg -q '^RmProfilingAdminOnly:[[:space:]]*1' /proc/driver/nvidia/params 2>/dev/null; then
  cat >&2 <<'MESSAGE'
Nsight Compute cannot read hardware counters for this user:
  /proc/driver/nvidia/params has RmProfilingAdminOnly: 1

Ask the machine administrator to enable non-admin GPU performance counters,
then rerun this script. CUDA Event timing and Nsight Systems still work now.
Official guidance: https://developer.nvidia.com/ERR_NVGPUCTRPERM
MESSAGE
  exit 2
fi

mkdir -p "${profile_dir}"
ncu --set basic --kernel-name 'regex:.*tiled_gemm_kernel.*' \
  --launch-count 1 --force-overwrite \
  --export "${profile_dir}/tiled_gemm" \
  "${build_dir}/bin/tiled_gemm" 512 512 512 1

