#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root}/build}"
profile_dir="${root}/profiles/vector_add_advanced/ncu"
results_dir="${root}/results/vector_add_advanced/ncu"

# sudo commonly replaces PATH with a secure_path that omits the CUDA bin
# directory. Resolve ncu explicitly so the documented `sudo ./script` path
# works as well as an ordinary user invocation.
if [[ -n ${NCU_BIN:-} ]]; then
  ncu_bin="${NCU_BIN}"
elif command -v ncu >/dev/null 2>&1; then
  ncu_bin="$(command -v ncu)"
elif [[ -x /usr/local/cuda/bin/ncu ]]; then
  ncu_bin=/usr/local/cuda/bin/ncu
elif [[ -x /usr/local/cuda-13.0/bin/ncu ]]; then
  ncu_bin=/usr/local/cuda-13.0/bin/ncu
else
  echo "Nsight Compute CLI (ncu) was not found." >&2
  exit 127
fi

if [[ ! -x "${build_dir}/bin/vector_add_advanced" ]]; then
  "${root}/scripts/build.sh"
fi

if [[ ${EUID} -ne 0 ]] && \
   grep -Eq '^RmProfilingAdminOnly:[[:space:]]*1' /proc/driver/nvidia/params; then
  cat >&2 <<MESSAGE
Nsight Compute counters are still admin-only.

Immediate one-shot collection:
  sudo ${root}/scripts/profile_vector_add_advanced_ncu.sh

Persistent access for this trusted workstation:
  sudo ${root}/scripts/enable_nvidia_performance_counters.sh
  sudo reboot
MESSAGE
  exit 2
fi

mkdir -p "${profile_dir}" "${results_dir}"

metrics='smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_bytes.sum,lts__t_bytes.sum.per_second,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum'

profiles=(
  'float scalar vector_add_float_scalar_advanced'
  'float vector vector_add_float4_advanced'
  'half scalar vector_add_half_scalar_advanced'
  'half vector vector_add_half2_advanced'
  'int scalar vector_add_int_scalar_advanced'
  'int vector vector_add_int4_advanced'
)

logs=()
for spec in "${profiles[@]}"; do
  read -r dtype variant kernel <<<"${spec}"
  log="${results_dir}/ncu_${dtype}_${variant}.csv"
  report="${profile_dir}/${dtype}_${variant}"
  logs+=("${log}")
  echo "Profiling ${dtype}/${variant} (${kernel})"
  "${ncu_bin}" --force-overwrite --export "${report}" --csv --page raw \
    --log-file "${log}" --kernel-name-base demangled \
    --kernel-name "regex:.*${kernel}.*" --launch-count 1 \
    --metrics "${metrics}" \
    "${build_dir}/bin/vector_add_advanced" \
    --n 16777219 --iterations 1 --rounds 1 --warmup 0 --block 256 \
    --types "${dtype}" --variants "${variant}" --offset 0 \
    --csv "${results_dir}/program_${dtype}_${variant}.csv"
done

/usr/bin/python3 "${root}/scripts/summarize_ncu_metrics.py" \
  "${logs[@]}" --output "${results_dir}/summary.csv"

if [[ -n ${SUDO_USER:-} ]]; then
  chown -R "${SUDO_USER}:${SUDO_USER}" "${profile_dir}" "${results_dir}"
fi

echo "NCU reports: ${profile_dir}"
echo "Metric table: ${results_dir}/summary.csv"
echo "Comparison:   ${results_dir}/comparison.csv"
