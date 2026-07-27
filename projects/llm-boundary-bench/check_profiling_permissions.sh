#!/usr/bin/env bash
set -uo pipefail

readonly NVIDIA_CONF="/etc/modprobe.d/99-nvidia-profiling.conf"
readonly CPU_PERF_CONF="/etc/sysctl.d/99-local-profiling.conf"

usage() {
  cat <<'EOF'
Usage: ./check_profiling_permissions.sh

Read-only checks for the running NVIDIA setting, its persistent configuration,
the CPU perf-event policy, and the Nsight Compute executable.

Exit codes:
  0  NVIDIA counters are enabled now and persistently configured; ncu works.
  1  Setup or a reboot is still required.
  2  A required driver interface or ncu executable could not be checked.
EOF
}

if (($# > 0)); then
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    *)
      printf 'error: unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
fi

needs_action=0
check_error=0

ok() {
  printf '[OK]     %s\n' "$*"
}

action() {
  printf '[ACTION] %s\n' "$*"
  needs_action=1
}

error() {
  printf '[ERROR]  %s\n' "$*" >&2
  check_error=1
}

note() {
  printf '[INFO]   %s\n' "$*"
}

find_ncu() {
  local candidate
  if command -v ncu >/dev/null 2>&1; then
    command -v ncu
    return 0
  fi
  for candidate in /usr/local/cuda/bin/ncu /usr/local/cuda-*/bin/ncu; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

printf 'NVIDIA runtime\n'
if [[ -r /proc/driver/nvidia/params ]]; then
  runtime_value="$(awk -F: '$1 == "RmProfilingAdminOnly" {gsub(/[[:space:]]/, "", $2); print $2; exit}' /proc/driver/nvidia/params)"
  case "$runtime_value" in
    0)
      ok "RmProfilingAdminOnly=0; non-root GPU performance counters are enabled"
      ;;
    1)
      action "RmProfilingAdminOnly=1; configure the module option and reboot"
      ;;
    *)
      error "could not parse RmProfilingAdminOnly from /proc/driver/nvidia/params"
      ;;
  esac
else
  error "/proc/driver/nvidia/params is unavailable; the NVIDIA driver may not be loaded"
fi

printf '\nPersistent NVIDIA configuration\n'
if [[ -r "$NVIDIA_CONF" ]] && grep -Eq '^[[:space:]]*options[[:space:]]+nvidia[[:space:]].*NVreg_RestrictProfilingToAdminUsers=0([[:space:]]|$)' "$NVIDIA_CONF"; then
  ok "$NVIDIA_CONF requests NVreg_RestrictProfilingToAdminUsers=0"
else
  action "$NVIDIA_CONF is missing or does not contain the required module option"
fi

conflicts="$(grep -HnE '^[[:space:]]*options[[:space:]]+nvidia[[:space:]].*NVreg_RestrictProfilingToAdminUsers=1([[:space:]]|$)' /etc/modprobe.d/*.conf /lib/modprobe.d/*.conf /usr/lib/modprobe.d/*.conf 2>/dev/null || true)"
if [[ -n "$conflicts" ]]; then
  note "other files request NVreg_RestrictProfilingToAdminUsers=1; verify module-option ordering:"
  while IFS= read -r conflict; do
    printf '         %s\n' "$conflict"
  done <<<"$conflicts"
fi

printf '\nCPU perf-event policy (independent of NVIDIA GPU counters)\n'
if [[ -r /proc/sys/kernel/perf_event_paranoid ]]; then
  cpu_perf_level="$(< /proc/sys/kernel/perf_event_paranoid)"
  note "running kernel.perf_event_paranoid=$cpu_perf_level"
  if ((cpu_perf_level > 2)); then
    note "CPU sampling is restricted; use setup --cpu-perf-event-paranoid LEVEL only if CPU sampling is required"
  fi
else
  note "/proc/sys/kernel/perf_event_paranoid is unavailable"
fi

if [[ -r "$CPU_PERF_CONF" ]]; then
  persisted_cpu="$(awk -F= '$1 ~ /^[[:space:]]*kernel\.perf_event_paranoid[[:space:]]*$/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$CPU_PERF_CONF")"
  if [[ -n "$persisted_cpu" ]]; then
    note "$CPU_PERF_CONF persists kernel.perf_event_paranoid=$persisted_cpu"
  else
    note "$CPU_PERF_CONF exists but has no recognizable perf-event setting"
  fi
else
  note "no CPU perf-event policy is managed by this project"
fi

printf '\nNsight Compute\n'
if ncu_path="$(find_ncu)"; then
  if ncu_output="$("$ncu_path" --version 2>&1)"; then
    ncu_first_line="$(printf '%s\n' "$ncu_output" | sed -n '1p')"
    ok "$ncu_path is executable (${ncu_first_line:-version output was empty})"
  else
    error "$ncu_path exists but 'ncu --version' failed"
  fi
else
  error "ncu was not found in PATH or under /usr/local/cuda*"
fi

printf '\n'
if ((check_error != 0)); then
  printf 'Result: check incomplete because a required component is unavailable (exit 2).\n'
  exit 2
fi
if ((needs_action != 0)); then
  printf 'Result: setup or a reboot is required (exit 1).\n'
  exit 1
fi
printf 'Result: NVIDIA performance-counter access is ready (exit 0).\n'

