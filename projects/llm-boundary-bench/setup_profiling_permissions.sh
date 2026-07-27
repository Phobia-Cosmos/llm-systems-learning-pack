#!/usr/bin/env bash
set -euo pipefail

readonly NVIDIA_CONF="/etc/modprobe.d/99-nvidia-profiling.conf"
readonly CPU_PERF_CONF="/etc/sysctl.d/99-local-profiling.conf"

cpu_perf_level=""

usage() {
  cat <<'EOF'
Usage:
  sudo ./setup_profiling_permissions.sh [--cpu-perf-event-paranoid LEVEL]

Permanently allow non-root access to NVIDIA GPU performance counters. The
NVIDIA setting takes effect after a reboot. This script updates initramfs, but
it never unloads the NVIDIA driver and never reboots the machine.

Options:
  --cpu-perf-event-paranoid LEVEL
      Also persist and apply kernel.perf_event_paranoid=LEVEL. Lower values
      grant broader CPU perf-event access. This setting is not changed unless
      the option is explicitly supplied. Accepted range: -1 through 4.
  -h, --help
      Show this help text.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

while (($# > 0)); do
  case "$1" in
    --cpu-perf-event-paranoid)
      (($# >= 2)) || die "--cpu-perf-event-paranoid requires a value"
      cpu_perf_level="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ -n "$cpu_perf_level" ]]; then
  [[ "$cpu_perf_level" =~ ^-?[0-9]+$ ]] || die "CPU perf-event level must be an integer"
  ((cpu_perf_level >= -1 && cpu_perf_level <= 4)) || die "CPU perf-event level must be between -1 and 4"
fi

((EUID == 0)) || die "run this script with sudo: sudo $0"
command -v update-initramfs >/dev/null 2>&1 || die "update-initramfs was not found"
if [[ -n "$cpu_perf_level" ]]; then
  command -v sysctl >/dev/null 2>&1 || die "sysctl was not found"
fi

tmp_nvidia="$(mktemp)"
tmp_cpu=""
cleanup() {
  rm -f "$tmp_nvidia"
  if [[ -n "$tmp_cpu" ]]; then
    rm -f "$tmp_cpu"
  fi
}
trap cleanup EXIT

cat >"$tmp_nvidia" <<'EOF'
# Managed by llm-boundary-bench/setup_profiling_permissions.sh.
# This module option is applied when the NVIDIA driver is loaded at boot.
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOF

if [[ -f "$NVIDIA_CONF" ]] && cmp -s "$tmp_nvidia" "$NVIDIA_CONF"; then
  printf 'unchanged: %s\n' "$NVIDIA_CONF"
else
  install -D -o root -g root -m 0644 "$tmp_nvidia" "$NVIDIA_CONF"
  printf 'updated:   %s\n' "$NVIDIA_CONF"
fi

# Always rebuild on an explicit administrator invocation. This also makes a
# rerun recover cleanly if an earlier initramfs update was interrupted.
printf 'running:   update-initramfs -u\n'
update-initramfs -u

if [[ -n "$cpu_perf_level" ]]; then
  tmp_cpu="$(mktemp)"
  cat >"$tmp_cpu" <<EOF
# Managed by llm-boundary-bench/setup_profiling_permissions.sh.
kernel.perf_event_paranoid = $cpu_perf_level
EOF

  if [[ -f "$CPU_PERF_CONF" ]] && cmp -s "$tmp_cpu" "$CPU_PERF_CONF"; then
    printf 'unchanged: %s\n' "$CPU_PERF_CONF"
  else
    install -D -o root -g root -m 0644 "$tmp_cpu" "$CPU_PERF_CONF"
    printf 'updated:   %s\n' "$CPU_PERF_CONF"
  fi
  printf 'applying:  kernel.perf_event_paranoid=%s\n' "$cpu_perf_level"
  sysctl -p "$CPU_PERF_CONF"
else
  printf 'unchanged: kernel.perf_event_paranoid (no CPU perf option supplied)\n'
fi

runtime_value=""
if [[ -r /proc/driver/nvidia/params ]]; then
  runtime_value="$(awk -F: '$1 == "RmProfilingAdminOnly" {gsub(/[[:space:]]/, "", $2); print $2; exit}' /proc/driver/nvidia/params)"
fi

if [[ "$runtime_value" == "0" ]]; then
  printf 'ready:     the running NVIDIA driver already permits non-root counters\n'
else
  printf '\nConfiguration is persistent, but the running NVIDIA driver still reports RmProfilingAdminOnly=%s.\n' "${runtime_value:-unknown}"
  printf 'Reboot at a convenient time, then run ./check_profiling_permissions.sh.\n'
fi

