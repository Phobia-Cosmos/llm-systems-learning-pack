#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_conf="${root}/config/nvidia-profiler.conf"
destination="/etc/modprobe.d/nvidia-profiler.conf"

if [[ ${EUID} -ne 0 ]]; then
  echo "This system change needs root. Run:" >&2
  echo "  sudo $0" >&2
  exit 2
fi

if [[ -e "${destination}" ]] && ! cmp -s "${source_conf}" "${destination}"; then
  echo "Refusing to overwrite an existing, different ${destination}." >&2
  echo "Inspect and merge that file manually if it contains settings you need." >&2
  exit 1
fi

install -o root -g root -m 0644 "${source_conf}" "${destination}"
update-initramfs -u -k all

cat <<'MESSAGE'
NVIDIA performance-counter access is configured for the next boot.
Reboot when convenient, then verify:
  rg '^RmProfilingAdminOnly: 0' /proc/driver/nvidia/params

Security note: all local users will be able to read low-level GPU counters.
To restore admin-only access, run scripts/disable_nvidia_performance_counters.sh.
MESSAGE
