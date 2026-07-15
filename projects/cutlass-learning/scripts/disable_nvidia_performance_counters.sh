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

if [[ -e "${destination}" ]]; then
  if ! cmp -s "${source_conf}" "${destination}"; then
    echo "Refusing to delete a modified ${destination}." >&2
    echo "Inspect it and restore admin-only settings manually." >&2
    exit 1
  fi
  rm -f "${destination}"
fi
update-initramfs -u -k all
echo "Admin-only NVIDIA performance counters will be restored after reboot."
