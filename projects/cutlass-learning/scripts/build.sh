#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root}/build}"

"${root}/scripts/bootstrap.sh"
"${root}/scripts/configure.sh"
cmake --build "${build_dir}" --parallel "${JOBS:-4}"

