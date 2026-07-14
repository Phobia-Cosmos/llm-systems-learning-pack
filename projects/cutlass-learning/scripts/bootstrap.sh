#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cutlass_dir="${root}/third_party/cutlass"
stamp_file="${root}/third_party/CUTLASS_COMMIT"
expected_tag="v4.5.3"
expected_commit="4552152794e8bd3bcfd63cf9b44369e590420dba"

"${root}/scripts/check_env.sh"

mkdir -p "${root}/third_party"
if [[ ! -f "${cutlass_dir}/include/cutlass/version.h" ]]; then
  if [[ -e "${cutlass_dir}" ]]; then
    echo "ERROR: ${cutlass_dir} exists but is not a CUTLASS checkout." >&2
    exit 1
  fi
  git clone --branch "${expected_tag}" --depth 1 --filter=blob:none \
    https://github.com/NVIDIA/cutlass.git "${cutlass_dir}"
fi

if [[ -d "${cutlass_dir}/.git" ]]; then
  actual_commit="$(git -C "${cutlass_dir}" rev-parse HEAD)"
  if ! git -C "${cutlass_dir}" diff --quiet || \
     ! git -C "${cutlass_dir}" diff --cached --quiet; then
    echo "ERROR: the pinned CUTLASS checkout has local modifications." >&2
    exit 1
  fi
  printf '%s\n' "${actual_commit}" > "${stamp_file}"
elif [[ -f "${stamp_file}" ]]; then
  # Some workspace transports retain a source snapshot but omit nested .git
  # metadata.  The adjacent immutable stamp keeps that snapshot verifiable.
  actual_commit="$(tr -d '[:space:]' < "${stamp_file}")"
else
  echo "ERROR: CUTLASS source has neither .git metadata nor ${stamp_file}." >&2
  exit 1
fi

if [[ "${actual_commit}" != "${expected_commit}" ]]; then
  echo "ERROR: ${cutlass_dir} is ${actual_commit}, expected ${expected_commit}." >&2
  echo "Move the existing checkout aside, then rerun bootstrap.sh." >&2
  exit 1
fi
if ! rg -q '^#define CUTLASS_MAJOR 4$' "${cutlass_dir}/include/cutlass/version.h" || \
   ! rg -q '^#define CUTLASS_MINOR 5$' "${cutlass_dir}/include/cutlass/version.h" || \
   ! rg -q '^#define CUTLASS_PATCH 3$' "${cutlass_dir}/include/cutlass/version.h"; then
  echo "ERROR: CUTLASS version header is not v4.5.3." >&2
  exit 1
fi

echo
echo "CUTLASS ${expected_tag} is ready at ${cutlass_dir}"
echo "commit: ${actual_commit}"
