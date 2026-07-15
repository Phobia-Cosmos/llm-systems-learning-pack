#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root}/build}"
binary="${build_dir}/bin/vector_add"
advanced_binary="${build_dir}/bin/vector_add_advanced"

if [[ ! -x "${binary}" || ! -x "${advanced_binary}" ]]; then
  "${root}/scripts/build.sh"
fi

echo "128-bit global-memory instructions found in vector_add:"
cuobjdump --dump-sass "${binary}" \
  | rg 'LDG[^;]*\.128|STG[^;]*\.128' \
  | sort -u

echo
echo "Packed-kernel evidence in vector_add_advanced:"
cuobjdump --dump-sass "${advanced_binary}" | awk '
  /Function :/ {
    current = ""
    if ($0 ~ /vector_add_float4_advanced/) current = "float4"
    if ($0 ~ /vector_add_half2_advanced/) current = "half2"
    if ($0 ~ /vector_add_int4_advanced/) current = "int4"
    if (current != "") print "\n" $0
    next
  }
  current != "" && $0 ~ /(LDG|STG|HADD2)/ {
    print
    if (current == "float4" && $0 ~ /LDG.*\.128/) float4_ld = 1
    if (current == "float4" && $0 ~ /STG.*\.128/) float4_st = 1
    if (current == "int4" && $0 ~ /LDG.*\.128/) int4_ld = 1
    if (current == "int4" && $0 ~ /STG.*\.128/) int4_st = 1
    if (current == "half2" && $0 ~ /LDG\.E\.CONSTANT/) half2_ld = 1
    if (current == "half2" && $0 ~ /STG\.E \[/) half2_st = 1
    if (current == "half2" && $0 ~ /HADD2/) half2_add = 1
  }
  END {
    if (!(float4_ld && float4_st && int4_ld && int4_st &&
          half2_ld && half2_st && half2_add)) {
      print "Missing expected float4/half2/int4 SASS evidence" > "/dev/stderr"
      exit 1
    }
  }
'
