#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-clang++}"
TINYBVH_DIR="${TINYBVH_DIR:-external/tinybvh}"
SOURCE="${SOURCE:-${SCRIPT_DIR}/bench_tinybvh.cpp}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/bench_tinybvh}"
OBJ_PATH="${1:-./assets/dragon/dragon.obj}"

if [[ ! -f "${TINYBVH_DIR}/tiny_bvh.h" ]]; then
    echo "error: ${TINYBVH_DIR}/tiny_bvh.h not found" >&2
    echo "Clone TinyBVH there, or set TINYBVH_DIR." >&2
    exit 1
fi

# clang++: warning: argument '-Ofast' is deprecated; use '-O3 -ffast-math'
"${CXX}" \
    -std=c++20 \
    -O3 \
    -ffast-math \
    -DNDEBUG \
    -march=native \
    -mavx2 \
    -mfma \
    -I"${TINYBVH_DIR}" \
    "${SOURCE}" \
    -o "${OUTPUT}"

"${OUTPUT}" "${OBJ_PATH}"
