#!/usr/bin/env bash
set -euo pipefail

taskset -c 0 mojo -I . bench/bvh/bench_cpu_bvh_grid.mojo
taskset -c 0 mojo -I . bench/bvh/bench_cpu_bvh_dragon.mojo
taskset -c 0 mojo -I . bench/bvh/bench_cpu_bvh_packets.mojo
g++ -O3 -march=native -DNDEBUG -std=c++20 \
  bench/bvh/bench_embree_cpu.cpp -lembree4 -o /tmp/bajo_bench_embree_cpu
taskset -c 0 /tmp/bajo_bench_embree_cpu assets/dragon/dragon.obj


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
