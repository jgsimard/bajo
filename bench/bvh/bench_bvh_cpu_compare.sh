#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
EMBREE_CXX="${EMBREE_CXX:-g++}"
CXX="${CXX:-clang++}"
TINYBVH_DIR="${TINYBVH_DIR:-external/tinybvh}"
SOURCE="${SOURCE:-${SCRIPT_DIR}/bench_tinybvh.cpp}"
EMBREE_OUTPUT="${EMBREE_OUTPUT:-/tmp/bajo_bench_embree_cpu}"
TINYBVH_SINGLE_OUTPUT="${TINYBVH_SINGLE_OUTPUT:-/tmp/bajo_bench_tinybvh_single}"
TINYBVH_ALL_OUTPUT="${TINYBVH_ALL_OUTPUT:-/tmp/bajo_bench_tinybvh_all}"
OBJ_PATH="${1:-./assets/dragon/dragon.obj}"
ALL_CPU_LIST="${BVH_BENCH_CPU_LIST:-$(taskset -pc $$ | sed 's/.*: //')}"
ALL_THREAD_COUNT="${BVH_BENCH_THREAD_COUNT:-$(taskset -c "${ALL_CPU_LIST}" nproc)}"
SINGLE_CPU="${BVH_BENCH_SINGLE_CPU:-${ALL_CPU_LIST%%[-,]*}}"

cd "${ROOT_DIR}"

if [[ ! -f "${TINYBVH_DIR}/tiny_bvh.h" ]]; then
    echo "error: ${TINYBVH_DIR}/tiny_bvh.h not found" >&2
    echo "Clone TinyBVH there, or set TINYBVH_DIR." >&2
    exit 1
fi

"${EMBREE_CXX}" \
    -std=c++20 \
    -O3 \
    -DNDEBUG \
    -march=native \
    bench/bvh/bench_embree_cpu.cpp \
    -lembree4 \
    -o "${EMBREE_OUTPUT}"

# clang++: warning: argument '-Ofast' is deprecated; use '-O3 -ffast-math'
compile_tinybvh() {
    local output="$1"
    shift
    "${CXX}" \
        -std=c++20 \
        -O3 \
        -ffast-math \
        -DNDEBUG \
        -march=native \
        -mavx2 \
        -mfma \
        -I"${TINYBVH_DIR}" \
        "$@" \
        "${SOURCE}" \
        -o "${output}"
}

run_suite() {
    local mode="$1"
    local available_cpus="$2"
    local affinity="$3"
    local embree_threads="$4"
    local tinybvh_output="$5"

    echo \
        "=== BVH build threads: ${mode}; available CPUs: ${available_cpus}; affinity: ${affinity} ==="
    taskset -c "${affinity}" mojo -I . bench/bvh/bench_cpu_bvh_grid.mojo
    taskset -c "${affinity}" mojo -I . bench/bvh/bench_cpu_bvh_dragon.mojo
    taskset -c "${affinity}" mojo -I . bench/bvh/bench_cpu_bvh_packets.mojo
    env BVH_BUILD_THREADS="${embree_threads}" taskset -c "${affinity}" \
        "${EMBREE_OUTPUT}" "${OBJ_PATH}"
    taskset -c "${affinity}" "${tinybvh_output}" "${OBJ_PATH}"
}

compile_tinybvh "${TINYBVH_SINGLE_OUTPUT}"
compile_tinybvh \
    "${TINYBVH_ALL_OUTPUT}" \
    -DBENCH_THREADED_BUILDS \
    -pthread

run_suite 1 1 "${SINGLE_CPU}" 1 "${TINYBVH_SINGLE_OUTPUT}"
run_suite \
    all \
    "${ALL_THREAD_COUNT}" \
    "${ALL_CPU_LIST}" \
    "${ALL_THREAD_COUNT}" \
    "${TINYBVH_ALL_OUTPUT}"
