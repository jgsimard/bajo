from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv

from bajo.sort.gpu import bitonic_sort, bitonic_sort_basic
from bajo.sort.gpu.radix_sort import (
    device_radix_sort_pairs,
    device_radix_sort_keys,
)


def check_validity(keys: DeviceBuffer[DType.uint32], size: Int) raises:
    with keys.map_to_host() as host:
        for i in range(size - 1):
            if host[i] > host[i + 1]:
                raise Error(
                    t"VALIDITY FAILED: Keys are not sorted at index {i}"
                )


def calculate_gks(ns_per_iter: Float64, SIZE: Int) -> Float64:
    return Float64(SIZE) / ns_per_iter


def fmt(ns: Float64, SIZE: Int) -> String:
    var ms = round(ns / 1_000_000.0, 4)
    var gks = round(calculate_gks(ns, SIZE), 3)
    return String(ms) + " | " + String(gks)


def benchmark_sorts_key_value[SIZE: Int = 1 << 20]() raises:
    comptime dtype = DType.uint32
    # comptime SIZE = 1 << 20
    comptime WARMUP = 5
    comptime N_ITERS = 10
    comptime THREADS_PER_BLOCK = 512

    print(t"\nSetting up benchmarks for {SIZE} elements...")

    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[dtype](SIZE)
        var values = ctx.enqueue_create_buffer[dtype](SIZE)

        def reset_data() capturing raises:
            with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
                for i in range(SIZE):
                    # Deterministic pseudo-random (xor-shift style)
                    var val = UInt32((i * 1103515245 + 12345) & 0x7FFFFFFF)
                    host_keys[i] = val
                    host_values[i] = UInt32(i)

        # Basic Bitonic Sort
        reset_data()
        bitonic_sort_basic[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
        ctx.synchronize()
        check_validity(keys, SIZE)

        t0 = perf_counter_ns()
        for _ in range(N_ITERS):
            bitonic_sort_basic[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()
        var basic_ns = Float64(perf_counter_ns() - t0) / N_ITERS

        # Shared Memory Bitonic Sort
        reset_data()
        bitonic_sort[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
        ctx.synchronize()
        check_validity(keys, SIZE)

        t0 = perf_counter_ns()
        for _ in range(N_ITERS):
            bitonic_sort[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()
        var opt_ns = Float64(perf_counter_ns() - t0) / N_ITERS

        # Radix Sort
        reset_data()
        device_radix_sort_pairs(ctx, keys, values, SIZE)
        ctx.synchronize()
        check_validity(keys, SIZE)

        t0 = perf_counter_ns()
        for _ in range(N_ITERS):
            device_radix_sort_pairs(ctx, keys, values, SIZE)
            ctx.synchronize()
        var radix_ns = Float64(perf_counter_ns() - t0) / N_ITERS

        print("==================== RESULTS ====================")
        print(t"Data Size: {SIZE} elements (32-bit keys)")
        print("Algorithm               | Time (ms) | Throughput (GK/s)")
        print("-------------------------------------------------")
        print(t"Basic Bitonic Sort      | {fmt(basic_ns, SIZE)}")
        print(t"Shared Mem Bitonic Sort | {fmt(opt_ns, SIZE)}")
        print(t"Radix Sort              | {fmt(radix_ns, SIZE)}")
        print("=================================================")


def benchmark_sort_key[SIZE: Int = 1 << 20]() raises:
    comptime dtype = DType.uint32
    comptime N_ITERS = 10
    comptime WARMUP = 5
    comptime THREADS_PER_BLOCK = 512

    print(t"\nSetting up benchmarks for {SIZE} elements...")

    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[dtype](SIZE)

        def reset_data() capturing raises:
            with keys.map_to_host() as host_keys:
                for i in range(SIZE):
                    # Deterministic pseudo-random (xor-shift style)
                    var val = UInt32((i * 1103515245 + 12345) & 0x7FFFFFFF)
                    host_keys[i] = val

        # Radix Sort
        reset_data()
        device_radix_sort_keys(ctx, keys, SIZE)
        ctx.synchronize()
        check_validity(keys, SIZE)

        t0 = perf_counter_ns()
        for _ in range(N_ITERS):
            device_radix_sort_keys(ctx, keys, SIZE)
            ctx.synchronize()
        var radix_ns = Float64(perf_counter_ns() - t0) / N_ITERS

        print("==================== RESULTS ====================")
        print(t"Data Size: {SIZE} elements (32-bit keys)")
        print("Algorithm               | Time (ms) | Throughput (GK/s)")
        print("-------------------------------------------------")
        # print(t"Basic Bitonic Sort      | {fmt(basic_ns)}")
        # print(t"Shared Mem Bitonic Sort | {fmt(opt_ns)}")
        print(t"Radix Sort              | {fmt(radix_ns, SIZE)}")
        print("=================================================")


def main() raises:
    benchmark_sorts_key_value[1 << 10]()
    benchmark_sorts_key_value[1 << 12]()
    benchmark_sorts_key_value[1 << 14]()
    benchmark_sorts_key_value[1 << 16]()
    benchmark_sorts_key_value[1 << 18]()
    benchmark_sorts_key_value[1 << 20]()
    benchmark_sorts_key_value[1 << 22]()
    benchmark_sorts_key_value[1 << 24]()
    benchmark_sorts_key_value[1 << 26]()
    benchmark_sorts_key_value[1 << 28]()

    # benchmark_sort_key[1 << 10]()
    # benchmark_sort_key[1 << 12]()
    # benchmark_sort_key[1 << 14]()
    # benchmark_sort_key[1 << 16]()
    # benchmark_sort_key[1 << 18]()
    # benchmark_sort_key[1 << 20]()
    # benchmark_sort_key[1 << 22]()
    # benchmark_sort_key[1 << 24]()
    # benchmark_sort_key[1 << 26]()
    # benchmark_sort_key[1 << 28]()


# current results
# Setting up benchmarks for 1024 elements...
# ==================== RESULTS ====================
# Data Size: 1024 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 0.3233 | 0.003
# Shared Mem Bitonic Sort | 0.0128 | 0.08
# Radix Sort              | 0.0941 | 0.011
# =================================================

# Setting up benchmarks for 4096 elements...
# ==================== RESULTS ====================
# Data Size: 4096 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 0.4787 | 0.009
# Shared Mem Bitonic Sort | 0.028 | 0.146
# Radix Sort              | 0.1272 | 0.032
# =================================================

# Setting up benchmarks for 16384 elements...
# ==================== RESULTS ====================
# Data Size: 16384 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 0.6504 | 0.025
# Shared Mem Bitonic Sort | 0.052 | 0.315
# Radix Sort              | 0.1627 | 0.101
# =================================================

# Setting up benchmarks for 65536 elements...
# ==================== RESULTS ====================
# Data Size: 65536 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 0.9691 | 0.068
# Shared Mem Bitonic Sort | 0.1013 | 0.647
# Radix Sort              | 0.1582 | 0.414
# =================================================

# Setting up benchmarks for 262144 elements...
# ==================== RESULTS ====================
# Data Size: 262144 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 1.4272 | 0.184
# Shared Mem Bitonic Sort | 0.3122 | 0.84
# Radix Sort              | 0.1893 | 1.385
# =================================================

# Setting up benchmarks for 1048576 elements...
# ==================== RESULTS ====================
# Data Size: 1048576 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 2.9376 | 0.357
# Shared Mem Bitonic Sort | 1.0087 | 1.04
# Radix Sort              | 0.339 | 3.093
# =================================================

# Setting up benchmarks for 4194304 elements...
# ==================== RESULTS ====================
# Data Size: 4194304 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 9.7529 | 0.43
# Shared Mem Bitonic Sort | 4.7613 | 0.881
# Radix Sort              | 1.2134 | 3.457
# =================================================

# Setting up benchmarks for 16777216 elements...
# ==================== RESULTS ====================
# Data Size: 16777216 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 135.9153 | 0.123
# Shared Mem Bitonic Sort | 65.4751 | 0.256
# Radix Sort              | 4.6132 | 3.637
# =================================================

# Setting up benchmarks for 67108864 elements...
# ==================== RESULTS ====================
# Data Size: 67108864 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 627.4224 | 0.107
# Shared Mem Bitonic Sort | 328.2056 | 0.204
# Radix Sort              | 18.4465 | 3.638
# =================================================

# Setting up benchmarks for 268435456 elements...
# ==================== RESULTS ====================
# Data Size: 268435456 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 2908.1129 | 0.092
# Shared Mem Bitonic Sort | 1628.0588 | 0.165
# Radix Sort              | 77.3159 | 3.472
# =================================================
