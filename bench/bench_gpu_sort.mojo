from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv

from bajo.sort.gpu import bitonic_sort, bitonic_sort_basic
from bajo.sort.gpu.radix_sort import device_radix_sort_pairs


def check_validity(keys: DeviceBuffer[DType.uint32], size: Int) raises:
    with keys.map_to_host() as host:
        for i in range(size - 1):
            if host[i] > host[i + 1]:
                raise Error(
                    "VALIDITY FAILED: Keys are not sorted at index " + String(i)
                )


def benchmark_gpu_sorts() raises:
    comptime dtype = DType.uint32
    comptime SIZE = 1 << 20  # 1,048,576 elements
    comptime WARMUP = 5
    comptime N_ITERS = 10
    comptime THREADS_PER_BLOCK = 1024

    print("Setting up benchmarks for " + String(SIZE) + " elements...")

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

        @always_inline
        def calculate_gks(ns_per_iter: Float64) -> Float64:
            return Float64(SIZE) / ns_per_iter

        # 1. Benchmark: Basic Bitonic Sort
        reset_data()
        bitonic_sort_basic[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
        ctx.synchronize()
        check_validity(keys, SIZE)

        t0 = perf_counter_ns()
        for _ in range(N_ITERS):
            bitonic_sort_basic[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()
        var basic_ns = Float64(perf_counter_ns() - t0) / N_ITERS

        # 2. Benchmark: Shared Memory Bitonic Sort
        reset_data()
        bitonic_sort[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
        ctx.synchronize()
        check_validity(keys, SIZE)

        t0 = perf_counter_ns()
        for _ in range(N_ITERS):
            bitonic_sort[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()
        var opt_ns = Float64(perf_counter_ns() - t0) / N_ITERS

        # 4. Benchmark: Radix Sort
        reset_data()
        device_radix_sort_pairs(ctx, keys, values, SIZE)
        ctx.synchronize()
        check_validity(keys, SIZE)

        t0 = perf_counter_ns()
        for _ in range(N_ITERS):
            device_radix_sort_pairs(ctx, keys, values, SIZE)
            ctx.synchronize()
        var radix_ns = Float64(perf_counter_ns() - t0) / N_ITERS

        # Results Calculation
        def fmt(ns: Float64) -> String:
            var ms = round(ns / 1_000_000.0, 4)
            var gks = round(calculate_gks(ns), 3)
            return String(ms) + " | " + String(gks)

        print("==================== RESULTS ====================")
        print("Data Size: " + String(SIZE) + " elements (32-bit keys)")
        print("Algorithm               | Time (ms) | Throughput (GK/s)")
        print("-------------------------------------------------")
        print("Basic Bitonic Sort      | " + fmt(basic_ns))
        print("Shared Mem Bitonic Sort | " + fmt(opt_ns))
        print("Radix Sort              | " + fmt(radix_ns))
        print("=================================================")


def main() raises:
    benchmark_gpu_sorts()


# current results
# Setting up benchmarks for 1024 elements...
# ==================== RESULTS ====================
# Data Size: 1024 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 0.3448 | 0.003
# Shared Mem Bitonic Sort | 0.021 | 0.049
# Radix Sort              | 0.1023 | 0.01
# =================================================
#
# Setting up benchmarks for 65536 elements...
# ==================== RESULTS ====================
# Data Size: 65536 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 0.8763 | 0.075
# Shared Mem Bitonic Sort | 0.1107 | 0.592
# Radix Sort              | 0.2318 | 0.283
# =================================================
#
# Setting up benchmarks for 1048576 elements...
# ==================== RESULTS ====================
# Data Size: 1048576 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 3.8401 | 0.273
# Shared Mem Bitonic Sort | 1.7617 | 0.595
# Radix Sort              | 0.454 | 2.309
# =================================================
#
# Setting up benchmarks for 67108864 elements...
# ==================== RESULTS ====================
# Data Size: 67108864 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 681.084 | 0.099
# Shared Mem Bitonic Sort | 358.8891 | 0.187
# Radix Sort              | 19.3657 | 3.465
# =================================================
