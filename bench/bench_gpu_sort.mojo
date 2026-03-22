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
                    t"VALIDITY FAILED: Keys are not sorted at index {i}"
                )


def benchmark_gpu_sorts[SIZE: Int = 1 << 20]() raises:
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
        print(t"Data Size: {SIZE} elements (32-bit keys)")
        print("Algorithm               | Time (ms) | Throughput (GK/s)")
        print("-------------------------------------------------")
        print(t"Basic Bitonic Sort      | {fmt(basic_ns)}")
        print(t"Shared Mem Bitonic Sort | {fmt(opt_ns)}")
        print(t"Radix Sort              | {fmt(radix_ns)}")
        print("=================================================")


def main() raises:
    benchmark_gpu_sorts[1 << 10]()
    benchmark_gpu_sorts[1 << 12]()
    benchmark_gpu_sorts[1 << 14]()
    benchmark_gpu_sorts[1 << 16]()
    benchmark_gpu_sorts[1 << 18]()
    benchmark_gpu_sorts[1 << 20]()
    benchmark_gpu_sorts[1 << 22]()
    benchmark_gpu_sorts[1 << 24]()
    benchmark_gpu_sorts[1 << 26]()
    # benchmark_gpu_sorts[1 << 28]()


# current results
# Setting up benchmarks for 1024 elements...
# ==================== RESULTS ====================
# Data Size: 1024 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 0.3254 | 0.003
# Shared Mem Bitonic Sort | 0.0134 | 0.076
# Radix Sort              | 0.1025 | 0.01
# =================================================

# Setting up benchmarks for 4096 elements...
# ==================== RESULTS ====================
# Data Size: 4096 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 0.4771 | 0.009
# Shared Mem Bitonic Sort | 0.0279 | 0.147
# Radix Sort              | 0.1583 | 0.026
# =================================================

# Setting up benchmarks for 16384 elements...
# ==================== RESULTS ====================
# Data Size: 16384 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 0.6536 | 0.025
# Shared Mem Bitonic Sort | 0.0517 | 0.317
# Radix Sort              | 0.2367 | 0.069
# =================================================

# Setting up benchmarks for 65536 elements...
# ==================== RESULTS ====================
# Data Size: 65536 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 0.9267 | 0.071
# Shared Mem Bitonic Sort | 0.1013 | 0.647
# Radix Sort              | 0.2317 | 0.283
# =================================================

# Setting up benchmarks for 262144 elements...
# ==================== RESULTS ====================
# Data Size: 262144 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 1.4019 | 0.187
# Shared Mem Bitonic Sort | 0.3499 | 0.749
# Radix Sort              | 0.2459 | 1.066
# =================================================

# Setting up benchmarks for 1048576 elements...
# ==================== RESULTS ====================
# Data Size: 1048576 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 2.9451 | 0.356
# Shared Mem Bitonic Sort | 1.0219 | 1.026
# Radix Sort              | 0.4465 | 2.349
# =================================================

# Setting up benchmarks for 4194304 elements...
# ==================== RESULTS ====================
# Data Size: 4194304 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 9.8762 | 0.425
# Shared Mem Bitonic Sort | 4.8728 | 0.861
# Radix Sort              | 1.4204 | 2.953
# =================================================

# Setting up benchmarks for 16777216 elements...
# ==================== RESULTS ====================
# Data Size: 16777216 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 133.278 | 0.126
# Shared Mem Bitonic Sort | 65.2862 | 0.257
# Radix Sort              | 5.0302 | 3.335
# =================================================

# Setting up benchmarks for 67108864 elements...
# ==================== RESULTS ====================
# Data Size: 67108864 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 621.8846 | 0.108
# Shared Mem Bitonic Sort | 327.1196 | 0.205
# Radix Sort              | 19.2547 | 3.485
# =================================================

# Setting up benchmarks for 268435456 elements...
# ==================== RESULTS ====================
# Data Size: 268435456 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 2917.6888 | 0.092
# Shared Mem Bitonic Sort | 1618.9657 | 0.166
# Radix Sort              | 76.7529 | 3.497
# =================================================
