from std.time import perf_counter_ns
from std.gpu import block_idx, block_dim, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv

from bajo.sort.gpu import bitonic_sort, bitonic_sort_basic
from bajo.sort.gpu.radix_sort import (
    device_radix_sort_pairs,
    device_radix_sort_keys,
    RadixSortWorkspace,
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


def copy_kernel(
    dst: UnsafePointer[UInt32, MutAnyOrigin],
    src: UnsafePointer[UInt32, MutAnyOrigin],
    size: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid < size:
        dst[tid] = src[tid]


def benchmark_sorts_key_value[SIZE: Int = 1 << 20]() raises:
    comptime dtype = DType.uint32
    comptime WARMUP = 5
    comptime N_ITERS = 10
    comptime THREADS_PER_BLOCK = 512

    print(t"\nSetting up benchmarks for {SIZE} elements...")

    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[dtype](SIZE)
        var values = ctx.enqueue_create_buffer[dtype](SIZE)

        var pristine_keys = ctx.enqueue_create_buffer[dtype](SIZE)
        var pristine_vals = ctx.enqueue_create_buffer[dtype](SIZE)

        def init_pristine_data() capturing raises:
            with pristine_keys.map_to_host() as host_keys, pristine_vals.map_to_host() as host_vals:
                for i in range(SIZE):
                    var val = UInt32((i * 1103515245 + 12345) & 0x7FFFFFFF)
                    host_keys[i] = val
                    host_vals[i] = UInt32(i)

        init_pristine_data()
        var gdim = ceildiv(SIZE, 256)

        def reset_data() capturing raises:
            ctx.enqueue_function[copy_kernel, copy_kernel](
                keys.unsafe_ptr(),
                pristine_keys.unsafe_ptr(),
                SIZE,
                grid_dim=gdim,
                block_dim=256,
            )
            ctx.enqueue_function[copy_kernel, copy_kernel](
                values.unsafe_ptr(),
                pristine_vals.unsafe_ptr(),
                SIZE,
                grid_dim=gdim,
                block_dim=256,
            )

        # Basic Bitonic Sort
        bitonic_sort_basic[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
        ctx.synchronize()
        check_validity(keys, SIZE)

        t_total_start = perf_counter_ns()
        for _ in range(N_ITERS):
            reset_data()
            bitonic_sort_basic[THREADS_PER_BLOCK](ctx, keys, values, SIZE)

        ctx.synchronize()
        var total_basic_ns = Float64(perf_counter_ns() - t_total_start)

        # # Shared Memory Bitonic Sort
        reset_data()
        bitonic_sort[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
        ctx.synchronize()
        check_validity(keys, SIZE)

        t_total_start = perf_counter_ns()
        for _ in range(N_ITERS):
            reset_data()
            bitonic_sort[THREADS_PER_BLOCK](ctx, keys, values, SIZE)

        ctx.synchronize()
        var total_opt_ns = Float64(perf_counter_ns() - t_total_start)

        # Radix Sort
        var radix_ws = RadixSortWorkspace[dtype](ctx, SIZE)
        reset_data()
        device_radix_sort_pairs(ctx, radix_ws, keys, values, SIZE)
        ctx.synchronize()
        check_validity(keys, SIZE)

        t_total_start = perf_counter_ns()
        for _ in range(N_ITERS):
            reset_data()
            device_radix_sort_pairs(ctx, radix_ws, keys, values, SIZE)

        ctx.synchronize()
        var radix_total_ns = Float64(perf_counter_ns() - t_total_start)

        # Copy only
        var t_copy_start = perf_counter_ns()
        for _ in range(N_ITERS):
            reset_data()

        ctx.synchronize()
        var copy_ns = Float64(perf_counter_ns() - t_copy_start)

        # Subtraction
        var basic_ns = (total_basic_ns - copy_ns) / N_ITERS
        var opt_ns = (total_opt_ns - copy_ns) / N_ITERS
        var radix_ns = (radix_total_ns - copy_ns) / N_ITERS

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

    benchmark_sort_key[1 << 10]()
    benchmark_sort_key[1 << 12]()
    benchmark_sort_key[1 << 14]()
    benchmark_sort_key[1 << 16]()
    benchmark_sort_key[1 << 18]()
    benchmark_sort_key[1 << 20]()
    benchmark_sort_key[1 << 22]()
    benchmark_sort_key[1 << 24]()
    benchmark_sort_key[1 << 26]()
    benchmark_sort_key[1 << 28]()


# current results
# Setting up benchmarks for 1024 elements...
# ==================== RESULTS ====================
# Data Size: 1024 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 0.3566 | 0.003
# Shared Mem Bitonic Sort | 0.014 | 0.073
# Radix Sort              | 0.0646 | 0.016
# =================================================

# Setting up benchmarks for 4096 elements...
# ==================== RESULTS ====================
# Data Size: 4096 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 0.4842 | 0.008
# Shared Mem Bitonic Sort | 0.0269 | 0.152
# Radix Sort              | 0.1403 | 0.029
# =================================================

# Setting up benchmarks for 16384 elements...
# ==================== RESULTS ====================
# Data Size: 16384 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 0.6589 | 0.025
# Shared Mem Bitonic Sort | 0.0496 | 0.33
# Radix Sort              | 0.1041 | 0.157
# =================================================

# Setting up benchmarks for 65536 elements...
# ==================== RESULTS ====================
# Data Size: 65536 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 1.0326 | 0.063
# Shared Mem Bitonic Sort | 0.1004 | 0.653
# Radix Sort              | 0.102 | 0.643
# =================================================

# Setting up benchmarks for 262144 elements...
# ==================== RESULTS ====================
# Data Size: 262144 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 1.5214 | 0.172
# Shared Mem Bitonic Sort | 0.3128 | 0.838
# Radix Sort              | 0.119 | 2.202
# =================================================

# Setting up benchmarks for 1048576 elements...
# ==================== RESULTS ====================
# Data Size: 1048576 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 3.349 | 0.313
# Shared Mem Bitonic Sort | 1.0233 | 1.025
# Radix Sort              | 0.3121 | 3.36
# =================================================

# Setting up benchmarks for 4194304 elements...
# ==================== RESULTS ====================
# Data Size: 4194304 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 14.0959 | 0.298
# Shared Mem Bitonic Sort | 5.126 | 0.818
# Radix Sort              | 0.9811 | 4.275
# =================================================

# Setting up benchmarks for 16777216 elements...
# ==================== RESULTS ====================
# Data Size: 16777216 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 160.2779 | 0.105
# Shared Mem Bitonic Sort | 66.7245 | 0.251
# Radix Sort              | 4.1089 | 4.083
# =================================================

# Setting up benchmarks for 67108864 elements...
# ==================== RESULTS ====================
# Data Size: 67108864 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 738.5024 | 0.091
# Shared Mem Bitonic Sort | 333.4763 | 0.201
# Radix Sort              | 16.9234 | 3.965
# =================================================

# Setting up benchmarks for 268435456 elements...
# ==================== RESULTS ====================
# Data Size: 268435456 elements (32-bit keys)
# Algorithm               | Time (ms) | Throughput (GK/s)
# -------------------------------------------------
# Basic Bitonic Sort      | 3403.3905 | 0.079
# Shared Mem Bitonic Sort | 1603.1786 | 0.167
# Radix Sort              | 64.7352 | 4.147
# =================================================
