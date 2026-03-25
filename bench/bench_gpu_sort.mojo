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


@fieldwise_init
struct SortResult(Copyable):
    var size: Int
    var time_ms: Float64
    var gks: Float64


def copy_kernel(
    dst: UnsafePointer[UInt32, MutAnyOrigin],
    src: UnsafePointer[UInt32, MutAnyOrigin],
    size: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid < size:
        dst[tid] = src[tid]


def print_results_table(title: String, results: List[SortResult]):
    print("\n== " + title + " ==")
    print("    N    | Time (ms) | Throughput (GK/s)")
    print("-----------------------------------------")
    for res in results:
        var size_str = String(res.size)
        if res.size >= 1_000_000:
            size_str = String(res.size // 1_000_000) + "M"
        elif res.size >= 1_000:
            size_str = String(res.size // 1_000) + "k"

        size_str = size_str.ascii_rjust(8)
        var ms_str = String(round(res.time_ms, 3)).ascii_rjust(9)
        var gks_str = String(round(res.gks, 3)).ascii_rjust(10)

        print(t"{size_str} | {ms_str} | {gks_str}")
    print("=========================================")


def benchmark_sorts_key_value(sizes: List[Int]) raises:
    comptime dtype = DType.uint32
    comptime N_ITERS = 10
    comptime THREADS_PER_BLOCK = 512

    # Lists to store results for each algorithm
    var basic_results = List[SortResult]()
    var opt_results = List[SortResult]()
    var radix_results = List[SortResult]()

    print("Starting benchmarks for key-value sorting...")

    with DeviceContext() as ctx:
        for SIZE in sizes:
            var keys = ctx.enqueue_create_buffer[dtype](SIZE)
            var values = ctx.enqueue_create_buffer[dtype](SIZE)
            var pristine_keys = ctx.enqueue_create_buffer[dtype](SIZE)
            var pristine_vals = ctx.enqueue_create_buffer[dtype](SIZE)

            with pristine_keys.map_to_host() as host_keys, pristine_vals.map_to_host() as host_vals:
                for i in range(SIZE):
                    var val = UInt32((i * 1103515245 + 12345) & 0x7FFFFFFF)
                    host_keys[i] = val
                    host_vals[i] = UInt32(i)

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

            # copy overhead
            var t_copy_start = perf_counter_ns()
            for _ in range(N_ITERS):
                reset_data()
            ctx.synchronize()
            var copy_overhead_total = Float64(perf_counter_ns() - t_copy_start)

            # Basic Bitonic Sort
            reset_data()
            bitonic_sort_basic[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()
            check_validity(keys, SIZE)

            var t0 = perf_counter_ns()
            for _ in range(N_ITERS):
                reset_data()
                bitonic_sort_basic[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()
            var basic_ns = (
                Float64(perf_counter_ns() - t0) - copy_overhead_total
            ) / N_ITERS
            basic_results.append(
                SortResult(SIZE, basic_ns / 1e6, Float64(SIZE) / basic_ns)
            )

            # Shared Mem Bitonic Sort
            reset_data()
            bitonic_sort[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()
            check_validity(keys, SIZE)

            t0 = perf_counter_ns()
            for _ in range(N_ITERS):
                reset_data()
                bitonic_sort[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()
            var opt_ns = (
                Float64(perf_counter_ns() - t0) - copy_overhead_total
            ) / N_ITERS
            opt_results.append(
                SortResult(SIZE, opt_ns / 1e6, Float64(SIZE) / opt_ns)
            )

            # Radix Sort
            var radix_ws = RadixSortWorkspace[dtype](ctx, SIZE)
            reset_data()
            device_radix_sort_pairs(ctx, radix_ws, keys, values, SIZE)
            ctx.synchronize()
            check_validity(keys, SIZE)

            t0 = perf_counter_ns()
            for _ in range(N_ITERS):
                reset_data()
                device_radix_sort_pairs(ctx, radix_ws, keys, values, SIZE)
            ctx.synchronize()
            var radix_ns = (
                Float64(perf_counter_ns() - t0) - copy_overhead_total
            ) / N_ITERS
            radix_results.append(
                SortResult(SIZE, radix_ns / 1e6, Float64(SIZE) / radix_ns)
            )

    print_results_table("Basic Bitonic Sort (Pairs)", basic_results)
    print_results_table("Shared Mem Bitonic Sort (Pairs)", opt_results)
    print_results_table("Radix Sort (Pairs)", radix_results)


def benchmark_sort_key(sizes: List[Int]) raises:
    comptime dtype = DType.uint32
    comptime N_ITERS = 10
    comptime WARMUP = 5
    comptime THREADS_PER_BLOCK = 512

    print("\nStarting benchmarks for keys only sorting...")
    var results = List[SortResult]()
    with DeviceContext() as ctx:
        for SIZE in sizes:
            var keys = ctx.enqueue_create_buffer[dtype](SIZE)
            var pristine_keys = ctx.enqueue_create_buffer[dtype](SIZE)

            with pristine_keys.map_to_host() as host_keys:
                for i in range(SIZE):
                    # Deterministic pseudo-random (xor-shift style)
                    var val = UInt32((i * 1103515245 + 12345) & 0x7FFFFFFF)
                    host_keys[i] = val

            var gdim = ceildiv(SIZE, 256)

            def reset_data() capturing raises:
                ctx.enqueue_function[copy_kernel, copy_kernel](
                    keys.unsafe_ptr(),
                    pristine_keys.unsafe_ptr(),
                    SIZE,
                    grid_dim=gdim,
                    block_dim=256,
                )

            # Radix Sort
            reset_data()
            device_radix_sort_keys(ctx, keys, SIZE)
            ctx.synchronize()
            check_validity(keys, SIZE)

            var t_copy_start = perf_counter_ns()
            for _ in range(N_ITERS):
                reset_data()
            ctx.synchronize()
            var copy_overhead_total = Float64(perf_counter_ns() - t_copy_start)

            var t0 = perf_counter_ns()
            for _ in range(N_ITERS):
                reset_data()
                device_radix_sort_keys(ctx, keys, SIZE)
            ctx.synchronize()
            var total_ns = Float64(perf_counter_ns() - t0)

            var pure_sort_ns_avg = (total_ns - copy_overhead_total) / N_ITERS
            var ms = pure_sort_ns_avg / 1_000_000.0
            var gks = Float64(SIZE) / pure_sort_ns_avg

            results.append(SortResult(SIZE, ms, gks))

    print("== Results for Keys only Radix Sort ==")
    print(" N       | Time (ms) | Throughput (GK/s)")
    print("----------------------------------------")
    for res in results:
        # Format elements with 'k' or 'M' for readability
        var size_str = String(res.size)
        if res.size >= 1_000_000:
            size_str = String(res.size // 1_000_000) + "M"
        elif res.size >= 1_000:
            size_str = String(res.size // 1_000) + "k"
        size_str = size_str.ascii_rjust(8)

        print(
            t"{size_str} | {String(round(res.time_ms, 3)).ascii_rjust(6)} |"
            t" {round(res.gks, 3)}"
        )
    print("=========================================")


def main() raises:
    sizes = [
        1 << 10,
        1 << 12,
        1 << 14,
        1 << 16,
        1 << 18,
        1 << 20,
        1 << 22,
        1 << 24,
        1 << 26,
        1 << 28,
    ]
    benchmark_sorts_key_value(sizes)
    # benchmark_sort_key(sizes)


# current results
# Starting benchmarks for key-value sorting...

# == Basic Bitonic Sort (Pairs) ==
#     N    | Time (ms) | Throughput (GK/s)
# -----------------------------------------
#       1k |     0.105 |       0.01
#       4k |     0.164 |      0.025
#      16k |     0.219 |      0.075
#      65k |     0.284 |      0.231
#     262k |     0.704 |      0.373
#       1M |     2.201 |      0.476
#       4M |    12.822 |      0.327
#      16M |   160.965 |      0.104
#      67M |    753.54 |      0.089
#     268M |  3513.673 |      0.076
# =========================================

# == Shared Mem Bitonic Sort (Pairs) ==
#     N    | Time (ms) | Throughput (GK/s)
# -----------------------------------------
#       1k |     0.003 |      0.325
#       4k |     0.028 |      0.147
#      16k |     0.049 |      0.333
#      65k |       0.1 |      0.654
#     262k |     0.386 |      0.678
#       1M |     1.078 |      0.973
#       4M |     4.979 |      0.842
#      16M |    66.956 |      0.251
#      67M |   334.704 |      0.201
#     268M |   1644.81 |      0.163
# =========================================

# == Radix Sort (Pairs) ==
#     N    | Time (ms) | Throughput (GK/s)
# -----------------------------------------
#       1k |     0.056 |      0.018
#       4k |     0.098 |      0.042
#      16k |     0.102 |       0.16
#      65k |     0.104 |      0.629
#     262k |     0.117 |      2.243
#       1M |      0.31 |      3.381
#       4M |     0.995 |      4.214
#      16M |      4.21 |      3.985
#      67M |    16.863 |       3.98
#     268M |    67.792 |       3.96
# =========================================

# Starting benchmarks for keys only sorting...
# == Results for Keys only Radix Sort ==
#  N       | Time (ms) | Throughput (GK/s)
# ----------------------------------------
#       1k |  0.072 | 0.014
#       4k |  0.095 | 0.043
#      16k |  0.107 | 0.153
#      65k |  0.098 | 0.671
#     262k |  0.115 | 2.28
#       1M |   0.27 | 3.879
#       4M |  0.912 | 4.597
#      16M |  3.351 | 5.007
#      67M | 13.518 | 4.964
#     268M | 54.593 | 4.917
# =========================================
