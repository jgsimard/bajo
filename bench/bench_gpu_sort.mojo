from std.time import perf_counter_ns
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv

from bajo.sort.gpu import gpu_sort_pairs
from bajo.sort.gpu.bitonic_sort import (
    bitonic_sort_pairs,
    naive_bitonic_sort_pairs,
)
from bajo.sort.gpu.radix_sort import (
    device_radix_sort_pairs,
    device_radix_sort_keys,
    RadixSortWorkspace,
)
from bajo.sort.gpu.onesweep import (
    onesweep_radix_sort_pairs,
    onesweep_radix_sort_keys,
    OneSweepWorkspace,
)


def check_validity[dtype: DType](keys: DeviceBuffer[dtype], size: Int) raises:
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
struct SortResult(ImplicitlyCopyable, Writable):
    var size: Int
    var time_ms: Float64
    var gks: Float64
    var name: String


def copy_kernel[
    dtype: DType
](
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int,
):
    var gid = global_idx.x
    if gid < size:
        dst[gid] = src[gid]


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


def benchmark_sorts_pairs(sizes: List[Int]) raises -> List[List[SortResult]]:
    comptime keys_dtype = DType.uint32
    comptime vals_dtype = DType.uint32
    comptime N_ITERS = 10
    comptime THREADS_PER_BLOCK = 512
    comptime KEYS_PER_THREAD = 9

    # Lists to store results for each algorithm
    var basic_results = List[SortResult]()
    var opt_results = List[SortResult]()
    var radix_results = List[SortResult]()
    var onesweep_results = List[SortResult]()
    var merge_results = List[SortResult]()

    print("Starting benchmarks for key-value sorting...")

    with DeviceContext() as ctx:
        for SIZE in sizes:
            var keys = ctx.enqueue_create_buffer[keys_dtype](SIZE)
            var values = ctx.enqueue_create_buffer[vals_dtype](SIZE)
            var pristine_keys = ctx.enqueue_create_buffer[keys_dtype](SIZE)
            var pristine_vals = ctx.enqueue_create_buffer[vals_dtype](SIZE)

            with pristine_keys.map_to_host() as host_keys, pristine_vals.map_to_host() as host_vals:
                for i in range(SIZE):
                    host_keys[i] = Scalar[keys_dtype](
                        (i * 1103515245 + 12345) & 0x7FFFFFFF
                    )
                    host_vals[i] = Scalar[vals_dtype](i)

            var gdim = ceildiv(SIZE, 256)

            def reset_data() capturing raises:
                ctx.enqueue_function[
                    copy_kernel[keys_dtype], copy_kernel[keys_dtype]
                ](
                    keys.unsafe_ptr(),
                    pristine_keys.unsafe_ptr(),
                    SIZE,
                    grid_dim=gdim,
                    block_dim=256,
                )
                ctx.enqueue_function[
                    copy_kernel[vals_dtype], copy_kernel[vals_dtype]
                ](
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
            naive_bitonic_sort_pairs[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()
            check_validity(keys, SIZE)

            var t0 = perf_counter_ns()
            for _ in range(N_ITERS):
                reset_data()
                naive_bitonic_sort_pairs[THREADS_PER_BLOCK](
                    ctx, keys, values, SIZE
                )
            ctx.synchronize()
            var basic_ns = (
                Float64(perf_counter_ns() - t0) - copy_overhead_total
            ) / N_ITERS
            basic_results.append(
                SortResult(
                    SIZE,
                    basic_ns / 1e6,
                    Float64(SIZE) / basic_ns,
                    "Basic_Bitonic_Sort_Pairs",
                )
            )

            # Shared Mem Bitonic Sort
            reset_data()
            bitonic_sort_pairs[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()
            check_validity(keys, SIZE)

            t0 = perf_counter_ns()
            for _ in range(N_ITERS):
                reset_data()
                bitonic_sort_pairs[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()
            var opt_ns = (
                Float64(perf_counter_ns() - t0) - copy_overhead_total
            ) / N_ITERS
            opt_results.append(
                SortResult(
                    SIZE,
                    opt_ns / 1e6,
                    Float64(SIZE) / opt_ns,
                    "Shared_Memory_Bitonic_Sort_Pairs",
                )
            )

            # Radix
            var radix_ws = RadixSortWorkspace[
                keys_dtype, vals_dtype, KEYS_PER_THREAD=KEYS_PER_THREAD
            ](ctx, SIZE)
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
                SortResult(
                    SIZE,
                    radix_ns / 1e6,
                    Float64(SIZE) / radix_ns,
                    "Radix_Pairs",
                )
            )

            # OneSweep
            comptime TPB = 512
            comptime KEYS_PER_THREAD_ONESWEEP = 15
            var onesweep_ws = OneSweepWorkspace[
                keys_dtype,
                vals_dtype,
                BLOCK_SIZE=TPB,
                KEYS_PER_THREAD=KEYS_PER_THREAD_ONESWEEP,
            ](ctx, SIZE)
            reset_data()
            onesweep_radix_sort_pairs[
                BINNING_TPB=TPB, KEYS_PER_THREAD=KEYS_PER_THREAD_ONESWEEP
            ](ctx, onesweep_ws, keys, values, SIZE)
            ctx.synchronize()
            check_validity(keys, SIZE)

            t0 = perf_counter_ns()
            for _ in range(N_ITERS):
                reset_data()
                onesweep_radix_sort_pairs[
                    BINNING_TPB=TPB, KEYS_PER_THREAD=KEYS_PER_THREAD_ONESWEEP
                ](ctx, onesweep_ws, keys, values, SIZE)
            ctx.synchronize()
            var onesweep_ns = (
                Float64(perf_counter_ns() - t0) - copy_overhead_total
            ) / N_ITERS
            onesweep_results.append(
                SortResult(
                    SIZE,
                    onesweep_ns / 1e6,
                    Float64(SIZE) / onesweep_ns,
                    "Onesweep_Pairs",
                )
            )

            # SMEM bitonic sort + OneSweep
            reset_data()
            gpu_sort_pairs(ctx, keys, values, SIZE)
            ctx.synchronize()
            check_validity(keys, SIZE)

            t0 = perf_counter_ns()
            for _ in range(N_ITERS):
                reset_data()
                gpu_sort_pairs(ctx, keys, values, SIZE)
            ctx.synchronize()
            var merge_ns = (
                Float64(perf_counter_ns() - t0) - copy_overhead_total
            ) / N_ITERS
            merge_results.append(
                SortResult(
                    SIZE,
                    merge_ns / 1e6,
                    Float64(SIZE) / merge_ns,
                    "Onesweep_Pairs",
                )
            )
    print_results_table("Basic Bitonic Sort (Pairs)", basic_results)
    print_results_table("Shared Mem Bitonic Sort (Pairs)", opt_results)
    print_results_table("Radix Sort (Pairs)", radix_results)
    print_results_table("OneSweep Radix Sort (Pairs)", onesweep_results)
    print_results_table("SMEM Bitonic + Onesweep Sort (Pairs)", merge_results)

    return [basic_results^, opt_results^, radix_results^, onesweep_results^]


def benchmark_sort_key(sizes: List[Int]) raises -> List[List[SortResult]]:
    comptime dtype = DType.uint32
    comptime N_ITERS = 10
    comptime WARMUP = 5
    comptime THREADS_PER_BLOCK = 512
    comptime KEYS_PER_THREAD = 9

    print("\nStarting benchmarks for keys only sorting...")
    var radix_results = List[SortResult]()
    var onesweep_results = List[SortResult]()
    with DeviceContext() as ctx:
        for SIZE in sizes:
            var keys = ctx.enqueue_create_buffer[dtype](SIZE)
            var pristine_keys = ctx.enqueue_create_buffer[dtype](SIZE)

            with pristine_keys.map_to_host() as host_keys:
                for i in range(SIZE):
                    # Deterministic pseudo-random (xor-shift style)
                    host_keys[i] = Scalar[dtype](
                        (i * 1103515245 + 12345) & 0x7FFFFFFF
                    )

            var gdim = ceildiv(SIZE, 256)

            def reset_data() capturing raises:
                ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
                    keys.unsafe_ptr(),
                    pristine_keys.unsafe_ptr(),
                    SIZE,
                    grid_dim=gdim,
                    block_dim=256,
                )

            # Radix
            reset_data()
            device_radix_sort_keys[KEYS_PER_THREAD=KEYS_PER_THREAD](
                ctx, keys, SIZE
            )
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
                device_radix_sort_keys[KEYS_PER_THREAD=KEYS_PER_THREAD](
                    ctx, keys, SIZE
                )
            ctx.synchronize()
            var total_ns = Float64(perf_counter_ns() - t0)

            var pure_sort_ns_avg = (total_ns - copy_overhead_total) / N_ITERS
            var ms = pure_sort_ns_avg / 1_000_000.0
            var gks = Float64(SIZE) / pure_sort_ns_avg

            radix_results.append(SortResult(SIZE, ms, gks, "Radix_Keys"))

            # Onesweep
            comptime TPB = 512
            comptime KEYS_PER_THREAD_ONESWEEP = 15
            reset_data()
            var onesweep_ws = OneSweepWorkspace[
                dtype,
                DType.invalid,
                BLOCK_SIZE=TPB,
                KEYS_PER_THREAD=KEYS_PER_THREAD_ONESWEEP,
            ](ctx, SIZE)
            onesweep_radix_sort_keys[
                BINNING_TPB=TPB, KEYS_PER_THREAD=KEYS_PER_THREAD_ONESWEEP
            ](ctx, onesweep_ws, keys, SIZE)
            ctx.synchronize()
            check_validity(keys, SIZE)

            t_copy_start = perf_counter_ns()
            for _ in range(N_ITERS):
                reset_data()
            ctx.synchronize()
            copy_overhead_total = Float64(perf_counter_ns() - t_copy_start)

            t0 = perf_counter_ns()
            for _ in range(N_ITERS):
                reset_data()
                onesweep_radix_sort_keys[
                    BINNING_TPB=TPB, KEYS_PER_THREAD=KEYS_PER_THREAD_ONESWEEP
                ](ctx, onesweep_ws, keys, SIZE)
            ctx.synchronize()
            total_ns = Float64(perf_counter_ns() - t0)

            pure_sort_ns_avg = (total_ns - copy_overhead_total) / N_ITERS
            ms = pure_sort_ns_avg / 1_000_000.0
            gks = Float64(SIZE) / pure_sort_ns_avg

            onesweep_results.append(SortResult(SIZE, ms, gks, "Onesweep_Keys"))

    print_results_table("Radix Sort (Keys)", radix_results)
    print_results_table("OneSweep Radix Sort (Keys)", onesweep_results)

    return [radix_results^, onesweep_results^]


def save_results_csv(filename: String, data: List[List[SortResult]]) raises:
    with open(filename, "w") as f:
        f.write("Algorithm,N,Time_ms,Throughput_GKs\n")
        for i in range(len(data)):
            if len(data[i]) == 0:
                continue
            var label = data[i][0].name
            var results = data[i].copy()
            for res in results:
                var line = t"{label},{res.size},{res.time_ms},{res.gks}\n"
                f.write(line)
    print("\n[INFO] Results saved to " + filename)


def main() raises:
    sizes = [
        1 << 10,
        1 << 12,
        1 << 14,
        1 << 16,
        1 << 18,
        1 << 20,
        # 1 << 22,
        # 1 << 24,
        # 1 << 26,
        # 1 << 28,
    ]
    res = benchmark_sorts_pairs(sizes)
    # res_keys = benchmark_sort_key(sizes)
    # res.extend(res_keys^)
    # res = benchmark_sort_key(sizes)

    save_results_csv("gpu_sort_benchmark_results.csv", res)


# current results on rtx 5060 ti 16 gb
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
#       1k |      0.03 |      0.034
#       4k |     0.054 |      0.075
#      16k |     0.052 |      0.312
#      65k |     0.054 |       1.21
#     262k |      0.07 |      3.729
#       1M |     0.147 |      7.154
#       4M |     0.769 |      5.453
#      16M |     3.599 |      4.662
#      67M |    14.748 |       4.55
#     268M |    59.414 |      4.518
# =========================================

# == OneSweep Radix Sort (Pairs) ==
#     N    | Time (ms) | Throughput (GK/s)
# -----------------------------------------
#       1k |     0.036 |      0.029
#       4k |     0.052 |      0.078
#      16k |     0.065 |      0.252
#      65k |     0.076 |      0.857
#     262k |     0.077 |      3.383
#       1M |      0.16 |      6.555
#       4M |     0.713 |      5.881
#      16M |     3.061 |      5.481
#      67M |    12.246 |       5.48
#     268M |    49.019 |      5.476
# =========================================

# Starting benchmarks for keys only sorting...

# == Radix Sort (Keys) ==
#     N    | Time (ms) | Throughput (GK/s)
# -----------------------------------------
#       1k |     0.057 |      0.018
#       4k |     0.069 |      0.059
#      16k |     0.063 |      0.262
#      65k |     0.061 |      1.077
#     262k |     0.078 |      3.356
#       1M |     0.134 |      7.827
#       4M |     0.448 |      9.363
#      16M |     2.059 |      8.148
#      67M |     8.834 |      7.597
#     268M |    36.125 |      7.431
# =========================================

# == OneSweep Radix Sort (Keys) ==
#     N    | Time (ms) | Throughput (GK/s)
# -----------------------------------------
#       1k |      0.04 |      0.026
#       4k |     0.045 |      0.091
#      16k |     0.058 |      0.283
#      65k |     0.067 |      0.977
#     262k |      0.07 |      3.747
#       1M |     0.136 |       7.73
#       4M |     0.415 |     10.096
#      16M |      1.64 |     10.228
#      67M |     6.565 |     10.222
#     268M |    26.139 |      10.27
# =========================================
