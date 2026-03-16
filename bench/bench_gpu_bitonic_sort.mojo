from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from bajo.core.sort.gpu import bitonic_sort, bitonic_sort_basic, radix_sort


def benchmark_bitonic_sorts() raises:
    comptime dtype = DType.uint32
    comptime SIZE = 1 << 20  # 2^20 (1M elements)
    comptime WARMUP = 10
    comptime ITERS = 100
    comptime THREADS_PER_BLOCK = 1024

    print(t"Setting up benchmarks for {SIZE} elements...")

    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[dtype](SIZE)
        var values = ctx.enqueue_create_buffer[dtype](SIZE)

        with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
            for i in range(SIZE):
                host_keys[i] = UInt32(SIZE - i)
                host_values[i] = UInt32(i)

        # ---------------------------------------------------------
        # Optimized Shared Memory
        # ---------------------------------------------------------
        for _ in range(WARMUP):
            bitonic_sort[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()

        print("shared-memory bitonic sort...")
        t0 = perf_counter_ns()
        for _ in range(ITERS):
            bitonic_sort[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()
        t1 = perf_counter_ns()
        delta_t = t1 - t0
        opt_ms = round(Float32(delta_t / ITERS) / 1_000_000.0, 2)

        var opt_valid = True
        with keys.map_to_host() as host_keys:
            for i in range(1, SIZE):
                if host_keys[i - 1] > host_keys[i]:
                    opt_valid = False
                    break

        # ---------------------------------------------------------
        #  Basic Global Memory
        # ---------------------------------------------------------

        for _ in range(WARMUP):
            bitonic_sort_basic[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()

        print("global-memory bitonic sort...")
        t0 = perf_counter_ns()
        for _ in range(ITERS):
            bitonic_sort_basic[THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()
        t1 = perf_counter_ns()
        delta_t = t1 - t0
        basic_ms = round(Float32(delta_t / ITERS) / 1_000_000.0, 2)

        var basic_valid = True
        with keys.map_to_host() as host_keys:
            for i in range(1, SIZE):
                if host_keys[i - 1] > host_keys[i]:
                    basic_valid = False
                    break

        # ---------------------------------------------------------
        #  Radix Basic Global Memory
        # ---------------------------------------------------------
        comptime N_BITS = 2
        for _ in range(WARMUP):
            radix_sort[N_BITS, THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()

        print("global-memory radix sort...")
        t0 = perf_counter_ns()
        for _ in range(ITERS):
            radix_sort[N_BITS, THREADS_PER_BLOCK](ctx, keys, values, SIZE)
            ctx.synchronize()
        t1 = perf_counter_ns()
        delta_t = t1 - t0
        radix_ms = round(Float32(delta_t / ITERS) / 1_000_000.0, 2)

        var radix_valid = True
        with keys.map_to_host() as host_keys:
            for i in range(1, SIZE):
                if host_keys[i - 1] > host_keys[i]:
                    radix_valid = False
                    break

        print("\n================ RESULTS ================")
        print(t"Data Size: {SIZE} elements")
        print(t"Basic Sort:     {basic_ms} ms (Valid: {basic_valid})")
        print(t"Optimized Sort: {opt_ms} ms (Valid: {opt_valid})")
        print(t"Radix Sort: {radix_ms} ms (Valid: {radix_valid})")
        print("-----------------------------------------")
        print(t"SPEEDUP:       {basic_ms / opt_ms} x")
        print("=========================================")


# current results
# Setting up benchmarks for 1048576 elements...
# shared-memory bitonic sort...
# global-memory bitonic sort...
# global-memory radix sort...

# ================ RESULTS ================
# Data Size: 1048576 elements
# Basic Sort:     3.78 ms (Valid: True)
# Optimized Sort: 1.78 ms (Valid: True)
# Radix Sort: 21.24 ms (Valid: True)
# -----------------------------------------
# SPEEDUP:       2.1235955 x
# =========================================


def main() raises:
    benchmark_bitonic_sorts()
