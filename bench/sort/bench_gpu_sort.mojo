from std.sys.arg import argv
from std.time import perf_counter_ns
from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceContext

from bajo.benchmark.timing import summarize_timings
from bajo.parser.number import parse_int_f32
from bajo.sort.gpu.radix_sort import (
    RadixSortWorkspace,
    device_radix_sort_keys,
    device_radix_sort_pairs,
)
from bajo.sort.gpu.onesweep import (
    OneSweepWorkspace,
    onesweep_radix_sort_keys,
    onesweep_radix_sort_pairs,
)
from bajo.sort.gpu.decoupled_fallback import (
    DecoupledFallbackWorkspace,
    decoupled_fallback_radix_sort_keys,
    decoupled_fallback_radix_sort_pairs,
)


def benchmark[
    algorithm: Int, pairs: Bool, Workspace: Movable
](ctx: DeviceContext, mut workspace: Workspace, size: Int) raises -> String:
    var keys = ctx.enqueue_create_buffer[.uint32](size)
    var values = ctx.enqueue_create_buffer[.uint32](size if pairs else 0)
    var input = ctx.enqueue_create_buffer[.uint32](size)
    var ids = ctx.enqueue_create_buffer[.uint32](size if pairs else 0)
    var checksum = UInt64(0)
    with input.map_to_host() as host:
        for i in range(size):
            var key = UInt32(i % 17) if i % 4 == 0 else (
                UInt32(i) * UInt32(2654435761) ^ UInt32(i >> 3)
            )
            host[i] = key
            checksum += UInt64(key)
    comptime if pairs:
        with ids.map_to_host() as host:
            for i in range(size):
                host[i] = UInt32(i)

    var times = List[Int]()
    for trial in range(4):
        input.enqueue_copy_to(keys)
        comptime if pairs:
            ids.enqueue_copy_to(values)
        ctx.synchronize()
        var start = perf_counter_ns()
        comptime if algorithm == 0:
            ref sort_workspace = rebind[RadixSortWorkspace[.uint32, .uint32]](
                workspace
            )
            comptime if pairs:
                device_radix_sort_pairs(ctx, sort_workspace, keys, values, size)
            else:
                device_radix_sort_keys(ctx, sort_workspace, keys, size)
        elif algorithm == 1:
            ref sort_workspace = rebind[
                OneSweepWorkspace[
                    .uint32, .uint32, BLOCK_SIZE=512, KEYS_PER_THREAD=9
                ]
            ](workspace)
            comptime if pairs:
                onesweep_radix_sort_pairs[
                    .uint32, .uint32, BINNING_TPB=512, KEYS_PER_THREAD=9
                ](ctx, sort_workspace, keys, values, size)
            else:
                onesweep_radix_sort_keys[
                    .uint32, BINNING_TPB=512, KEYS_PER_THREAD=9
                ](ctx, sort_workspace, keys, size)
        else:
            ref sort_workspace = rebind[
                DecoupledFallbackWorkspace[
                    .uint32, .uint32, BLOCK_SIZE=512, KEYS_PER_THREAD=9
                ]
            ](workspace)
            comptime if pairs:
                decoupled_fallback_radix_sort_pairs[
                    .uint32, .uint32, BINNING_TPB=512, KEYS_PER_THREAD=9
                ](ctx, sort_workspace, keys, values, size)
            else:
                decoupled_fallback_radix_sort_keys[
                    .uint32, BINNING_TPB=512, KEYS_PER_THREAD=9
                ](ctx, sort_workspace, keys, size)

        ctx.synchronize()
        if trial > 0:
            times.append(Int(perf_counter_ns() - start))
    # Keep correctness checks outside the measured region.
    with keys.map_to_host() as host:
        var actual = UInt64(0)
        for i in range(size):
            actual += UInt64(host[i])
            if i > 0:
                assert_true(host[i - 1] <= host[i])
        assert_equal(actual, checksum)
    var timing = summarize_timings(times)
    var label = String(
        "Radix" if algorithm
        == 0 else "Onesweep" if algorithm
        == 1 else "Decoupled_Fallback"
    ) + ("_Pairs" if pairs else "_Keys")
    var row = String(
        t"{label},{size},{Float64(timing.median_ns) * 1.0e-6},"
        t"{Float64(size) / Float64(timing.median_ns)}\n"
    )
    print(row, end="")
    return row


def main() raises:
    var args = argv()
    var sizes: List[Int] = [4097, 1000003]
    if len(args) > 1:
        sizes.clear()
        for i in range(1, len(args)):
            var size = parse_int_f32(StringSpan(args[i]), "size")
            if size < 1 or size > Int(Int32.MAX):
                raise Error("size must be positive and fit Int32")
            sizes.append(size)
    var csv = String("Algorithm,N,Time_ms,Throughput_GKs\n")
    print(csv, end="")
    with DeviceContext() as ctx:
        for size in sizes:
            var radix = RadixSortWorkspace[.uint32, .uint32](ctx, size)
            comptime for pairs in [False, True]:
                csv += benchmark[0, pairs](ctx, radix, size)
            var onesweep = OneSweepWorkspace[
                .uint32, .uint32, BLOCK_SIZE=512, KEYS_PER_THREAD=9
            ](ctx, size)
            comptime for pairs in [False, True]:
                csv += benchmark[1, pairs](ctx, onesweep, size)
            var fallback = DecoupledFallbackWorkspace[
                .uint32, .uint32, BLOCK_SIZE=512, KEYS_PER_THREAD=9
            ](ctx, size)
            comptime for pairs in [False, True]:
                csv += benchmark[2, pairs](ctx, fallback, size)
    with open("gpu_sort_benchmark_results.csv", "w") as file:
        file.write(csv)
