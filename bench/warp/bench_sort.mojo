"""Warp-matched stable pair sorting with nonnegative int32-compatible keys.

Run with --exact to exercise completely filled final histogram tiles.
Input copies and warmup are excluded from each 11-sample median.
Patterns: 0 = mixed 31-bit keys, 1 = duplicates, 2 = zero high byte.
Pairs and OneSweep retain scratch; radix keys uses the allocating API.
"""

from std.time import perf_counter_ns
from std.sys import argv
from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceContext
from bajo.sort.gpu.radix_sort import (
    RadixSortWorkspace,
    device_radix_sort_pairs,
    device_radix_sort_keys,
)
from bajo.sort.gpu.onesweep import (
    OneSweepWorkspace,
    onesweep_radix_sort_pairs,
    onesweep_radix_sort_keys,
)


def key_for(i: UInt32, pattern: Int) -> UInt32:
    if pattern == 1:
        return i * UInt32(37) % UInt32(257)
    var key = (i * UInt32(2654435761) ^ (i >> UInt32(3))) & UInt32(0x7FFFFFFF)
    if pattern == 2:
        return key & UInt32(0x00FFFFFF)
    return key


def run[
    one: Bool, keys_only: Bool
](ctx: DeviceContext, size: Int, pattern: Int) raises:
    var keys = ctx.enqueue_create_buffer[.uint32](size)
    var values = ctx.enqueue_create_buffer[.uint32](size)
    var pristine = ctx.enqueue_create_buffer[.uint32](size)
    var indices = ctx.enqueue_create_buffer[.uint32](size)
    with pristine.map_to_host() as host, indices.map_to_host() as ids:
        for i in range(size):
            host[i] = key_for(UInt32(i), pattern)
            ids[i] = UInt32(i)
    var radix = RadixSortWorkspace[.uint32, .uint32](ctx, size)
    var sweep = OneSweepWorkspace[
        .uint32, .uint32, BLOCK_SIZE=512, KEYS_PER_THREAD=15
    ](ctx, size)
    var expected_sum = UInt64(0)
    for i in range(size):
        expected_sum += UInt64(key_for(UInt32(i), pattern))
    var times = List[Int]()
    for repeat in range(14):
        pristine.enqueue_copy_to(keys)
        indices.enqueue_copy_to(values)
        ctx.synchronize()
        var start = perf_counter_ns()
        comptime if keys_only:
            comptime if one:
                onesweep_radix_sort_keys[
                    .uint32, BINNING_TPB=512, KEYS_PER_THREAD=15
                ](ctx, sweep, keys, size)
            else:
                device_radix_sort_keys(ctx, keys, size)
        else:
            comptime if one:
                onesweep_radix_sort_pairs[
                    .uint32, .uint32, BINNING_TPB=512, KEYS_PER_THREAD=15
                ](ctx, sweep, keys, values, size)
            else:
                device_radix_sort_pairs(ctx, radix, keys, values, size)
        ctx.synchronize()
        if repeat >= 3:
            var elapsed = Int(perf_counter_ns() - start)
            times.append(elapsed)
            print(
                "SAMPLE",
                "onesweep" if one else "radix",
                size,
                pattern,
                elapsed,
                sep="\t",
            )
    with keys.map_to_host() as host, values.map_to_host() as ids:
        var actual_sum = UInt64(0)
        for i in range(size):
            actual_sum += UInt64(host[i])
            comptime if not keys_only:
                assert_equal(host[i], key_for(ids[i], pattern))
                assert_true(Int(ids[i]) < size)
            if i > 0:
                assert_true(host[i - 1] <= host[i])
                comptime if not keys_only:
                    if host[i - 1] == host[i]:
                        assert_true(ids[i - 1] < ids[i])
        assert_equal(actual_sum, expected_sum)
    sort(times)
    print(
        "SORTKEYS" if keys_only else "SORT",
        "onesweep" if one else "radix",
        size,
        pattern,
        times[len(times) // 2],
        sep="\t",
    )


def main() raises:
    var args = argv()
    var sizes: List[Int] = [65536, 1048576]
    if len(args) > 1 and args[1] == "--exact":
        sizes = [16384, 131072, 262144, 1048576, 4194304]
    with DeviceContext() as ctx:
        for size in sizes:
            for pattern in range(3):
                run[False, False](ctx, size, pattern)
                run[True, False](ctx, size, pattern)
