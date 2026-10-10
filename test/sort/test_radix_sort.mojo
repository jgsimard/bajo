from test.sort.helpers import _byte_pass_keys, _stable_byte_order

from std.math import ceildiv
from std.testing import (
    TestSuite,
    assert_true,
    assert_equal,
)
from max.gpu.host import DeviceContext

from bajo.sort.gpu.radix_sort import (
    device_radix_sort_pairs,
    device_radix_sort_keys,
    RadixSortWorkspace,
    upsweep,
    scan,
    downsweep,
)


def test_upsweep() raises:
    """Tests the Upsweep pass of Radix Sorting."""
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime RADIX = 256
        comptime RADIX_MASK = RADIX - 1
        comptime VEC_WIDTH = 4
        comptime KEYS_PER_THREAD = 8
        comptime PART_SIZE = 512 * KEYS_PER_THREAD
        var size = 10_000

        var keys = ctx.enqueue_create_buffer[dtype](size)
        var global_hist = ctx.enqueue_create_buffer[dtype](1024)

        var gdim = ceildiv(size, PART_SIZE)
        var bdim = 128
        var pass_hist = ctx.enqueue_create_buffer[dtype](gdim * RADIX)

        global_hist.enqueue_fill(0)
        pass_hist.enqueue_fill(0)

        # CPU ground-truth histogram
        var expected_counts = List[UInt32](capacity=RADIX)
        for _ in range(RADIX):
            expected_counts.append(0)

        with keys.map_to_host() as host_keys:
            for i in range(size):
                var val = UInt32((i * 13) ^ (i << 16))
                host_keys[i] = val
                expected_counts[Int(val & RADIX_MASK)] += 1

        # Execute
        ctx.enqueue_function[
            upsweep[dtype, 128, RADIX, VEC_WIDTH, KEYS_PER_THREAD],
        ](
            keys,
            global_hist,
            pass_hist,
            Int32(size),
            UInt32(0),  # Testing radixShift = 0 first
            grid_dim=gdim,
            block_dim=bdim,
        )

        ctx.synchronize()

        # Build expected exclusive prefix sum array using ground truth
        var expected_prefix = List[UInt32](capacity=RADIX)
        var current_sum: UInt32 = 0
        for i in range(RADIX):
            expected_prefix.append(current_sum)
            current_sum += expected_counts[i]

        # Verify
        with global_hist.map_to_host() as host_hist:
            for i in range(RADIX):
                assert_equal(
                    host_hist[i],
                    expected_prefix[i],
                    msg=String("Mismatch at globalHist bin ") + String(i),
                )


def test_scan() raises:
    """Tests the Scan pass of Radix Sorting."""
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime RADIX = 256

        # 200 blocks used to force testing both the main full-warp loop (0..127)
        # and the partial-warp tail handling (128..199).
        var thread_blocks = 200
        var total_size = RADIX * thread_blocks

        var pass_hist = ctx.enqueue_create_buffer[dtype](total_size)
        var expected = List[UInt32](capacity=total_size)

        # We fill passHist with 1s. The exclusive prefix sum of [1, 1, 1...]
        # should be[0, 1, 2, 3...] independent for each radix block.
        with pass_hist.map_to_host() as host_hist:
            for bid in range(RADIX):
                var current_sum: UInt32 = 0
                var offset = bid * thread_blocks
                for i in range(thread_blocks):
                    host_hist[offset + i] = 1
                    expected.append(current_sum)
                    current_sum += 1

        # Execute
        ctx.enqueue_function[scan[128]](
            pass_hist,
            Int32(thread_blocks),
            grid_dim=RADIX,
            block_dim=128,
        )

        ctx.synchronize()

        # Verify
        with pass_hist.map_to_host() as host_hist:
            for bid in range(RADIX):
                var offset = bid * thread_blocks
                for i in range(thread_blocks):
                    assert_equal(
                        host_hist[offset + i],
                        expected[offset + i],
                        msg=String(
                            t"Mismatch at Radix bin {bid}  thread_block"
                            t" offset {i}"
                        ),
                    )


def _test_downsweep[pairs: Bool]() raises:
    with DeviceContext() as ctx:
        comptime RADIX = 256
        comptime BLOCK_SIZE = 512
        comptime KEYS_PER_THREAD = 8
        var size = 20_000
        var gdim = ceildiv(size, BLOCK_SIZE * KEYS_PER_THREAD)
        var original = _byte_pass_keys[pairs](size)
        var expected = _stable_byte_order(original)

        var keys = ctx.enqueue_create_buffer[.uint32](size)
        var alternate = ctx.enqueue_create_buffer[.uint32](size)
        var values = ctx.enqueue_create_buffer[.uint32](size if pairs else 0)
        var alternate_values = ctx.enqueue_create_buffer[.uint32](
            size if pairs else 0
        )
        var global_hist = ctx.enqueue_create_buffer[.uint32](1024)
        var pass_hist = ctx.enqueue_create_buffer[.uint32](gdim * RADIX)
        global_hist.enqueue_fill(0)
        pass_hist.enqueue_fill(0)
        with keys.map_to_host() as host:
            for i in range(size):
                host[i] = original[i]
        var values_ptr = Optional[Pointer[UInt32, origin_of(values)]]()
        var alternate_values_ptr = Optional[
            Pointer[UInt32, origin_of(alternate_values)]
        ]()
        comptime if pairs:
            with values.map_to_host() as host:
                for i in range(size):
                    host[i] = UInt32(i)
            values_ptr = Optional(values.unsafe_ptr())
            alternate_values_ptr = Optional(alternate_values.unsafe_ptr())

        ctx.enqueue_function[upsweep[.uint32, 128, RADIX, 4, KEYS_PER_THREAD]](
            keys,
            global_hist,
            pass_hist,
            Int32(size),
            UInt32(0),
            grid_dim=gdim,
            block_dim=128,
        )
        ctx.synchronize()
        ctx.enqueue_function[scan[128]](
            pass_hist, Int32(gdim), grid_dim=RADIX, block_dim=128
        )
        ctx.synchronize()
        ctx.enqueue_function[
            downsweep[.uint32, .uint32, 8, BLOCK_SIZE, KEYS_PER_THREAD, pairs]
        ](
            keys,
            alternate,
            values_ptr,
            alternate_values_ptr,
            global_hist,
            pass_hist,
            Int32(size),
            UInt32(0),
            grid_dim=gdim,
            block_dim=BLOCK_SIZE,
        )
        ctx.synchronize()
        with alternate.map_to_host() as host:
            for i in range(size):
                assert_equal(
                    host[i], original[expected[i]], msg=String(t"key at {i}")
                )
        comptime if pairs:
            with alternate_values.map_to_host() as host:
                for i in range(size):
                    assert_equal(
                        host[i],
                        UInt32(expected[i]),
                        msg=String(t"payload at {i}"),
                    )


def test_downsweep_end_to_end() raises:
    _test_downsweep[False]()


def test_downsweep_pairs_end_to_end() raises:
    _test_downsweep[True]()


def _composite_key(original_idx: UInt32) -> UInt64:
    var segment = original_idx * UInt32(7) % UInt32(19)
    var morton = (
        original_idx * UInt32(2_654_435_761)
        ^ (original_idx << UInt32(11))
        ^ (original_idx >> UInt32(3))
    )
    return (UInt64(segment) << 32) | UInt64(morton)


def test_uint64_composite_key_pairs_end_to_end() raises:
    with DeviceContext() as ctx:
        var size = 10_003
        var keys = ctx.enqueue_create_buffer[.uint64](size)
        var values = ctx.enqueue_create_buffer[.uint32](size)
        with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
            for i in range(size):
                host_values[i] = UInt32(i)
                host_keys[i] = _composite_key(UInt32(i))

        device_radix_sort_pairs[.uint64, .uint32](ctx, keys, values, size)
        ctx.synchronize()

        with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
            for i in range(size):
                assert_equal(host_keys[i], _composite_key(host_values[i]))
                if i > 0:
                    assert_true(host_keys[i - 1] <= host_keys[i])


def _workspace_key(index: Int) -> UInt64:
    if index % 257 == 0:
        return UInt64.MAX
    return (UInt64(index % 11) << UInt64(32)) | UInt64(index * 37 % 257)


def _assert_radix_keys_workspace[dtype: DType](size: Int) raises:
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[dtype](size)
        var reference = ctx.enqueue_create_buffer[dtype](size)
        var values = ctx.enqueue_create_buffer[.uint32](size)
        with reference.map_to_host() as host, values.map_to_host() as ids:
            for i in range(size):
                host[i] = Scalar[dtype](_workspace_key(i))
                ids[i] = UInt32(i)
        reference.enqueue_copy_to(keys)
        device_radix_sort_pairs[dtype, .uint32](ctx, reference, values, size)
        ctx.synchronize()
        var workspace = RadixSortWorkspace[dtype, dtype](
            ctx, size, with_values=False
        )
        assert_equal(len(workspace.vals_alternate), 0)
        for repeat in range(3):
            if repeat > 0:
                with keys.map_to_host() as host:
                    for i in range(size):
                        host[i] = Scalar[dtype](_workspace_key(i))
            if repeat == 2:
                device_radix_sort_keys(ctx, keys, size)
            else:
                device_radix_sort_keys(ctx, workspace, keys, size)
            ctx.synchronize()
            with keys.map_to_host() as host, reference.map_to_host() as expected, values.map_to_host() as ids:
                for i in range(size):
                    assert_equal(host[i], expected[i])
                    var original = Int(ids[i])
                    assert_true(original < size)
                    assert_equal(
                        host[i], Scalar[dtype](_workspace_key(original))
                    )
                    if i > 0:
                        assert_true(host[i - 1] <= host[i])
                        if host[i - 1] == host[i]:
                            assert_true(ids[i - 1] < ids[i])


def test_radix_keys_workspace_matches_stable_pairs() raises:
    comptime for dtype in [DType.uint32, DType.uint64]:
        for size in [1, 33, 4608, 4609, 20003, 294912]:
            _assert_radix_keys_workspace[dtype](size)


def test_radix_empty_input_preserves_buffers() raises:
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[.uint32](1)
        var values = ctx.enqueue_create_buffer[.uint32](1)
        keys.enqueue_fill(99)
        values.enqueue_fill(88)
        device_radix_sort_keys(ctx, keys, 0)
        device_radix_sort_pairs(ctx, keys, values, 0)
        ctx.synchronize()
        with keys.map_to_host() as host, values.map_to_host() as ids:
            assert_equal(host[0], UInt32(99))
            assert_equal(ids[0], UInt32(88))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
