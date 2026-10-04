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


def test_downsweep_end_to_end() raises:
    """Tests Downsweep by executing a full single-byte radix pass and comparing to a CPU stable sort.
    """
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime BITS_PER_PASS = 8
        comptime RADIX = 256
        comptime RADIX_MASK = RADIX - 1
        comptime VEC_WIDTH = 4
        comptime BLOCK_SIZE = 512
        comptime KEYS_PER_THREAD = 8
        comptime PART_SIZE = BLOCK_SIZE * KEYS_PER_THREAD

        var _dummy_ptr = Optional[Pointer[UInt32, MutAnyOrigin]]()

        var size = 20_000
        var gdim = ceildiv(size, PART_SIZE)

        var keys = ctx.enqueue_create_buffer[dtype](size)
        var keys_alternate = ctx.enqueue_create_buffer[dtype](size)
        var global_hist = ctx.enqueue_create_buffer[dtype](1024)
        var pass_hist = ctx.enqueue_create_buffer[dtype](gdim * RADIX)

        global_hist.enqueue_fill(0)
        pass_hist.enqueue_fill(0)

        # Generate random keys
        with keys.map_to_host() as host_keys:
            for i in range(size):
                host_keys[i] = UInt32((i * 17) ^ (i << 13) ^ (i >> 5))

        # 1. UPSWEEP
        ctx.enqueue_function[
            upsweep[dtype, 128, RADIX, VEC_WIDTH, KEYS_PER_THREAD],
        ](
            keys,
            global_hist,
            pass_hist,
            Int32(size),
            UInt32(0),
            grid_dim=gdim,
            block_dim=128,
        )
        ctx.synchronize()

        # 2. SCAN
        ctx.enqueue_function[scan[128]](
            pass_hist, Int32(gdim), grid_dim=RADIX, block_dim=128
        )
        ctx.synchronize()

        # 3. DOWNSWEEP
        comptime _downsweep = downsweep[
            dtype, dtype, BITS_PER_PASS, 512, KEYS_PER_THREAD, False
        ]
        ctx.enqueue_function[_downsweep](
            keys,
            keys_alternate,
            _dummy_ptr,
            _dummy_ptr,
            global_hist,
            pass_hist,
            Int32(size),
            UInt32(0),
            grid_dim=gdim,
            block_dim=BLOCK_SIZE,
        )
        ctx.synchronize()

        # Build CPU 1-pass Stable Sort ground truth
        var cpu_sorted = List[UInt32](length=size, fill=0)
        var counts = List[UInt32](length=RADIX, fill=0)

        with keys.map_to_host() as host_keys:
            for i in range(size):
                counts[Int(host_keys[i] & RADIX_MASK)] += 1

            var current_sum: UInt32 = 0
            for i in range(RADIX):
                var cnt = counts[i]
                counts[i] = current_sum
                current_sum += cnt

            for i in range(size):
                var val = host_keys[i]
                var digit = Int(val & RADIX_MASK)
                var dst = counts[digit]
                cpu_sorted[Int(dst)] = val
                counts[digit] += 1

        # Verify
        with keys_alternate.map_to_host() as host_alt:
            for i in range(size):
                assert_equal(
                    host_alt[i],
                    cpu_sorted[i],
                    msg=String(t"Mismatch after downsweep at sorted index {i}"),
                )


def test_downsweep_pairs_end_to_end() raises:
    """Tests sorting key-value pairs."""
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime BITS_PER_PASS = 8
        comptime RADIX = 256
        comptime RADIX_MASK = RADIX - 1
        comptime VEC_WIDTH = 4
        comptime BLOCK_SIZE = 512
        comptime KEYS_PER_THREAD = 8
        comptime PART_SIZE = BLOCK_SIZE * KEYS_PER_THREAD

        var size = 20_000
        var gdim = ceildiv(size, PART_SIZE)
        var bdim = 512

        var keys = ctx.enqueue_create_buffer[dtype](size)
        var vals = ctx.enqueue_create_buffer[dtype](size)
        var keys_alternate_keys = ctx.enqueue_create_buffer[dtype](size)
        var keys_alternate_vals = ctx.enqueue_create_buffer[dtype](size)
        var global_hist = ctx.enqueue_create_buffer[dtype](1024)
        var pass_hist = ctx.enqueue_create_buffer[dtype](gdim * RADIX)

        global_hist.enqueue_fill(0)
        pass_hist.enqueue_fill(0)

        # Generate keys and payloads
        with keys.map_to_host() as host_keys, vals.map_to_host() as host_vals:
            for i in range(size):
                host_keys[i] = UInt32((i * 17) ^ (i << 13))
                host_vals[i] = UInt32(i)  # Payload is original index

        # 1. UPSWEEP
        ctx.enqueue_function[
            upsweep[dtype, 128, RADIX, VEC_WIDTH, KEYS_PER_THREAD],
        ](
            keys,
            global_hist,
            pass_hist,
            Int32(size),
            UInt32(0),
            grid_dim=gdim,
            block_dim=128,
        )
        ctx.synchronize()

        # 2. SCAN
        ctx.enqueue_function[scan[128]](
            pass_hist, Int32(gdim), grid_dim=RADIX, block_dim=128
        )
        ctx.synchronize()

        # 3. DOWNSWEEP PAIRS
        ctx.enqueue_function[
            downsweep[dtype, dtype, BITS_PER_PASS, 512, KEYS_PER_THREAD, True],
        ](
            keys,
            keys_alternate_keys,
            Optional(vals.unsafe_ptr()),
            Optional(keys_alternate_vals.unsafe_ptr()),
            global_hist,
            pass_hist,
            Int32(size),
            UInt32(0),
            grid_dim=gdim,
            block_dim=bdim,
        )
        ctx.synchronize()

        # Verify against CPU stable sort
        with keys_alternate_keys.map_to_host() as host_alt_keys, keys_alternate_vals.map_to_host() as host_alt_vals:
            # Simple check: Keys should be non-decreasing for the first byte
            # and payload should match the original position's index to ensure stability.
            for i in range(size - 1):
                var k1 = host_alt_keys[i] & RADIX_MASK
                var k2 = host_alt_keys[i + 1] & RADIX_MASK
                assert_true(k1 <= k2, msg="Keys not sorted")
                if k1 == k2:
                    # If keys are same, original index (payload) must be increasing (stability)
                    assert_true(
                        host_alt_vals[i] < host_alt_vals[i + 1],
                        msg="Sort not stable",
                    )


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


def _assert_radix_keys_workspace[dtype: DType](size: Int) raises:
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[dtype](size)
        var reference = ctx.enqueue_create_buffer[dtype](size)
        var values = ctx.enqueue_create_buffer[.uint32](size)
        with reference.map_to_host() as host, values.map_to_host() as ids:
            for i in range(size):
                var value = (UInt64(i % 11) << UInt64(32)) | UInt64(
                    i * 37 % 257
                )
                if i % 257 == 0:
                    value = UInt64.MAX
                host[i] = Scalar[dtype](value)
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
                        var value = (UInt64(i % 11) << UInt64(32)) | UInt64(
                            i * 37 % 257
                        )
                        if i % 257 == 0:
                            value = UInt64.MAX
                        host[i] = Scalar[dtype](value)
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
                    var value = (UInt64(original % 11) << UInt64(32)) | UInt64(
                        original * 37 % 257
                    )
                    if original % 257 == 0:
                        value = UInt64.MAX
                    assert_equal(host[i], Scalar[dtype](value))
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
