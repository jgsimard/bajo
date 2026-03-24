from std.math import ceildiv
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_true,
    assert_false,
    assert_equal,
)
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext

from bajo.sort.gpu.radix_sort import (
    upsweep,
    RADIX_MASK,
    PART_SIZE,
    RADIX,
    scan,
    downsweep,
    KEYS_PER_THREAD,
)


def test_upsweep() raises:
    """Tests the Upsweep pass of Radix Sorting."""
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        var size = 10_000

        var d_keys = ctx.enqueue_create_buffer[dtype](size)
        var d_globalHist = ctx.enqueue_create_buffer[dtype](1024)

        var gdim = ceildiv(size, PART_SIZE)
        var bdim = 128
        var d_passHist = ctx.enqueue_create_buffer[dtype](gdim * RADIX)

        d_globalHist.enqueue_fill(0)
        d_passHist.enqueue_fill(0)

        # CPU ground-truth histogram
        var expected_counts = List[UInt32](capacity=RADIX)
        for _ in range(RADIX):
            expected_counts.append(0)

        with d_keys.map_to_host() as host_keys:
            for i in range(size):
                var val = UInt32((i * 13) ^ (i << 16))
                host_keys[i] = val
                expected_counts[Int(val & RADIX_MASK)] += 1

        # Execute
        ctx.enqueue_function[upsweep[128], upsweep[128]](
            d_keys.unsafe_ptr(),
            d_globalHist.unsafe_ptr(),
            d_passHist.unsafe_ptr(),
            size,
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
        with d_globalHist.map_to_host() as host_hist:
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

        var d_passHist = ctx.enqueue_create_buffer[dtype](total_size)
        var expected = List[UInt32](capacity=total_size)

        # We fill passHist with 1s. The exclusive prefix sum of [1, 1, 1...]
        # should be[0, 1, 2, 3...] independent for each radix block.
        with d_passHist.map_to_host() as host_hist:
            for bid in range(RADIX):
                var current_sum: UInt32 = 0
                var offset = bid * thread_blocks
                for i in range(thread_blocks):
                    host_hist[offset + i] = 1
                    expected.append(current_sum)
                    current_sum += 1

        # Execute
        ctx.enqueue_function[scan[128], scan[128]](
            d_passHist.unsafe_ptr(),
            thread_blocks,
            grid_dim=RADIX,  # One block per digit
            block_dim=128,  # Must be 128 as designed
        )

        ctx.synchronize()

        # Verify
        with d_passHist.map_to_host() as host_hist:
            for bid in range(RADIX):
                var offset = bid * thread_blocks
                for i in range(thread_blocks):
                    assert_equal(
                        host_hist[offset + i],
                        expected[offset + i],
                        msg=String("Mismatch at Radix bin ")
                        + String(bid)
                        + String(" thread_block offset ")
                        + String(i),
                    )


def test_downsweep_end_to_end() raises:
    """Tests Downsweep by executing a full single-byte radix pass and comparing to a CPU stable sort.
    """
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime BLOCK_SIZE = 512
        comptime PART_SIZE = BLOCK_SIZE * KEYS_PER_THREAD

        var _dummy_ptr = UnsafePointer[UInt32, MutAnyOrigin]()

        var size = 20_000
        var gdim = ceildiv(Int(size), PART_SIZE)

        var d_keys = ctx.enqueue_create_buffer[dtype](Int(size))
        var d_alt = ctx.enqueue_create_buffer[dtype](Int(size))
        var d_globalHist = ctx.enqueue_create_buffer[dtype](1024)
        var d_passHist = ctx.enqueue_create_buffer[dtype](gdim * RADIX)

        d_globalHist.enqueue_fill(0)
        d_passHist.enqueue_fill(0)

        # Generate random keys
        with d_keys.map_to_host() as host_keys:
            for i in range(Int(size)):
                host_keys[i] = UInt32((i * 17) ^ (i << 13) ^ (i >> 5))

        # 1. UPSWEEP
        ctx.enqueue_function[upsweep[128], upsweep[128]](
            d_keys.unsafe_ptr(),
            d_globalHist.unsafe_ptr(),
            d_passHist.unsafe_ptr(),
            size,
            UInt32(0),
            grid_dim=gdim,
            block_dim=128,
        )
        ctx.synchronize()

        # 2. SCAN
        ctx.enqueue_function[scan[128], scan[128]](
            d_passHist.unsafe_ptr(), gdim, grid_dim=RADIX, block_dim=128
        )
        ctx.synchronize()

        # 3. DOWNSWEEP
        ctx.enqueue_function[downsweep[512, False], downsweep[512, False]](
            d_keys.unsafe_ptr(),
            _dummy_ptr,
            d_alt.unsafe_ptr(),
            _dummy_ptr,
            d_globalHist.unsafe_ptr(),
            d_passHist.unsafe_ptr(),
            size,
            UInt32(0),
            grid_dim=gdim,
            block_dim=BLOCK_SIZE,
        )
        ctx.synchronize()

        # Build CPU 1-pass Stable Sort ground truth
        var cpu_sorted = List[UInt32](capacity=Int(size))
        var counts = List[UInt32](capacity=RADIX)
        for _ in range(RADIX):
            counts.append(0)
        for _ in range(Int(size)):
            cpu_sorted.append(0)

        with d_keys.map_to_host() as host_keys:
            for i in range(Int(size)):
                counts[Int(host_keys[i] & RADIX_MASK)] += 1

            var current_sum: UInt32 = 0
            for i in range(RADIX):
                var cnt = counts[i]
                counts[i] = current_sum
                current_sum += cnt

            for i in range(Int(size)):
                var val = host_keys[i]
                var digit = Int(val & RADIX_MASK)
                var dst = counts[digit]
                cpu_sorted[Int(dst)] = val
                counts[digit] += 1

        # Verify
        with d_alt.map_to_host() as host_alt:
            for i in range(Int(size)):
                assert_equal(
                    host_alt[i],
                    cpu_sorted[i],
                    msg=String("Mismatch after downsweep at sorted index ")
                    + String(i),
                )


def test_downsweep_pairs_end_to_end() raises:
    """Tests sorting key-value pairs."""
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime BLOCK_SIZE = 512
        comptime PART_SIZE = BLOCK_SIZE * KEYS_PER_THREAD

        var size = 20_000
        var gdim = ceildiv(size, PART_SIZE)
        var bdim = 512

        var d_keys = ctx.enqueue_create_buffer[dtype](Int(size))
        var d_vals = ctx.enqueue_create_buffer[dtype](Int(size))
        var d_alt_keys = ctx.enqueue_create_buffer[dtype](Int(size))
        var d_alt_vals = ctx.enqueue_create_buffer[dtype](Int(size))
        var d_globalHist = ctx.enqueue_create_buffer[dtype](1024)
        var d_passHist = ctx.enqueue_create_buffer[dtype](gdim * RADIX)

        d_globalHist.enqueue_fill(0)
        d_passHist.enqueue_fill(0)

        # Generate keys and payloads
        with d_keys.map_to_host() as host_keys, d_vals.map_to_host() as host_vals:
            for i in range(Int(size)):
                host_keys[i] = UInt32((i * 17) ^ (i << 13))
                host_vals[i] = UInt32(i)  # Payload is original index

        # 1. UPSWEEP
        ctx.enqueue_function[upsweep[128], upsweep[128]](
            d_keys.unsafe_ptr(),
            d_globalHist.unsafe_ptr(),
            d_passHist.unsafe_ptr(),
            size,
            UInt32(0),
            grid_dim=gdim,
            block_dim=128,
        )
        ctx.synchronize()

        # 2. SCAN
        ctx.enqueue_function[scan[128], scan[128]](
            d_passHist.unsafe_ptr(), gdim, grid_dim=RADIX, block_dim=128
        )
        ctx.synchronize()

        # 3. DOWNSWEEP PAIRS
        ctx.enqueue_function[downsweep[512, True], downsweep[512, True]](
            d_keys.unsafe_ptr(),
            d_vals.unsafe_ptr(),
            d_alt_keys.unsafe_ptr(),
            d_alt_vals.unsafe_ptr(),
            d_globalHist.unsafe_ptr(),
            d_passHist.unsafe_ptr(),
            size,
            UInt32(0),
            grid_dim=gdim,
            block_dim=bdim,
        )
        ctx.synchronize()

        # Verify against CPU stable sort
        with d_alt_keys.map_to_host() as host_alt_keys, d_alt_vals.map_to_host() as host_alt_vals:
            # Simple check: Keys should be non-decreasing for the first byte
            # and payload should match the original position's index to ensure stability.
            for i in range(Int(size) - 1):
                var k1 = host_alt_keys[i] & RADIX_MASK
                var k2 = host_alt_keys[i + 1] & RADIX_MASK
                assert_true(k1 <= k2, msg="Keys not sorted")
                if k1 == k2:
                    # If keys are same, original index (payload) must be increasing (stability)
                    assert_true(
                        host_alt_vals[i] < host_alt_vals[i + 1],
                        msg="Sort not stable",
                    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
