from std.math import ceildiv
from std.testing import (
    TestSuite,
    assert_true,
    assert_equal,
)
from std.gpu.host import DeviceContext

from bajo.sort.gpu.radix_sort import (
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
            upsweep[dtype, 128, RADIX, VEC_WIDTH, KEYS_PER_THREAD],
        ](
            keys,
            global_hist,
            pass_hist,
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
        ctx.enqueue_function[scan[128], scan[128]](
            pass_hist,
            thread_blocks,
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

        var _dummy_ptr = Optional[UnsafePointer[UInt32, MutAnyOrigin]]()

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
            upsweep[dtype, 128, RADIX, VEC_WIDTH, KEYS_PER_THREAD],
        ](
            keys,
            global_hist,
            pass_hist,
            size,
            UInt32(0),
            grid_dim=gdim,
            block_dim=128,
        )
        ctx.synchronize()

        # 2. SCAN
        ctx.enqueue_function[scan[128], scan[128]](
            pass_hist, gdim, grid_dim=RADIX, block_dim=128
        )
        ctx.synchronize()

        # 3. DOWNSWEEP
        comptime _downsweep = downsweep[
            dtype, dtype, BITS_PER_PASS, 512, KEYS_PER_THREAD, False
        ]
        ctx.enqueue_function[_downsweep, _downsweep](
            keys,
            keys_alternate,
            _dummy_ptr,
            _dummy_ptr,
            global_hist,
            pass_hist,
            size,
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
            upsweep[dtype, 128, RADIX, VEC_WIDTH, KEYS_PER_THREAD],
        ](
            keys,
            global_hist,
            pass_hist,
            size,
            UInt32(0),
            grid_dim=gdim,
            block_dim=128,
        )
        ctx.synchronize()

        # 2. SCAN
        ctx.enqueue_function[scan[128], scan[128]](
            pass_hist, gdim, grid_dim=RADIX, block_dim=128
        )
        ctx.synchronize()

        # 3. DOWNSWEEP PAIRS
        ctx.enqueue_function[
            downsweep[dtype, dtype, BITS_PER_PASS, 512, KEYS_PER_THREAD, True],
            downsweep[dtype, dtype, BITS_PER_PASS, 512, KEYS_PER_THREAD, True],
        ](
            keys,
            keys_alternate_keys,
            Optional(vals.unsafe_ptr()),
            Optional(keys_alternate_vals.unsafe_ptr()),
            global_hist,
            pass_hist,
            size,
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
