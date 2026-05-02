from std.math import ceildiv
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_true,
    assert_false,
    assert_equal,
)
from std.gpu import DeviceContext

from bajo.sort.gpu.onesweep import (
    global_histogram,
    scan_global,
    digit_binning,
)


def test_global_histogram() raises:
    """Tests the fused 4-pass Global Histogram of OneSweep."""
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime RADIX = 256
        comptime TPB = 128
        comptime ITEMS_PER_THREAD = 128
        comptime PART_SIZE = TPB * ITEMS_PER_THREAD
        comptime VEC_WIDTH = 4

        var size = 100_000
        var gdim = ceildiv(size, PART_SIZE)

        var keys = ctx.enqueue_create_buffer[dtype](size)
        var global_hist = ctx.enqueue_create_buffer[dtype](RADIX * 4)
        global_hist.enqueue_fill(0)

        var exp_h0 = List[UInt32](length=RADIX, fill=0)
        var exp_h1 = List[UInt32](length=RADIX, fill=0)
        var exp_h2 = List[UInt32](length=RADIX, fill=0)
        var exp_h3 = List[UInt32](length=RADIX, fill=0)

        with keys.map_to_host() as host_keys:
            for i in range(size):
                var val = (i * 13) ^ (i << 16) ^ (i >> 3)
                host_keys[i] = UInt32(val)
                exp_h0[val & 255] += 1
                exp_h1[(val >> 8) & 255] += 1
                exp_h2[(val >> 16) & 255] += 1
                exp_h3[(val >> 24) & 255] += 1

        comptime _hist = global_histogram[
            dtype,
            BLOCK_SIZE=TPB,
            RADIX=RADIX,
            VEC_WIDTH=VEC_WIDTH,
            ITEMS_PER_THREAD=ITEMS_PER_THREAD,
        ]
        ctx.enqueue_function[_hist, _hist](
            keys,
            global_hist,
            size,
            grid_dim=gdim,
            block_dim=TPB,
        )

        ctx.synchronize()

        with global_hist.map_to_host() as host_hist:
            for i in range(RADIX):
                assert_equal(host_hist[i], exp_h0[i])
                assert_equal(host_hist[i + RADIX], exp_h1[i])
                assert_equal(host_hist[i + RADIX * 2], exp_h2[i])
                assert_equal(host_hist[i + RADIX * 3], exp_h3[i])


def test_scan_global() raises:
    """Tests the block scan that seeds the Decoupled Look-back flags."""
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime RADIX = 256
        comptime FLAG_INCLUSIVE = 2

        var global_hist = ctx.enqueue_create_buffer[dtype](RADIX * 4)
        var pass_hist = ctx.enqueue_create_buffer[dtype](RADIX * 4)

        with global_hist.map_to_host() as host_hist:
            for i in range(RADIX * 4):
                host_hist[i] = 1

        comptime _scan = scan_global
        ctx.enqueue_function[_scan, _scan](
            global_hist,
            pass_hist,
            1,
            grid_dim=4,
            block_dim=RADIX,
        )

        ctx.synchronize()

        with pass_hist.map_to_host() as h0:
            for i in range(RADIX * 4):
                var expected = (UInt32(i % RADIX) << 2) | FLAG_INCLUSIVE
                assert_equal(h0[i], expected)


def test_digit_binning_end_to_end() raises:
    """Tests Decoupled Look-back with correct buffer padding."""
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime RADIX = 256
        comptime RADIX_MASK = RADIX - 1
        comptime BIN_PART_SIZE = 7680
        comptime BLOCK_SIZE = 512
        comptime KEYS_PER_THREAD = 15
        comptime TPB = 128
        comptime ITEMS_PER_THREAD = 128
        comptime PART_SIZE = TPB * ITEMS_PER_THREAD
        comptime VEC_WIDTH = 4

        var _dummy_ptr = Optional[UnsafePointer[UInt32, MutAnyOrigin]]()
        var size = 20_000
        var binning_blocks = ceildiv(Int(size), BIN_PART_SIZE)

        var keys = ctx.enqueue_create_buffer[dtype](Int(size))
        var keys_alternate = ctx.enqueue_create_buffer[dtype](Int(size))
        var global_hist = ctx.enqueue_create_buffer[dtype](RADIX * 4)

        var pass_hist = ctx.enqueue_create_buffer[dtype](
            (binning_blocks + 1) * RADIX * 4
        )
        var d_index = ctx.enqueue_create_buffer[dtype](4)

        global_hist.enqueue_fill(0)
        pass_hist.enqueue_fill(0)
        d_index.enqueue_fill(0)

        with keys.map_to_host() as host_keys:
            for i in range(Int(size)):
                host_keys[i] = UInt32((i * 17) ^ (i << 13) ^ (i >> 5))

        var g_hist_blocks = ceildiv(size, PART_SIZE)
        comptime _hist = global_histogram[
            dtype,
            BLOCK_SIZE=TPB,
            RADIX=RADIX,
            VEC_WIDTH=VEC_WIDTH,
            ITEMS_PER_THREAD=ITEMS_PER_THREAD,
        ]
        ctx.enqueue_function[_hist, _hist](
            keys,
            global_hist,
            size,
            grid_dim=g_hist_blocks,
            block_dim=TPB,
        )

        comptime _scan = scan_global
        ctx.enqueue_function[_scan, _scan](
            global_hist,
            pass_hist,
            binning_blocks + 1,
            grid_dim=1,
            block_dim=RADIX,
        )

        comptime _bin = digit_binning[
            dtype,
            dtype,
            BITS_PER_PASS=8,
            BLOCK_SIZE=BLOCK_SIZE,
            KEYS_PER_THREAD=KEYS_PER_THREAD,
            HAVE_PAYLOAD=False,
        ]
        ctx.enqueue_function[_bin, _bin](
            keys,
            keys_alternate,
            _dummy_ptr,
            _dummy_ptr,
            pass_hist,
            d_index,
            size,
            UInt32(0),
            grid_dim=binning_blocks,
            block_dim=BLOCK_SIZE,
        )
        ctx.synchronize()

        # Stable Sort Verification
        var counts = List[UInt32](length=RADIX, fill=0)
        var cpu_sorted = List[UInt32](length=size, fill=0)

        with keys.map_to_host() as host_keys:
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
                cpu_sorted[Int(counts[digit])] = val
                counts[digit] += 1

        with keys_alternate.map_to_host() as host_alt:
            for i in range(Int(size)):
                assert_equal(host_alt[i], cpu_sorted[i])


def test_digit_binning_pairs_end_to_end() raises:
    """Tests Decoupled Look-back pairs with correct buffer padding."""
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime RADIX = 256
        comptime RADIX_MASK = RADIX - 1
        comptime BIN_PART_SIZE = 7680
        comptime BLOCK_SIZE = 512
        comptime KEYS_PER_THREAD = 15
        comptime TPB = 128
        comptime ITEMS_PER_THREAD = 128
        comptime PART_SIZE = TPB * ITEMS_PER_THREAD
        comptime VEC_WIDTH = 4

        var size = 20_000
        var binning_blocks = ceildiv(size, BIN_PART_SIZE)

        var keys = ctx.enqueue_create_buffer[dtype](Int(size))
        var d_vals = ctx.enqueue_create_buffer[dtype](Int(size))
        var keys_alternate = ctx.enqueue_create_buffer[dtype](Int(size))
        var vals_alternate = ctx.enqueue_create_buffer[dtype](Int(size))
        var global_hist = ctx.enqueue_create_buffer[dtype](RADIX * 4)

        var pass_hist = ctx.enqueue_create_buffer[dtype](
            (binning_blocks + 1) * RADIX
        )
        var d_index = ctx.enqueue_create_buffer[dtype](4)

        global_hist.enqueue_fill(0)
        pass_hist.enqueue_fill(0)
        d_index.enqueue_fill(0)

        with keys.map_to_host() as host_keys, d_vals.map_to_host() as host_vals:
            for i in range(Int(size)):
                host_keys[i] = UInt32((i * 17) ^ (i << 13))
                host_vals[i] = UInt32(i)

        var _dummy_ptr = Optional[UnsafePointer[UInt32, MutAnyOrigin]]()
        var g_hist_blocks = ceildiv(size, PART_SIZE)

        comptime _hist = global_histogram[
            dtype,
            BLOCK_SIZE=TPB,
            RADIX=RADIX,
            VEC_WIDTH=VEC_WIDTH,
            ITEMS_PER_THREAD=ITEMS_PER_THREAD,
        ]
        ctx.enqueue_function[_hist, _hist](
            keys,
            global_hist,
            size,
            grid_dim=g_hist_blocks,
            block_dim=TPB,
        )

        comptime _scan = scan_global
        ctx.enqueue_function[_scan, _scan](
            global_hist,
            pass_hist,
            binning_blocks + 1,
            grid_dim=1,
            block_dim=RADIX,
        )

        comptime _bin = digit_binning[
            dtype,
            dtype,
            BITS_PER_PASS=8,
            BLOCK_SIZE=BLOCK_SIZE,
            KEYS_PER_THREAD=KEYS_PER_THREAD,
            HAVE_PAYLOAD=True,
        ]
        ctx.enqueue_function[_bin, _bin](
            keys,
            keys_alternate,
            Optional(d_vals.unsafe_ptr()),
            Optional(vals_alternate.unsafe_ptr()),
            pass_hist,
            d_index,
            size,
            UInt32(0),
            grid_dim=binning_blocks,
            block_dim=BLOCK_SIZE,
        )
        ctx.synchronize()

        with keys_alternate.map_to_host() as host_alt_keys, vals_alternate.map_to_host() as host_alt_vals:
            for i in range(Int(size) - 1):
                var k1 = host_alt_keys[i] & RADIX_MASK
                var k2 = host_alt_keys[i + 1] & RADIX_MASK
                assert_true(k1 <= k2)
                if k1 == k2:
                    assert_true(host_alt_vals[i] < host_alt_vals[i + 1])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
