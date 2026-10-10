from test.sort.helpers import _byte_pass_keys, _stable_byte_order

from std.math import ceildiv
from std.testing import (
    TestSuite,
    assert_true,
    assert_equal,
)
from max.gpu.host import DeviceContext

from bajo.sort.gpu.onesweep import (
    global_histogram,
    scan_global,
    digit_binning,
    OneSweepWorkspace,
    onesweep_radix_sort_pairs,
    onesweep_radix_sort_keys,
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

        var expected = List[UInt32](length=RADIX * 4, fill=0)

        with keys.map_to_host() as host_keys:
            for i in range(size):
                var val = (i * 13) ^ (i << 16) ^ (i >> 3)
                host_keys[i] = UInt32(val)
                for byte in range(4):
                    expected[byte * RADIX + ((val >> (byte * 8)) & 255)] += 1

        comptime _hist = global_histogram[
            dtype,
            BLOCK_SIZE=TPB,
            RADIX=RADIX,
            VEC_WIDTH=VEC_WIDTH,
            ITEMS_PER_THREAD=ITEMS_PER_THREAD,
        ]
        ctx.enqueue_function[_hist](
            keys,
            global_hist,
            Int32(size),
            grid_dim=gdim,
            block_dim=TPB,
        )

        ctx.synchronize()

        with global_hist.map_to_host() as host_hist:
            for i in range(RADIX * 4):
                assert_equal(host_hist[i], expected[i])


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
        ctx.enqueue_function[_scan](
            global_hist,
            pass_hist,
            Int32(1),
            grid_dim=4,
            block_dim=RADIX,
        )

        ctx.synchronize()

        with pass_hist.map_to_host() as h0:
            for i in range(RADIX * 4):
                var expected = (UInt32(i % RADIX) << 2) | FLAG_INCLUSIVE
                assert_equal(h0[i], expected)


def _test_digit_binning[pairs: Bool]() raises:
    """Exercise look-back across blocks and the padded sentinel block."""
    with DeviceContext() as ctx:
        comptime RADIX = 256
        comptime BLOCK_SIZE = 512
        comptime KEYS_PER_THREAD = 15
        var size = 20_000
        var binning_blocks = ceildiv(size, BLOCK_SIZE * KEYS_PER_THREAD)
        var original = _byte_pass_keys[pairs](size)
        var expected = _stable_byte_order(original)

        var keys = ctx.enqueue_create_buffer[.uint32](size)
        var alternate = ctx.enqueue_create_buffer[.uint32](size)
        var values = ctx.enqueue_create_buffer[.uint32](size if pairs else 0)
        var alternate_values = ctx.enqueue_create_buffer[.uint32](
            size if pairs else 0
        )
        var global_hist = ctx.enqueue_create_buffer[.uint32](RADIX * 4)
        # Key-only coverage retains four passes of padding; pairs need one.
        var pass_hist = ctx.enqueue_create_buffer[.uint32](
            (binning_blocks + 1) * RADIX * (1 if pairs else 4)
        )
        var index = ctx.enqueue_create_buffer[.uint32](4)
        global_hist.enqueue_fill(0)
        pass_hist.enqueue_fill(0)
        index.enqueue_fill(0)
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

        ctx.enqueue_function[
            global_histogram[
                .uint32,
                BLOCK_SIZE=128,
                RADIX=RADIX,
                VEC_WIDTH=4,
                ITEMS_PER_THREAD=128,
            ]
        ](
            keys,
            global_hist,
            Int32(size),
            grid_dim=ceildiv(size, 128 * 128),
            block_dim=128,
        )
        ctx.enqueue_function[scan_global](
            global_hist,
            pass_hist,
            Int32(binning_blocks + 1),
            grid_dim=1,
            block_dim=RADIX,
        )
        ctx.enqueue_function[
            digit_binning[
                .uint32,
                .uint32,
                BITS_PER_PASS=8,
                BLOCK_SIZE=BLOCK_SIZE,
                KEYS_PER_THREAD=KEYS_PER_THREAD,
                HAVE_PAYLOAD=pairs,
            ]
        ](
            keys,
            alternate,
            values_ptr,
            alternate_values_ptr,
            pass_hist,
            index,
            Int32(size),
            UInt32(0),
            grid_dim=binning_blocks,
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


def test_digit_binning_end_to_end() raises:
    _test_digit_binning[False]()


def test_digit_binning_pairs_end_to_end() raises:
    _test_digit_binning[True]()


def _onesweep_test_key(index: UInt32) -> UInt64:
    if index % UInt32(257) == 0:
        return UInt64.MAX
    return (UInt64(index % UInt32(11)) << UInt64(32)) | UInt64(
        index * UInt32(37) % UInt32(257)
    )


def _assert_onesweep_full_width_pairs[
    keys_dtype: DType, vals_dtype: DType
](size: Int) raises:
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[keys_dtype](size)
        var values = ctx.enqueue_create_buffer[vals_dtype](size)
        with keys.map_to_host() as host, values.map_to_host() as ids:
            for i in range(size):
                host[i] = Scalar[keys_dtype](_onesweep_test_key(UInt32(i)))
                ids[i] = Scalar[vals_dtype](
                    (UInt64(0xFEDCBA98) << UInt64(32)) | UInt64(i)
                )
        var workspace = OneSweepWorkspace[
            keys_dtype, vals_dtype, BLOCK_SIZE=512, KEYS_PER_THREAD=9
        ](ctx, size)
        for rebuild in range(3):
            if rebuild > 0:
                with keys.map_to_host() as host, values.map_to_host() as ids:
                    for i in range(size):
                        host[i] = Scalar[keys_dtype](
                            _onesweep_test_key(UInt32(i))
                        )
                        ids[i] = Scalar[vals_dtype](
                            (UInt64(0xFEDCBA98) << UInt64(32)) | UInt64(i)
                        )
            if rebuild == 2:
                onesweep_radix_sort_pairs[keys_dtype, vals_dtype](
                    ctx, keys, values, size
                )
            else:
                onesweep_radix_sort_pairs[
                    keys_dtype, vals_dtype, BINNING_TPB=512, KEYS_PER_THREAD=9
                ](ctx, workspace, keys, values, size)
            ctx.synchronize()
            with keys.map_to_host() as host, values.map_to_host() as ids:
                for i in range(size):
                    var original = UInt32(ids[i])
                    assert_true(Int(original) < size)
                    assert_equal(
                        ids[i],
                        Scalar[vals_dtype](
                            (UInt64(0xFEDCBA98) << UInt64(32))
                            | UInt64(original)
                        ),
                    )
                    assert_equal(
                        host[i],
                        Scalar[keys_dtype](_onesweep_test_key(original)),
                    )
                    if i > 0:
                        assert_true(host[i - 1] <= host[i])
                        if host[i - 1] == host[i]:
                            assert_true(UInt32(ids[i - 1]) < original)


def test_onesweep_full_width_keys_and_payloads() raises:
    comptime for keys_dtype in [DType.uint32, DType.uint64]:
        comptime for vals_dtype in [DType.uint32, DType.uint64]:
            for size in [1, 33, 4608, 4609, 16384, 20003]:
                _assert_onesweep_full_width_pairs[keys_dtype, vals_dtype](size)


def test_onesweep_uint64_keys_only_matches_pairs() raises:
    var size = 20003
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[.uint64](size)
        var pair_keys = ctx.enqueue_create_buffer[.uint64](size)
        var values = ctx.enqueue_create_buffer[.uint64](size)
        with keys.map_to_host() as host, pair_keys.map_to_host() as pair, values.map_to_host() as ids:
            for i in range(size):
                host[i] = _onesweep_test_key(UInt32(i))
                pair[i] = host[i]
                ids[i] = UInt64(i)
        var workspace = OneSweepWorkspace[
            .uint64, .uint64, BLOCK_SIZE=512, KEYS_PER_THREAD=9
        ](ctx, size)
        onesweep_radix_sort_keys[.uint64, BINNING_TPB=512, KEYS_PER_THREAD=9](
            ctx, workspace, keys, size
        )
        onesweep_radix_sort_pairs[
            .uint64, .uint64, BINNING_TPB=512, KEYS_PER_THREAD=9
        ](ctx, workspace, pair_keys, values, size)
        ctx.synchronize()
        with keys.map_to_host() as host, pair_keys.map_to_host() as pair:
            for i in range(size):
                assert_equal(host[i], pair[i])


def test_onesweep_empty_and_single_input_preserves_buffers() raises:
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[.uint64](1)
        var values = ctx.enqueue_create_buffer[.uint64](1)
        var key = UInt64(0xFEDCBA9876543210)
        var value = UInt64(0xABCDEF1234567890)
        keys.enqueue_fill(key)
        values.enqueue_fill(value)
        var workspace = OneSweepWorkspace[
            .uint64, .uint64, BLOCK_SIZE=512, KEYS_PER_THREAD=9
        ](ctx, 1)
        for size in [0, 1]:
            onesweep_radix_sort_keys[
                .uint64, BINNING_TPB=512, KEYS_PER_THREAD=9
            ](ctx, workspace, keys, size)
            onesweep_radix_sort_pairs[.uint64, .uint64](ctx, keys, values, size)
        ctx.synchronize()
        with keys.map_to_host() as host, values.map_to_host() as ids:
            assert_equal(host[0], key)
            assert_equal(ids[0], value)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
