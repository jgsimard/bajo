from test.sort.helpers import _full_width_key

from std.testing import (
    TestSuite,
    assert_true,
    assert_equal,
)
from max.gpu.host import DeviceContext

from bajo.sort.gpu.onesweep import (
    OneSweepWorkspace,
    onesweep_radix_sort_pairs,
    onesweep_radix_sort_keys,
)


def _assert_onesweep_full_width_pairs[
    keys_dtype: DType, vals_dtype: DType
](size: Int) raises:
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[keys_dtype](size)
        var values = ctx.enqueue_create_buffer[vals_dtype](size)
        with keys.map_to_host() as host, values.map_to_host() as ids:
            for i in range(size):
                host[i] = Scalar[keys_dtype](_full_width_key(UInt32(i)))
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
                        host[i] = Scalar[keys_dtype](_full_width_key(UInt32(i)))
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
                        Scalar[keys_dtype](_full_width_key(original)),
                    )
                    if i > 0:
                        assert_true(host[i - 1] <= host[i])
                        if host[i - 1] == host[i]:
                            assert_true(UInt32(ids[i - 1]) < original)


def test_onesweep_full_width_keys_and_payloads() raises:
    comptime for keys_dtype in [DType.uint32, DType.uint64]:
        comptime for vals_dtype in [DType.uint32, DType.uint64]:
            for size in [1, 4608, 4609, 20003]:
                _assert_onesweep_full_width_pairs[keys_dtype, vals_dtype](size)


def test_onesweep_uint64_keys_only_matches_pairs() raises:
    var size = 20003
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[.uint64](size)
        var pair_keys = ctx.enqueue_create_buffer[.uint64](size)
        var values = ctx.enqueue_create_buffer[.uint64](size)
        with keys.map_to_host() as host, pair_keys.map_to_host() as pair, values.map_to_host() as ids:
            for i in range(size):
                host[i] = _full_width_key(UInt32(i))
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
