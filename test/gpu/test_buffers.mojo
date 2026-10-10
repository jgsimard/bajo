"""Retained buffer copies preserve bits and reject size changes atomically."""

from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_raises
from bajo.gpu.buffers import copy_to_device, upload_list


def test_copy_preserves_bits_and_storage() raises:
    with DeviceContext() as ctx:
        comptime for dtype in [DType.uint32, DType.float32]:
            var values = List[Scalar[dtype]]()
            for word in [
                UInt32(0),
                UInt32(0x80000000),
                UInt32(0x7FC01234),
                UInt32(0xFFFFFFFF),
            ]:
                values.append(bitcast[dtype](word))
            var buffer = upload_list(ctx, List[Scalar[dtype]](length=4, fill=0))
            var address = Int(buffer.unsafe_ptr())
            ctx.synchronize()
            copy_to_device(buffer, values)
            ctx.synchronize()
            assert_equal(Int(buffer.unsafe_ptr()), address)
            with assert_raises():
                copy_to_device(buffer, values[:3])
            with buffer.map_to_host() as data:
                for i in range(len(values)):
                    assert_equal(
                        bitcast[.uint32](data[i]), bitcast[.uint32](values[i])
                    )


def test_empty_buffer_copy() raises:
    with DeviceContext() as ctx:
        var values = List[UInt32]()
        var buffer = upload_list(ctx, values)
        ctx.synchronize()
        copy_to_device(buffer, values)
        assert_equal(len(buffer), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
