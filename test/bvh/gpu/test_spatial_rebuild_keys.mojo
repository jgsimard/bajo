"""Cached UInt64 sort keys must exactly match Float64 centroid comparison."""
from std.testing import TestSuite, assert_equal
from std.memory import bitcast
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.gpu.utils import upload_list, _device_span
from bajo.bvh.spatial.parallel_refit import validate_mesh_refit_kernel
from bajo.bvh.spatial.device_update import _item_less


def _compare_kernel[
    count: Int
](
    items: ImmSpan[Float32, ImmutAnyOrigin],
    output: MutSpan[UInt32, MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i >= (count + 1) * (count + 1):
        return
    var ai = i // (count + 1)
    var bi = i % (count + 1)
    var a = UInt32(ai) if ai < count else EMPTY_LANE
    var b = UInt32(bi) if bi < count else EMPTY_LANE
    output[i] = UInt32(_item_less[True](items, a, b) != _item_less(items, a, b))


def test_cached_keys_exact_pairwise_order_and_signed_zero_ties() raises:
    var values: List[Float32] = [
        -3e38,
        -1e35,
        -1e8,
        -1.0,
        -1e-8,
        -1e-35,
        -1e-38,
        bitcast[.float32](UInt32(0x80000001)),
        bitcast[.float32](UInt32(0x80000000)),
        0.0,
        bitcast[.float32](UInt32(1)),
        1e-38,
        1e-35,
        1e-8,
        1.0,
        1e8,
        1e35,
        3e38,
    ]
    var points = List[Float32]()
    var indices = List[UInt32]()
    # Point faces, cancellation faces and duplicates test ordering and ID ties.
    for shape in range(3):
        for value in values:
            var face = len(indices) // 3
            for corner in range(3):
                var x = value
                if shape == 1 and corner == 1:
                    x = -value
                points.extend([x, Float32(corner), 0.0])
                indices.append(UInt32(face * 3 + corner))
    var count = len(indices) // 3
    assert_equal(count, 54)
    with DeviceContext() as ctx:
        var input = upload_list(ctx, points)
        var ids = upload_list(ctx, indices)
        var items = ctx.enqueue_create_buffer[.float32](count * 8)
        var status = upload_list(ctx, List[UInt32](length=1, fill=0))
        var output = ctx.enqueue_create_buffer[.uint32](
            (count + 1) * (count + 1)
        )
        ctx.enqueue_function[validate_mesh_refit_kernel[True]](
            _device_span[mut=False](input),
            _device_span[mut=False](ids),
            _device_span[mut=True](items),
            _device_span[mut=True](status),
            grid_dim=(len(points) + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_function[_compare_kernel[54]](
            _device_span[mut=False](items),
            _device_span[mut=True](output),
            grid_dim=((count + 1) * (count + 1) + 255) // 256,
            block_dim=256,
        )
        ctx.synchronize()
        with status.map_to_host() as data:
            assert_equal(data[0], UInt32(0))
        with output.map_to_host() as data:
            for i in range(len(data)):
                assert_equal(data[i], UInt32(0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
