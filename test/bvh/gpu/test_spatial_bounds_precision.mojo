"""Filtered GPU slabs/spheres must agree with Float64 at boundaries/scales."""

from std.testing import TestSuite, assert_equal
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import AABB, Point3f32, Vec3f32, Rayf32, Vec3
from bajo.bvh.gpu.utils import upload_list, _device_span
from bajo.bvh.spatial.query import (
    ray_overlaps_bounds,
    ray_overlaps_bounds_cached,
    sphere_overlaps_bounds,
)


@always_inline
def _case(
    data: ImmSpan[Float32, _], lane: Int
) -> Tuple[AABB[.WORLD], Rayf32[.WORLD], Float32, Float32]:
    var i = lane * 16
    return (
        AABB(
            Point3f32[.WORLD](data[i], data[i + 1], data[i + 2]),
            Point3f32[.WORLD](data[i + 3], data[i + 4], data[i + 5]),
        ),
        Rayf32[.WORLD](
            Point3f32[.WORLD](data[i + 6], data[i + 7], data[i + 8]),
            Vec3f32[.WORLD](data[i + 9], data[i + 10], data[i + 11]),
            data[i + 12],
            data[i + 13],
        ),
        data[i + 14],
        data[i + 15],
    )


def _precision_kernel[
    block_size: Int
](
    data: ImmSpan[Float32, ImmutAnyOrigin],
    output: MutSpan[UInt32, MutAnyOrigin],
    count: Int32,
):
    var lane = global_idx.x
    if lane >= Int(count):
        return
    var box, ray, radius, cutoff = _case(data, lane)
    var inverse = Vec3[.float64, .WORLD](0.0)
    comptime for axis in range(3):
        if ray.d[axis] != 0.0:
            inverse[axis] = 1.0 / Float64(ray.d[axis])
    output[lane * 2] = UInt32(
        ray_overlaps_bounds_cached(ray, box, cutoff, radius, inverse)[0]
    )
    output[lane * 2 + 1] = UInt32(sphere_overlaps_bounds(ray.o, radius, box)[0])


def test_gpu_filtered_bounds_match_double_precision() raises:
    var data = List[Float32]()
    var scales: List[Float32] = [1e-35, 1e-20, 1e-5, 1.0, 1e5, 1e20, 1e35]
    var offsets: List[Float32] = [
        -2.0,
        -1.000001,
        -1.0,
        -0.999999,
        0.0,
        0.999999,
        1.0,
        1.000001,
        2.0,
    ]
    for scale in scales:
        for offset in offsets:
            for variant in range(64):
                var radius = scale * Float32(variant % 4) * 0.25
                var direction_scale = scale
                if variant % 8 == 0:
                    direction_scale = 1.0
                var fields: List[Float32] = [
                    -scale,
                    -scale,
                    -scale,
                    scale,
                    scale,
                    scale,
                    offset * scale,
                    Float32(variant % 5 - 2) * scale,
                    Float32(variant % 7 - 3) * scale,
                    direction_scale * Float32(variant % 3 - 1),
                    direction_scale * Float32(variant % 5 - 2),
                    direction_scale * Float32(variant % 7 - 3),
                    Float32(-1.0 if variant % 2 == 0 else 0.0),
                    Float32(1.0),
                    radius,
                    Float32(0.0 if variant % 4 == 0 else 1.0),
                ]
                for value in fields:
                    data.append(value)
    var count = len(data) // 16
    with DeviceContext() as ctx:
        var device_data = upload_list(ctx, data)
        var output = ctx.enqueue_create_buffer[.uint32](count * 2)
        ctx.enqueue_function[_precision_kernel[128]](
            _device_span[mut=False](device_data),
            _device_span[mut=True](output),
            Int32(count),
            grid_dim=(count + 127) // 128,
            block_dim=128,
        )
        ctx.synchronize()
        with output.map_to_host() as result:
            for lane in range(count):
                var box, ray, radius, cutoff = _case(data, lane)
                assert_equal(
                    result[lane * 2],
                    UInt32(ray_overlaps_bounds(ray, box, cutoff, radius)),
                )
                assert_equal(
                    result[lane * 2 + 1],
                    UInt32(
                        box.distance2(ray.o)[0]
                        <= Float64(radius) * Float64(radius)
                    ),
                )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
