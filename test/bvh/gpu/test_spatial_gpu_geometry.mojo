"""Filtered GPU geometry agrees with robust geometry across regions and scales."""

from std.testing import TestSuite, assert_equal, assert_true
from std.math import abs, max
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import Point3f32, Vec3f32, Rayf32
from bajo.core.intersect import (
    project_triangle,
    intersect_triangle,
    triangle_normal,
    ray_triangle_interval,
    project_triangle_filtered,
    triangle_normal_f32,
)

from bajo.bvh.gpu.utils import _device_span, upload_list


@always_inline
def _case(
    data: ImmSpan[Float32, _], lane: Int
) -> Tuple[
    Point3f32[.WORLD], Point3f32[.WORLD], Point3f32[.WORLD], Rayf32[.WORLD]
]:
    var base = lane * 17
    return (
        Point3f32[.WORLD](data[base], data[base + 1], data[base + 2]),
        Point3f32[.WORLD](data[base + 3], data[base + 4], data[base + 5]),
        Point3f32[.WORLD](data[base + 6], data[base + 7], data[base + 8]),
        Rayf32[.WORLD](
            Point3f32[.WORLD](data[base + 9], data[base + 10], data[base + 11]),
            Vec3f32[.WORLD](data[base + 12], data[base + 13], data[base + 14]),
            data[base + 15],
            data[base + 16],
        ),
    )


def _geometry_kernel[
    count: Int
](
    data: ImmSpan[Float32, ImmutAnyOrigin],
    output: MutSpan[Float64, MutAnyOrigin],
):
    var lane = Int(global_idx.x)
    if lane >= count:
        return
    var a, b, c, ray = _case(data, lane)
    var projection, error = project_triangle_filtered(ray.o, a, b, c)
    output[lane * 8] = projection.distance2
    output[lane * 8 + 1] = error
    var filtered, ray_t, ray_error = ray_triangle_interval(ray, a, b, c)
    output[lane * 8 + 2] = Float64(filtered)
    output[lane * 8 + 6] = Float64(ray_t)
    output[lane * 8 + 7] = Float64(ray_error)
    var normal = triangle_normal_f32(a, b, c)
    comptime for axis in range(3):
        output[lane * 8 + 3 + axis] = Float64(normal[axis])


def test_gpu_geometry_filters_regions_skinny_faces_and_scales() raises:
    var data = List[Float32]()
    var scales: List[Float32] = [1e-20, 1e-5, 1.0, 1e5, 1e20]
    for scale in scales:
        for i in range(256):
            var origin = Point3f32[.WORLD](Float32(i % 7 - 3) * scale)
            var e1 = Vec3f32[.WORLD](scale, 0.2 * scale, 0.0)
            var e2 = Vec3f32[.WORLD](0.1 * scale, scale, 0.1 * scale)
            if i % 5 == 0:
                e2 = Vec3f32[.WORLD](scale, scale * 1e-6, 0.0)
            if i % 13 == 0:
                e2 = e1
            var sx = Float32(1.0 if i % 2 else -1.0)
            var sy = Float32(1.0 if (i // 2) % 2 else -1.0)
            var sz = Float32(1.0 if (i // 3) % 2 else -1.0)
            if i % 17 == 0:
                origin = Point3f32[.WORLD](0.0)
                e1 = Vec3f32[.WORLD](sx * scale, 0.0, 0.0)
                e2 = Vec3f32[.WORLD](0.0, sy * scale, 0.0)
            if i % 19 == 0:
                origin = Point3f32[.WORLD](0.0)
                e1 = Vec3f32[.WORLD](sx * scale, -sy * scale, 0.0)
                e2 = Vec3f32[.WORLD](sx * scale, 0.0, -sz * scale)
                if (i // 7) % 2:
                    e2 = e2 * -1.0
            if i % 23 == 0:
                origin = Point3f32[.WORLD](0.0)
                e1 = Vec3f32[.WORLD](sx * scale, -sy * scale, 0.0)
                e2 = Vec3f32[.WORLD](0.0, 0.0, sz * scale)
            if i % 29 == 0:
                origin = Point3f32[.WORLD](0.0)
                e1 = Vec3f32[.WORLD](scale, -scale, 0.0)
                e2 = Vec3f32[.WORLD](scale, 0.0, -scale * 1.0000001)
            var p = (
                origin
                + e1 * Float32(i % 4 - 1) * 0.23
                + e2 * Float32(i % 3 - 1) * 0.31
                + Vec3f32[.WORLD](0.0, 0.0, scale)
            )
            var b = origin + e1
            var c = origin + e2
            var direction = Vec3f32[.WORLD](0.0, 0.0, -scale)
            if i % 17 == 0:
                direction = e1
            for vertex in List[Point3f32[.WORLD]]([origin, b, c, p]):
                comptime for axis in range(3):
                    data.append(vertex[(axis + i % 3) % 3])
            comptime for axis in range(3):
                data.append(direction[(axis + i % 3) % 3])
            data.append(0.0)
            data.append(8.0)
    with DeviceContext() as ctx:
        var input = upload_list(ctx, data)
        var output = ctx.enqueue_create_buffer[.float64](1280 * 8)
        ctx.enqueue_function[_geometry_kernel[1280]](
            _device_span[mut=False](input),
            _device_span[mut=True](output),
            grid_dim=40,
            block_dim=32,
        )
        ctx.synchronize()
        var classified = 0
        var certified_hits = 0
        with output.map_to_host() as values:
            for lane in range(1280):
                var a, b, c, ray = _case(data, lane)
                var normal = triangle_normal(a, b, c)
                comptime for axis in range(3):
                    assert_equal(
                        values[lane * 8 + 3 + axis],
                        Float64(Float32(normal[axis])),
                    )
                var exact = project_triangle(ray.o, a, b, c)
                var error = values[lane * 8 + 1]
                assert_true(
                    abs(values[lane * 8] - exact.distance2)
                    <= error
                    + max(abs(exact.distance2), Float64(1e-300)) * 1e-12
                )
                var result = values[lane * 8 + 2]
                if result == 1.0:
                    certified_hits += 1
                    var hit = intersect_triangle(ray, a, b, c)
                    assert_true(hit.result)
                    assert_true(
                        abs(values[lane * 8 + 6] - hit.t)
                        <= values[lane * 8 + 7]
                    )
                if result >= 0.0:
                    classified += 1
                    assert_equal(
                        result == 1.0, intersect_triangle(ray, a, b, c).result
                    )
        assert_true(classified > 100)
        assert_true(certified_hits > 10)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
