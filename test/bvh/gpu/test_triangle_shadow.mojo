"""Compressed shadow masks agree with determinant-scaled reference tests."""
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.memory import bitcast
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_true
from bajo.bvh.constants import TraceMode
from bajo.bvh.gpu.triangle_trace import (
    _intersect_cwbvh_triangle,
    _intersect_cwbvh_indexed_triangle,
    _intersect_triangle_leaf,
)
from bajo.bvh.gpu.utils import upload_list
from bajo.bvh.types import Hit
from bajo.core import Frame, Point3f32, Rayf32, Vec3f32, cross
from bajo.core.intersect import intersect_ray_tri_edges_scaled
from bajo.core.random import Rng


def _compare_masks[
    triangle_count: Int, frame: Frame
](
    triangles: ImmPointer[Float32, ImmutAnyOrigin],
    wide_triangles: ImmPointer[Float32, ImmutAnyOrigin],
    vertices: ImmPointer[Float32, ImmutAnyOrigin],
    primitive_ids: ImmPointer[UInt32, ImmutAnyOrigin],
    results: MutPointer[UInt32, MutAnyOrigin],
):
    var index = global_idx.x
    if index >= triangle_count * 64:
        return
    var triangle = index // 64
    var ray_case = index % 64
    var base = triangle * 12
    var e1 = Vec3f32[frame](
        triangles[unsafe_offset=base],
        triangles[unsafe_offset=base + 1],
        triangles[unsafe_offset=base + 2],
    )
    var e2 = Vec3f32[frame](
        triangles[unsafe_offset=base + 4],
        triangles[unsafe_offset=base + 5],
        triangles[unsafe_offset=base + 6],
    )
    var v0 = Point3f32[frame](
        triangles[unsafe_offset=base + 8],
        triangles[unsafe_offset=base + 9],
        triangles[unsafe_offset=base + 10],
    )
    var u = Float32(ray_case % 6 - 1) * 0.25
    var v = Float32((ray_case // 6) % 6 - 1) * 0.25
    var normal = cross(e1, e2)
    var direction = normal
    if ray_case % 2:
        direction = -direction
    var origin = v0 + e1 * u + e2 * v - direction * 2
    var t_min = Float32(0)
    var t_max = Float32(10)
    if ray_case >= 36 and ray_case < 40:
        t_min = bitcast[.float32](UInt32(0x40000000 + ray_case - 37))
    elif ray_case >= 40 and ray_case < 44:
        t_max = bitcast[.float32](UInt32(0x40000000 + ray_case - 41))
    elif ray_case == 44:
        t_min = -1
    elif ray_case >= 45 and ray_case < 48:
        direction = direction * Float32(1.0e-10)
    elif ray_case >= 48 and ray_case < 51:
        direction = direction * Float32(1.0e20)
    elif ray_case >= 51 and ray_case < 54:
        direction = e1
    elif ray_case >= 54 and ray_case < 57:
        direction = Vec3f32[frame](0, bitcast[.float32](UInt32(0x80000000)), 0)
    elif ray_case == 57:
        direction.x = bitcast[.float32](UInt32(0x7FC00000))
    elif ray_case == 58:
        direction.x = bitcast[.float32](UInt32(0x7F800000))
    elif ray_case == 59:
        direction.x = bitcast[.float32](UInt32(0xFF800000))
    elif ray_case == 60:
        t_min = 2
        t_max = 2
    elif ray_case == 61:
        t_min = 3
        t_max = 1
    elif ray_case == 62:
        t_max = bitcast[.float32](UInt32(0x7F800000))
    elif ray_case == 63:
        t_max = bitcast[.float32](UInt32(0x7FC00000))
    var ray = Rayf32[frame](origin, direction, t_min, t_max)
    var hit = Hit[frame].miss(t_max)
    var reference = intersect_ray_tri_edges_scaled(
        origin, direction, v0, e1, e2, t_max, t_min
    )
    var candidate = _intersect_cwbvh_triangle[frame, .ANY_HIT](
        triangles, UInt32(triangle), ray, hit
    )
    var wide_candidate = _intersect_triangle_leaf[frame, 1, .ANY_HIT](
        wide_triangles, UInt32(triangle), UInt32(1), ray, hit
    )
    var vertex_base = triangle * 9
    var indexed_v0 = Point3f32[frame](
        vertices[unsafe_offset=vertex_base],
        vertices[unsafe_offset=vertex_base + 1],
        vertices[unsafe_offset=vertex_base + 2],
    )
    var indexed_e1 = Vec3f32[frame](
        vertices[unsafe_offset=vertex_base + 3] - indexed_v0.x,
        vertices[unsafe_offset=vertex_base + 4] - indexed_v0.y,
        vertices[unsafe_offset=vertex_base + 5] - indexed_v0.z,
    )
    var indexed_e2 = Vec3f32[frame](
        vertices[unsafe_offset=vertex_base + 6] - indexed_v0.x,
        vertices[unsafe_offset=vertex_base + 7] - indexed_v0.y,
        vertices[unsafe_offset=vertex_base + 8] - indexed_v0.z,
    )
    var indexed_reference = intersect_ray_tri_edges_scaled(
        origin, direction, indexed_v0, indexed_e1, indexed_e2, t_max, t_min
    )
    var indexed_candidate = _intersect_cwbvh_indexed_triangle[frame, .ANY_HIT](
        primitive_ids, vertices, UInt32(triangle), ray, hit
    )
    results[unsafe_offset=index * 5] = UInt32(reference.mask)
    results[unsafe_offset=index * 5 + 1] = UInt32(candidate)
    results[unsafe_offset=index * 5 + 2] = UInt32(wide_candidate)
    results[unsafe_offset=index * 5 + 3] = UInt32(indexed_reference.mask)
    results[unsafe_offset=index * 5 + 4] = UInt32(indexed_candidate)


def test_shadow_masks_match_reference() raises:
    comptime count = 16384
    var rng = Rng(123, 456)
    var triangles = List[Float32](capacity=count * 12)
    var wide_triangles = List[Float32](capacity=count * 12)
    var vertices = List[Float32](capacity=count * 9)
    var primitive_ids = List[UInt32](capacity=count)
    for index in range(count):
        var e1 = Vec3f32[.WORLD](rng.f32(-2, 2), rng.f32(-2, 2), rng.f32(-2, 2))
        var e2 = Vec3f32[.WORLD](rng.f32(-2, 2), rng.f32(-2, 2), rng.f32(-2, 2))
        var v0 = Point3f32[.WORLD](
            rng.f32(-2, 2), rng.f32(-2, 2), rng.f32(-2, 2)
        )
        if index == 0:
            e1 = Vec3f32[.WORLD](1, 0, 0)
            e2 = Vec3f32[.WORLD](0, 1, 0)
            v0 = Point3f32[.WORLD](0)
        elif index % 257 == 0:
            e2 = e1
        triangles.extend(
            [
                e1.x,
                e1.y,
                e1.z,
                Float32(0),
                e2.x,
                e2.y,
                e2.z,
                Float32(0),
                v0.x,
                v0.y,
                v0.z,
                Float32(0),
            ]
        )
        wide_triangles.extend(
            [
                v0.x,
                v0.y,
                v0.z,
                bitcast[.float32](UInt32(index)),
                e1.x,
                e1.y,
                e1.z,
                Float32(0),
                e2.x,
                e2.y,
                e2.z,
                Float32(0),
            ]
        )
        var v1 = v0 + e1
        var v2 = v0 + e2
        vertices.extend([v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z])
        primitive_ids.append(UInt32(index))
    with DeviceContext() as ctx:
        var data = upload_list(ctx, triangles)
        var wide_data = upload_list(ctx, wide_triangles)
        var vertex_data = upload_list(ctx, vertices)
        var id_data = upload_list(ctx, primitive_ids)
        var results = ctx.enqueue_create_buffer[.uint32](count * 64 * 5)
        comptime for frame in [Frame.WORLD, Frame.LOCAL]:
            ctx.enqueue_function[_compare_masks[count, frame]](
                data,
                wide_data,
                vertex_data,
                id_data,
                results,
                grid_dim=ceildiv(count * 64, 128),
                block_dim=128,
            )
            ctx.synchronize()
            var hits = 0
            var misses = 0
            with results.map_to_host() as values:
                for index in range(count * 64):
                    assert_equal(values[index * 5], values[index * 5 + 1])
                    assert_equal(values[index * 5], values[index * 5 + 2])
                    assert_equal(values[index * 5 + 3], values[index * 5 + 4])
                    if values[index * 5]:
                        hits += 1
                    else:
                        misses += 1
            assert_true(hits > 0)
            assert_true(misses > 0)


def run_tests[_discovery: Bool = True]() raises:
    comptime if not has_accelerator():
        raise "No accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
