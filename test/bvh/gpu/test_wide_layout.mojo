"""GPU wide-node slabs and segmented compaction preserve query semantics."""

from std.math import ceildiv
from std.memory import bitcast
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import (
    AxisAlignedBoundingBox,
    Point3f32,
    Rayf32,
    SegmentOffsets,
    Vec3f32,
)
from bajo.core.intersect import intersect_ray_aabb_rcp
from bajo.bvh.constants import EMPTY_LANE, f32_max
from bajo.bvh.gpu.ray_launch import _load_packed_ray
from bajo.bvh.gpu.utils import upload_list, upload_rays
from bajo.bvh.gpu.wide_layout import (
    _intersect_wide_node_precomputed,
    enqueue_compact_segmented_buffer,
)


def _compare_wide_slabs[
    width: SIMDLength
](
    nodes: ImmPointer[Float32, ImmutAnyOrigin],
    rays: ImmPointer[Float32, ImmutAnyOrigin],
    results: MutPointer[UInt32, MutAnyOrigin],
    ray_count: Int32,
):
    var index = global_idx.x
    if index >= Int(ray_count) * 5:
        return
    var node = UInt32(index % 5)
    var ray = _load_packed_ray[.WORLD](rays, Int(ray_count), index // 5)
    var origin = ray.origin[width]()
    var reciprocal = ray.reciprocal_direction[width]()
    var actual = _intersect_wide_node_precomputed[.WORLD, width](
        nodes, node, origin, reciprocal, ray.t_max
    )
    var bounds = AxisAlignedBoundingBox[.float32, .WORLD, width].invalid()
    var base = Int(node) * 7 * width
    comptime for lane in range(width):
        bounds._min.x[lane] = nodes[unsafe_offset=base + lane]
        bounds._min.y[lane] = nodes[unsafe_offset=base + width + lane]
        bounds._min.z[lane] = nodes[unsafe_offset=base + 2 * width + lane]
        bounds._max.x[lane] = nodes[unsafe_offset=base + 3 * width + lane]
        bounds._max.y[lane] = nodes[unsafe_offset=base + 4 * width + lane]
        bounds._max.z[lane] = nodes[unsafe_offset=base + 5 * width + lane]
    var reference = intersect_ray_aabb_rcp(
        origin, reciprocal, bounds, SIMD[.float32, width](ray.t_max)
    )
    var failures = UInt32(0)
    comptime for lane in range(width):
        var meta = bitcast[.uint32](
            nodes[unsafe_offset=base + 6 * width + lane]
        )
        if actual.meta[lane] != meta:
            failures |= 1
        if actual.bounds_hit.mask[lane] != reference.mask[lane]:
            failures |= 2
        if (
            reference.mask[lane]
            and actual.bounds_hit.t[lane] != reference.t[lane]
        ):
            failures |= 4
    results[unsafe_offset=index] = failures


def _check_wide_slabs[width: SIMDLength]() raises:
    # Full, partially empty, planar, point, and completely empty nodes.
    var nodes = List[Float32](length=5 * 7 * width, fill=0.0)
    for node in range(5):
        for lane in range(width):
            var base = node * 7 * width + lane
            var empty = node == 4 or (node == 1 and lane != 0)
            var lo = [-1.25 + Float32(lane) * 0.125, -0.125, 0.5]
            var hi = [lo[0] + 0.5, 0.25, 1.0]
            if node == 2:
                hi[0] = lo[0]
            if node == 3:
                lo = [Float32(0), 0.0, 0.0]
                hi = lo.copy()
            for axis in range(3):
                nodes[base + axis * width] = f32_max if empty else lo[axis]
                nodes[base + (axis + 3) * width] = -f32_max if empty else hi[
                    axis
                ]
            nodes[base + 6 * width] = bitcast[.float32](
                EMPTY_LANE if empty else UInt32(1 + lane + node * width)
            )
    var rays = List[Rayf32[.WORLD]]()
    for x in [-2.0, -1.25, 0.0, 2.0, 8.0]:
        for y in [-0.25, -0.125, 0.0, 0.25]:
            for z in [0.0, 0.5, 1.0, 4.0]:
                for dx in [-1.0, -1.0e-12, -0.0, 1.0e-12, 1.0]:
                    for dy in [-1.0, -1.0e-12, -0.0, 1.0e-12, 1.0]:
                        for dz in [-1.0, -1.0e-12, -0.0, 1.0e-12, 1.0]:
                            for limit in [
                                Float32(0),
                                1.0e-9,
                                0.25,
                                1.0,
                                4.0,
                                f32_max,
                            ]:
                                rays.append(
                                    Rayf32[.WORLD](
                                        Point3f32[.WORLD](
                                            Float32(x), Float32(y), Float32(z)
                                        ),
                                        Vec3f32[.WORLD](
                                            Float32(dx),
                                            Float32(dy),
                                            Float32(dz),
                                        ),
                                        0.0,
                                        limit,
                                    )
                                )
    with DeviceContext() as ctx:
        var results = ctx.enqueue_create_buffer[.uint32](len(rays) * 5)
        ctx.enqueue_function[_compare_wide_slabs[width]](
            upload_list(ctx, nodes),
            upload_rays(ctx, rays),
            results,
            Int32(len(rays)),
            grid_dim=ceildiv(len(rays) * 5, 128),
            block_dim=128,
        )
        ctx.synchronize()
        with results.map_to_host() as values:
            for index in range(len(values)):
                assert_equal(values[index], UInt32(0))


def test_wide_slabs_match_core_reference_at_boundaries() raises:
    _check_wide_slabs[2]()
    _check_wide_slabs[4]()
    _check_wide_slabs[8]()


def _check_compaction[stride: Int]() raises:
    var capacities = SegmentOffsets.from_counts([3, 0, 5, 2, 1, 2049])
    var counts = SegmentOffsets.from_counts([1, 0, 3, 1, 0, 2047])
    var values = [Float32(i) for i in range(capacities.item_count() * stride)]
    with DeviceContext() as ctx:
        var source = upload_list(ctx, values)
        var source_offsets = upload_list(ctx, capacities.offsets)
        var target_offsets = upload_list(ctx, counts.offsets)
        var target = enqueue_compact_segmented_buffer[.float32, stride](
            ctx,
            source,
            source_offsets,
            target_offsets,
            counts.item_count(),
            counts.segment_count(),
        )
        with target.map_to_host() as result:
            assert_equal(len(result), counts.item_count() * stride)
            for segment in range(counts.segment_count()):
                for item in range(Int(counts.count(segment))):
                    for field in range(stride):
                        var target_idx = (
                            Int(counts.offsets[segment]) + item
                        ) * stride + field
                        var source_idx = (
                            Int(capacities.offsets[segment]) + item
                        ) * stride + field
                        assert_equal(result[target_idx], values[source_idx])


def test_compact_segmented_fields() raises:
    comptime for stride in [1, 7, 112, 192, 253, 640]:
        _check_compaction[stride]()


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
