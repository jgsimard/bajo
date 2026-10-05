"""Nested traversal counters use embedded instance IDs and preserve hits."""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_true

from bajo.bvh.constants import EMPTY_LANE, TraceMode
from bajo.bvh.gpu import GpuBvhLayout
from bajo.bvh.gpu.ray_launch import _load_packed_ray, _store_packed_hit
from bajo.bvh.gpu.tlas import _trace_tlas_ray, build_gpu_tlas
from bajo.bvh.gpu.tlas_diagnostics import (
    GpuTlasTraversalStats,
    _trace_tlas2_leaf1_cwbvh8_with_stats,
)
from bajo.bvh.gpu.triangle_bvh import build_gpu_triangle_blas_set
from bajo.bvh.gpu.triangle_trace import _intersect_triangle_leaf
from bajo.bvh.gpu.utils import upload_list, upload_rays
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Hit, Instance
from bajo.core import Affine3f32, Point3f32, Quat, Rayf32, Vec3f32


def _compare_diagnostic_hits[
    instance_count: Int
](
    nodes: ImmPointer[Float32, ImmutAnyOrigin],
    legacy_instances: ImmPointer[UInt32, ImmutAnyOrigin],
    inverse: ImmPointer[Float32, ImmutAnyOrigin],
    indices: ImmPointer[UInt32, ImmutAnyOrigin],
    descriptors: ImmPointer[UInt32, ImmutAnyOrigin],
    blas_nodes: ImmPointer[Float32, ImmutAnyOrigin],
    blas_leaves: ImmPointer[Float32, ImmutAnyOrigin],
    rays: ImmPointer[Float32, ImmutAnyOrigin],
    hits: MutPointer[Float32, MutAnyOrigin],
    counters: MutPointer[UInt32, MutAnyOrigin],
    root: UInt32,
    ray_count: Int32,
):
    var index = global_idx.x
    if index >= Int(ray_count):
        return
    var ray = _load_packed_ray[.WORLD](rays, Int(ray_count), index)
    var production = _trace_tlas_ray[
        2,
        1,
        8,
        4,
        .CLOSEST_HIT,
        _intersect_triangle_leaf[.LOCAL, 4, TraceMode.CLOSEST_HIT],
        .CWBVH8,
    ](
        nodes,
        legacy_instances,
        inverse,
        indices,
        descriptors,
        blas_nodes,
        blas_leaves,
        instance_count,
        1,
        root,
        ray,
    )
    var diagnostic = _trace_tlas2_leaf1_cwbvh8_with_stats(
        nodes,
        legacy_instances,
        inverse,
        indices,
        descriptors,
        blas_nodes,
        blas_leaves,
        instance_count,
        root,
        ray,
    )
    _store_packed_hit(production, hits, Int(ray_count), index)
    _store_packed_hit(
        diagnostic[0], hits, Int(ray_count), index + Int(ray_count)
    )
    diagnostic[1].store(counters, index)
    var shadow = _trace_tlas_ray[
        2,
        1,
        8,
        4,
        .ANY_HIT,
        _intersect_triangle_leaf[.LOCAL, 4, TraceMode.ANY_HIT],
        .CWBVH8,
    ](
        nodes,
        legacy_instances,
        inverse,
        indices,
        descriptors,
        blas_nodes,
        blas_leaves,
        instance_count,
        1,
        root,
        ray,
    )
    counters[
        unsafe_offset=Int(ray_count) * GpuTlasTraversalStats.STRIDE + index
    ] = UInt32(shadow.is_occluded())


def test_embedded_instance_diagnostics_match_production() raises:
    var vertices = List[Point3f32[.LOCAL]](
        [
            Point3f32[.LOCAL](-0.4, -0.4, 0),
            Point3f32[.LOCAL](0.4, -0.4, 0),
            Point3f32[.LOCAL](0, 0.4, 0),
        ]
    )
    var bounds = compute_bounds(vertices)
    var instances = List[Instance]()
    for translation in [
        Vec3f32[.WORLD](0, 0, 6),
        Vec3f32[.WORLD](-1.5, 0, 4),
        Vec3f32[.WORLD](1.5, 0, 2),
        Vec3f32[.WORLD](0, 0, 3),
    ]:
        var transform = Affine3f32[
            .LOCAL, .WORLD
        ].from_rotation_scale_translation(
            Quat.from_axis_angle(Vec3f32[.LOCAL](0, 1, 0), 0.37),
            Vec3f32[.LOCAL](0.75, 1.1, 1.25),
            translation,
        )
        instances.append(Instance(transform, UInt32(0), bounds, .TRIANGLE))
    var rays = List[Rayf32[.WORLD]]()
    for x in [-1.5, 1.5, 0.0, 3.0]:
        rays.append(
            Rayf32[.WORLD](
                Point3f32[.WORLD](Float32(x), 0, 0), Vec3f32[.WORLD](0, 0, 1)
            )
        )
    rays.append(
        Rayf32[.WORLD](
            Point3f32[.WORLD](0, 0, 0), Vec3f32[.WORLD](0, 0, 1), 4, 100
        )
    )
    rays.append(
        Rayf32[.WORLD](
            Point3f32[.WORLD](0, 0, 0), Vec3f32[.WORLD](0, 0, 1), 0, 1
        )
    )
    rays.append(
        Rayf32[.WORLD](Point3f32[.WORLD](0, 0, 3.5), Vec3f32[.WORLD](0, 0, 1))
    )
    rays.append(
        Rayf32[.WORLD](Point3f32[.WORLD](0, 0, 10), Vec3f32[.WORLD](0, 0, -1))
    )
    var expected_instances = [UInt32(1), 2, 3, EMPTY_LANE, 0, EMPTY_LANE, 0, 0]
    with DeviceContext() as ctx:
        var vertex_sets = [vertices^]
        var blases = build_gpu_triangle_blas_set[8, 4, .HPLOC, .CWBVH8](
            ctx, vertex_sets
        )
        var tlas = build_gpu_tlas[
            .TRIANGLE, 2, 8, 1, 4, .LBVH, GpuBvhLayout.CWBVH8
        ](ctx, instances)
        var hits = ctx.enqueue_create_buffer[.float32](
            len(rays) * Hit.STRIDE * 2
        )
        var counters = ctx.enqueue_create_buffer[.uint32](
            len(rays) * (GpuTlasTraversalStats.STRIDE + 1)
        )
        # Neither traversal may read the obsolete table, even if it is poisoned.
        var legacy_instances = upload_list(
            ctx, List[UInt32](length=4, fill=EMPTY_LANE)
        )
        ctx.enqueue_function[_compare_diagnostic_hits[4]](
            tlas._tree.wide_nodes,
            legacy_instances,
            tlas._inst_inv_transform,
            tlas._inst_blas_indices,
            blases.descs,
            blases.nodes,
            blases.leaves,
            upload_rays(ctx, rays),
            hits,
            counters,
            tlas._tree.root_idx,
            Int32(len(rays)),
            grid_dim=1,
            block_dim=128,
        )
        ctx.synchronize()
        with hits.map_to_host() as values:
            var span = Span(unsafe_ptr=values.unsafe_ptr(), length=len(values))
            for index in range(len(rays)):
                assert_equal(
                    Hit[.WORLD].load(span, index).inst,
                    expected_instances[index],
                )
                for field in range(Hit.STRIDE):
                    assert_equal(
                        bitcast[.uint32](values[index * Hit.STRIDE + field]),
                        bitcast[.uint32](
                            values[(index + len(rays)) * Hit.STRIDE + field]
                        ),
                    )
        with counters.map_to_host() as values:
            for index in range(len(rays)):
                assert_equal(
                    values[len(rays) * GpuTlasTraversalStats.STRIDE + index],
                    UInt32(expected_instances[index] != EMPTY_LANE),
                )
                var base = index * GpuTlasTraversalStats.STRIDE
                var dispatches = values[
                    base + GpuTlasTraversalStats.BLAS_DISPATCHES
                ]
                assert_equal(
                    dispatches,
                    values[base + GpuTlasTraversalStats.BLAS_HITS]
                    + values[base + GpuTlasTraversalStats.BLAS_MISSES],
                )
                if expected_instances[index] != EMPTY_LANE:
                    assert_true(dispatches > 0)
                    assert_true(
                        values[base + GpuTlasTraversalStats.BLAS_NODE_VISITS]
                        > 0
                    )
                    assert_true(
                        values[base + GpuTlasTraversalStats.HIT_REPLACEMENTS]
                        > 0
                    )
                    assert_equal(
                        values[base + GpuTlasTraversalStats.WINNING_BLAS],
                        UInt32(0),
                    )


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
