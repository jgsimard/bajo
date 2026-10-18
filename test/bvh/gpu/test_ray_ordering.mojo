"""Ordered traversal preserves ray bits and original hit indices."""

from max.gpu.host import DeviceContext
from std.math import ceildiv
from bajo.bvh.gpu.ray_ordering import _copy_ray_buffer
from std.memory import bitcast
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_true

from bajo.bvh.constants import PrimitiveKind, TraceMode
from bajo.bvh.gpu import (
    GpuBlasSet,
    GpuBvhLayout,
    GpuRayOrderWorkspace,
    build_gpu_sphere_blas_set,
    build_gpu_tlas,
    build_gpu_triangle_blas_set,
)
from bajo.bvh.gpu.utils import upload_list, upload_rays
from bajo.bvh.host_utils import compute_bounds, sphere_bounds
from bajo.bvh.types import Hit, Instance, Sphere
from bajo.core import AABB, Affine3f32, Point3f32, Quat, Rayf32, Vec3f32


def test_ordering_preserves_source_bits_and_reuses_capacity() raises:
    with DeviceContext() as ctx:
        # Cross multiple OneSweep tiles, then reuse the workspace for smaller
        # non-power-of-two batches and a single ray.
        var ordering = GpuRayOrderWorkspace(ctx, 65537)
        var bounds = AABB[.WORLD](
            Point3f32[.WORLD](0, -1, 0), Point3f32[.WORLD](0, 1, 0)
        )
        var patterns = [
            UInt32(0),
            UInt32(0x80000000),
            UInt32(0x7F800000),
            UInt32(0xFF800000),
            UInt32(0x7FC01234),
            UInt32(0x7FA01234),
            UInt32(0xFFC01234),
            UInt32(0xFFA01234),
            UInt32(1),
            UInt32(0x80000001),
            UInt32(0xFFFFFFFF),
        ]
        for count in [65537, 8193, 31, 257, 1]:
            var rays = List[Rayf32[.WORLD]]()
            for index in range(count):
                rays.append(
                    Rayf32[.WORLD](
                        Point3f32[.WORLD](
                            Float32(index % 9 - 4),
                            Float32(index % 7 - 3),
                            Float32(index % 5 - 2),
                        ),
                        Vec3f32[.WORLD](
                            Float32(-0.0) if index % 2 else Float32(0.0),
                            Float32(-1) if index % 3 else Float32(1),
                            Float32(-1) if index % 5 else Float32(1),
                        ),
                        bitcast[.float32](patterns[index % len(patterns)]),
                        bitcast[.float32](
                            patterns[(index + 7) % len(patterns)]
                        ),
                    )
                )
            var source = upload_rays(ctx, rays)
            ordering._enqueue_order(ctx, source, count, bounds)
            ctx.synchronize()
            with source.map_to_host() as original:
                with ordering._indices.map_to_host() as indices:
                    with ordering._keys.map_to_host() as keys:
                        var seen = List[Bool](length=count, fill=False)
                        for index in range(count):
                            var old = Int(indices[index])
                            assert_true(old >= 0 and old < count)
                            assert_true(not seen[old])
                            seen[old] = True
                            if index > 0:
                                assert_true(keys[index - 1] <= keys[index])
                            var expected = rays[old].flatten()
                            for field in range(Rayf32.STRIDE):
                                assert_equal(
                                    bitcast[.uint32](
                                        original[field * count + old]
                                    ),
                                    bitcast[.uint32](expected[field]),
                                )
            # All keys collide here, exercising stable sorting on reuse.
            var tied = List[Rayf32[.WORLD]](length=count, fill=rays[0])
            var tied_source = upload_rays(ctx, tied)
            ordering._enqueue_order(ctx, tied_source, count, bounds)
            ctx.synchronize()
            with ordering._indices.map_to_host() as indices:
                for index in range(count):
                    assert_equal(indices[index], UInt32(index))


def _check_ordered_hits[
    kind: PrimitiveKind,
    layout: GpuBvhLayout,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    mut ctx: DeviceContext,
    blases: GpuBlasSet[kind, layout, node_width, leaf_width],
    bounds_list: List[AABB[.LOCAL]],
) raises:
    var instances = List[Instance]()
    var scene_bounds = AABB[.WORLD].invalid()
    for index in range(4):
        var blas_idx = UInt32(index % 2)
        var transform = Affine3f32[
            .LOCAL, .WORLD
        ].from_rotation_scale_translation(
            Quat.from_axis_angle(Vec3f32[.LOCAL](0, 1, 0), 0.3),
            Vec3f32[.LOCAL](0.75, 1.1, 1.25),
            Vec3f32[.WORLD](Float32(index % 2) * 2 - 1, 0, Float32(index + 2)),
        )
        var instance = Instance(
            transform, blas_idx, bounds_list[Int(blas_idx)], kind
        )
        scene_bounds.grow(instance.bounds)
        instances.append(instance^)
    var tlas = build_gpu_tlas[
        kind, 2, node_width, 1, leaf_width, .LBVH, layout
    ](ctx, instances)
    var ordering = GpuRayOrderWorkspace(ctx, 65537)
    var alias_base = Int(0)
    for count in [65537, 8193, 31, 257, 1]:
        var rays = List[Rayf32[.WORLD]]()
        for index in range(count):
            var x = Float32(index % 7 - 3)
            var y = Float32(index % 5 - 2) * 0.2
            var backwards = index % 3 == 0
            rays.append(
                Rayf32[.WORLD](
                    Point3f32[.WORLD](
                        x, y, Float32(8) if backwards else Float32(0)
                    ),
                    Vec3f32[.WORLD](
                        0, 0, Float32(-1) if backwards else Float32(1)
                    ),
                    Float32(index % 4) * 0.5,
                    Float32(1) if index % 11 == 0 else Float32(100),
                )
            )
        var source = upload_rays(ctx, rays)
        var seed = List[Float32](length=count * Hit.STRIDE + 16, fill=-42.0)
        var direct = upload_list(ctx, seed)
        var ordered = upload_list(ctx, seed)
        comptime for any_hit in [False, True]:
            comptime mode = TraceMode.ANY_HIT if any_hit else TraceMode.CLOSEST_HIT
            tlas.launch_rays[mode](ctx, blases, source, direct, count)
            tlas.launch_ordered_rays[mode](
                ctx, blases, ordering, source, ordered, count, scene_bounds
            )
            var in_place = upload_rays(ctx, rays)
            tlas.launch_ordered_rays[mode](
                ctx, blases, ordering, in_place, in_place, count, scene_bounds
            )
            assert_true(ordering._alias_rays)
            var base = Int(ordering._alias_rays.value().unsafe_ptr())
            if alias_base == 0:
                alias_base = base
            else:
                assert_equal(base, alias_base)
            ctx.synchronize()
            with in_place.map_to_host() as alias:
                with direct.map_to_host() as expected:
                    for index in range(count * Hit.STRIDE):
                        assert_equal(
                            bitcast[.uint32](alias[index]),
                            bitcast[.uint32](expected[index]),
                        )
            with direct.map_to_host() as a:
                with ordered.map_to_host() as b:
                    var hit_count = 0
                    var miss_count = 0
                    for index in range(len(seed)):
                        assert_equal(
                            bitcast[.uint32](a[index]),
                            bitcast[.uint32](b[index]),
                        )
                        if index >= count * Hit.STRIDE:
                            assert_equal(b[index], Float32(-42))
                    var span = Span(unsafe_ptr=a.unsafe_ptr(), length=len(a))
                    for index in range(count):
                        var hit = Hit[.WORLD].load(span, index)
                        if hit.is_hit() or hit.is_occluded():
                            hit_count += 1
                        else:
                            miss_count += 1
                    if count > 1:
                        assert_true(hit_count > 0 and miss_count > 0)


def _check_triangle[
    layout: GpuBvhLayout, node_width: SIMDLength, leaf_width: SIMDLength
]() raises:
    with DeviceContext() as ctx:
        var vertex_sets = List[List[Point3f32[.LOCAL]]]()
        var bounds_list = List[AABB[.LOCAL]]()
        for z in [Float32(0), Float32(0.3)]:
            var vertices = List[Point3f32[.LOCAL]](
                [
                    Point3f32[.LOCAL](-0.5, -0.5, z),
                    Point3f32[.LOCAL](0.5, -0.5, z),
                    Point3f32[.LOCAL](0, 0.5, z),
                ]
            )
            bounds_list.append(compute_bounds(vertices))
            vertex_sets.append(vertices^)
        var blases = build_gpu_triangle_blas_set[
            node_width, leaf_width, .HPLOC, layout
        ](ctx, vertex_sets)
        _check_ordered_hits[.TRIANGLE, layout, node_width, leaf_width](
            ctx, blases, bounds_list
        )


def test_compressed_triangle_ordered_hits() raises:
    _check_triangle[.CWBVH8, 8, 4]()


def test_wide_triangle_ordered_hits() raises:
    _check_triangle[.WIDE, 4, 2]()


def test_sphere_ordered_hits() raises:
    with DeviceContext() as ctx:
        var sphere_sets = List[List[Sphere[.LOCAL]]]()
        var bounds_list = List[AABB[.LOCAL]]()
        for z in [Float32(0), Float32(0.3)]:
            var spheres = List[Sphere[.LOCAL]](
                [Sphere[.LOCAL](Point3f32[.LOCAL](0, 0, z), 0.5)]
            )
            bounds_list.append(sphere_bounds(spheres))
            sphere_sets.append(spheres^)
        var blases = build_gpu_sphere_blas_set[4, 2, .LBVH](ctx, sphere_sets)
        _check_ordered_hits[.SPHERE, .WIDE, 4, 2](ctx, blases, bounds_list)


def _copy_at_offset[
    copy_width: Int
](
    source: ImmPointer[Float32, ImmutAnyOrigin],
    destination: MutPointer[Float32, MutAnyOrigin],
    count: Int32,
    offset: Int32,
):
    comptime assert copy_width == 4
    _copy_ray_buffer(source.unsafe_offset(Int(offset)), destination, count)


def test_alias_copy_preserves_unaligned_source_words() raises:
    var patterns = [
        UInt32(0),
        UInt32(0x80000000),
        UInt32(0x7F800000),
        UInt32(0xFF800000),
        UInt32(0x7FC01234),
        UInt32(0x7FA01234),
        UInt32(0xFFC01234),
        UInt32(0xFFA01234),
        UInt32(1),
        UInt32(0x80000001),
        UInt32(0xFFFFFFFF),
    ]
    with DeviceContext() as ctx:
        for count in [1, 31, 257, 8193]:
            var words = List[Float32]()
            for index in range(count * Rayf32.STRIDE + 16):
                var value = UInt32(index) * UInt32(1664525) + UInt32(1013904223)
                if index < len(patterns) * Rayf32.STRIDE:
                    value = patterns[
                        (index + index // Rayf32.STRIDE) % len(patterns)
                    ]
                words.append(bitcast[.float32](value))
            var source = upload_list(ctx, words)
            for offset in range(4):
                var guard = bitcast[.float32](UInt32(0xDEADBEEF))
                var output = upload_list(
                    ctx,
                    List[Float32](
                        length=count * Rayf32.STRIDE + 16, fill=guard
                    ),
                )
                ctx.enqueue_function[_copy_at_offset[4]](
                    source,
                    output,
                    Int32(count),
                    Int32(offset),
                    grid_dim=ceildiv(count * Rayf32.STRIDE // 4, 256),
                    block_dim=256,
                )
                ctx.synchronize()
                with output.map_to_host() as values:
                    for index in range(count * Rayf32.STRIDE):
                        assert_equal(
                            bitcast[.uint32](values[index]),
                            bitcast[.uint32](words[offset + index]),
                        )
                    for index in range(count * Rayf32.STRIDE, len(values)):
                        assert_equal(
                            bitcast[.uint32](values[index]), UInt32(0xDEADBEEF)
                        )


def test_alias_copy_cache_is_lazy_and_reused() raises:
    with DeviceContext() as ctx:
        var ordering = GpuRayOrderWorkspace(ctx, 8193)
        assert_true(not ordering._alias_rays)
        var base = Int(0)
        for count in [8193, 1, 31, 257]:
            var words = List[Float32]()
            for index in range(count * Rayf32.STRIDE):
                words.append(
                    bitcast[.float32](
                        UInt32(index) * UInt32(1664525) + UInt32(1013904223)
                    )
                )
            var source = upload_list(ctx, words)
            if ordering._alias_rays:
                with ordering._alias_rays.value().map_to_host() as values:
                    for index in range(len(values)):
                        values[index] = bitcast[.float32](UInt32(0xDEADBEEF))
            ordering._enqueue_alias_copy(ctx, source, count)
            var current = Int(ordering._alias_rays.value().unsafe_ptr())
            if base == 0:
                base = current
            else:
                assert_equal(current, base)
            ctx.synchronize()
            with ordering._alias_rays.value().map_to_host() as values:
                for index in range(count * Rayf32.STRIDE):
                    assert_equal(
                        bitcast[.uint32](values[index]),
                        bitcast[.uint32](words[index]),
                    )
                for index in range(count * Rayf32.STRIDE, len(values)):
                    assert_equal(
                        bitcast[.uint32](values[index]), UInt32(0xDEADBEEF)
                    )


def run_tests[_discovery: Bool = True]() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
