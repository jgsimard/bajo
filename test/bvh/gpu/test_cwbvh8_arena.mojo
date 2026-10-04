"""Correctness tests for reusable CWBVH8 construction storage."""

from std.sys import has_accelerator
from std.memory import bitcast
from std.math import ceil, clamp, floor, fma, max
from max.gpu import global_idx
from bajo.bvh.gpu.compressed_bounds_bvh import (
    pack_segmented_cwbvh_triangles_kernel,
    _scale_from_exponent,
    _inverse_scale_from_exponent,
    _quantize_lower,
    _quantize_upper,
)
from std.testing import TestSuite, assert_equal, assert_true
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.benchmark.gpu_bvh_fixtures import (
    create_cwbvh8_bench_arena,
    trace_cwbvh8_camera_kernel,
    trace_cwbvh8_indexed_camera_kernel,
    trace_cwbvh8_camera_legacy_decode_kernel,
)
from bajo.bvh.constants import (
    BinaryBvhNode,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    GPU_STACK_SIZE,
)
from bajo.bvh.gpu.utils import _device_span, upload_list, upload_vertices
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Hit
from bajo.core import Point3f32
from test.bvh.fixtures import (
    _append_tri,
    _make_camera_rays_and_params,
    _make_duplicate_centroid_scene,
    _make_small_scene,
)


def _buffer_word_checksum(buffer: DeviceBuffer[.float32]) raises -> UInt64:
    var checksum = UInt64(0)
    with buffer.map_to_host() as host:
        var words = host.unsafe_ptr().unsafe_bitcast[UInt32]()
        for i in range(len(host)):
            checksum = (checksum * UInt64(1099511628211)) ^ UInt64(
                words[unsafe_offset=i]
            )
    return checksum


def _assert_rebuild_is_stable[
    max_leaf_size: Int,
    direct_conversion: Bool = False,
    indexed_triangles: Bool = False,
](duplicate: Bool) raises:
    var vertices = _make_duplicate_centroid_scene[
        .WORLD
    ]() if duplicate else _make_small_scene[.WORLD]()
    with DeviceContext() as ctx:
        var d_vertices = upload_vertices(ctx, vertices)
        var arena = create_cwbvh8_bench_arena[
            max_leaf_size, direct_conversion, indexed_triangles
        ](ctx, d_vertices)
        ctx.synchronize()
        var internal_capacity = len(vertices) // 3 - 1
        comptime if direct_conversion:
            var scratch_records = max(internal_capacity, 1)
            comptime if max_leaf_size == 1:
                scratch_records += len(vertices) // 3
            assert_equal(len(arena.hploc.scratch_bounds), scratch_records * 6)
            assert_equal(len(arena.binary.node_meta), 1)
            assert_equal(len(arena.binary.node_bounds), 1)
            assert_equal(len(arena.workspace.topology.value().node_flags), 1)
            assert_equal(len(arena.workspace.topology.value().leaf_parent), 1)
            comptime if max_leaf_size == 1:
                assert_equal(len(arena.binary.node_leaf_counts), 1)
            else:
                assert_equal(
                    len(arena.binary.node_leaf_counts), internal_capacity
                )
        else:
            assert_equal(
                len(arena.binary.node_meta),
                internal_capacity * BinaryBvhNode.META_STRIDE,
            )
            assert_equal(
                len(arena.binary.node_bounds),
                internal_capacity * BinaryBvhNode.BOUNDS_STRIDE,
            )
            assert_equal(len(arena.binary.node_leaf_counts), internal_capacity)
        comptime if direct_conversion:
            assert_true(not Bool(arena.wide_nodes))
            assert_true(not Bool(arena.leaf_block_indices))
        else:
            assert_true(Bool(arena.wide_nodes))
            assert_true(Bool(arena.leaf_block_indices))
        arena.finish_synchronized()
        var node_checksum = _buffer_word_checksum(arena.nodes)
        var triangle_checksum = _buffer_word_checksum(arena.triangles)

        for _ in range(3):
            _ = arena.enqueue_rebuild(ctx, d_vertices)
            ctx.synchronize()
            arena.finish_synchronized()
            assert_equal(_buffer_word_checksum(arena.nodes), node_checksum)
            assert_equal(
                _buffer_word_checksum(arena.triangles), triangle_checksum
            )
        assert_true(node_checksum != 0)
        assert_true(triangle_checksum != 0)


def test_cwbvh8_arena_reuses_all_leaf_configurations() raises:
    comptime for max_leaf_size in [1, 2, 3]:
        _assert_rebuild_is_stable[max_leaf_size](False)


def test_cwbvh8_arena_handles_duplicate_morton_codes() raises:
    _assert_rebuild_is_stable[1](True)


def test_direct_cwbvh8_conversion_matches_staged_hits() raises:
    var vertices = _make_small_scene[.WORLD]()
    var camera = _make_camera_rays_and_params(
        compute_bounds(vertices), 16, 12, 1
    )
    var ray_count = len(camera[0])
    with DeviceContext() as ctx:
        var d_vertices = upload_vertices(ctx, vertices)
        var d_camera = upload_list(ctx, camera[1])
        var staged = create_cwbvh8_bench_arena[1, False](ctx, d_vertices)
        var direct = create_cwbvh8_bench_arena[1, True](ctx, d_vertices)
        var staged_hits = ctx.enqueue_create_buffer[.float32](
            ray_count * Hit.STRIDE
        )
        var direct_hits = ctx.enqueue_create_buffer[.float32](
            ray_count * Hit.STRIDE
        )
        ctx.enqueue_function[trace_cwbvh8_camera_kernel[3]](
            staged.nodes,
            staged.triangles,
            UInt32(0),
            d_camera,
            staged_hits,
            Int32(ray_count),
            Int32(16),
            Int32(12),
            Float32(1.0) / Float32(12),
            grid_dim=1,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.enqueue_function[trace_cwbvh8_camera_kernel[3]](
            direct.nodes,
            direct.triangles,
            UInt32(0),
            d_camera,
            direct_hits,
            Int32(ray_count),
            Int32(16),
            Int32(12),
            Float32(1.0) / Float32(12),
            grid_dim=1,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        staged.finish_synchronized()
        direct.finish_synchronized()
        with staged_hits.map_to_host() as lhs, direct_hits.map_to_host() as rhs:
            var lhs_words = lhs.unsafe_ptr().unsafe_bitcast[UInt32]()
            var rhs_words = rhs.unsafe_ptr().unsafe_bitcast[UInt32]()
            for i in range(len(lhs)):
                assert_equal(
                    lhs_words[unsafe_offset=i], rhs_words[unsafe_offset=i]
                )


def test_direct_cwbvh8_conversion_rebuilds_all_leaf_sizes() raises:
    comptime for max_leaf_size in [1, 2, 3]:
        _assert_rebuild_is_stable[max_leaf_size, True](False)


def _assert_cross_block_primitive_remap[max_leaf_size: Int]() raises:
    comptime triangle_count = 4097
    var vertices = List[Point3f32[.WORLD]](capacity=triangle_count * 3)
    for i in range(triangle_count):
        _append_tri(
            vertices,
            Float32((i % 19) * 3),
            Float32(((i // 19) % 13) * 3),
            Float32((i * 7) % 11 + 2),
        )
    with DeviceContext() as ctx:
        var d_vertices = upload_vertices(ctx, vertices)
        var arena = create_cwbvh8_bench_arena[max_leaf_size, True, True](
            ctx, d_vertices
        )
        for rebuild in range(3):
            if rebuild != 0:
                _ = arena.enqueue_rebuild(ctx, d_vertices)
            ctx.synchronize()
            arena.finish_synchronized()
            var seen = List[Bool](length=triangle_count, fill=False)
            with arena.representation_workspace.compact_primitive_ids.map_to_host() as primitive_ids:
                assert_equal(len(primitive_ids), triangle_count)
                for i in range(triangle_count):
                    var primitive = Int(primitive_ids[i])
                    assert_true(0 <= primitive < triangle_count)
                    assert_true(not seen[primitive])
                    seen[primitive] = True
            for present in seen:
                assert_true(present)


def test_direct_cwbvh8_cross_block_primitive_remap_is_complete() raises:
    comptime for max_leaf_size in [1, 2, 3]:
        _assert_cross_block_primitive_remap[max_leaf_size]()


def test_compact_hploc_children_match_full_production_layout() raises:
    var vertices = _make_duplicate_centroid_scene[.WORLD]()
    with DeviceContext() as ctx:
        var d_vertices = upload_vertices(ctx, vertices)
        var full = create_cwbvh8_bench_arena[1, False](ctx, d_vertices)
        var compact = create_cwbvh8_bench_arena[1, True](ctx, d_vertices)
        ctx.synchronize()
        full.finish_synchronized()
        compact.finish_synchronized()
        assert_equal(
            len(compact.hploc.compact_children),
            compact.binary.internal_count * 2,
        )
        with full.binary.node_meta.map_to_host() as node_meta, compact.hploc.compact_children.map_to_host() as children:
            for node_idx in range(compact.binary.internal_count):
                assert_equal(
                    children[node_idx * 2],
                    node_meta[
                        node_idx * BinaryBvhNode.META_STRIDE
                        + BinaryBvhNode.LEFT
                    ],
                )
                assert_equal(
                    children[node_idx * 2 + 1],
                    node_meta[
                        node_idx * BinaryBvhNode.META_STRIDE
                        + BinaryBvhNode.RIGHT
                    ],
                )


def test_packed_cwbvh8_decoder_matches_legacy_hits() raises:
    var vertices = _make_small_scene[.WORLD]()
    var camera = _make_camera_rays_and_params(
        compute_bounds(vertices), 32, 24, 3
    )
    var ray_count = len(camera[0])
    with DeviceContext() as ctx:
        var d_vertices = upload_vertices(ctx, vertices)
        var d_camera = upload_list(ctx, camera[1])
        var arena = create_cwbvh8_bench_arena[3, True](ctx, d_vertices)
        var packed_hits = ctx.enqueue_create_buffer[.float32](
            ray_count * Hit.STRIDE
        )
        var legacy_hits = ctx.enqueue_create_buffer[.float32](
            ray_count * Hit.STRIDE
        )
        ctx.enqueue_function[
            trace_cwbvh8_camera_kernel[3, GPU_STACK_SIZE, False]
        ](
            arena.nodes,
            arena.triangles,
            UInt32(0),
            d_camera,
            packed_hits,
            Int32(ray_count),
            Int32(32),
            Int32(24),
            Float32(1.0) / Float32(24),
            grid_dim=(ray_count + GPU_BOUNDS_BVH_BLOCK_SIZE - 1)
            // GPU_BOUNDS_BVH_BLOCK_SIZE,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.enqueue_function[trace_cwbvh8_camera_legacy_decode_kernel[False]](
            arena.nodes,
            arena.triangles,
            UInt32(0),
            d_camera,
            legacy_hits,
            Int32(ray_count),
            Int32(32),
            Int32(24),
            Float32(1.0) / Float32(24),
            grid_dim=(ray_count + GPU_BOUNDS_BVH_BLOCK_SIZE - 1)
            // GPU_BOUNDS_BVH_BLOCK_SIZE,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        arena.finish_synchronized()
        with packed_hits.map_to_host() as lhs, legacy_hits.map_to_host() as rhs:
            var lhs_words = lhs.unsafe_ptr().unsafe_bitcast[UInt32]()
            var rhs_words = rhs.unsafe_ptr().unsafe_bitcast[UInt32]()
            for i in range(len(lhs)):
                assert_equal(
                    lhs_words[unsafe_offset=i], rhs_words[unsafe_offset=i]
                )


def _assert_indexed_cwbvh8_triangles_match_packed_hits[direct: Bool]() raises:
    var vertices = _make_small_scene[.WORLD]()
    var camera = _make_camera_rays_and_params(
        compute_bounds(vertices), 32, 24, 1
    )
    var ray_count = len(camera[0])
    with DeviceContext() as ctx:
        var d_vertices = upload_vertices(ctx, vertices)
        var d_camera = upload_list(ctx, camera[1])
        var packed = create_cwbvh8_bench_arena[1, True, False](ctx, d_vertices)
        var indexed = create_cwbvh8_bench_arena[1, direct, True](
            ctx, d_vertices
        )
        assert_equal(len(indexed.triangles), len(d_vertices))
        var packed_hits = ctx.enqueue_create_buffer[.float32](
            ray_count * Hit.STRIDE
        )
        var indexed_hits = ctx.enqueue_create_buffer[.float32](
            ray_count * Hit.STRIDE
        )
        ctx.enqueue_function[trace_cwbvh8_camera_kernel[1]](
            packed.nodes,
            packed.triangles,
            UInt32(0),
            d_camera,
            packed_hits,
            Int32(ray_count),
            Int32(32),
            Int32(24),
            Float32(1.0) / Float32(24),
            grid_dim=(ray_count + GPU_BOUNDS_BVH_BLOCK_SIZE - 1)
            // GPU_BOUNDS_BVH_BLOCK_SIZE,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.enqueue_function[trace_cwbvh8_indexed_camera_kernel[1]](
            indexed.nodes,
            indexed.representation_workspace.compact_primitive_ids,
            d_vertices,
            UInt32(0),
            d_camera,
            indexed_hits,
            Int32(ray_count),
            Int32(32),
            Int32(24),
            Float32(1.0) / Float32(24),
            grid_dim=(ray_count + GPU_BOUNDS_BVH_BLOCK_SIZE - 1)
            // GPU_BOUNDS_BVH_BLOCK_SIZE,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        packed.finish_synchronized()
        indexed.finish_synchronized()
        with packed_hits.map_to_host() as lhs, indexed_hits.map_to_host() as rhs:
            var lhs_words = lhs.unsafe_ptr().unsafe_bitcast[UInt32]()
            var rhs_words = rhs.unsafe_ptr().unsafe_bitcast[UInt32]()
            for i in range(len(lhs)):
                assert_equal(
                    lhs_words[unsafe_offset=i], rhs_words[unsafe_offset=i]
                )


def test_indexed_cwbvh8_triangles_match_packed_hits() raises:
    comptime for direct in [False, True]:
        _assert_indexed_cwbvh8_triangles_match_packed_hits[direct]()
        _assert_rebuild_is_stable[1, direct, True](False)


def test_reciprocal_quantization_matches_division_on_gpu() raises:
    def _quantization_edge_kernel(
        values: Pointer[Float32, ImmutAnyOrigin],
        results: Pointer[UInt32, MutAnyOrigin],
    ):
        var idx = global_idx.x
        if idx >= 254 * 9:
            return
        var scale = _scale_from_exponent(UInt32(idx // 9 + 1))
        var value = values[unsafe_offset=idx]
        var inverse = _inverse_scale_from_exponent(UInt32(idx // 9 + 1))
        results[unsafe_offset=idx * 8] = _quantize_lower(value, 0, scale)
        results[unsafe_offset=idx * 8 + 1] = _quantize_lower[True](
            value, 0, scale, inverse
        )
        results[unsafe_offset=idx * 8 + 2] = _quantize_upper(value, 0, scale)
        results[unsafe_offset=idx * 8 + 3] = _quantize_upper[True](
            value, 0, scale, inverse
        )
        # Keep the former 64-bit integer conversion as a device reference.
        var scaled = value / scale
        var lower = clamp(Int(floor(scaled)), 0, 255)
        var upper = clamp(Int(ceil(scaled)), 0, 255)
        if fma(Float32(lower), scale, Float32(0)) > value and lower > 0:
            lower -= 1
        if fma(Float32(upper), scale, Float32(0)) < value and upper < 255:
            upper += 1
        results[unsafe_offset=idx * 8 + 4] = UInt32(lower)
        results[unsafe_offset=idx * 8 + 5] = UInt32(upper)
        results[unsafe_offset=idx * 8 + 6] = bitcast[.uint32](inverse)
        results[unsafe_offset=idx * 8 + 7] = bitcast[.uint32](
            Float32(1) / scale
        )

    var values = List[Float32](capacity=254 * 9)
    for exponent in range(1, 255):
        var scale = _scale_from_exponent(UInt32(exponent))
        for fraction in [Float32(0.125), Float32(0.5), Float32(1)]:
            var bits = bitcast[.uint32](fraction * scale)
            for delta in [-1, 0, 1]:
                values.append(bitcast[.float32](UInt32(Int(bits) + delta)))
    with DeviceContext() as ctx:
        var inputs = upload_list(ctx, values)
        var results = ctx.enqueue_create_buffer[.uint32](len(values) * 8)
        ctx.enqueue_function[_quantization_edge_kernel](
            inputs,
            results,
            grid_dim=(len(values) + GPU_BOUNDS_BVH_BLOCK_SIZE - 1)
            // GPU_BOUNDS_BVH_BLOCK_SIZE,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        with results.map_to_host() as host:
            for idx in range(len(values)):
                assert_equal(host[idx * 8 + 6], host[idx * 8 + 7])
                assert_equal(host[idx * 8], host[idx * 8 + 1])
                assert_equal(host[idx * 8 + 2], host[idx * 8 + 3])
                assert_equal(host[idx * 8], host[idx * 8 + 4])
                assert_equal(host[idx * 8 + 2], host[idx * 8 + 5])


def test_vector_triangle_packing_keeps_segment_local_primitive_ids() raises:
    var values = List[Float32](capacity=36)
    for value in range(1, 37):
        values.append(Float32(value))
    var order: List[UInt32] = [1, 0, 3, 2]
    var offsets: List[UInt32] = [0, 2, 4]
    with DeviceContext() as ctx:
        var vertices = upload_list(ctx, values)
        var ids = upload_list(ctx, order)
        var segments = upload_list(ctx, offsets)
        var triangles = ctx.enqueue_create_buffer[.float32](48)
        ctx.enqueue_function[pack_segmented_cwbvh_triangles_kernel[False]](
            vertices,
            ids,
            _device_span[mut=False](segments),
            triangles,
            Int32(4),
            grid_dim=1,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        with triangles.map_to_host() as host:
            for triangle in range(4):
                var base = triangle * 12
                for axis in range(3):
                    assert_equal(host[base + axis], Float32(3))
                    assert_equal(host[base + 4 + axis], Float32(6))
                    assert_equal(
                        host[base + 8 + axis],
                        Float32(Int(order[triangle]) * 9 + axis + 1),
                    )
                assert_equal(host[base + 3], Float32(0))
                assert_equal(host[base + 7], Float32(0))
                assert_equal(
                    bitcast[.uint32](host[base + 11]),
                    order[triangle] - UInt32((triangle // 2) * 2),
                )


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
