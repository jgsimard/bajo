"""Correctness tests for reusable CWBVH8 construction storage."""

from std.sys import has_accelerator
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
from bajo.bvh.gpu.utils import upload_list, upload_vertices
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Hit
from test.bvh.fixtures import (
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
](duplicate: Bool) raises:
    var vertices = _make_duplicate_centroid_scene[
        .WORLD
    ]() if duplicate else _make_small_scene[.WORLD]()
    with DeviceContext() as ctx:
        var d_vertices = upload_vertices(ctx, vertices)
        var arena = create_cwbvh8_bench_arena[max_leaf_size, direct_conversion](
            ctx, d_vertices
        )
        ctx.synchronize()
        arena.finish_synchronized()
        var node_checksum = _buffer_word_checksum(arena.nodes)
        var triangle_checksum = _buffer_word_checksum(arena.triangles)

        for _ in range(3):
            arena.enqueue_rebuild(ctx, d_vertices)
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


def test_indexed_cwbvh8_triangles_match_packed_hits() raises:
    var vertices = _make_small_scene[.WORLD]()
    var camera = _make_camera_rays_and_params(
        compute_bounds(vertices), 32, 24, 1
    )
    var ray_count = len(camera[0])
    with DeviceContext() as ctx:
        var d_vertices = upload_vertices(ctx, vertices)
        var d_camera = upload_list(ctx, camera[1])
        var packed = create_cwbvh8_bench_arena[1, True, False](ctx, d_vertices)
        var indexed = create_cwbvh8_bench_arena[1, True, True](ctx, d_vertices)
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


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
