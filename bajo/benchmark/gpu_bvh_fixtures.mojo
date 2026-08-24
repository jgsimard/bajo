"""Shared inputs and adapters for GPU BVH diagnostics."""

from std.gpu import global_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import (
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    GPU_STACK_SIZE,
    TRI_LEAF_VERTEX_STRIDE,
)
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.builder.segmented_build import enqueue_segmented_wide_build
from bajo.bvh.gpu.camera_launch import (
    _camera_ray,
    _camera_ray_single_view,
    _store_camera_hit,
)
from bajo.bvh.gpu.compressed_bounds_bvh import (
    CWBVH_NODE_WORDS,
    CWBVH_TRIANGLE_WORDS,
    enqueue_segmented_cwbvh8_representation,
)
from bajo.bvh.gpu.cwbvh8_builder import GpuCwbvh8BuildArena
from bajo.bvh.gpu.builder.hploc_layout import HPLOC_MERGING_THRESHOLD
from bajo.bvh.gpu.triangle_bvh import (
    compute_triangle_bounds_kernel,
    trace_cwbvh8_indexed_triangles,
    trace_cwbvh8_triangles,
)
from bajo.bvh.gpu.tlas_diagnostics import _trace_cwbvh8_with_stats_in_frame
from bajo.bvh.gpu.trace import GpuTraversalStats
from bajo.bvh.gpu.utils import _device_span
from bajo.core import AABB, Point3f32, Rayf32, SegmentOffsets


@fieldwise_init
struct Cwbvh8BenchBvh(Copyable):
    var nodes: DeviceBuffer[.float32]
    var triangles: DeviceBuffer[.float32]
    var root_idx: UInt32


def create_cwbvh8_bench_arena[
    max_leaf_size: Int = 3,
    direct_conversion: Bool = True,
    indexed_triangle_layout: Bool = False,
    spatial_slots: Bool = True,
    merging_threshold: Int = HPLOC_MERGING_THRESHOLD,
](
    mut ctx: DeviceContext, d_vertices: DeviceBuffer[.float32]
) raises -> GpuCwbvh8BuildArena[
    max_leaf_size,
    direct_conversion,
    indexed_triangle_layout,
    spatial_slots,
    4,
    merging_threshold,
]:
    """Create a fixed-capacity H-PLOC arena and queue its initial build."""
    var tri_count = len(d_vertices) / TRI_LEAF_VERTEX_STRIDE
    var leaf_bounds = ctx.enqueue_create_buffer[.float32](tri_count * 6)
    var payloads = ctx.enqueue_create_buffer[.uint32](tri_count)
    ctx.enqueue_function[compute_triangle_bounds_kernel[.WORLD]](
        _device_span[mut=False](d_vertices),
        _device_span[mut=True](leaf_bounds),
        _device_span[mut=True](payloads),
        grid_dim=ceildiv(tri_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    return GpuCwbvh8BuildArena[
        max_leaf_size,
        direct_conversion,
        indexed_triangle_layout,
        spatial_slots,
        4,
        merging_threshold,
    ](ctx, leaf_bounds^, payloads^, d_vertices)


def build_cwbvh8_bench_bvh[
    method: GpuBvhBuildMethod = .HPLOC,
    max_leaf_size: Int = 3,
](
    mut ctx: DeviceContext, d_vertices: DeviceBuffer[.float32]
) raises -> Cwbvh8BenchBvh:
    """Build one CWBVH8 using the selected binary hierarchy method."""
    comptime assert 1 <= max_leaf_size <= 3
    var tri_count = len(d_vertices) / TRI_LEAF_VERTEX_STRIDE
    var leaf_bounds = ctx.enqueue_create_buffer[.float32](tri_count * 6)
    var payloads = ctx.enqueue_create_buffer[.uint32](tri_count)
    ctx.enqueue_function[compute_triangle_bounds_kernel[.WORLD]](
        _device_span[mut=False](d_vertices),
        _device_span[mut=True](leaf_bounds),
        _device_span[mut=True](payloads),
        grid_dim=ceildiv(tri_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    var build = enqueue_segmented_wide_build[
        8, 4, max_leaf_size, method, True, True
    ](ctx, SegmentOffsets.single(tri_count), leaf_bounds^, payloads^)
    var nodes = ctx.enqueue_create_buffer[.float32](
        build.wide.node_segments.item_count() * CWBVH_NODE_WORDS
    )
    var triangles = ctx.enqueue_create_buffer[.float32](
        tri_count * CWBVH_TRIANGLE_WORDS
    )
    var encoded_counts = enqueue_segmented_cwbvh8_representation[4](
        ctx,
        build.wide.wide_nodes,
        build.wide.leaf_block_indices,
        build.wide.node_segment_offsets,
        build.wide.leaf_block_segment_offsets,
        build.binary.segment_offsets,
        build.wide.node_counts,
        d_vertices,
        nodes,
        triangles,
    )
    ctx.synchronize()
    build.finish_synchronized()
    with encoded_counts.map_to_host() as counts:
        if counts[0] != UInt32(tri_count):
            raise "unified CWBVH8 encoding lost triangles"
    return Cwbvh8BenchBvh(nodes^, triangles^, UInt32(0))


def trace_cwbvh8_camera_kernel[
    max_leaf_size: Int = 3,
    stack_capacity: Int = GPU_STACK_SIZE,
    single_view: Bool = True,
](
    nodes: Pointer[Float32, ImmutAnyOrigin],
    triangles: Pointer[Float32, ImmutAnyOrigin],
    root_idx: UInt32,
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    ray_count: Int32,
    width_px: Int32,
    height_px: Int32,
    inv_height: Float32,
):
    """Trace closest hits through one CWBVH8 for generated camera rays."""
    var ray_count_int = Int(ray_count)
    var ray_idx = global_idx.x
    if ray_idx >= ray_count_int:
        return
    var ray: Rayf32[.WORLD]
    comptime if single_view:
        ray = _camera_ray_single_view(
            camera_params,
            Int32(ray_idx),
            width_px,
            inv_height,
        )
    else:
        ray = _camera_ray(
            camera_params,
            ray_count_int,
            ray_idx,
            Int(width_px),
            Int(height_px),
            inv_height,
        )
    var hit = trace_cwbvh8_triangles[
        .WORLD, .CLOSEST_HIT, True, max_leaf_size, stack_capacity
    ](nodes, triangles, root_idx, ray)
    _store_camera_hit(hit, hits, ray_count_int, ray_idx)


def trace_cwbvh8_camera_legacy_decode_kernel[
    single_view: Bool = True,
](
    nodes: Pointer[Float32, ImmutAnyOrigin],
    triangles: Pointer[Float32, ImmutAnyOrigin],
    root_idx: UInt32,
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    ray_count: Int32,
    width_px: Int32,
    height_px: Int32,
    inv_height: Float32,
):
    """Trace through the retained pre-packed decoder for A/B tests."""
    var ray_count_int = Int(ray_count)
    var ray_idx = global_idx.x
    if ray_idx >= ray_count_int:
        return
    var ray: Rayf32[.WORLD]
    comptime if single_view:
        ray = _camera_ray_single_view(
            camera_params,
            Int32(ray_idx),
            width_px,
            inv_height,
        )
    else:
        ray = _camera_ray(
            camera_params,
            ray_count_int,
            ray_idx,
            Int(width_px),
            Int(height_px),
            inv_height,
        )
    var hit = trace_cwbvh8_triangles[.WORLD, .CLOSEST_HIT, False](
        nodes, triangles, root_idx, ray
    )
    _store_camera_hit(hit, hits, ray_count_int, ray_idx)


def trace_cwbvh8_indexed_camera_kernel[
    max_leaf_size: Int = 3,
    stack_capacity: Int = GPU_STACK_SIZE,
](
    nodes: Pointer[Float32, ImmutAnyOrigin],
    primitive_ids: Pointer[UInt32, ImmutAnyOrigin],
    vertices: Pointer[Float32, ImmutAnyOrigin],
    root_idx: UInt32,
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    ray_count: Int32,
    width_px: Int32,
    height_px: Int32,
    inv_height: Float32,
):
    var ray_idx = global_idx.x
    if ray_idx >= Int(ray_count):
        return
    var ray = _camera_ray_single_view(
        camera_params, Int32(ray_idx), width_px, inv_height
    )
    var hit = trace_cwbvh8_indexed_triangles[
        .WORLD, .CLOSEST_HIT, max_leaf_size, stack_capacity
    ](nodes, primitive_ids, vertices, root_idx, ray)
    _store_camera_hit(hit, hits, Int(ray_count), ray_idx)


def trace_cwbvh8_camera_stats_kernel(
    nodes: Pointer[Float32, ImmutAnyOrigin],
    triangles: Pointer[Float32, ImmutAnyOrigin],
    root_idx: UInt32,
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    stats: Pointer[UInt32, MutAnyOrigin],
    ray_count: Int32,
    width_px: Int32,
    height_px: Int32,
    inv_height: Float32,
):
    """Trace camera rays and record per-ray CWBVH8 work counters."""
    var ray_count_int = Int(ray_count)
    var ray_idx = global_idx.x
    if ray_idx >= ray_count_int:
        return
    var ray = _camera_ray_single_view(
        camera_params,
        Int32(ray_idx),
        width_px,
        inv_height,
    )
    var result = _trace_cwbvh8_with_stats_in_frame[.WORLD](
        nodes, triangles, root_idx, ray
    )
    _store_camera_hit(result.hit, hits, ray_count_int, ray_idx)
    result.stats.store(stats, ray_idx)


def flatten_triangle_bounds(
    vertices: List[Point3f32[.WORLD]],
) -> Tuple[List[Float32], List[UInt32]]:
    var triangle_count = len(vertices) / 3
    var bounds = List[Float32](capacity=triangle_count * 6)
    var payloads = List[UInt32](capacity=triangle_count)
    for i in range(triangle_count):
        var box = AABB(
            vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]
        )
        bounds.append(box._min.x)
        bounds.append(box._min.y)
        bounds.append(box._min.z)
        bounds.append(box._max.x)
        bounds.append(box._max.y)
        bounds.append(box._max.z)
        payloads.append(UInt32(i))
    return (bounds^, payloads^)
