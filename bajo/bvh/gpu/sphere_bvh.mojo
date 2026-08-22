from std.math import ceildiv, max
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import global_idx

from bajo.bvh.constants import (
    EMPTY_LANE,
    TRACE,
    f32_max,
    SPHERE_LEAF_PACKED_STRIDE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    WideNode,
)
from bajo.core.utils import min_argmin
from bajo.core import (
    AABB,
    Vec3,
    Point3,
    Point3f32,
    Frame,
    GeoKind,
    normalize,
    Rayf32,
    SegmentOffsets,
)
from bajo.core.intersect import intersect_ray_sphere
from bajo.bvh.types import GpuBlasSet, Hit, Sphere
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvh
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.builder.binary_layout import _segment_for_item
from bajo.bvh.gpu.builder.segmented_build import enqueue_segmented_wide_build
from bajo.bvh.gpu.blas_desc import enqueue_segmented_blas_descriptors
from bajo.bvh.gpu.camera_launch import (
    validate_camera_launch,
    _camera_ray,
    _store_camera_hit,
)
from bajo.bvh.gpu.trace import trace_bounds_bvh
from bajo.bvh.gpu.utils import (
    GpuBuildTimings,
    _device_span,
    upload_list,
)


def build_sphere_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
](
    mut ctx: DeviceContext,
    sphere_sets: ImmSpan[List[Sphere[Frame.LOCAL]], _],
) raises -> GpuBlasSet[node_width, leaf_width]:
    debug_assert["safe", _use_compiler_assume=True](len(sphere_sets) > 0)
    var primitive_counts = List[Int](capacity=len(sphere_sets))
    var total_sphere_count = 0
    for spheres in sphere_sets:
        var sphere_count = len(spheres)
        primitive_counts.append(sphere_count)
        total_sphere_count += sphere_count
    if total_sphere_count == 0:
        return GpuBlasSet[node_width, leaf_width].empty(ctx, len(sphere_sets))

    var flat_spheres = List[Float32](
        capacity=total_sphere_count * Sphere.STRIDE
    )
    var leaf_bounds = List[Float32](
        capacity=total_sphere_count * AABB[Frame.LOCAL].STRIDE
    )
    var payloads = List[UInt32](capacity=total_sphere_count)
    for spheres in sphere_sets:
        for sphere in spheres:
            flat_spheres.append(sphere.center.x)
            flat_spheres.append(sphere.center.y)
            flat_spheres.append(sphere.center.z)
            flat_spheres.append(sphere.radius)
            leaf_bounds.append(sphere.center.x - sphere.radius)
            leaf_bounds.append(sphere.center.y - sphere.radius)
            leaf_bounds.append(sphere.center.z - sphere.radius)
            leaf_bounds.append(sphere.center.x + sphere.radius)
            leaf_bounds.append(sphere.center.y + sphere.radius)
            leaf_bounds.append(sphere.center.z + sphere.radius)
            payloads.append(UInt32(len(payloads)))

    var segments = SegmentOffsets.from_counts(primitive_counts)
    var d_spheres = upload_list(ctx, flat_spheres)
    var build = enqueue_segmented_wide_build[
        node_width, leaf_width, Int(leaf_width), build_method, True
    ](
        ctx,
        segments,
        upload_list(ctx, leaf_bounds),
        upload_list(ctx, payloads),
    )
    ref binary = build.binary
    ref wide = build.wide

    var leaf_lane_capacity = wide.leaf_block_segments.item_count() * leaf_width
    var leaf_spheres = ctx.enqueue_create_buffer[DType.float32](
        wide.leaf_block_segments.item_count()
        * leaf_width
        * SPHERE_LEAF_PACKED_STRIDE
    )
    ctx.enqueue_function[pack_segmented_sphere_leaf_lanes_kernel[leaf_width]](
        _device_span[mut=False](d_spheres),
        _device_span[mut=False](binary.segment_offsets),
        _device_span[mut=False](wide.leaf_block_segment_offsets),
        wide.leaf_block_indices,
        wide.leaf_block_counts,
        leaf_spheres,
        Int32(leaf_lane_capacity),
        grid_dim=ceildiv(leaf_lane_capacity, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )

    var descs = enqueue_segmented_blas_descriptors[
        node_width * WideNode.CHILD_STRIDE,
        leaf_width * SPHERE_LEAF_PACKED_STRIDE,
    ](
        ctx,
        wide.node_segment_offsets,
        wide.leaf_block_segment_offsets,
        binary.segment_offsets,
        wide.node_counts,
        wide.leaf_block_counts,
        segments.segment_count(),
    )

    ctx.synchronize()
    build.finish_synchronized()

    return GpuBlasSet[node_width, leaf_width](
        descs^,
        wide.wide_nodes.copy(),
        leaf_spheres^,
        segments.segment_count(),
    )


struct GpuSphereBvh[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
]:
    var tree: GpuWideBoundsBvh[Self.node_width, Self.leaf_width]
    var leaf_spheres: DeviceBuffer[DType.float32]
    var sphere_count: Int

    def __init__(
        out self,
        var tree: GpuWideBoundsBvh[Self.node_width, Self.leaf_width],
        var leaf_spheres: DeviceBuffer[DType.float32],
        sphere_count: Int,
    ):
        self.tree = tree^
        self.leaf_spheres = leaf_spheres^
        self.sphere_count = sphere_count

    def launch_camera(
        self,
        ctx: DeviceContext,
        d_camera_params: DeviceBuffer[DType.float32],
        d_hits: DeviceBuffer[DType.float32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        comptime assert Self.frame == Frame.WORLD
        validate_camera_launch(
            d_camera_params, d_hits, ray_count, cwidth, cheight
        )
        ctx.enqueue_function[
            trace_sphere_bvh_camera_kernel[Self.node_width, Self.leaf_width]
        ](
            self.tree.wide_nodes,
            self.leaf_spheres,
            self.tree.root_idx,
            d_camera_params,
            d_hits,
            Int32(ray_count),
            Int32(cwidth),
            Int32(cheight),
            Float32(1.0) / Float32(cheight),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


def build_sphere_bvh[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
](
    mut ctx: DeviceContext,
    spheres: ImmSpan[Sphere[frame], _],
) raises -> GpuSphereBvh[frame, node_width, leaf_width]:
    var timings = GpuBuildTimings(0, 0, 0, 0, 0, 0, 0)
    return _build_sphere_bvh[frame, node_width, leaf_width](
        ctx, spheres, timings, False
    )


def build_sphere_bvh_measured[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
](
    mut ctx: DeviceContext,
    spheres: ImmSpan[Sphere[frame], _],
    mut timings: GpuBuildTimings,
) raises -> GpuSphereBvh[frame, node_width, leaf_width]:
    return _build_sphere_bvh[frame, node_width, leaf_width](
        ctx, spheres, timings, True
    )


def _build_sphere_bvh[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    mut ctx: DeviceContext,
    spheres: ImmSpan[Sphere[frame], _],
    mut timings: GpuBuildTimings,
    measure_build: Bool,
) raises -> GpuSphereBvh[frame, node_width, leaf_width]:
    var sphere_count = len(spheres)
    debug_assert["safe", _use_compiler_assume=True](
        sphere_count > 0, "standalone sphere BVH requires nonempty input"
    )
    var flat_spheres = _flatten_spheres(spheres)
    var d_spheres = upload_list(ctx, flat_spheres)
    var leaf_bounds = List[Float32](
        capacity=max(sphere_count, 1) * AABB[frame].STRIDE
    )
    var payloads = List[UInt32](capacity=max(sphere_count, 1))
    for i, s in enumerate(spheres):
        leaf_bounds.append(s.center.x - s.radius)
        leaf_bounds.append(s.center.y - s.radius)
        leaf_bounds.append(s.center.z - s.radius)
        leaf_bounds.append(s.center.x + s.radius)
        leaf_bounds.append(s.center.y + s.radius)
        leaf_bounds.append(s.center.z + s.radius)
        payloads.append(UInt32(i))

    var build = enqueue_segmented_wide_build[
        node_width,
        leaf_width,
        Int(leaf_width),
        GpuBvhBuildMethod.LBVH,
        True,
    ](
        ctx,
        SegmentOffsets.single(sphere_count),
        upload_list(ctx, leaf_bounds),
        upload_list(ctx, payloads),
        measure_build,
    )
    var leaf_lane_capacity = (
        build.wide.leaf_block_segments.item_count() * leaf_width
    )
    var leaf_spheres = ctx.enqueue_create_buffer[DType.float32](
        leaf_lane_capacity * SPHERE_LEAF_PACKED_STRIDE
    )
    if measure_build:
        build.wait(ctx)
    var leaf_pack_start = Int(0)
    if measure_build:
        leaf_pack_start = perf_counter_ns()
    ctx.enqueue_function[pack_segmented_sphere_leaf_lanes_kernel[leaf_width]](
        _device_span[mut=False](d_spheres),
        _device_span[mut=False](build.binary.segment_offsets),
        _device_span[mut=False](build.wide.leaf_block_segment_offsets),
        build.wide.leaf_block_indices,
        build.wide.leaf_block_counts,
        leaf_spheres,
        Int32(leaf_lane_capacity),
        grid_dim=ceildiv(leaf_lane_capacity, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )

    if measure_build:
        ctx.synchronize()
        var leaf_pack_ns = Int(perf_counter_ns() - leaf_pack_start)
        var tree = build^.take_single_segment_synchronized(timings)
        timings.leaf_pack_ns = leaf_pack_ns
        return GpuSphereBvh[frame, node_width, leaf_width](
            tree^, leaf_spheres^, sphere_count
        )

    var tree = build^.wait_into_single_segment(ctx, timings)
    return GpuSphereBvh[frame, node_width, leaf_width](
        tree^, leaf_spheres^, sphere_count
    )


def trace_sphere_bvh_camera_kernel[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    leaf_spheres: Pointer[Float32, ImmutAnyOrigin],
    root_idx: UInt32,
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    ray_count: Int32,
    width_px: Int32,
    height_px: Int32,
    inv_height: Float32,
):
    var ray_count_int = Int(ray_count)
    var width_px_int = Int(width_px)
    var height_px_int = Int(height_px)
    var ray_idx = global_idx.x
    if ray_idx >= ray_count_int:
        return

    var ray = _camera_ray(
        camera_params,
        ray_count_int,
        ray_idx,
        width_px_int,
        height_px_int,
        inv_height,
    )

    # extra distance stack benchmarks positively for sphere BVH2
    # BVH4 and BVH8 retain the lower-memory stack specialization
    var hit = trace_bounds_bvh[
        Frame.WORLD,
        node_width,
        TRACE.CLOSEST_HIT,
        _intersect_sphere_leaf[
            Frame.WORLD,
            leaf_width,
            TRACE.CLOSEST_HIT,
        ],
        True,
        node_width == 2,
    ](
        wide_nodes,
        leaf_spheres,
        root_idx,
        ray,
    )
    _store_camera_hit(hit, hits, ray_count_int, ray_idx)


def _intersect_sphere_leaf[
    frame: Frame,
    width: SIMDLength,
    mode: TRACE,
](
    leaf_spheres: ImmPointer[Float32, _],
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: Rayf32[frame],
    mut hit: Hit[frame],
) -> Bool:
    _ = item_count
    var block_base = Int(leaf_block_idx) * SPHERE_LEAF_PACKED_STRIDE * width
    var leaf_spheres_u32 = leaf_spheres.unsafe_bitcast[UInt32]()

    var center = Point3[DType.float32, frame, width](
        leaf_spheres.unsafe_load[width=width](block_base + 0 * width),
        leaf_spheres.unsafe_load[width=width](block_base + 1 * width),
        leaf_spheres.unsafe_load[width=width](block_base + 2 * width),
    )
    var radius = leaf_spheres.unsafe_load[width=width](block_base + 3 * width)
    var prim_indices = leaf_spheres_u32.unsafe_load[width=width](
        block_base + 4 * width
    )

    var O = ray.origin[width]()
    var D = ray.direction[width]()

    var hit_sphere = intersect_ray_sphere(
        O, D, center, radius, hit.t, ray.t_min
    )
    var valid_lanes = prim_indices.ne(EMPTY_LANE)
    var hit_mask = hit_sphere.mask & valid_lanes

    if not hit_mask.reduce_or():
        return False

    comptime if mode == TRACE.CLOSEST_HIT:
        var _t = hit_mask.select(hit_sphere.t, f32_max)
        var min_t, lane = min_argmin(_t)

        hit.t = min_t
        hit.u = 0.0
        hit.v = 0.0
        hit.inst = EMPTY_LANE
        hit.prim = prim_indices[lane]
        var lane_center = Point3f32[frame](
            center.x[lane], center.y[lane], center.z[lane]
        )
        var p = ray.o + min_t * ray.d
        hit.normal = normalize(p - lane_center).unsafe_convert[
            new_kind=GeoKind.NORMAL
        ]()

    return True


def pack_segmented_sphere_leaf_lanes_kernel[
    width: SIMDLength,
](
    spheres: ImmSpan[Float32, ImmutAnyOrigin],
    primitive_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    leaf_block_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    leaf_block_indices: Pointer[UInt32, ImmutAnyOrigin],
    leaf_block_counts: Pointer[UInt32, ImmutAnyOrigin],
    leaf_spheres: Pointer[Float32, MutAnyOrigin],
    leaf_lane_capacity: Int32,
):
    """Pack all sphere BLAS leaf ranges and restore local primitive IDs."""
    var lane_idx = global_idx.x
    if lane_idx >= Int(leaf_lane_capacity):
        return

    var physical_block = lane_idx / width
    var segment_idx = _segment_for_item(
        leaf_block_segment_offsets, physical_block
    )
    var block_begin = Int(leaf_block_segment_offsets.unsafe_get(segment_idx))
    if physical_block - block_begin >= Int(
        leaf_block_counts[unsafe_offset=segment_idx]
    ):
        return

    var lane = lane_idx % width
    var out_base = physical_block * SPHERE_LEAF_PACKED_STRIDE * width
    var prim = leaf_block_indices[unsafe_offset=lane_idx]
    var leaf_spheres_u32 = leaf_spheres.unsafe_bitcast[UInt32]()
    if prim == EMPTY_LANE:
        leaf_spheres_u32[unsafe_offset=out_base + 4 * width + lane] = prim
        return

    var primitive_begin = primitive_segment_offsets.unsafe_get(segment_idx)
    leaf_spheres_u32[unsafe_offset=out_base + 4 * width + lane] = (
        prim - primitive_begin
    )
    var in_base = Int(prim) * Sphere.STRIDE
    leaf_spheres[
        unsafe_offset=out_base + 0 * width + lane
    ] = spheres.unsafe_get(in_base + 0)
    leaf_spheres[
        unsafe_offset=out_base + 1 * width + lane
    ] = spheres.unsafe_get(in_base + 1)
    leaf_spheres[
        unsafe_offset=out_base + 2 * width + lane
    ] = spheres.unsafe_get(in_base + 2)
    leaf_spheres[
        unsafe_offset=out_base + 3 * width + lane
    ] = spheres.unsafe_get(in_base + 3)


def _flatten_spheres[
    frame: Frame
](spheres: ImmSpan[Sphere[frame], _]) -> List[Float32]:
    var out = List[Float32](capacity=max(len(spheres), 1) * Sphere.STRIDE)
    for sphere in spheres:
        out.append(sphere.center.x)
        out.append(sphere.center.y)
        out.append(sphere.center.z)
        out.append(sphere.radius)
    return out^
