from std.math import ceildiv
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import global_idx

from bajo.bvh.constants import (
    EMPTY_LANE,
    Primitive,
    TRACE,
    f32_max,
    SPHERE_LEAF_PACKED_STRIDE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
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
from bajo.bvh.gpu.blas_storage import GpuBlasSet, GpuBvhLayout
from bajo.bvh.types import Hit, Sphere
from bajo.bvh.gpu.wide_layout import (
    GpuCompactWideLayout,
    GpuWideBoundsBvh,
    enqueue_compact_segmented_buffer,
)
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.builder.binary_layout import _segment_for_item
from bajo.bvh.gpu.builder.segmented_build import (
    GpuSegmentedWideBuildTicket,
    enqueue_segmented_wide_build,
)
from bajo.bvh.gpu.blas_finalize import finalize_ordinary_wide_blas_set
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


@fieldwise_init
struct _SphereHostSegments:
    var segments: SegmentOffsets
    var flat_spheres: List[Float32]
    var leaf_bounds: List[Float32]
    var payloads: List[UInt32]


def _append_sphere_record[
    frame: Frame,
](
    sphere: Sphere[frame],
    mut flat_spheres: List[Float32],
    mut leaf_bounds: List[Float32],
    mut payloads: List[UInt32],
):
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


def _flatten_sphere_sets[
    frame: Frame,
](sphere_sets: ImmSpan[List[Sphere[frame]], _],) -> _SphereHostSegments:
    """Prepare the one host representation used by all sphere build owners."""
    var primitive_counts = List[Int](capacity=len(sphere_sets))
    var sphere_count = 0
    for spheres in sphere_sets:
        primitive_counts.append(len(spheres))
        sphere_count += len(spheres)

    var flat_spheres = List[Float32](capacity=sphere_count * Sphere.STRIDE)
    var leaf_bounds = List[Float32](capacity=sphere_count * AABB[frame].STRIDE)
    var payloads = List[UInt32](capacity=sphere_count)
    for spheres in sphere_sets:
        for sphere in spheres:
            _append_sphere_record(sphere, flat_spheres, leaf_bounds, payloads)
    return _SphereHostSegments(
        SegmentOffsets.from_counts(primitive_counts^),
        flat_spheres^,
        leaf_bounds^,
        payloads^,
    )


def _flatten_sphere_segment[
    frame: Frame,
](spheres: ImmSpan[Sphere[frame], _]) -> _SphereHostSegments:
    """Prepare one segment without copying it through a nested owner."""
    var flat_spheres = List[Float32](capacity=len(spheres) * Sphere.STRIDE)
    var leaf_bounds = List[Float32](capacity=len(spheres) * AABB[frame].STRIDE)
    var payloads = List[UInt32](capacity=len(spheres))
    for sphere in spheres:
        _append_sphere_record(sphere, flat_spheres, leaf_bounds, payloads)
    return _SphereHostSegments(
        SegmentOffsets.single(len(spheres)),
        flat_spheres^,
        leaf_bounds^,
        payloads^,
    )


@fieldwise_init
struct _SegmentedSphereWideBuild[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    build_method: GpuBvhBuildMethod,
]:
    var hierarchy: GpuSegmentedWideBuildTicket[
        Self.node_width,
        Self.leaf_width,
        Int(Self.leaf_width),
        Self.build_method,
        True,
        False,
    ]
    var leaf_spheres: DeviceBuffer[DType.float32]
    var leaf_pack_start_ns: Int

    def into_blas_set(
        deinit self, mut ctx: DeviceContext
    ) raises -> GpuBlasSet[
        Primitive.SPHERE,
        GpuBvhLayout.WIDE,
        Self.node_width,
        Self.leaf_width,
    ]:
        """Finalize the adapter as a descriptor-backed sphere BLAS set."""
        return finalize_ordinary_wide_blas_set[
            Self.node_width,
            Self.leaf_width,
            Self.build_method,
            Primitive.SPHERE,
            GpuBvhLayout.WIDE,
            SPHERE_LEAF_PACKED_STRIDE,
        ](ctx, self.hierarchy^, self.leaf_spheres^)

    def into_bvh(
        deinit self,
        mut ctx: DeviceContext,
        mut timings: GpuBuildTimings,
        measure_build: Bool,
    ) raises -> GpuSphereBvh[Self.frame, Self.node_width, Self.leaf_width]:
        """Finalize the adapter's only segment as a standalone sphere BVH."""
        if measure_build:
            ctx.synchronize()
        else:
            self.hierarchy.wait(ctx)
        ref wide = self.hierarchy.wide
        var layout = GpuCompactWideLayout(
            ctx, wide.node_counts, wide.leaf_block_counts, 1
        )
        var compact_leaves = enqueue_compact_segmented_buffer[
            DType.float32,
            Self.leaf_width * SPHERE_LEAF_PACKED_STRIDE,
        ](
            ctx,
            self.leaf_spheres,
            wide.leaf_block_segment_offsets,
            layout.leaf_block_segment_offsets,
            layout.leaf_block_segments.item_count(),
            1,
        )
        var tree = self.hierarchy^.take_single_segment_synchronized(
            ctx, timings
        )
        if measure_build:
            timings.leaf_pack_ns = Int(
                perf_counter_ns() - self.leaf_pack_start_ns
            )
        return GpuSphereBvh[Self.frame, Self.node_width, Self.leaf_width](
            tree^, compact_leaves^
        )


def _enqueue_segmented_sphere_wide[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    build_method: GpuBvhBuildMethod,
](
    mut ctx: DeviceContext,
    var inputs: _SphereHostSegments,
    measure_build: Bool = False,
) raises -> _SegmentedSphereWideBuild[
    frame, node_width, leaf_width, build_method
]:
    """Run the one ordinary-wide sphere adapter for any segment count."""
    var sphere_count = inputs.segments.item_count()
    debug_assert["safe", _use_compiler_assume=True](
        sphere_count > 0, "sphere wide build requires at least one primitive"
    )
    var d_spheres = upload_list(ctx, inputs.flat_spheres)
    var hierarchy = enqueue_segmented_wide_build[
        node_width, leaf_width, Int(leaf_width), build_method, True
    ](
        ctx,
        inputs.segments,
        upload_list(ctx, inputs.leaf_bounds),
        upload_list(ctx, inputs.payloads),
        measure_build,
    )
    var leaf_lane_capacity = (
        hierarchy.wide.leaf_block_segments.item_count() * leaf_width
    )
    var leaf_spheres = ctx.enqueue_create_buffer[DType.float32](
        leaf_lane_capacity * SPHERE_LEAF_PACKED_STRIDE
    )
    if measure_build:
        hierarchy.wait(ctx)
    var leaf_pack_start_ns = Int(0)
    if measure_build:
        leaf_pack_start_ns = perf_counter_ns()
    ctx.enqueue_function[pack_segmented_sphere_leaf_lanes_kernel[leaf_width]](
        _device_span[mut=False](d_spheres),
        _device_span[mut=False](hierarchy.binary.segment_offsets),
        _device_span[mut=False](hierarchy.wide.leaf_block_segment_offsets),
        hierarchy.wide.leaf_block_indices,
        hierarchy.wide.leaf_block_counts,
        leaf_spheres,
        Int32(leaf_lane_capacity),
        grid_dim=ceildiv(leaf_lane_capacity, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    return _SegmentedSphereWideBuild[
        frame, node_width, leaf_width, build_method
    ](hierarchy^, leaf_spheres^, leaf_pack_start_ns)


def build_sphere_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
](
    mut ctx: DeviceContext,
    sphere_sets: ImmSpan[List[Sphere[Frame.LOCAL]], _],
) raises -> GpuBlasSet[
    Primitive.SPHERE, GpuBvhLayout.WIDE, node_width, leaf_width
]:
    debug_assert["safe", _use_compiler_assume=True](len(sphere_sets) > 0)
    var inputs = _flatten_sphere_sets(sphere_sets)
    if inputs.segments.item_count() == 0:
        return GpuBlasSet[
            Primitive.SPHERE, GpuBvhLayout.WIDE, node_width, leaf_width
        ].empty(ctx, len(sphere_sets))
    var adapter = _enqueue_segmented_sphere_wide[
        Frame.LOCAL, node_width, leaf_width, build_method
    ](ctx, inputs^)
    return adapter^.into_blas_set(ctx)


struct GpuSphereBvh[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
]:
    var tree: GpuWideBoundsBvh[Self.node_width, Self.leaf_width]
    var leaf_spheres: DeviceBuffer[DType.float32]

    def __init__(
        out self,
        var tree: GpuWideBoundsBvh[Self.node_width, Self.leaf_width],
        var leaf_spheres: DeviceBuffer[DType.float32],
    ):
        self.tree = tree^
        self.leaf_spheres = leaf_spheres^

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
    var inputs = _flatten_sphere_segment(spheres)
    var adapter = _enqueue_segmented_sphere_wide[
        frame,
        node_width,
        leaf_width,
        GpuBvhBuildMethod.LBVH,
    ](ctx, inputs^, measure_build)
    return adapter^.into_bvh(ctx, timings, measure_build)


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
