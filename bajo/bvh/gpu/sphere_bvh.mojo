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
)
from bajo.core.intersect import intersect_ray_sphere
from bajo.bvh.types import Sphere, Hit, BlasSet
from bajo.bvh.gpu.bounds_bvh import build_bounds_bvh
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvh
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
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
](
    mut ctx: DeviceContext,
    sphere_sets: List[List[Sphere[Frame.LOCAL]]],
) raises -> BlasSet[node_width, leaf_width]:
    debug_assert["safe", _use_compiler_assume=True](len(sphere_sets) > 0)

    var descs = List[UInt32](capacity=len(sphere_sets) * BlasSet.STRIDE)

    var total_wide_nodes = 0
    var total_leaf_spheres = 0

    # First pass: compute final packed offsets without building/downloading.
    for blas_idx in range(len(sphere_sets)):
        var sphere_count = len(sphere_sets[blas_idx])
        debug_assert["safe", _use_compiler_assume=True](sphere_count > 0)

        var internal_count = sphere_count - 1
        var max_wide_nodes = max(internal_count, 1)
        var max_leaf_blocks = max(sphere_count, 1)

        var wide_node_base = UInt32(total_wide_nodes)
        var leaf_f32_base = UInt32(total_leaf_spheres)

        descs.append(wide_node_base)
        descs.append(leaf_f32_base)

        # Filled after the actual GPU BLAS build.
        descs.append(UInt32(0))  # BlasSet.ROOT_IDX

        descs.append(UInt32(max_wide_nodes))
        descs.append(UInt32(max_leaf_blocks))
        descs.append(UInt32(sphere_count))

        total_wide_nodes += max_wide_nodes * node_width * WideNode.CHILD_STRIDE
        total_leaf_spheres += (
            max_leaf_blocks * leaf_width * SPHERE_LEAF_PACKED_STRIDE
        )

    var wide_nodes = ctx.enqueue_create_buffer[DType.float32](total_wide_nodes)
    var leaf_spheres = ctx.enqueue_create_buffer[DType.float32](
        total_leaf_spheres
    )

    # Second pass: build each BLAS, then copy its device buffers into the
    # final packed device buffers.
    for blas_idx in range(len(sphere_sets)):
        var blas = build_sphere_bvh[Frame.LOCAL, node_width, leaf_width](
            ctx, sphere_sets[blas_idx]
        )

        var desc_base = blas_idx * BlasSet.STRIDE

        descs[desc_base + BlasSet.ROOT_IDX] = blas.tree.root_idx
        descs[desc_base + BlasSet.NODE_COUNT] = UInt32(blas.tree.node_count)
        descs[desc_base + BlasSet.LEAF_BLOCK_COUNT] = UInt32(
            blas.tree.leaf_block_count
        )

        var wide_node_base = Int(descs[desc_base + BlasSet.WIDE_NODE_BASE])
        var leaf_f32_base = Int(descs[desc_base + BlasSet.LEAF_F32_BASE])

        blas.tree.wide_nodes.enqueue_copy_to(
            wide_nodes.unsafe_ptr().unsafe_offset(wide_node_base)
        )
        blas.leaf_spheres.enqueue_copy_to(
            leaf_spheres.unsafe_ptr().unsafe_offset(leaf_f32_base)
        )

        ctx.synchronize()

    return BlasSet[node_width, leaf_width](
        upload_list(ctx, descs),
        wide_nodes,
        leaf_spheres,
        len(sphere_sets),
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
    spheres: List[Sphere[frame]],
) raises -> GpuSphereBvh[
    frame, node_width, leaf_width
]:
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
    spheres: List[Sphere[frame]],
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
    spheres: List[Sphere[frame]],
    mut timings: GpuBuildTimings,
    measure_build: Bool,
) raises -> GpuSphereBvh[frame, node_width, leaf_width]:
    var sphere_count = len(spheres)
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

    var d_payloads = upload_list(ctx, payloads)
    var d_leaf_bounds = upload_list(ctx, leaf_bounds)
    var tree = GpuWideBoundsBvh[node_width, leaf_width](ctx, sphere_count)
    timings = build_bounds_bvh[
        node_width,
        leaf_width,
        Int(leaf_width),
        GpuBvhBuildMethod.LBVH,
        True,
    ](
        ctx,
        tree,
        d_leaf_bounds,
        d_payloads,
        measure_build=measure_build,
    )
    var leaf_block_capacity = max(tree.leaf_block_count, 1)
    var leaf_spheres = ctx.enqueue_create_buffer[DType.float32](
        leaf_block_capacity * leaf_width * SPHERE_LEAF_PACKED_STRIDE
    )
    var start = Int(0)
    if measure_build:
        start = perf_counter_ns()
    var leaf_lane_count = max(tree.leaf_block_count * leaf_width, 1)
    var blocks = ceildiv(leaf_lane_count, GPU_BOUNDS_BVH_BLOCK_SIZE)
    ctx.enqueue_function[pack_sphere_leaf_lanes_kernel[leaf_width]](
        _device_span[mut=False](d_spheres),
        _device_span[mut=False](tree.leaf_block_indices),
        _device_span[mut=True](leaf_spheres),
        grid_dim=blocks,
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    ctx.synchronize()
    if measure_build:
        timings.leaf_pack_ns = Int(perf_counter_ns() - start)
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
    leaf_spheres: Pointer[mut=False, Float32, _],
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


def pack_sphere_leaf_lanes_kernel[
    width: SIMDLength,
](
    spheres: Span[mut=False, Float32, ImmutAnyOrigin],
    leaf_block_indices: Span[mut=False, UInt32, ImmutAnyOrigin],
    leaf_spheres: Span[mut=True, Float32, MutAnyOrigin],
):
    var leaf_lane_count = len(leaf_spheres) / SPHERE_LEAF_PACKED_STRIDE
    var lane_idx = global_idx.x
    if lane_idx >= leaf_lane_count:
        return

    var lane = lane_idx % width
    var block_idx = lane_idx / width

    var out_base = block_idx * SPHERE_LEAF_PACKED_STRIDE * width
    debug_assert["safe", _use_compiler_assume=True](
        lane_idx < len(leaf_block_indices)
        and out_base <= len(leaf_spheres) - SPHERE_LEAF_PACKED_STRIDE * width,
        "packed sphere output block is outside a device span",
    )
    var prim = UInt32(leaf_block_indices.unsafe_get(lane_idx))
    var leaf_spheres_ptr = leaf_spheres.unsafe_ptr()
    var leaf_spheres_u32 = leaf_spheres_ptr.unsafe_bitcast[UInt32]()

    # AoSoA: [block][field][lane]
    # Packed fields:
    #   0..2 = center.xyz
    #   3    = radius
    #   4    = prim id bits
    leaf_spheres_u32[unsafe_offset=out_base + 4 * width + lane] = prim

    # traversal checks packed prim != EMPTY_LANE
    if prim == EMPTY_LANE:
        return

    var in_base = Int(prim) * Sphere.STRIDE
    debug_assert["safe", _use_compiler_assume=True](
        in_base >= 0 and in_base <= len(spheres) - Sphere.STRIDE,
        "sphere input record is outside the sphere span",
    )

    leaf_spheres_ptr[
        unsafe_offset=out_base + 0 * width + lane
    ] = spheres.unsafe_get(in_base + 0)
    leaf_spheres_ptr[
        unsafe_offset=out_base + 1 * width + lane
    ] = spheres.unsafe_get(in_base + 1)
    leaf_spheres_ptr[
        unsafe_offset=out_base + 2 * width + lane
    ] = spheres.unsafe_get(in_base + 2)
    leaf_spheres_ptr[
        unsafe_offset=out_base + 3 * width + lane
    ] = spheres.unsafe_get(in_base + 3)


def _flatten_spheres[
    frame: Frame
](spheres: List[Sphere[frame]]) -> List[Float32]:
    var out = List[Float32](capacity=max(len(spheres), 1) * Sphere.STRIDE)
    for sphere in spheres:
        out.append(sphere.center.x)
        out.append(sphere.center.y)
        out.append(sphere.center.z)
        out.append(sphere.radius)
    return out^
