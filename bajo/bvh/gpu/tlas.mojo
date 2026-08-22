from std.math import ceildiv, min
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import global_idx

from bajo.core import AABB, Affine3f32, Frame, Rayf32
from bajo.bvh.constants import (
    TRACE,
    EMPTY_LANE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)
from bajo.bvh.types import BlasDesc, GpuBlasSet, Hit, Instance
from bajo.bvh.tlas_common import (
    finalize_tlas_hit_normal,
    promote_tlas_local_hit,
)
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvh
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.builder.segmented_build import build_single_segment_wide
from bajo.bvh.gpu.camera_launch import (
    validate_camera_launch,
    _camera_ray,
    _store_camera_hit,
)
from bajo.bvh.gpu.sphere_bvh import _intersect_sphere_leaf
from bajo.bvh.gpu.triangle_bvh import (
    _intersect_triangle_leaf,
    trace_cwbvh8_triangles,
)
from bajo.bvh.gpu.trace import (
    GpuLeafFn,
    trace_bounds_bvh,
    trace_bounds_bvh_state,
)
from bajo.bvh.gpu.utils import GpuBuildTimings, upload_list


@fieldwise_init
struct GpuTlasLeafState(TrivialRegisterPassable):
    """Explicit state consumed by the shared BVH traversal loop."""

    var leaf_instances: Pointer[UInt32, ImmUntrackedOrigin]
    var inst_inv_transform: Pointer[Float32, ImmUntrackedOrigin]
    var inst_blas_indices: Pointer[UInt32, ImmUntrackedOrigin]
    var blas_descs: Pointer[UInt32, ImmUntrackedOrigin]
    var blas_wide_nodes: Pointer[Float32, ImmUntrackedOrigin]
    var blas_leaves: Pointer[Float32, ImmUntrackedOrigin]
    var instance_count: Int
    var blas_count: Int


@always_inline
def _intersect_tlas_instance_block[
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    mode: TRACE,
    blas_leaf_fn: GpuLeafFn[Frame.LOCAL],
    blas_compressed: Bool = False,
](
    tlas_leaf_instances: ImmPointer[UInt32, _],
    inst_inv_transform: ImmPointer[Float32, _],
    inst_blas_indices: ImmPointer[UInt32, _],
    blas_descs: ImmPointer[UInt32, _],
    blas_wide_nodes: ImmPointer[Float32, _],
    blas_leaves: ImmPointer[Float32, _],
    instance_count: Int,
    blas_count: Int,
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: Rayf32[Frame.WORLD],
    mut hit: Hit[Frame.WORLD],
) -> Bool:
    var hit_any = False
    var inst_inv_transform_span = Span(
        unsafe_ptr=inst_inv_transform,
        length=instance_count * Affine3f32.STRIDE,
    )

    for lane in range(min(tlas_leaf_width, Int(item_count))):
        var idx = Int(leaf_block_idx) * tlas_leaf_width + lane
        var inst_idx = UInt32(tlas_leaf_instances[unsafe_offset=idx])

        if inst_idx != EMPTY_LANE:
            debug_assert["safe", _use_compiler_assume=True](
                Int(inst_idx) < instance_count,
                "GPU TLAS instance index is out of range",
            )
            var blas_idx = UInt32(
                inst_blas_indices[unsafe_offset=Int(inst_idx)]
            )
            debug_assert["safe", _use_compiler_assume=True](
                Int(blas_idx) < blas_count,
                "GPU TLAS BLAS index is out of range",
            )
            var blas_desc = BlasDesc.load(blas_descs, blas_idx)
            if blas_desc.prim_count == 0:
                continue
            var transform_base = Int(inst_idx) * Affine3f32.STRIDE
            var inverse = Affine3f32[Frame.WORLD, Frame.LOCAL].load(
                inst_inv_transform_span, transform_base
            )

            var local_ray = inverse.ray(ray, hit.t)

            var local_nodes = blas_wide_nodes.unsafe_offset(
                Int(blas_desc.node_f32_base)
            )
            var local_leaves = blas_leaves.unsafe_offset(
                Int(blas_desc.leaf_f32_base)
            )
            var local_root = blas_desc.root_idx
            var local_hit: Hit[Frame.LOCAL]
            comptime if blas_compressed:
                local_hit = trace_cwbvh8_triangles[Frame.LOCAL, mode](
                    local_nodes, local_leaves, local_root, local_ray
                )
            else:
                local_hit = trace_bounds_bvh[
                    Frame.LOCAL,
                    blas_node_width,
                    mode,
                    blas_leaf_fn,
                ](local_nodes, local_leaves, local_root, local_ray)

            comptime if mode == TRACE.ANY_HIT:
                if local_hit.is_occluded():
                    hit = Hit[Frame.WORLD].shadow_hit()
                    hit.inst = inst_idx
                    return True
            else:
                if promote_tlas_local_hit(local_hit, inst_idx, hit):
                    hit_any = True
    return hit_any


@always_inline
def _intersect_tlas_leaf_state[
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    mode: TRACE,
    blas_leaf_fn: GpuLeafFn[Frame.LOCAL],
    blas_compressed: Bool = False,
](
    state: GpuTlasLeafState,
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: Rayf32[Frame.WORLD],
    mut hit: Hit[Frame.WORLD],
) -> Bool:
    return _intersect_tlas_instance_block[
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        mode,
        blas_leaf_fn,
        blas_compressed,
    ](
        state.leaf_instances,
        state.inst_inv_transform,
        state.inst_blas_indices,
        state.blas_descs,
        state.blas_wide_nodes,
        state.blas_leaves,
        state.instance_count,
        state.blas_count,
        leaf_block_idx,
        item_count,
        ray,
        hit,
    )


def _trace_tlas_ray[
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    mode: TRACE,
    blas_leaf_fn: GpuLeafFn[Frame.LOCAL],
    blas_compressed: Bool = False,
](
    tlas_wide_nodes: ImmPointer[Float32, _],
    tlas_leaf_instances: ImmPointer[UInt32, _],
    inst_inv_transform: ImmPointer[Float32, _],
    inst_blas_indices: ImmPointer[UInt32, _],
    blas_descs: ImmPointer[UInt32, _],
    blas_wide_nodes: ImmPointer[Float32, _],
    blas_leaves: ImmPointer[Float32, _],
    instance_count: Int,
    blas_count: Int,
    tlas_root_idx: UInt32,
    ray: Rayf32[Frame.WORLD],
) -> Hit[Frame.WORLD]:
    var leaf_state = GpuTlasLeafState(
        tlas_leaf_instances.unsafe_origin_cast[ImmUntrackedOrigin](),
        inst_inv_transform.unsafe_origin_cast[ImmUntrackedOrigin](),
        inst_blas_indices.unsafe_origin_cast[ImmUntrackedOrigin](),
        blas_descs.unsafe_origin_cast[ImmUntrackedOrigin](),
        blas_wide_nodes.unsafe_origin_cast[ImmUntrackedOrigin](),
        blas_leaves.unsafe_origin_cast[ImmUntrackedOrigin](),
        instance_count,
        blas_count,
    )
    var hit = trace_bounds_bvh_state[
        Frame.WORLD,
        tlas_node_width,
        mode,
        GpuTlasLeafState,
        _intersect_tlas_leaf_state[
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            mode,
            blas_leaf_fn,
            blas_compressed,
        ],
        tlas_node_width == 2 and mode == TRACE.CLOSEST_HIT,
    ](tlas_wide_nodes, leaf_state, tlas_root_idx, ray)
    comptime if mode == TRACE.CLOSEST_HIT:
        if hit.is_hit():
            var inverse_span = Span(
                unsafe_ptr=inst_inv_transform,
                length=instance_count * Affine3f32.STRIDE,
            )
            var inverse = Affine3f32[Frame.WORLD, Frame.LOCAL].load(
                inverse_span, Int(hit.inst) * Affine3f32.STRIDE
            )
            finalize_tlas_hit_normal(hit, inverse)
    return hit


def trace_triangle_tlas_camera_kernel[
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    blas_compressed: Bool = False,
](
    tlas_wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    tlas_leaf_instances: Pointer[UInt32, ImmutAnyOrigin],
    inst_inv_transform: Pointer[Float32, ImmutAnyOrigin],
    inst_blas_indices: Pointer[UInt32, ImmutAnyOrigin],
    blas_descs: Pointer[UInt32, ImmutAnyOrigin],
    blas_wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    blas_leaf_vertices: Pointer[Float32, ImmutAnyOrigin],
    tlas_root_idx: UInt32,
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    instance_count: Int32,
    blas_count: Int32,
    ray_count: Int32,
    width: Int32,
    height: Int32,
    inv_height: Float32,
):
    var ray_count_int = Int(ray_count)
    var width_int = Int(width)
    var height_int = Int(height)
    var ray_idx = global_idx.x
    if ray_idx >= ray_count_int:
        return

    var ray = _camera_ray(
        camera_params,
        ray_count_int,
        ray_idx,
        width_int,
        height_int,
        inv_height,
    )

    var hit = _trace_tlas_ray[
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        TRACE.CLOSEST_HIT,
        _intersect_triangle_leaf[
            Frame.LOCAL,
            blas_leaf_width,
            TRACE.CLOSEST_HIT,
            blas_leaf_width > blas_node_width or blas_leaf_width == 8,
        ],
        blas_compressed,
    ](
        tlas_wide_nodes,
        tlas_leaf_instances,
        inst_inv_transform,
        inst_blas_indices,
        blas_descs,
        blas_wide_nodes,
        blas_leaf_vertices,
        Int(instance_count),
        Int(blas_count),
        tlas_root_idx,
        ray,
    )
    _store_camera_hit(hit, hits, ray_count_int, ray_idx)


def trace_sphere_tlas_camera_kernel[
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
](
    tlas_wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    tlas_leaf_instances: Pointer[UInt32, ImmutAnyOrigin],
    inst_inv_transform: Pointer[Float32, ImmutAnyOrigin],
    inst_blas_indices: Pointer[UInt32, ImmutAnyOrigin],
    blas_descs: Pointer[UInt32, ImmutAnyOrigin],
    blas_wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    blas_leaf_spheres: Pointer[Float32, ImmutAnyOrigin],
    tlas_root_idx: UInt32,
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    instance_count: Int32,
    blas_count: Int32,
    ray_count: Int32,
    width: Int32,
    height: Int32,
    inv_height: Float32,
):
    var ray_count_int = Int(ray_count)
    var width_int = Int(width)
    var height_int = Int(height)
    var ray_idx = global_idx.x
    if ray_idx >= ray_count_int:
        return

    var ray = _camera_ray(
        camera_params,
        ray_count_int,
        ray_idx,
        width_int,
        height_int,
        inv_height,
    )

    var hit = _trace_tlas_ray[
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        TRACE.CLOSEST_HIT,
        _intersect_sphere_leaf[
            Frame.LOCAL,
            blas_leaf_width,
            TRACE.CLOSEST_HIT,
        ],
    ](
        tlas_wide_nodes,
        tlas_leaf_instances,
        inst_inv_transform,
        inst_blas_indices,
        blas_descs,
        blas_wide_nodes,
        blas_leaf_spheres,
        Int(instance_count),
        Int(blas_count),
        tlas_root_idx,
        ray,
    )
    _store_camera_hit(hit, hits, ray_count_int, ray_idx)


struct GpuTypedTlasCore[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
]:
    """GPU TLAS core shared by typed TLAS wrappers.

    Instance leaves are packed by the generic wide collapse:
    `tree.leaf_block_indices[leaf_block * leaf_width + lane]` stores the
    instance id.
    """

    var tree: GpuWideBoundsBvh[Self.node_width, Self.leaf_width]
    var inst_inv_transform: DeviceBuffer[DType.float32]
    var inst_blas_indices: DeviceBuffer[DType.uint32]
    var inst_count: Int

    def __init__(
        out self,
        var tree: GpuWideBoundsBvh[Self.node_width, Self.leaf_width],
        var inst_inv_transform: DeviceBuffer[DType.float32],
        var inst_blas_indices: DeviceBuffer[DType.uint32],
        inst_count: Int,
    ):
        self.tree = tree^
        self.inst_inv_transform = inst_inv_transform^
        self.inst_blas_indices = inst_blas_indices^
        self.inst_count = inst_count


def build_typed_tlas_core[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
](
    mut ctx: DeviceContext,
    instances: ImmSpan[Instance, _],
) raises -> GpuTypedTlasCore[node_width, leaf_width]:
    var timings = GpuBuildTimings(0, 0, 0, 0, 0, 0, 0)
    return _build_typed_tlas_core[node_width, leaf_width, method](
        ctx, instances, timings, False
    )


def build_typed_tlas_core_measured[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
](
    mut ctx: DeviceContext,
    instances: ImmSpan[Instance, _],
    mut timings: GpuBuildTimings,
) raises -> GpuTypedTlasCore[node_width, leaf_width]:
    return _build_typed_tlas_core[node_width, leaf_width, method](
        ctx, instances, timings, True
    )


def _build_typed_tlas_core[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    method: GpuBvhBuildMethod,
](
    mut ctx: DeviceContext,
    instances: ImmSpan[Instance, _],
    mut timings: GpuBuildTimings,
    measure_build: Bool,
) raises -> GpuTypedTlasCore[node_width, leaf_width]:
    var inst_count = len(instances)
    debug_assert["safe", _use_compiler_assume=True](
        inst_count > 0, "passed empty input."
    )
    var leaf_bounds = List[Float32](
        capacity=inst_count * AABB[Frame.WORLD].STRIDE
    )
    var payloads = List[UInt32](capacity=inst_count)
    var inv_transforms = List[Float32](capacity=inst_count * Affine3f32.STRIDE)
    var blas_indices = List[UInt32](capacity=inst_count)
    for i, inst in enumerate(instances):
        leaf_bounds.append(inst.bounds._min.x)
        leaf_bounds.append(inst.bounds._min.y)
        leaf_bounds.append(inst.bounds._min.z)
        leaf_bounds.append(inst.bounds._max.x)
        leaf_bounds.append(inst.bounds._max.y)
        leaf_bounds.append(inst.bounds._max.z)
        payloads.append(UInt32(i))
        inv_transforms.extend(inst.inv_transform.flatten())
        blas_indices.append(inst.blas_idx)

    var tree = build_single_segment_wide[
        node_width,
        leaf_width,
        Int(leaf_width),
        method,
        True,
    ](
        ctx,
        inst_count,
        upload_list(ctx, leaf_bounds),
        upload_list(ctx, payloads),
        timings,
        measure_build,
    )
    var inst_inv_transform = upload_list(ctx, inv_transforms)
    var inst_blas_indices = upload_list(ctx, blas_indices)
    return GpuTypedTlasCore[node_width, leaf_width](
        tree^,
        inst_inv_transform^,
        inst_blas_indices^,
        inst_count,
    )


struct GpuTriangleTlas[
    tlas_node_width: SIMDLength,
    blas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength = tlas_node_width,
    blas_leaf_width: SIMDLength = blas_node_width,
    blas_compressed: Bool = False,
]:
    """Typed triangle TLAS over a descriptor-backed triangle BLAS set."""

    var core: GpuTypedTlasCore[Self.tlas_node_width, Self.tlas_leaf_width]

    def __init__(
        out self,
        var core: GpuTypedTlasCore[Self.tlas_node_width, Self.tlas_leaf_width],
    ):
        self.core = core^

    def launch_camera(
        self,
        ctx: DeviceContext,
        blases: GpuBlasSet[Self.blas_node_width, Self.blas_leaf_width],
        d_camera_params: DeviceBuffer[DType.float32],
        d_hits: DeviceBuffer[DType.float32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        validate_camera_launch(
            d_camera_params, d_hits, ray_count, cwidth, cheight
        )
        debug_assert["safe", _use_compiler_assume=True](
            blases.blas_count > 0,
            "GPU TLAS requires at least one BLAS descriptor",
        )
        ctx.enqueue_function[
            trace_triangle_tlas_camera_kernel[
                Self.tlas_node_width,
                Self.tlas_leaf_width,
                Self.blas_node_width,
                Self.blas_leaf_width,
                Self.blas_compressed,
            ]
        ](
            self.core.tree.wide_nodes,
            self.core.tree.leaf_block_indices,
            self.core.inst_inv_transform,
            self.core.inst_blas_indices,
            blases.descs,
            blases.nodes,
            blases.leaves,
            self.core.tree.root_idx,
            d_camera_params,
            d_hits,
            Int32(self.core.inst_count),
            Int32(blases.blas_count),
            Int32(ray_count),
            Int32(cwidth),
            Int32(cheight),
            Float32(1.0) / Float32(cheight),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


struct GpuSphereTlas[
    tlas_node_width: SIMDLength,
    blas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength = tlas_node_width,
    blas_leaf_width: SIMDLength = blas_node_width,
]:
    """Typed sphere TLAS over a descriptor-backed sphere BLAS set."""

    var core: GpuTypedTlasCore[Self.tlas_node_width, Self.tlas_leaf_width]

    def __init__(
        out self,
        var core: GpuTypedTlasCore[Self.tlas_node_width, Self.tlas_leaf_width],
    ):
        self.core = core^

    def launch_camera(
        self,
        ctx: DeviceContext,
        blases: GpuBlasSet[Self.blas_node_width, Self.blas_leaf_width],
        d_camera_params: DeviceBuffer[DType.float32],
        d_hits: DeviceBuffer[DType.float32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        validate_camera_launch(
            d_camera_params, d_hits, ray_count, cwidth, cheight
        )
        debug_assert["safe", _use_compiler_assume=True](
            blases.blas_count > 0,
            "GPU TLAS requires at least one BLAS descriptor",
        )
        ctx.enqueue_function[
            trace_sphere_tlas_camera_kernel[
                Self.tlas_node_width,
                Self.tlas_leaf_width,
                Self.blas_node_width,
                Self.blas_leaf_width,
            ]
        ](
            self.core.tree.wide_nodes,
            self.core.tree.leaf_block_indices,
            self.core.inst_inv_transform,
            self.core.inst_blas_indices,
            blases.descs,
            blases.nodes,
            blases.leaves,
            self.core.tree.root_idx,
            d_camera_params,
            d_hits,
            Int32(self.core.inst_count),
            Int32(blases.blas_count),
            Int32(ray_count),
            Int32(cwidth),
            Int32(cheight),
            Float32(1.0) / Float32(cheight),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


def build_triangle_tlas[
    tlas_node_width: SIMDLength,
    blas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength = tlas_node_width,
    blas_leaf_width: SIMDLength = blas_node_width,
    method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
    blas_compressed: Bool = False,
](
    mut ctx: DeviceContext,
    instances: ImmSpan[Instance, _],
) raises -> GpuTriangleTlas[
    tlas_node_width,
    blas_node_width,
    tlas_leaf_width,
    blas_leaf_width,
    blas_compressed,
]:
    var core = build_typed_tlas_core[tlas_node_width, tlas_leaf_width, method](
        ctx, instances
    )
    return GpuTriangleTlas[
        tlas_node_width,
        blas_node_width,
        tlas_leaf_width,
        blas_leaf_width,
        blas_compressed,
    ](core^)


def build_triangle_tlas_measured[
    tlas_node_width: SIMDLength,
    blas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength = tlas_node_width,
    blas_leaf_width: SIMDLength = blas_node_width,
    method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
    blas_compressed: Bool = False,
](
    mut ctx: DeviceContext,
    instances: ImmSpan[Instance, _],
    mut timings: GpuBuildTimings,
) raises -> GpuTriangleTlas[
    tlas_node_width,
    blas_node_width,
    tlas_leaf_width,
    blas_leaf_width,
    blas_compressed,
]:
    var core = build_typed_tlas_core_measured[
        tlas_node_width, tlas_leaf_width, method
    ](ctx, instances, timings)
    return GpuTriangleTlas[
        tlas_node_width,
        blas_node_width,
        tlas_leaf_width,
        blas_leaf_width,
        blas_compressed,
    ](core^)


def build_sphere_tlas[
    tlas_node_width: SIMDLength,
    blas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength = tlas_node_width,
    blas_leaf_width: SIMDLength = blas_node_width,
    method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
](
    mut ctx: DeviceContext,
    instances: ImmSpan[Instance, _],
) raises -> GpuSphereTlas[
    tlas_node_width, blas_node_width, tlas_leaf_width, blas_leaf_width
]:
    var core = build_typed_tlas_core[tlas_node_width, tlas_leaf_width, method](
        ctx, instances
    )
    return GpuSphereTlas[
        tlas_node_width, blas_node_width, tlas_leaf_width, blas_leaf_width
    ](core^)


def build_sphere_tlas_measured[
    tlas_node_width: SIMDLength,
    blas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength = tlas_node_width,
    blas_leaf_width: SIMDLength = blas_node_width,
    method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
](
    mut ctx: DeviceContext,
    instances: ImmSpan[Instance, _],
    mut timings: GpuBuildTimings,
) raises -> GpuSphereTlas[
    tlas_node_width, blas_node_width, tlas_leaf_width, blas_leaf_width
]:
    var core = build_typed_tlas_core_measured[
        tlas_node_width, tlas_leaf_width, method
    ](ctx, instances, timings)
    return GpuSphereTlas[
        tlas_node_width, blas_node_width, tlas_leaf_width, blas_leaf_width
    ](core^)
