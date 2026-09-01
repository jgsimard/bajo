from std.math import ceildiv, min
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu import global_idx

from bajo.core import AABB, Affine3f32, Rayf32
from bajo.bvh.constants import (
    TraceMode,
    EMPTY_LANE,
    PrimitiveKind,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)
from bajo.bvh.gpu.blas_storage import GpuBlasSet, GpuBvhLayout
from bajo.bvh.types import BlasDesc, Hit, Instance
from bajo.bvh.tlas_common import (
    finalize_tlas_hit_normal,
    promote_tlas_local_hit,
)
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvh
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.builder.segmented_build import (
    build_single_segment_wide,
    build_single_segment_wide_embedded_leaf1,
)
from bajo.bvh.gpu.camera_launch import (
    validate_camera_launch,
    _camera_ray,
    _camera_ray_single_view,
    _store_camera_hit,
)
from bajo.bvh.gpu.sphere_bvh import _intersect_sphere_leaf
from bajo.bvh.gpu.blas_trace import trace_gpu_blas
from bajo.bvh.gpu.triangle_bvh import (
    _intersect_triangle_leaf,
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
    mode: TraceMode,
    blas_leaf_fn: GpuLeafFn[.LOCAL],
    blas_layout: GpuBvhLayout = .WIDE,
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
    ray: Rayf32[.WORLD],
    mut hit: Hit[.WORLD],
) -> Bool:
    var hit_any = False
    var inst_inv_transform_span = Span(
        unsafe_ptr=inst_inv_transform,
        length=instance_count * Affine3f32.STRIDE,
    )

    var first_inst_idx = UInt32(0)
    comptime if tlas_leaf_width == 1:
        # Leaf1 embeds its sole instance id in wide-node metadata.
        first_inst_idx = leaf_block_idx

    for lane in range(min(tlas_leaf_width, Int(item_count))):
        var inst_idx = first_inst_idx
        comptime if tlas_leaf_width != 1:
            var idx = Int(leaf_block_idx) * tlas_leaf_width + lane
            inst_idx = UInt32(tlas_leaf_instances[unsafe_offset=idx])

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
            var inverse = Affine3f32[.WORLD, .LOCAL].load(
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
            var local_hit = trace_gpu_blas[
                .LOCAL,
                blas_node_width,
                mode,
                blas_leaf_fn,
                blas_layout,
            ](local_nodes, local_leaves, local_root, local_ray)

            comptime if mode == .ANY_HIT:
                if local_hit.is_occluded():
                    hit = Hit[.WORLD].shadow_hit()
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
    mode: TraceMode,
    blas_leaf_fn: GpuLeafFn[.LOCAL],
    blas_layout: GpuBvhLayout = .WIDE,
](
    state: GpuTlasLeafState,
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: Rayf32[.WORLD],
    mut hit: Hit[.WORLD],
) -> Bool:
    return _intersect_tlas_instance_block[
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        mode,
        blas_leaf_fn,
        blas_layout,
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
    mode: TraceMode,
    blas_leaf_fn: GpuLeafFn[.LOCAL],
    blas_layout: GpuBvhLayout = .WIDE,
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
    ray: Rayf32[.WORLD],
) -> Hit[.WORLD]:
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
        .WORLD,
        tlas_node_width,
        mode,
        GpuTlasLeafState,
        _intersect_tlas_leaf_state[
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            mode,
            blas_leaf_fn,
            blas_layout,
        ],
        tlas_node_width == 2 and mode == .CLOSEST_HIT,
    ](tlas_wide_nodes, leaf_state, tlas_root_idx, ray)
    comptime if mode == .CLOSEST_HIT:
        if hit.is_hit():
            var inverse_span = Span(
                unsafe_ptr=inst_inv_transform,
                length=instance_count * Affine3f32.STRIDE,
            )
            var inverse = Affine3f32[.WORLD, .LOCAL].load(
                inverse_span, Int(hit.inst) * Affine3f32.STRIDE
            )
            finalize_tlas_hit_normal(hit, inverse)
    return hit


def _trace_tlas_camera_kernel[
    kind: PrimitiveKind,
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    blas_layout: GpuBvhLayout = .WIDE,
    single_view: Bool = False,
](
    tlas_wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    tlas_leaf_instances: Pointer[UInt32, ImmutAnyOrigin],
    inst_inv_transform: Pointer[Float32, ImmutAnyOrigin],
    inst_blas_indices: Pointer[UInt32, ImmutAnyOrigin],
    blas_descs: Pointer[UInt32, ImmutAnyOrigin],
    blas_wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    blas_leaves: Pointer[Float32, ImmutAnyOrigin],
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
    var ray_idx = global_idx.x
    if ray_idx >= ray_count_int:
        return

    var ray: Rayf32[.WORLD]
    comptime if single_view:
        ray = _camera_ray_single_view(
            camera_params,
            Int32(ray_idx),
            width,
            inv_height,
        )
    else:
        ray = _camera_ray(
            camera_params,
            ray_count_int,
            ray_idx,
            Int(width),
            Int(height),
            inv_height,
        )

    comptime assert kind == .TRIANGLE or kind == .SPHERE
    comptime if kind == .SPHERE:
        comptime assert blas_layout == .WIDE
    comptime trace_ray = (
        _trace_tlas_ray[
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            .CLOSEST_HIT,
            _intersect_sphere_leaf[
                .LOCAL,
                blas_leaf_width,
                .CLOSEST_HIT,
            ],
        ] if kind
        == .SPHERE else _trace_tlas_ray[
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            .CLOSEST_HIT,
            _intersect_triangle_leaf[
                .LOCAL,
                blas_leaf_width,
                .CLOSEST_HIT,
                blas_leaf_width > blas_node_width or blas_leaf_width == 8,
            ],
            blas_layout,
        ]
    )
    var hit = trace_ray(
        tlas_wide_nodes,
        tlas_leaf_instances,
        inst_inv_transform,
        inst_blas_indices,
        blas_descs,
        blas_wide_nodes,
        blas_leaves,
        Int(instance_count),
        Int(blas_count),
        tlas_root_idx,
        ray,
    )
    _store_camera_hit(hit, hits, ray_count_int, ray_idx)


@fieldwise_init
struct GpuTlas[
    kind: PrimitiveKind,
    tlas_node_width: SIMDLength,
    blas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength = tlas_node_width,
    blas_leaf_width: SIMDLength = blas_node_width,
    blas_layout: GpuBvhLayout = GpuBvhLayout.WIDE,
]:
    """Primitive-typed TLAS owning its tree and instance buffers.

    Instance leaves are packed by the generic wide collapse:
    `_tree.leaf_block_indices[leaf_block * tlas_leaf_width + lane]` stores the
    instance id.
    """

    var _tree: GpuWideBoundsBvh[Self.tlas_node_width, Self.tlas_leaf_width]
    var _inst_inv_transform: DeviceBuffer[.float32]
    var _inst_blas_indices: DeviceBuffer[.uint32]
    var _inst_count: Int

    def launch_camera(
        self,
        ctx: DeviceContext,
        blases: GpuBlasSet[
            Self.kind,
            Self.blas_layout,
            Self.blas_node_width,
            Self.blas_leaf_width,
        ],
        d_camera_params: DeviceBuffer[.float32],
        d_hits: DeviceBuffer[.float32],
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
        if ray_count == cwidth * cheight:
            comptime single_view_kernel = _trace_tlas_camera_kernel[
                Self.kind,
                Self.tlas_node_width,
                Self.tlas_leaf_width,
                Self.blas_node_width,
                Self.blas_leaf_width,
                Self.blas_layout,
                True,
            ]
            ctx.enqueue_function[single_view_kernel](
                self._tree.wide_nodes,
                self._tree.leaf_block_indices,
                self._inst_inv_transform,
                self._inst_blas_indices,
                blases.descs,
                blases.nodes,
                blases.leaves,
                self._tree.root_idx,
                d_camera_params,
                d_hits,
                Int32(self._inst_count),
                Int32(blases.blas_count),
                Int32(ray_count),
                Int32(cwidth),
                Int32(cheight),
                Float32(1.0) / Float32(cheight),
                grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            return

        comptime kernel = _trace_tlas_camera_kernel[
            Self.kind,
            Self.tlas_node_width,
            Self.tlas_leaf_width,
            Self.blas_node_width,
            Self.blas_leaf_width,
            Self.blas_layout,
        ]
        ctx.enqueue_function[kernel](
            self._tree.wide_nodes,
            self._tree.leaf_block_indices,
            self._inst_inv_transform,
            self._inst_blas_indices,
            blases.descs,
            blases.nodes,
            blases.leaves,
            self._tree.root_idx,
            d_camera_params,
            d_hits,
            Int32(self._inst_count),
            Int32(blases.blas_count),
            Int32(ray_count),
            Int32(cwidth),
            Int32(cheight),
            Float32(1.0) / Float32(cheight),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


def _build_gpu_tlas[
    kind: PrimitiveKind,
    tlas_node_width: SIMDLength,
    blas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    method: GpuBvhBuildMethod,
    blas_layout: GpuBvhLayout,
](
    mut ctx: DeviceContext,
    instances: ImmSpan[Instance, _],
    mut timings: GpuBuildTimings,
    measure_build: Bool,
) raises -> GpuTlas[
    kind,
    tlas_node_width,
    blas_node_width,
    tlas_leaf_width,
    blas_leaf_width,
    blas_layout,
]:
    var inst_count = len(instances)
    debug_assert["safe", _use_compiler_assume=True](
        inst_count > 0, "passed empty input."
    )
    var leaf_bounds = List[Float32](capacity=inst_count * AABB[.WORLD].STRIDE)
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

    var d_leaf_bounds = upload_list(ctx, leaf_bounds)
    var d_payloads = upload_list(ctx, payloads)
    var tree: GpuWideBoundsBvh[
        tlas_node_width, tlas_leaf_width, Int(tlas_leaf_width)
    ]
    comptime if tlas_leaf_width == 1:
        tree = build_single_segment_wide_embedded_leaf1[
            tlas_node_width,
            tlas_leaf_width,
            Int(tlas_leaf_width),
            method,
        ](
            ctx,
            inst_count,
            d_leaf_bounds^,
            d_payloads^,
            timings,
            measure_build,
        )
    else:
        tree = build_single_segment_wide[
            tlas_node_width,
            tlas_leaf_width,
            Int(tlas_leaf_width),
            method,
            True,
            False,
        ](
            ctx,
            inst_count,
            d_leaf_bounds^,
            d_payloads^,
            timings,
            measure_build,
        )
    var inst_inv_transform = upload_list(ctx, inv_transforms)
    var inst_blas_indices = upload_list(ctx, blas_indices)
    return GpuTlas[
        kind,
        tlas_node_width,
        blas_node_width,
        tlas_leaf_width,
        blas_leaf_width,
        blas_layout,
    ](
        tree^,
        inst_inv_transform^,
        inst_blas_indices^,
        inst_count,
    )


def build_gpu_tlas[
    kind: PrimitiveKind,
    tlas_node_width: SIMDLength,
    blas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength = tlas_node_width,
    blas_leaf_width: SIMDLength = blas_node_width,
    method: GpuBvhBuildMethod = .LBVH,
    blas_layout: GpuBvhLayout = GpuBvhLayout.WIDE,
](
    mut ctx: DeviceContext,
    instances: ImmSpan[Instance, _],
) raises -> GpuTlas[
    kind,
    tlas_node_width,
    blas_node_width,
    tlas_leaf_width,
    blas_leaf_width,
    blas_layout,
]:
    var timings = GpuBuildTimings(0, 0, 0, 0, 0, 0, 0)
    return _build_gpu_tlas[
        kind,
        tlas_node_width,
        blas_node_width,
        tlas_leaf_width,
        blas_leaf_width,
        method,
        blas_layout,
    ](ctx, instances, timings, False)


def build_gpu_tlas_measured[
    kind: PrimitiveKind,
    tlas_node_width: SIMDLength,
    blas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength = tlas_node_width,
    blas_leaf_width: SIMDLength = blas_node_width,
    method: GpuBvhBuildMethod = .LBVH,
    blas_layout: GpuBvhLayout = GpuBvhLayout.WIDE,
](
    mut ctx: DeviceContext,
    instances: ImmSpan[Instance, _],
    mut timings: GpuBuildTimings,
) raises -> GpuTlas[
    kind,
    tlas_node_width,
    blas_node_width,
    tlas_leaf_width,
    blas_leaf_width,
    blas_layout,
]:
    return _build_gpu_tlas[
        kind,
        tlas_node_width,
        blas_node_width,
        tlas_leaf_width,
        blas_leaf_width,
        method,
        blas_layout,
    ](ctx, instances, timings, True)
