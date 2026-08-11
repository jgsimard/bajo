from std.math import ceildiv, max, min
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import global_idx

from bajo.core import AABB, Affine3f32, Frame, Rayf32
from bajo.bvh.constants import (
    TRACE,
    GPU_STACK_SIZE,
    EMPTY_LANE,
    f32_max,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)
from bajo.bvh.types import Hit, Instance, BlasSet
from bajo.bvh.gpu.bounds_bvh import (
    GpuBoundsBvh,
    _intersect_wide_node,
)
from bajo.bvh.gpu.wide_meta import _wide_meta_count, _wide_meta_data
from bajo.bvh.camera import Camera
from bajo.bvh.gpu.sphere_bvh import _intersect_sphere_leaf
from bajo.bvh.gpu.triangle_bvh import _intersect_triangle_leaf
from bajo.bvh.gpu.trace import trace_bounds_bvh
from bajo.bvh.gpu.utils import GpuBuildTimings, upload_list


comptime BlasLeafFn[frame: Frame] = def(
    Pointer[mut=False, Float32, _],
    UInt32,
    Rayf32[frame],
    mut Hit[frame],
) capturing -> Bool


def _flatten_instance_inv_transforms(
    instances: List[Instance],
) -> List[Float32]:
    debug_assert["safe", _use_compiler_assume=True](len(instances) > 0)

    var out = List[Float32](
        capacity=len(instances) * Affine3f32[Frame.WORLD, Frame.LOCAL].STRIDE
    )
    for instance in instances:
        out.extend(instance.inv_transform.flatten())
    return out^


def _flatten_instance_transforms(
    instances: List[Instance],
) -> List[Float32]:
    debug_assert["safe", _use_compiler_assume=True](len(instances) > 0)

    var out = List[Float32](
        capacity=len(instances) * Affine3f32[Frame.LOCAL, Frame.WORLD].STRIDE
    )
    for instance in instances:
        out.extend(instance.transform.flatten())
    return out^


def _flatten_instance_blas_indices(
    instances: List[Instance],
) -> List[UInt32]:
    debug_assert["safe", _use_compiler_assume=True](len(instances) > 0)
    return [instance.blas_idx for instance in instances]


@always_inline
def _intersect_tlas_instance_block[
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    mode: TRACE,
    direct_continuation: Bool,
    blas_leaf_fn: BlasLeafFn[Frame.LOCAL],
](
    tlas_leaf_instances: Pointer[mut=False, UInt32, _],
    inst_transform: Pointer[mut=False, Float32, _],
    inst_inv_transform: Pointer[mut=False, Float32, _],
    inst_blas_indices: Pointer[mut=False, UInt32, _],
    blas_descs: Pointer[mut=False, UInt32, _],
    blas_wide_nodes: Pointer[mut=False, Float32, _],
    blas_leaves: Pointer[mut=False, Float32, _],
    instance_count: Int,
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: Rayf32[Frame.WORLD],
    mut hit: Hit[Frame.WORLD],
) -> Bool:
    var hit_any = False
    var inst_transform_span = Span(
        unsafe_ptr=inst_transform,
        length=instance_count * Affine3f32.STRIDE,
    )
    var inst_inv_transform_span = Span(
        unsafe_ptr=inst_inv_transform,
        length=instance_count * Affine3f32.STRIDE,
    )

    for lane in range(min(tlas_leaf_width, Int(item_count))):
        var idx = Int(leaf_block_idx) * tlas_leaf_width + lane
        var inst_idx = UInt32(tlas_leaf_instances[unsafe_offset=idx])

        if inst_idx != EMPTY_LANE:
            var blas_idx = UInt32(
                inst_blas_indices[unsafe_offset=Int(inst_idx)]
            )
            var desc_base = Int(blas_idx) * BlasSet.STRIDE
            var transform_base = Int(inst_idx) * Affine3f32.STRIDE
            var inverse = Affine3f32[Frame.WORLD, Frame.LOCAL].load(
                inst_inv_transform_span, transform_base
            )

            var local_ray = inverse.ray(ray, hit.t)

            var local_hit = trace_bounds_bvh[
                Frame.LOCAL,
                blas_node_width,
                mode,
                blas_leaf_fn,
                True,
                False,
                direct_continuation,
            ](
                blas_wide_nodes.unsafe_offset(
                    Int(
                        blas_descs[
                            unsafe_offset=desc_base + BlasSet.WIDE_NODE_BASE
                        ]
                    )
                ),
                blas_leaves.unsafe_offset(
                    Int(
                        blas_descs[
                            unsafe_offset=desc_base + BlasSet.LEAF_F32_BASE
                        ]
                    )
                ),
                UInt32(blas_descs[unsafe_offset=desc_base + BlasSet.ROOT_IDX]),
                local_ray,
            )

            comptime if mode == TRACE.ANY_HIT:
                if local_hit.is_occluded():
                    hit = Hit[Frame.WORLD].shadow_hit()
                    hit.inst = inst_idx
                    return True
            else:
                if local_hit.t < hit.t and local_hit.prim != EMPTY_LANE:
                    var transform = Affine3f32[Frame.LOCAL, Frame.WORLD].load(
                        inst_transform_span,
                        transform_base,
                    )

                    hit.t = local_hit.t
                    hit.u = local_hit.u
                    hit.v = local_hit.v
                    hit.prim = local_hit.prim
                    hit.inst = inst_idx
                    hit.normal = transform.normal(local_hit.normal, inverse)
                    hit_any = True
    return hit_any


def _trace_tlas_ray[
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    mode: TRACE,
    direct_continuation: Bool,
    blas_leaf_fn: BlasLeafFn[Frame.LOCAL],
](
    tlas_wide_nodes: Pointer[mut=False, Float32, _],
    tlas_leaf_instances: Pointer[mut=False, UInt32, _],
    inst_transform: Pointer[mut=False, Float32, _],
    inst_inv_transform: Pointer[mut=False, Float32, _],
    inst_blas_indices: Pointer[mut=False, UInt32, _],
    blas_descs: Pointer[mut=False, UInt32, _],
    blas_wide_nodes: Pointer[mut=False, Float32, _],
    blas_leaves: Pointer[mut=False, Float32, _],
    instance_count: Int,
    tlas_root_idx: UInt32,
    ray: Rayf32[Frame.WORLD],
) -> Hit[Frame.WORLD]:
    var hit = Hit[Frame.WORLD].miss(ray.t_max)

    var stack = Array[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var current = tlas_root_idx

    while True:
        var node_hit = _intersect_wide_node[Frame.WORLD, tlas_node_width](
            tlas_wide_nodes,
            current,
            ray,
            hit.t,
        )
        var bounds_hit = node_hit.bounds_hit

        var child_valid = Array[Bool, tlas_node_width](fill=False)
        var child_data = Array[UInt32, tlas_node_width](fill=0)
        var child_t = Array[Float32, tlas_node_width](fill=0.0)

        comptime for node_lane in range(tlas_node_width):
            var meta = node_hit.meta[node_lane]
            var count = _wide_meta_count(meta)

            if count != EMPTY_LANE and bounds_hit.mask[node_lane]:
                var data = _wide_meta_data(meta)

                if count == 0:
                    child_valid[node_lane] = True
                    child_data[node_lane] = data
                    child_t[node_lane] = bounds_hit.t[node_lane]
                else:
                    var leaf_hit = _intersect_tlas_instance_block[
                        tlas_leaf_width,
                        blas_node_width,
                        blas_leaf_width,
                        mode,
                        direct_continuation,
                        blas_leaf_fn,
                    ](
                        tlas_leaf_instances,
                        inst_transform,
                        inst_inv_transform,
                        inst_blas_indices,
                        blas_descs,
                        blas_wide_nodes,
                        blas_leaves,
                        instance_count,
                        data,
                        count,
                        ray,
                        hit,
                    )

                    comptime if mode == TRACE.ANY_HIT:
                        if leaf_hit:
                            return hit

        # Keep the nearest child as the direct continuation. Only deferred
        # siblings consume stack entries.
        var nearest_lane = -1
        var nearest_t = f32_max
        comptime for lane in range(tlas_node_width):
            if child_valid[lane] and child_t[lane] < nearest_t:
                nearest_lane = lane
                nearest_t = child_t[lane]

        comptime for lane in range(tlas_node_width):
            if child_valid[lane] and lane != nearest_lane:
                comptime if mode != TRACE.ANY_HIT:
                    if child_t[lane] > hit.t:
                        continue

                debug_assert["safe", _use_compiler_assume=True](
                    stack_ptr < GPU_STACK_SIZE,
                    "GPU TLAS traversal stack overflow",
                )
                stack[stack_ptr] = child_data[lane]
                stack_ptr += 1

        if nearest_lane != -1:
            comptime if mode != TRACE.ANY_HIT:
                if nearest_t > hit.t:
                    nearest_lane = -1

            if nearest_lane != -1:
                comptime if direct_continuation:
                    current = child_data[nearest_lane]
                    continue
                else:
                    debug_assert["safe", _use_compiler_assume=True](
                        stack_ptr < GPU_STACK_SIZE,
                        "GPU TLAS traversal stack overflow",
                    )
                    stack[stack_ptr] = child_data[nearest_lane]
                    stack_ptr += 1

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        current = stack[stack_ptr]

    return hit


def trace_triangle_tlas_camera_kernel[
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
](
    tlas_wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    tlas_leaf_instances: Pointer[UInt32, ImmutAnyOrigin],
    inst_transform: Pointer[Float32, ImmutAnyOrigin],
    inst_inv_transform: Pointer[Float32, ImmutAnyOrigin],
    inst_blas_indices: Pointer[UInt32, ImmutAnyOrigin],
    blas_descs: Pointer[UInt32, ImmutAnyOrigin],
    blas_wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    blas_leaf_vertices: Pointer[Float32, ImmutAnyOrigin],
    tlas_root_idx: UInt32,
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    instance_count: Int32,
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

    var pixels_per_view = width_int * height_int
    var view_idx = ray_idx / pixels_per_view
    var local_idx = ray_idx - view_idx * pixels_per_view
    var px_i = local_idx % width_int
    var py_i = local_idx / width_int

    var camera_params_span = Span(
        unsafe_ptr=camera_params,
        length=ceildiv(ray_count_int, pixels_per_view) * Camera.STRIDE,
    )
    var camera = Camera(camera_params_span, view_idx * Camera.STRIDE)
    var ray = camera.make_ray_raster(px_i, py_i, width_int, inv_height)

    var hit = _trace_tlas_ray[
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        TRACE.CLOSEST_HIT,
        True,
        _intersect_triangle_leaf[
            Frame.LOCAL,
            blas_leaf_width,
            TRACE.CLOSEST_HIT,
            blas_leaf_width > blas_node_width or blas_leaf_width == 8,
        ],
    ](
        tlas_wide_nodes,
        tlas_leaf_instances,
        inst_transform,
        inst_inv_transform,
        inst_blas_indices,
        blas_descs,
        blas_wide_nodes,
        blas_leaf_vertices,
        Int(instance_count),
        tlas_root_idx,
        ray,
    )
    var hits_span = Span(unsafe_ptr=hits, length=ray_count_int * Hit.STRIDE)
    hit._store_unchecked(hits_span, ray_idx)


def trace_sphere_tlas_camera_kernel[
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
](
    tlas_wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    tlas_leaf_instances: Pointer[UInt32, ImmutAnyOrigin],
    inst_transform: Pointer[Float32, ImmutAnyOrigin],
    inst_inv_transform: Pointer[Float32, ImmutAnyOrigin],
    inst_blas_indices: Pointer[UInt32, ImmutAnyOrigin],
    blas_descs: Pointer[UInt32, ImmutAnyOrigin],
    blas_wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    blas_leaf_spheres: Pointer[Float32, ImmutAnyOrigin],
    tlas_root_idx: UInt32,
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    instance_count: Int32,
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

    var pixels_per_view = width_int * height_int
    var view_idx = ray_idx / pixels_per_view
    var local_idx = ray_idx - view_idx * pixels_per_view
    var px_i = local_idx % width_int
    var py_i = local_idx / width_int

    var camera_params_span = Span(
        unsafe_ptr=camera_params,
        length=ceildiv(ray_count_int, pixels_per_view) * Camera.STRIDE,
    )
    var camera = Camera(camera_params_span, view_idx * Camera.STRIDE)
    var ray = camera.make_ray_raster(px_i, py_i, width_int, inv_height)

    var hit = _trace_tlas_ray[
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        TRACE.CLOSEST_HIT,
        False,
        _intersect_sphere_leaf[
            Frame.LOCAL,
            blas_leaf_width,
            TRACE.CLOSEST_HIT,
        ],
    ](
        tlas_wide_nodes,
        tlas_leaf_instances,
        inst_transform,
        inst_inv_transform,
        inst_blas_indices,
        blas_descs,
        blas_wide_nodes,
        blas_leaf_spheres,
        Int(instance_count),
        tlas_root_idx,
        ray,
    )
    var hits_span = Span(unsafe_ptr=hits, length=ray_count_int * Hit.STRIDE)
    hit._store_unchecked(hits_span, ray_idx)


struct GpuTypedTlasCore[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
]:
    """GPU TLAS core shared by typed TLAS wrappers.

    Instance leaves are packed by the generic wide collapse:
    `tree.leaf_block_indices[leaf_block * leaf_width + lane]` stores the
    instance id.
    """

    var tree: GpuBoundsBvh[Self.node_width, Self.leaf_width]
    var inst_transform: DeviceBuffer[DType.float32]
    var inst_inv_transform: DeviceBuffer[DType.float32]
    var inst_blas_indices: DeviceBuffer[DType.uint32]
    var inst_count: Int
    var timings: GpuBuildTimings

    def __init__(
        out self,
        mut ctx: DeviceContext,
        instances: List[Instance],
        measure_build: Bool = False,
    ) raises:
        self.inst_count = len(instances)
        debug_assert["safe", _use_compiler_assume=True](
            self.inst_count > 0, "passed empty input."
        )

        var leaf_bounds = List[Float32](
            capacity=self.inst_count * AABB[Frame.WORLD].STRIDE
        )
        var payloads = List[UInt32](capacity=self.inst_count)
        for i, inst in enumerate(instances):
            leaf_bounds.append(inst.bounds._min.x)
            leaf_bounds.append(inst.bounds._min.y)
            leaf_bounds.append(inst.bounds._min.z)
            leaf_bounds.append(inst.bounds._max.x)
            leaf_bounds.append(inst.bounds._max.y)
            leaf_bounds.append(inst.bounds._max.z)
            payloads.append(UInt32(i))

        var d_leaf_bounds = upload_list(ctx, leaf_bounds)
        var d_payloads = upload_list(ctx, payloads)

        self.tree = GpuBoundsBvh[Self.node_width, Self.leaf_width](
            ctx, self.inst_count
        )
        self.timings = self.tree.build(
            ctx,
            d_leaf_bounds,
            d_payloads,
            measure_build=measure_build,
        )

        self.inst_transform = upload_list(
            ctx, _flatten_instance_transforms(instances)
        )
        self.inst_inv_transform = upload_list(
            ctx, _flatten_instance_inv_transforms(instances)
        )
        self.inst_blas_indices = upload_list(
            ctx, _flatten_instance_blas_indices(instances)
        )


struct GpuTriangleTlas[
    tlas_node_width: SIMDLength,
    blas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength = tlas_node_width,
    blas_leaf_width: SIMDLength = blas_node_width,
]:
    """Typed triangle TLAS over a descriptor-backed triangle BLAS set."""

    var core: GpuTypedTlasCore[Self.tlas_node_width, Self.tlas_leaf_width]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        instances: List[Instance],
        measure_build: Bool = False,
    ) raises:
        self.core = GpuTypedTlasCore[
            Self.tlas_node_width, Self.tlas_leaf_width
        ](
            ctx,
            instances,
            measure_build=measure_build,
        )

    def launch_camera(
        self,
        ctx: DeviceContext,
        blases: BlasSet[Self.blas_node_width, Self.blas_leaf_width],
        d_camera_params: DeviceBuffer[DType.float32],
        d_hits: DeviceBuffer[DType.float32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        debug_assert["safe", _use_compiler_assume=True](
            ray_count > 0 and cwidth > 0 and cheight > 0,
            "camera launch dimensions must be positive",
        )
        var pixels_per_view = cwidth * cheight
        debug_assert["safe", _use_compiler_assume=True](
            len(d_camera_params)
            >= ceildiv(ray_count, pixels_per_view) * Camera.STRIDE,
            "camera parameter buffer is too short",
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(d_hits) >= ray_count * Hit.STRIDE,
            "hit output buffer is too short",
        )
        ctx.enqueue_function[
            trace_triangle_tlas_camera_kernel[
                Self.tlas_node_width,
                Self.tlas_leaf_width,
                Self.blas_node_width,
                Self.blas_leaf_width,
            ]
        ](
            self.core.tree.wide_nodes,
            self.core.tree.leaf_block_indices,
            self.core.inst_transform,
            self.core.inst_inv_transform,
            self.core.inst_blas_indices,
            blases.descs,
            blases.wide_nodes,
            blases.leaves,
            self.core.tree.root_idx,
            d_camera_params,
            d_hits,
            Int32(self.core.inst_count),
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
        mut ctx: DeviceContext,
        instances: List[Instance],
        measure_build: Bool = False,
    ) raises:
        self.core = GpuTypedTlasCore[
            Self.tlas_node_width, Self.tlas_leaf_width
        ](
            ctx,
            instances,
            measure_build=measure_build,
        )

    def launch_camera(
        self,
        ctx: DeviceContext,
        blases: BlasSet[Self.blas_node_width, Self.blas_leaf_width],
        d_camera_params: DeviceBuffer[DType.float32],
        d_hits: DeviceBuffer[DType.float32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        debug_assert["safe", _use_compiler_assume=True](
            ray_count > 0 and cwidth > 0 and cheight > 0,
            "camera launch dimensions must be positive",
        )
        var pixels_per_view = cwidth * cheight
        debug_assert["safe", _use_compiler_assume=True](
            len(d_camera_params)
            >= ceildiv(ray_count, pixels_per_view) * Camera.STRIDE,
            "camera parameter buffer is too short",
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(d_hits) >= ray_count * Hit.STRIDE,
            "hit output buffer is too short",
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
            self.core.inst_transform,
            self.core.inst_inv_transform,
            self.core.inst_blas_indices,
            blases.descs,
            blases.wide_nodes,
            blases.leaves,
            self.core.tree.root_idx,
            d_camera_params,
            d_hits,
            Int32(self.core.inst_count),
            Int32(ray_count),
            Int32(cwidth),
            Int32(cheight),
            Float32(1.0) / Float32(cheight),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
