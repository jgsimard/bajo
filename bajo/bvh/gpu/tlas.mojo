from std.math import ceildiv, max, min
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.core import AABB, Affine3f32
from bajo.bvh.constants import (
    TRACE,
    GPU_STACK_SIZE,
    EMPTY_LANE,
    f32_max,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)
from bajo.bvh.types import Ray, Hit, Instance, BlasSet
from bajo.bvh.gpu.bounds_bvh import (
    GpuBoundsBvh,
    _wide_lane_base,
    _intersect_wide_node_bounds,
)
from bajo.bvh.camera import Camera
from bajo.bvh.gpu.sphere_bvh import _intersect_sphere_leaf
from bajo.bvh.gpu.triangle_bvh import _intersect_triangle_leaf
from bajo.bvh.gpu.trace import trace_bounds_bvh
from bajo.bvh.gpu.utils import GpuBuildTimings, upload_list


comptime BlasLeafFn = def(
    UnsafePointer[Float32, ImmutAnyOrigin],
    UnsafePointer[UInt32, ImmutAnyOrigin],
    UInt32,
    Ray,
    mut Hit,
) capturing -> Bool


def _flatten_instance_inv_transforms(
    instances: List[Instance],
) -> List[Float32]:
    if len(instances) == 0:
        return Affine3f32.identity().flatten()

    var out = List[Float32](capacity=len(instances) * Affine3f32.STRIDE)
    for instance in instances:
        out.extend(instance.inv_transform.flatten())
    return out^


def _flatten_instance_blas_indices(
    instances: List[Instance],
) -> List[UInt32]:
    if len(instances) == 0:
        return [0]
    return [instance.blas_idx for instance in instances]


def transform_ray[
    origin: ImmutOrigin
](
    transforms: UnsafePointer[Float32, origin],
    idx: UInt32,
    ray: Ray,
    t_max: Float32,
) -> Ray:
    var base = Int(idx) * Affine3f32.STRIDE
    var transform = Affine3f32.load(transforms, base)
    var o = transform.point(ray.o)
    var d = transform.vector(ray.d)
    return Ray(o, d, ray.t_min, t_max)


@always_inline
def _intersect_tlas_instance_block[
    tlas_width: Int,
    blas_width: Int,
    mode: TRACE,
    blas_leaf_fn: BlasLeafFn,
](
    tlas_leaf_instances: UnsafePointer[UInt32, ImmutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, ImmutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, ImmutAnyOrigin],
    blas_descs: UnsafePointer[UInt32, ImmutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, ImmutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, ImmutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, ImmutAnyOrigin],
    blas_leaf_data_f32: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_data_u32: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: Ray,
    mut hit: Hit,
) -> Bool:
    var hit_any = False

    for lane in range(min(tlas_width, Int(item_count))):
        var idx = Int(leaf_block_idx) * tlas_width + lane
        var inst_idx = UInt32(tlas_leaf_instances[idx])

        if inst_idx != EMPTY_LANE:
            var blas_idx = UInt32(inst_blas_indices[Int(inst_idx)])
            var desc_base = Int(blas_idx) * BlasSet.STRIDE

            var local_ray = transform_ray(
                inst_inv_transform,
                inst_idx,
                ray,
                hit.t,
            )

            var local_hit = trace_bounds_bvh[
                blas_width,
                mode,
                blas_leaf_fn,
            ](
                blas_wide_bounds
                + Int(blas_descs[desc_base + BlasSet.WIDE_BOUNDS_BASE]),
                blas_wide_data
                + Int(blas_descs[desc_base + BlasSet.WIDE_LANE_BASE]),
                blas_wide_counts
                + Int(blas_descs[desc_base + BlasSet.WIDE_LANE_BASE]),
                blas_leaf_data_f32
                + Int(blas_descs[desc_base + BlasSet.LEAF_F32_BASE]),
                blas_leaf_data_u32
                + Int(blas_descs[desc_base + BlasSet.LEAF_U32_BASE]),
                UInt32(blas_descs[desc_base + BlasSet.ROOT_IDX]),
                local_ray,
            )

            comptime if mode == TRACE.ANY_HIT:
                if local_hit.is_occluded():
                    hit = Hit.shadow_hit()
                    hit.inst = inst_idx
                    return True
            else:
                if local_hit.t < hit.t and local_hit.prim != EMPTY_LANE:
                    var inv_transform = Affine3f32.load(
                        inst_inv_transform,
                        Int(inst_idx) * Affine3f32.STRIDE,
                    )

                    hit = local_hit
                    hit.inst = inst_idx
                    hit.normal = inv_transform.vector(local_hit.normal)
                    hit_any = True
    return hit_any


def _trace_tlas_ray[
    tlas_width: Int,
    blas_width: Int,
    mode: TRACE,
    blas_leaf_fn: BlasLeafFn,
](
    tlas_wide_bounds: UnsafePointer[Float32, ImmutAnyOrigin],
    tlas_wide_data: UnsafePointer[UInt32, ImmutAnyOrigin],
    tlas_wide_counts: UnsafePointer[UInt32, ImmutAnyOrigin],
    tlas_leaf_instances: UnsafePointer[UInt32, ImmutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, ImmutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, ImmutAnyOrigin],
    blas_descs: UnsafePointer[UInt32, ImmutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, ImmutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, ImmutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, ImmutAnyOrigin],
    blas_leaf_data_f32: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_data_u32: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_root_idx: UInt32,
    ray: Ray,
) -> Hit:
    var hit = Hit.miss(ray.t_max)

    var stack = InlineArray[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var current = tlas_root_idx

    while True:
        var bounds_hit = _intersect_wide_node_bounds[tlas_width](
            tlas_wide_bounds,
            current,
            ray,
            hit.t,
        )

        var child_valid = InlineArray[Bool, tlas_width](fill=False)
        var child_data = InlineArray[UInt32, tlas_width](fill=0)
        var child_t = InlineArray[Float32, tlas_width](fill=0.0)

        comptime for node_lane in range(tlas_width):
            var lane_base = _wide_lane_base[tlas_width](current, node_lane)
            var count = UInt32(tlas_wide_counts[lane_base])

            if count != EMPTY_LANE and bounds_hit.mask[node_lane]:
                var data = UInt32(tlas_wide_data[lane_base])

                if count == 0:
                    child_valid[node_lane] = True
                    child_data[node_lane] = data
                    child_t[node_lane] = bounds_hit.t[node_lane]
                else:
                    var leaf_hit = _intersect_tlas_instance_block[
                        tlas_width,
                        blas_width,
                        mode,
                        blas_leaf_fn,
                    ](
                        tlas_leaf_instances,
                        inst_inv_transform,
                        inst_blas_indices,
                        blas_descs,
                        blas_wide_bounds,
                        blas_wide_data,
                        blas_wide_counts,
                        blas_leaf_data_f32,
                        blas_leaf_data_u32,
                        data,
                        count,
                        ray,
                        hit,
                    )

                    comptime if mode == TRACE.ANY_HIT:
                        if leaf_hit:
                            return hit

        # Push internal children far-to-near.  The stack is LIFO, so the nearest
        # surviving child is popped first.
        comptime for _ in range(tlas_width):
            var far_lane = -1
            var far_t = Float32(-f32_max)

            comptime for lane in range(tlas_width):
                if child_valid[lane]:
                    var t = child_t[lane]
                    if t >= far_t:
                        far_t = t
                        far_lane = lane

            if far_lane != -1:
                child_valid[far_lane] = False

                comptime if mode != TRACE.ANY_HIT:
                    if far_t <= hit.t:
                        if stack_ptr < GPU_STACK_SIZE:
                            stack[stack_ptr] = child_data[far_lane]
                            stack_ptr += 1
                else:
                    if stack_ptr < GPU_STACK_SIZE:
                        stack[stack_ptr] = child_data[far_lane]
                        stack_ptr += 1

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        current = stack[stack_ptr]

    return hit


def trace_triangle_tlas_camera_kernel[
    tlas_width: Int,
    blas_width: Int,
](
    tlas_wide_bounds: UnsafePointer[Float32, ImmutAnyOrigin],
    tlas_wide_data: UnsafePointer[UInt32, ImmutAnyOrigin],
    tlas_wide_counts: UnsafePointer[UInt32, ImmutAnyOrigin],
    tlas_leaf_instances: UnsafePointer[UInt32, ImmutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, ImmutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, ImmutAnyOrigin],
    blas_descs: UnsafePointer[UInt32, ImmutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, ImmutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, ImmutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, ImmutAnyOrigin],
    blas_leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_root_idx: UInt32,
    camera_params: UnsafePointer[Float32, ImmutAnyOrigin],
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
    width: Int,
    height: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var pixels_per_view = width * height
    var view_idx = ray_idx / pixels_per_view
    var local_idx = ray_idx - view_idx * pixels_per_view
    var px_i = local_idx % width
    var py_i = local_idx / width

    var camera = Camera(camera_params, view_idx * Camera.STRIDE)
    var ray = camera.make_ray(px_i, py_i, width, height)

    var hit = _trace_tlas_ray[
        tlas_width,
        blas_width,
        TRACE.CLOSEST_HIT,
        _intersect_triangle_leaf[
            blas_width,
            TRACE.CLOSEST_HIT,
        ],
    ](
        tlas_wide_bounds,
        tlas_wide_data,
        tlas_wide_counts,
        tlas_leaf_instances,
        inst_inv_transform,
        inst_blas_indices,
        blas_descs,
        blas_wide_bounds,
        blas_wide_data,
        blas_wide_counts,
        blas_leaf_vertices,
        blas_leaf_prims,
        tlas_root_idx,
        ray,
    )

    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = hit.t
    hits_f32[hit_base + 1] = hit.u
    hits_f32[hit_base + 2] = hit.v

    var ubase = ray_idx * 2
    hits_u32[ubase + 0] = hit.prim
    hits_u32[ubase + 1] = hit.inst


def trace_sphere_tlas_camera_kernel[
    tlas_width: Int,
    blas_width: Int,
](
    tlas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_leaf_instances: UnsafePointer[UInt32, MutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, MutAnyOrigin],
    blas_descs: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    blas_leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_root_idx: UInt32,
    camera_params: UnsafePointer[Float32, MutAnyOrigin],
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
    width: Int,
    height: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var pixels_per_view = width * height
    var view_idx = ray_idx / pixels_per_view
    var local_idx = ray_idx - view_idx * pixels_per_view
    var px_i = local_idx % width
    var py_i = local_idx / width

    var camera = Camera(camera_params, view_idx * Camera.STRIDE)
    var ray = camera.make_ray(px_i, py_i, width, height)

    var hit = _trace_tlas_ray[
        tlas_width,
        blas_width,
        TRACE.CLOSEST_HIT,
        _intersect_sphere_leaf[
            blas_width,
            TRACE.CLOSEST_HIT,
        ],
    ](
        tlas_wide_bounds,
        tlas_wide_data,
        tlas_wide_counts,
        tlas_leaf_instances,
        inst_inv_transform,
        inst_blas_indices,
        blas_descs,
        blas_wide_bounds,
        blas_wide_data,
        blas_wide_counts,
        blas_leaf_spheres,
        blas_leaf_prims,
        tlas_root_idx,
        ray,
    )

    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = hit.t
    hits_f32[hit_base + 1] = hit.u
    hits_f32[hit_base + 2] = hit.v

    var ubase = ray_idx * 2
    hits_u32[ubase + 0] = hit.prim
    hits_u32[ubase + 1] = hit.inst


struct GpuTypedTlasCore[width: Int]:
    """GPU TLAS core shared by typed TLAS wrappers.

    Instance leaves are packed by the generic wide collapse:
    `tree.leaf_block_indices[leaf_block * width + lane]` stores the instance id.
    """

    var tree: GpuBoundsBvh[Self.width]
    var inst_inv_transform: DeviceBuffer[DType.float32]
    var inst_blas_indices: DeviceBuffer[DType.uint32]
    var inst_count: Int
    var timings: GpuBuildTimings

    def __init__(
        out self,
        mut ctx: DeviceContext,
        instances: List[Instance],
    ) raises:
        self.inst_count = len(instances)

        var leaf_bounds = List[Float32](
            capacity=max(self.inst_count, 1) * AABB.STRIDE
        )
        var payloads = List[UInt32](capacity=max(self.inst_count, 1))

        for i in range(self.inst_count):
            ref inst = instances[i]
            leaf_bounds.append(inst.bounds._min.x)
            leaf_bounds.append(inst.bounds._min.y)
            leaf_bounds.append(inst.bounds._min.z)
            leaf_bounds.append(inst.bounds._max.x)
            leaf_bounds.append(inst.bounds._max.y)
            leaf_bounds.append(inst.bounds._max.z)
            payloads.append(UInt32(i))

        if self.inst_count == 0:
            for _ in range(AABB.STRIDE):
                leaf_bounds.append(0.0)
            payloads.append(EMPTY_LANE)

        var d_leaf_bounds = upload_list(ctx, leaf_bounds)
        var d_payloads = upload_list(ctx, payloads)

        self.tree = GpuBoundsBvh[Self.width](ctx, d_leaf_bounds, d_payloads)
        self.timings = self.tree.build(ctx)

        self.inst_inv_transform = upload_list(
            ctx, _flatten_instance_inv_transforms(instances)
        )
        self.inst_blas_indices = upload_list(
            ctx, _flatten_instance_blas_indices(instances)
        )


struct GpuTriangleTlas[tlas_width: Int, blas_width: Int]:
    """Typed triangle TLAS over a descriptor-backed triangle BLAS set."""

    var core: GpuTypedTlasCore[Self.tlas_width]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        instances: List[Instance],
    ) raises:
        self.core = GpuTypedTlasCore[Self.tlas_width](ctx, instances)

    def launch_camera(
        self,
        ctx: DeviceContext,
        blases: BlasSet[Self.blas_width],
        d_camera_params: DeviceBuffer[DType.float32],
        d_hits_f32: DeviceBuffer[DType.float32],
        d_hits_u32: DeviceBuffer[DType.uint32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        ctx.enqueue_function[
            trace_triangle_tlas_camera_kernel[
                Self.tlas_width,
                Self.blas_width,
            ]
        ](
            self.core.tree.wide_bounds,
            self.core.tree.wide_data,
            self.core.tree.wide_counts,
            self.core.tree.leaf_block_indices,
            self.core.inst_inv_transform,
            self.core.inst_blas_indices,
            blases.descs,
            blases.wide_bounds,
            blases.wide_data,
            blases.wide_counts,
            blases.leaf_data_f32,
            blases.leaf_prims,
            self.core.tree.root_idx,
            d_camera_params,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            cwidth,
            cheight,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


struct GpuSphereTlas[tlas_width: Int, blas_width: Int]:
    """Typed sphere TLAS over a descriptor-backed sphere BLAS set."""

    var core: GpuTypedTlasCore[Self.tlas_width]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        instances: List[Instance],
    ) raises:
        self.core = GpuTypedTlasCore[Self.tlas_width](ctx, instances)

    def launch_camera(
        self,
        ctx: DeviceContext,
        blases: BlasSet[Self.blas_width],
        d_camera_params: DeviceBuffer[DType.float32],
        d_hits_f32: DeviceBuffer[DType.float32],
        d_hits_u32: DeviceBuffer[DType.uint32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        ctx.enqueue_function[
            trace_sphere_tlas_camera_kernel[
                Self.tlas_width,
                Self.blas_width,
            ]
        ](
            self.core.tree.wide_bounds,
            self.core.tree.wide_data,
            self.core.tree.wide_counts,
            self.core.tree.leaf_block_indices,
            self.core.inst_inv_transform,
            self.core.inst_blas_indices,
            blases.descs,
            blases.wide_bounds,
            blases.wide_data,
            blases.wide_counts,
            blases.leaf_data_f32,
            blases.leaf_prims,
            self.core.tree.root_idx,
            d_camera_params,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            cwidth,
            cheight,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
