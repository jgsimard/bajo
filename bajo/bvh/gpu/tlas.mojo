from std.math import ceildiv, max, clamp
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.core.vec import Vec3f32, normalize
from bajo.core.transform import Affine3f32
from bajo.bvh.constants import (
    TRACE,
    GPU_STACK_SIZE,
    EMPTY_LANE,
    f32_max,
    BOUNDS_STRIDE,
    TRANSFORM_STRIDE,
)
from bajo.bvh.types import Ray, Hit, Instance
from bajo.bvh.gpu.bounds_bvh import (
    GpuBoundsBvh,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    _wide_lane_base,
    _intersect_wide_node_bounds,
)
from bajo.bvh.camera import Camera, CAMERA_STRIDE
from bajo.bvh.gpu.sphere_bvh import _intersect_sphere_leaf
from bajo.bvh.gpu.triangle_bvh import _intersect_triangle_leaf
from bajo.bvh.gpu.trace import trace_bounds_bvh
from bajo.bvh.gpu.utils import GpuBuildTimings, upload_list


comptime BlasLeafFn = def(
    UnsafePointer[Float32, MutAnyOrigin],
    UnsafePointer[UInt32, MutAnyOrigin],
    UInt32,
    UInt32,
    Ray,
    mut Hit,
) capturing -> Bool


def _flatten_instance_inv_transforms(
    instances: List[Instance],
) -> List[Float32]:
    var out = List[Float32](capacity=len(instances) * TRANSFORM_STRIDE)
    for instance in instances:
        out.extend(instance.inv_transform.flatten())
    return out^


def _flatten_instance_blas_indices(
    instances: List[Instance],
) -> List[UInt32]:
    return [instance.blas_idx for instance in instances]


def transform_ray(
    transforms: UnsafePointer[Float32, MutAnyOrigin],
    idx: UInt32,
    ray: Ray,
    t_max: Float32,
) -> Ray:
    var base = Int(idx) * TRANSFORM_STRIDE
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
    tlas_leaf_instances: UnsafePointer[UInt32, MutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    blas_leaf_data_f32: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_data_u32: UnsafePointer[UInt32, MutAnyOrigin],
    blas_root_idx: UInt32,
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: Ray,
    mut hit: Hit,
) -> Bool:
    var hit_any = False

    for lane in range(tlas_width):
        if lane < Int(item_count):
            var idx = Int(leaf_block_idx) * tlas_width + lane
            var inst_idx = UInt32(tlas_leaf_instances[idx])

            if inst_idx != EMPTY_LANE:
                var blas_idx = UInt32(inst_blas_indices[Int(inst_idx)])

                debug_assert["safe"](
                    blas_idx == 0,
                    "current implementation assumes a single BLAS root",
                )

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
                    blas_wide_bounds,
                    blas_wide_data,
                    blas_wide_counts,
                    blas_leaf_data_f32,
                    blas_leaf_data_u32,
                    blas_root_idx,
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
                            Int(inst_idx) * TRANSFORM_STRIDE,
                        )

                        hit = local_hit
                        hit.inst = inst_idx
                        hit.normal = inv_transform.vector(local_hit.normal)
                        hit_any = True

    return hit_any


def trace_tlas_ray[
    primitive: String,
    tlas_width: Int,
    blas_width: Int,
    mode: TRACE = TRACE.CLOSEST_HIT,
](
    tlas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_leaf_instances: UnsafePointer[UInt32, MutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    blas_leaf_data_f32: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_data_u32: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_root_idx: UInt32,
    blas_root_idx: UInt32,
    ray: Ray,
) -> Hit:
    comptime assert primitive in ["triangle", "sphere"]

    comptime leaf_fn: BlasLeafFn = (
        _intersect_triangle_leaf[blas_width, mode] if primitive
        == "triangle" else _intersect_sphere_leaf[blas_width, mode]
    )

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
                        leaf_fn,
                    ](
                        tlas_leaf_instances,
                        inst_inv_transform,
                        inst_blas_indices,
                        blas_wide_bounds,
                        blas_wide_data,
                        blas_wide_counts,
                        blas_leaf_data_f32,
                        blas_leaf_data_u32,
                        blas_root_idx,
                        data,
                        count,
                        ray,
                        hit,
                    )

                    comptime if mode == TRACE.ANY_HIT:
                        if leaf_hit:
                            return hit

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


struct GpuTlas[width: Int]:
    """GPU TLAS built with the same generic GpuBoundsBvh[width] as BLAS.

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
            capacity=max(self.inst_count, 1) * BOUNDS_STRIDE
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
            for _ in range(BOUNDS_STRIDE):
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

    def launch_camera[
        primitive: String,
        blas_width: Int,
    ](
        self,
        ctx: DeviceContext,
        blas_wide_bounds: DeviceBuffer[DType.float32],
        blas_wide_data: DeviceBuffer[DType.uint32],
        blas_wide_counts: DeviceBuffer[DType.uint32],
        blas_leaf_data_f32: DeviceBuffer[DType.float32],
        blas_leaf_data_u32: DeviceBuffer[DType.uint32],
        blas_root_idx: UInt32,
        d_camera_params: DeviceBuffer[DType.float32],
        d_hits_f32: DeviceBuffer[DType.float32],
        d_hits_u32: DeviceBuffer[DType.uint32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        comptime assert primitive in ["triangle", "sphere"]

        ctx.enqueue_function[
            trace_tlas_camera_kernel[
                primitive,
                Self.width,
                blas_width,
            ]
        ](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.tree.leaf_block_indices.unsafe_ptr(),
            self.inst_inv_transform.unsafe_ptr(),
            self.inst_blas_indices.unsafe_ptr(),
            blas_wide_bounds.unsafe_ptr(),
            blas_wide_data.unsafe_ptr(),
            blas_wide_counts.unsafe_ptr(),
            blas_leaf_data_f32.unsafe_ptr(),
            blas_leaf_data_u32.unsafe_ptr(),
            self.tree.root_idx,
            blas_root_idx,
            d_camera_params.unsafe_ptr(),
            d_hits_f32.unsafe_ptr(),
            d_hits_u32.unsafe_ptr(),
            ray_count,
            cwidth,
            cheight,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


def trace_tlas_camera_kernel[
    primitive: String,
    tlas_width: Int,
    blas_width: Int,
](
    tlas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_leaf_instances: UnsafePointer[UInt32, MutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    blas_leaf_data_f32: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_data_u32: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_root_idx: UInt32,
    blas_root_idx: UInt32,
    camera_params: UnsafePointer[Float32, MutAnyOrigin],
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
    width: Int,
    height: Int,
):
    comptime assert primitive in ["triangle", "sphere"]

    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var pixels_per_view = width * height
    var view_idx = ray_idx / pixels_per_view
    var local_idx = ray_idx - view_idx * pixels_per_view
    var px_i = local_idx % width
    var py_i = local_idx / width

    var camera = Camera(camera_params, view_idx * CAMERA_STRIDE)
    var ray = camera.make_ray(px_i, py_i, width, height)

    var hit = trace_tlas_ray[
        primitive,
        tlas_width,
        blas_width,
        TRACE.CLOSEST_HIT,
    ](
        tlas_wide_bounds,
        tlas_wide_data,
        tlas_wide_counts,
        tlas_leaf_instances,
        inst_inv_transform,
        inst_blas_indices,
        blas_wide_bounds,
        blas_wide_data,
        blas_wide_counts,
        blas_leaf_data_f32,
        blas_leaf_data_u32,
        tlas_root_idx,
        blas_root_idx,
        ray,
    )

    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = hit.t
    hits_f32[hit_base + 1] = hit.u
    hits_f32[hit_base + 2] = hit.v

    var ubase = ray_idx * 2
    hits_u32[ubase + 0] = hit.prim
    hits_u32[ubase + 1] = hit.inst
