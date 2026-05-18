from std.math import ceildiv, max
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.core.aabb import AABB
from bajo.core.mat import transform_point, transform_vector
from bajo.core.vec import Vec3f32
from bajo.core.bvh.constants import (
    TRACE_PRIMARY_FULL,
    TRACE_SHADOW,
    TRACE_PRIMARY_T,
    GPU_TRAVERSAL_STACK_SIZE,
    EMPTY_LANE,
    _gpu_inf_t,
)
from bajo.core.bvh.types import Ray, Hit, Instance
from bajo.core.bvh.gpu.bounds_bvh import (
    GpuBoundsBvh,
    _copy_f32_to_device,
    _copy_u32_to_device,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    GPU_WIDE_BOUNDS_STRIDE,
    _gpu_miss_prim,
    _wide_lane_base,
    _intersect_wide_node_bounds,
)
from bajo.core.bvh.gpu.camera import _make_camera_ray
from bajo.core.bvh.gpu.sphere_bvh import (
    GpuSphereBvh,
    _intersect_sphere_leaf,
)
from bajo.core.bvh.gpu.triangle_bvh import (
    GpuTriangleBvh,
    _intersect_triangle_leaf,
)
from bajo.core.bvh.gpu.traverse import trace_gpu_wide_ray


comptime GPU_TLAS_TRANSFORM_STRIDE = 16


def _flatten_instance_inv_transforms(
    instances: List[Instance],
) -> List[Float32]:
    var out = List[Float32](
        capacity=max(len(instances), 1) * GPU_TLAS_TRANSFORM_STRIDE
    )
    for i in range(len(instances)):
        ref m = instances[i].inv_transform
        comptime for j in range(GPU_TLAS_TRANSFORM_STRIDE):
            comptime row = j / 4
            comptime col = j - row * 4
            out.append(m[row][col])
    return out^


def _flatten_instance_blas_indices(
    instances: List[Instance],
) -> List[UInt32]:
    return [instance.blas_idx for instance in instances]


@always_inline
def _make_tlas_local_ray(
    inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_idx: UInt32,
    ray: Ray,
    t_max: Float32,
) -> Ray:
    var base = Int(inst_idx) * GPU_TLAS_TRANSFORM_STRIDE
    var o = transform_point(inst_inv_transform, base, ray.o)
    var d = transform_vector(inst_inv_transform, base, ray.d)

    return Ray(
        o,
        d,
        ray.t_min,
        t_max,
        ray.mask,
    )


@always_inline
def _intersect_tlas_instance_block[
    tlas_width: Int,
    blas_width: Int,
    mode: String,
    blas_leaf_fn: def(
        UnsafePointer[Float32, MutAnyOrigin],
        UnsafePointer[UInt32, MutAnyOrigin],
        UInt32,
        UInt32,
        Ray,
        mut Hit,
    ) capturing -> Bool,
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
    mut best_hit: Hit,
) -> Bool:
    comptime assert mode in [TRACE_PRIMARY_FULL, TRACE_PRIMARY_T, TRACE_SHADOW]

    var hit_any = False

    comptime for lane in range(tlas_width):
        if lane < Int(item_count):
            var idx = Int(leaf_block_idx) * tlas_width + lane
            var inst_idx = UInt32(tlas_leaf_instances[idx])

            if inst_idx != EMPTY_LANE:
                var blas_idx = UInt32(inst_blas_indices[Int(inst_idx)])

                debug_assert["safe"](
                    blas_idx == 0,
                    "current implementation assumes a single BLAS root",
                )
                if blas_idx == UInt32(0):
                    var local_t_max = best_hit.t
                    comptime if mode == TRACE_SHADOW:
                        local_t_max = ray.t_max

                    var local_ray = _make_tlas_local_ray(
                        inst_inv_transform,
                        inst_idx,
                        ray,
                        local_t_max,
                    )

                    var local_hit = trace_gpu_wide_ray[
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

                    comptime if mode == TRACE_SHADOW:
                        if local_hit.occluded != UInt32(0):
                            best_hit = Hit.shadow_hit()
                            best_hit.inst = inst_idx
                            return True
                    else:
                        if (
                            local_hit.t < best_hit.t
                            and local_hit.prim != EMPTY_LANE
                        ):
                            best_hit = local_hit
                            best_hit.inst = inst_idx
                            hit_any = True

    return hit_any


@always_inline
def trace_gpu_wide_tlas_ray[
    tlas_width: Int,
    blas_width: Int,
    mode: String,
    blas_leaf_fn: def(
        UnsafePointer[Float32, MutAnyOrigin],
        UnsafePointer[UInt32, MutAnyOrigin],
        UInt32,
        UInt32,
        Ray,
        mut Hit,
    ) capturing -> Bool,
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
    comptime assert mode in [TRACE_PRIMARY_FULL, TRACE_PRIMARY_T, TRACE_SHADOW]

    var best_hit = Hit.miss()
    best_hit.t = ray.t_max
    best_hit.prim = _gpu_miss_prim
    best_hit.inst = _gpu_miss_prim

    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var current = tlas_root_idx

    while True:
        var node_t_max = best_hit.t
        comptime if mode == TRACE_SHADOW:
            node_t_max = ray.t_max

        var bounds_hit = _intersect_wide_node_bounds[tlas_width](
            tlas_wide_bounds,
            current,
            ray,
            node_t_max,
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
                    child_t[node_lane] = bounds_hit.tmin[node_lane]
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
                        blas_wide_bounds,
                        blas_wide_data,
                        blas_wide_counts,
                        blas_leaf_data_f32,
                        blas_leaf_data_u32,
                        blas_root_idx,
                        data,
                        count,
                        ray,
                        best_hit,
                    )

                    comptime if mode == TRACE_SHADOW:
                        if leaf_hit:
                            return best_hit

        comptime for _ in range(tlas_width):
            var far_lane = -1
            var far_t = Float32(-_gpu_inf_t)

            comptime for lane in range(tlas_width):
                if child_valid[lane]:
                    var t = child_t[lane]
                    if t >= far_t:
                        far_t = t
                        far_lane = lane

            if far_lane != -1:
                child_valid[far_lane] = False

                comptime if mode != TRACE_SHADOW:
                    if far_t <= best_hit.t:
                        if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                            stack[stack_ptr] = child_data[far_lane]
                            stack_ptr += 1
                else:
                    if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                        stack[stack_ptr] = child_data[far_lane]
                        stack_ptr += 1

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        current = stack[stack_ptr]

    return best_hit


@always_inline
def trace_gpu_wide_tlas_triangle_ray[
    tlas_width: Int,
    blas_width: Int,
    mode: String = TRACE_PRIMARY_FULL,
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
    blas_leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_root_idx: UInt32,
    blas_root_idx: UInt32,
    ray: Ray,
) -> Hit:
    return trace_gpu_wide_tlas_ray[
        tlas_width,
        blas_width,
        mode,
        _intersect_triangle_leaf[
            blas_width,
            mode,
        ],
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
        blas_leaf_vertices,
        blas_leaf_prims,
        tlas_root_idx,
        blas_root_idx,
        ray,
    )


@always_inline
def trace_gpu_wide_tlas_sphere_ray[
    tlas_width: Int,
    blas_width: Int,
    mode: String = TRACE_PRIMARY_FULL,
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
    blas_leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_root_idx: UInt32,
    blas_root_idx: UInt32,
    ray: Ray,
) -> Hit:
    return trace_gpu_wide_tlas_ray[
        tlas_width,
        blas_width,
        mode,
        _intersect_sphere_leaf[
            blas_width,
            mode,
        ],
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
        blas_leaf_spheres,
        blas_leaf_prims,
        tlas_root_idx,
        blas_root_idx,
        ray,
    )


struct GpuTlas[width: Int]:
    """GPU TLAS built with the same generic GpuBoundsBvh[width] as BLAS.

    Instance leaves are packed by the generic wide collapse:
    `tree.leaf_block_indices[leaf_block * width + lane]` stores the instance id.
    """

    var tree: GpuBoundsBvh[Self.width]
    var inst_inv_transform: DeviceBuffer[DType.float32]
    var inst_blas_indices: DeviceBuffer[DType.uint32]
    var inst_count: Int

    def __init__(
        out self,
        mut ctx: DeviceContext,
        instances: List[Instance],
    ) raises:
        self.inst_count = len(instances)

        var leaf_bounds = List[Float32](
            capacity=max(self.inst_count, 1) * GPU_WIDE_BOUNDS_STRIDE
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

        self.tree = GpuBoundsBvh[Self.width](ctx, leaf_bounds, payloads)
        _ = self.tree.build(ctx)

        self.inst_inv_transform = _copy_f32_to_device(
            ctx, _flatten_instance_inv_transforms(instances)
        )
        self.inst_blas_indices = _copy_u32_to_device(
            ctx, _flatten_instance_blas_indices(instances)
        )

    def launch_uploaded_triangle_primary[
        blas_width: Int,
    ](
        self,
        ctx: DeviceContext,
        blas: GpuTriangleBvh[blas_width],
        d_rays: DeviceBuffer[DType.float32],
        d_hits_f32: DeviceBuffer[DType.float32],
        d_hits_u32: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[
            trace_gpu_tlas_primary_kernel[
                Self.width,
                blas_width,
                _intersect_triangle_leaf[
                    blas_width,
                    TRACE_PRIMARY_FULL,
                ],
            ]
        ](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.tree.leaf_block_indices.unsafe_ptr(),
            self.inst_inv_transform.unsafe_ptr(),
            self.inst_blas_indices.unsafe_ptr(),
            blas.tree.wide_bounds.unsafe_ptr(),
            blas.tree.wide_data.unsafe_ptr(),
            blas.tree.wide_counts.unsafe_ptr(),
            blas.leaf_vertices.unsafe_ptr(),
            blas.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            blas.tree.root_idx,
            d_rays.unsafe_ptr(),
            d_hits_f32.unsafe_ptr(),
            d_hits_u32.unsafe_ptr(),
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    def launch_uploaded_triangle_shadow[
        blas_width: Int,
    ](
        self,
        ctx: DeviceContext,
        blas: GpuTriangleBvh[blas_width],
        d_rays: DeviceBuffer[DType.float32],
        d_flags: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[
            trace_gpu_tlas_shadow_kernel[
                Self.width,
                blas_width,
                _intersect_triangle_leaf[
                    blas_width,
                    TRACE_SHADOW,
                ],
            ]
        ](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.tree.leaf_block_indices.unsafe_ptr(),
            self.inst_inv_transform.unsafe_ptr(),
            self.inst_blas_indices.unsafe_ptr(),
            blas.tree.wide_bounds.unsafe_ptr(),
            blas.tree.wide_data.unsafe_ptr(),
            blas.tree.wide_counts.unsafe_ptr(),
            blas.leaf_vertices.unsafe_ptr(),
            blas.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            blas.tree.root_idx,
            d_rays.unsafe_ptr(),
            d_flags.unsafe_ptr(),
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    def launch_uploaded_sphere_primary[
        blas_width: Int,
    ](
        self,
        ctx: DeviceContext,
        blas: GpuSphereBvh[blas_width],
        d_rays: DeviceBuffer[DType.float32],
        d_hits_f32: DeviceBuffer[DType.float32],
        d_hits_u32: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[
            trace_gpu_tlas_primary_kernel[
                Self.width,
                blas_width,
                _intersect_sphere_leaf[
                    blas_width,
                    TRACE_PRIMARY_FULL,
                ],
            ]
        ](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.tree.leaf_block_indices.unsafe_ptr(),
            self.inst_inv_transform.unsafe_ptr(),
            self.inst_blas_indices.unsafe_ptr(),
            blas.tree.wide_bounds.unsafe_ptr(),
            blas.tree.wide_data.unsafe_ptr(),
            blas.tree.wide_counts.unsafe_ptr(),
            blas.leaf_spheres.unsafe_ptr(),
            blas.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            blas.tree.root_idx,
            d_rays.unsafe_ptr(),
            d_hits_f32.unsafe_ptr(),
            d_hits_u32.unsafe_ptr(),
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    def launch_uploaded_sphere_shadow[
        blas_width: Int,
    ](
        self,
        ctx: DeviceContext,
        blas: GpuSphereBvh[blas_width],
        d_rays: DeviceBuffer[DType.float32],
        d_flags: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[
            trace_gpu_tlas_shadow_kernel[
                Self.width,
                blas_width,
                _intersect_sphere_leaf[
                    blas_width,
                    TRACE_SHADOW,
                ],
            ]
        ](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.tree.leaf_block_indices.unsafe_ptr(),
            self.inst_inv_transform.unsafe_ptr(),
            self.inst_blas_indices.unsafe_ptr(),
            blas.tree.wide_bounds.unsafe_ptr(),
            blas.tree.wide_data.unsafe_ptr(),
            blas.tree.wide_counts.unsafe_ptr(),
            blas.leaf_spheres.unsafe_ptr(),
            blas.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            blas.tree.root_idx,
            d_rays.unsafe_ptr(),
            d_flags.unsafe_ptr(),
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    def launch_camera_triangle_primary[
        blas_width: Int,
    ](
        self,
        ctx: DeviceContext,
        blas: GpuTriangleBvh[blas_width],
        d_camera_params: DeviceBuffer[DType.float32],
        d_hits_f32: DeviceBuffer[DType.float32],
        d_hits_u32: DeviceBuffer[DType.uint32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        ctx.enqueue_function[
            trace_gpu_tlas_camera_primary_kernel[
                Self.width,
                blas_width,
                _intersect_triangle_leaf[
                    blas_width,
                    TRACE_PRIMARY_FULL,
                ],
            ]
        ](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.tree.leaf_block_indices.unsafe_ptr(),
            self.inst_inv_transform.unsafe_ptr(),
            self.inst_blas_indices.unsafe_ptr(),
            blas.tree.wide_bounds.unsafe_ptr(),
            blas.tree.wide_data.unsafe_ptr(),
            blas.tree.wide_counts.unsafe_ptr(),
            blas.leaf_vertices.unsafe_ptr(),
            blas.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            blas.tree.root_idx,
            d_camera_params.unsafe_ptr(),
            d_hits_f32.unsafe_ptr(),
            d_hits_u32.unsafe_ptr(),
            ray_count,
            cwidth,
            cheight,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


def trace_gpu_tlas_camera_primary_kernel[
    tlas_width: Int,
    blas_width: Int,
    blas_leaf_fn: def(
        UnsafePointer[Float32, MutAnyOrigin],
        UnsafePointer[UInt32, MutAnyOrigin],
        UInt32,
        UInt32,
        Ray,
        mut Hit,
    ) capturing -> Bool,
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
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var ray = _make_camera_ray(camera_params, ray_idx, width, height)

    var hit = trace_gpu_wide_tlas_ray[
        tlas_width,
        blas_width,
        TRACE_PRIMARY_FULL,
        blas_leaf_fn,
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


def trace_gpu_tlas_primary_kernel[
    tlas_width: Int,
    blas_width: Int,
    blas_leaf_fn: def(
        UnsafePointer[Float32, MutAnyOrigin],
        UnsafePointer[UInt32, MutAnyOrigin],
        UInt32,
        UInt32,
        Ray,
        mut Hit,
    ) capturing -> Bool,
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
    rays: UnsafePointer[Float32, MutAnyOrigin],
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var ray = Ray(rays, ray_idx)
    var hit = trace_gpu_wide_tlas_ray[
        tlas_width,
        blas_width,
        TRACE_PRIMARY_FULL,
        blas_leaf_fn,
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


def trace_gpu_tlas_shadow_kernel[
    tlas_width: Int,
    blas_width: Int,
    blas_leaf_fn: def(
        UnsafePointer[Float32, MutAnyOrigin],
        UnsafePointer[UInt32, MutAnyOrigin],
        UInt32,
        UInt32,
        Ray,
        mut Hit,
    ) capturing -> Bool,
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
    rays: UnsafePointer[Float32, MutAnyOrigin],
    flags: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var ray = Ray(rays, ray_idx)
    var hit = trace_gpu_wide_tlas_ray[
        tlas_width,
        blas_width,
        TRACE_SHADOW,
        blas_leaf_fn,
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

    flags[ray_idx] = hit.occluded
