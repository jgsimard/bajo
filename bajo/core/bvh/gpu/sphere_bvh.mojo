from std.math import ceildiv, sqrt
from std.time import perf_counter_ns
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.core.intersect import intersect_ray_sphere
from bajo.core.vec import Vec3f32, Vec3
from bajo.core.bvh.types import Sphere, RayFlat, Hit
from bajo.core.bvh.gpu.bounds_bvh import (
    GpuBoundsBvh,
    _copy_f32_to_device,
    _copy_u32_to_device,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    GPU_WIDE_BOUNDS_STRIDE,
    GPU_SPHERE_STRIDE,
    TRACE_PRIMARY_FULL,
    TRACE_SHADOW,
    TRACE_PRIMARY_T,
    _gpu_miss_prim,
    GPU_TRAVERSAL_STACK_SIZE,
    _wide_lane_base,
    GPU_WIDE_EMPTY_LANE,
    _intersect_wide_lane_bounds,
)


struct GpuSphereBvh[width: Int]:
    var tree: GpuBoundsBvh[Self.width]
    var spheres: DeviceBuffer[DType.float32]
    var leaf_spheres: DeviceBuffer[DType.float32]
    var leaf_prims: DeviceBuffer[DType.uint32]
    var sphere_count: Int
    var leaf_pack_ns: Int

    def __init__(
        out self,
        mut ctx: DeviceContext,
        spheres: List[Sphere],
    ) raises:
        self.sphere_count = len(spheres)
        self.leaf_pack_ns = 0

        var flat_spheres = _flatten_spheres(spheres)
        self.spheres = _copy_f32_to_device(ctx, flat_spheres)

        var leaf_bounds = List[Float32](
            capacity=max(self.sphere_count, 1) * GPU_WIDE_BOUNDS_STRIDE
        )
        var payloads = List[UInt32](capacity=max(self.sphere_count, 1))

        for i in range(self.sphere_count):
            ref s = spheres[i]
            var r = s.radius

            leaf_bounds.append(s.center.x - r)
            leaf_bounds.append(s.center.y - r)
            leaf_bounds.append(s.center.z - r)
            leaf_bounds.append(s.center.x + r)
            leaf_bounds.append(s.center.y + r)
            leaf_bounds.append(s.center.z + r)
            payloads.append(UInt32(i))

        self.tree = GpuBoundsBvh[Self.width](ctx, leaf_bounds, payloads)
        _ = self.tree.build(ctx)

        self.leaf_spheres = ctx.enqueue_create_buffer[DType.float32](
            self.tree.max_leaf_blocks * Self.width * GPU_SPHERE_STRIDE
        )
        self.leaf_prims = ctx.enqueue_create_buffer[DType.uint32](
            self.tree.max_leaf_blocks * Self.width
        )
        self._pack_leaf_blocks(ctx)

    def _pack_leaf_blocks(
        mut self,
        ctx: DeviceContext,
    ) raises:
        var start = perf_counter_ns()
        var blocks = ceildiv(
            max(self.tree.leaf_block_count, 1), GPU_BOUNDS_BVH_BLOCK_SIZE
        )
        ctx.enqueue_function[pack_sphere_leaf_blocks_kernel[Self.width]](
            self.spheres,
            self.tree.leaf_block_indices,
            self.leaf_spheres,
            self.leaf_prims,
            self.tree.leaf_block_count,
            grid_dim=blocks,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        self.leaf_pack_ns = Int(perf_counter_ns() - start)

    def launch_uploaded_primary(
        self,
        ctx: DeviceContext,
        d_rays: DeviceBuffer[DType.float32],
        d_hits_f32: DeviceBuffer[DType.float32],
        d_hits_u32: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[trace_gpu_sphere_bvh_primary_kernel[Self.width]](
            self.tree.wide_bounds,
            self.tree.wide_data,
            self.tree.wide_counts,
            self.leaf_spheres,
            self.leaf_prims,
            self.tree.root_idx,
            d_rays,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    def launch_uploaded_shadow(
        self,
        ctx: DeviceContext,
        d_rays: DeviceBuffer[DType.float32],
        d_flags: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[trace_gpu_sphere_bvh_shadow_kernel[Self.width]](
            self.tree.wide_bounds,
            self.tree.wide_data,
            self.tree.wide_counts,
            self.leaf_spheres,
            self.leaf_prims,
            self.tree.root_idx,
            d_rays,
            d_flags,
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


def trace_gpu_sphere_bvh_primary_kernel[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    root_idx: UInt32,
    rays: UnsafePointer[Float32, MutAnyOrigin],
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var ray = RayFlat(rays, ray_idx)
    var hit = trace_gpu_wide_sphere_ray[width, TRACE_PRIMARY_FULL](
        wide_bounds,
        wide_data,
        wide_counts,
        leaf_spheres,
        leaf_prims,
        root_idx,
        ray,
    )

    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = hit.t
    hits_f32[hit_base + 1] = hit.u
    hits_f32[hit_base + 2] = hit.v
    hits_u32[ray_idx] = hit.prim


def trace_gpu_sphere_bvh_shadow_kernel[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    root_idx: UInt32,
    rays: UnsafePointer[Float32, MutAnyOrigin],
    flags: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var ray = RayFlat(rays, ray_idx)
    var hit = trace_gpu_wide_sphere_ray[width, TRACE_SHADOW](
        wide_bounds,
        wide_data,
        wide_counts,
        leaf_spheres,
        leaf_prims,
        root_idx,
        ray,
    )

    flags[ray_idx] = hit.occluded


@always_inline
def trace_gpu_wide_sphere_ray[
    width: Int,
    mode: String = TRACE_PRIMARY_FULL,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    root_idx: UInt32,
    ray: RayFlat,
) -> Hit:
    comptime assert mode in [TRACE_PRIMARY_FULL, TRACE_PRIMARY_T, TRACE_SHADOW]

    var best_t = ray.t_max
    var best_u = Float32(0.0)
    var best_v = Float32(0.0)
    var best_prim = _gpu_miss_prim

    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var current = root_idx

    while True:
        comptime for lane in range(width):
            var lane_base = _wide_lane_base[width](current, lane)
            var count = UInt32(wide_counts[lane_base])

            if count != GPU_WIDE_EMPTY_LANE:
                if _intersect_wide_lane_bounds[width](
                    wide_bounds, current, lane, ray, best_t
                ):
                    var data = UInt32(wide_data[lane_base])
                    if count == 0:
                        if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                            stack[stack_ptr] = data
                            stack_ptr += 1
                    else:
                        var leaf_hit = _intersect_sphere_leaf_block[
                            width, mode
                        ](
                            leaf_spheres,
                            leaf_prims,
                            data,
                            count,
                            ray,
                            best_t,
                            best_u,
                            best_v,
                            best_prim,
                        )
                        comptime if mode == TRACE_SHADOW:
                            if leaf_hit:
                                return Hit(0.0, 0.0, 0.0, best_prim, UInt32(1))

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        current = stack[stack_ptr]

    return Hit(best_t, best_u, best_v, best_prim, UInt32(0))


@always_inline
def _intersect_sphere_leaf_block[
    width: Int,
    mode: String,
](
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: RayFlat,
    mut best_t: Float32,
    mut best_u: Float32,
    mut best_v: Float32,
    mut best_prim: UInt32,
) -> Bool:
    comptime assert mode in [TRACE_PRIMARY_FULL, TRACE_PRIMARY_T, TRACE_SHADOW]

    var hit_any = False

    comptime for lane in range(width):
        if lane < Int(item_count):
            var idx = Int(leaf_block_idx) * width + lane
            var prim = UInt32(leaf_prims[idx])

            if prim != GPU_WIDE_EMPTY_LANE:
                var base = idx * GPU_SPHERE_STRIDE
                var center = Vec3f32(
                    leaf_spheres[base + 0],
                    leaf_spheres[base + 1],
                    leaf_spheres[base + 2],
                )
                var radius = leaf_spheres[base + 3]

                var O = Vec3[DType.float32](ray.o.x, ray.o.y, ray.o.z)
                var D = Vec3[DType.float32](ray.d.x, ray.d.y, ray.d.z)

                var h = intersect_ray_sphere(O, D, center, radius, best_t)

                if h.t > 1.0e-4 and h.t < best_t:
                    comptime if mode == TRACE_SHADOW:
                        return True
                    else:
                        hit_any = True
                        best_t = h.t
                        comptime if mode == TRACE_PRIMARY_FULL:
                            best_u = 0.0
                            best_v = 0.0
                            best_prim = prim

    return hit_any


def pack_sphere_leaf_blocks_kernel[
    width: Int,
](
    spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_count: Int,
):
    var block_idx = global_idx.x
    if block_idx >= leaf_block_count:
        return

    comptime for lane in range(width):
        var idx = block_idx * width + lane
        var prim = UInt32(leaf_block_indices[idx])
        leaf_prims[idx] = prim

        var out_base = idx * GPU_SPHERE_STRIDE
        if prim == GPU_WIDE_EMPTY_LANE:
            for k in range(GPU_SPHERE_STRIDE):
                leaf_spheres[out_base + k] = 0.0
        else:
            var in_base = Int(prim) * GPU_SPHERE_STRIDE
            for k in range(GPU_SPHERE_STRIDE):
                leaf_spheres[out_base + k] = spheres[in_base + k]


def _flatten_spheres(spheres: List[Sphere]) -> List[Float32]:
    var out = List[Float32](capacity=max(len(spheres), 1) * GPU_SPHERE_STRIDE)
    for i in range(len(spheres)):
        out.append(spheres[i].center.x)
        out.append(spheres[i].center.y)
        out.append(spheres[i].center.z)
        out.append(spheres[i].radius)
    return out^
