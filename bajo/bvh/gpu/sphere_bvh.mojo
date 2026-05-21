from std.math import ceildiv
from std.time import perf_counter_ns
from std.gpu import DeviceBuffer, DeviceContext, global_idx
from std.utils.numerics import max_finite

from bajo.bvh.constants import (
    EMPTY_LANE,
    TRACE_CLOSEST_HIT,
    TRACE_ANY_HIT,
    SPHERE_STRIDE,
    BOUNDS_STRIDE,
)
from bajo.core.intersect import intersect_ray_sphere
from bajo.core.vec import Vec3f32, Vec3
from bajo.bvh.types import Sphere, Ray, Hit, SphereLeafBlock
from bajo.bvh.gpu.bounds_bvh import (
    GpuBoundsBvh,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)
from bajo.bvh.gpu.trace import trace_bounds_bvh
from bajo.bvh.host_utils import copy_list_to_device


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
        self.spheres = copy_list_to_device(ctx, flat_spheres)

        var leaf_bounds = List[Float32](
            capacity=max(self.sphere_count, 1) * BOUNDS_STRIDE
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
            self.tree.max_leaf_blocks * Self.width * SPHERE_STRIDE
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

    var ray = Ray(rays, ray_idx)
    var hit = trace_bounds_bvh[
        width,
        TRACE_CLOSEST_HIT,
        _intersect_sphere_leaf[
            width,
            TRACE_CLOSEST_HIT,
        ],
    ](
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

    var ray = Ray(rays, ray_idx)
    var hit = trace_bounds_bvh[
        width,
        TRACE_ANY_HIT,
        _intersect_sphere_leaf[
            width,
            TRACE_ANY_HIT,
        ],
    ](
        wide_bounds,
        wide_data,
        wide_counts,
        leaf_spheres,
        leaf_prims,
        root_idx,
        ray,
    )

    flags[ray_idx] = hit.occluded


def _load_sphere_leaf_packet[
    width: Int,
](
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_idx: UInt32,
    item_count: UInt32,
) -> SphereLeafBlock[width]:
    var block = SphereLeafBlock[width]()

    comptime for lane in range(width):
        if lane < Int(item_count):
            var idx = Int(leaf_block_idx) * width + lane
            var prim = UInt32(leaf_prims[idx])

            if prim != EMPTY_LANE:
                var base = idx * SPHERE_STRIDE

                block.center.x[lane] = leaf_spheres[base + 0]
                block.center.y[lane] = leaf_spheres[base + 1]
                block.center.z[lane] = leaf_spheres[base + 2]
                block.radius[lane] = leaf_spheres[base + 3]

                block.prim_indices[lane] = prim
                block.valid_lane[lane] = True

    return block^


def _intersect_sphere_leaf[
    width: Int,
    mode: String,
](
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: Ray,
    mut hit: Hit,
) capturing -> Bool:
    comptime assert mode in [TRACE_CLOSEST_HIT, TRACE_ANY_HIT]

    var block = _load_sphere_leaf_packet[width](
        leaf_spheres,
        leaf_prims,
        leaf_block_idx,
        item_count,
    )

    var O = Vec3[DType.float32, width](ray.o.x, ray.o.y, ray.o.z)
    var D = Vec3[DType.float32, width](ray.d.x, ray.d.y, ray.d.z)

    var h = intersect_ray_sphere(
        O,
        D,
        block.center,
        block.radius,
        hit.t,
    )

    var eps_mask = h.t.gt(1.0e-4)
    var t_min_mask = h.t.ge(ray.t_min)
    var hit_mask = h.mask & block.valid_lane & eps_mask & t_min_mask

    if not hit_mask.reduce_or():
        return False

    comptime if mode == TRACE_ANY_HIT:
        return True
    else:
        comptime f32_max = max_finite[DType.float32]()
        var min_t = hit_mask.select(h.t, f32_max).reduce_min()

        if min_t < hit.t:
            hit.t = min_t

            comptime if mode == TRACE_CLOSEST_HIT:
                hit.u = 0.0
                hit.v = 0.0
                hit.inst = EMPTY_LANE
                hit.occluded = UInt32(0)

                comptime for lane in range(width):
                    if hit_mask[lane] and h.t[lane] == min_t:
                        hit.prim = block.prim_indices[lane]

        return True


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

        var out_base = idx * SPHERE_STRIDE
        if prim == EMPTY_LANE:
            for k in range(SPHERE_STRIDE):
                leaf_spheres[out_base + k] = 0.0
        else:
            var in_base = Int(prim) * SPHERE_STRIDE
            for k in range(SPHERE_STRIDE):
                leaf_spheres[out_base + k] = spheres[in_base + k]


def _flatten_spheres(spheres: List[Sphere]) -> List[Float32]:
    var out = List[Float32](capacity=max(len(spheres), 1) * SPHERE_STRIDE)
    for sphere in spheres:
        out.append(sphere.center.x)
        out.append(sphere.center.y)
        out.append(sphere.center.z)
        out.append(sphere.radius)
    return out^
