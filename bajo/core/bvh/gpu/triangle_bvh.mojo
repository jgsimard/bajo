from std.math import ceildiv, sqrt
from std.time import perf_counter_ns
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.core.vec import Vec3f32, vmin, vmax
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
    GPU_TRI_LEAF_VERTEX_STRIDE,
)
from bajo.core.intersect import intersect_ray_tri


struct GpuTriangleBvh[width: Int]:
    var tree: GpuBoundsBvh[Self.width]
    var vertices: DeviceBuffer[DType.float32]
    var leaf_vertices: DeviceBuffer[DType.float32]
    var leaf_prims: DeviceBuffer[DType.uint32]
    var tri_count: Int
    var leaf_pack_ns: Int

    def __init__(
        out self,
        mut ctx: DeviceContext,
        tri_vertices: List[Vec3f32],
    ) raises:
        self.tri_count = len(tri_vertices) / 3
        self.leaf_pack_ns = 0

        var flat_vertices = _flatten_vertices(tri_vertices)
        self.vertices = _copy_f32_to_device(ctx, flat_vertices)

        var leaf_bounds = List[Float32](
            capacity=max(self.tri_count, 1) * GPU_WIDE_BOUNDS_STRIDE
        )
        var payloads = List[UInt32](capacity=max(self.tri_count, 1))

        for i in range(self.tri_count):
            ref v0 = tri_vertices[i * 3 + 0]
            ref v1 = tri_vertices[i * 3 + 1]
            ref v2 = tri_vertices[i * 3 + 2]
            var bmin = vmin(vmin(v0, v1), v2)
            var bmax = vmax(vmax(v0, v1), v2)
            leaf_bounds.append(bmin.x)
            leaf_bounds.append(bmin.y)
            leaf_bounds.append(bmin.z)
            leaf_bounds.append(bmax.x)
            leaf_bounds.append(bmax.y)
            leaf_bounds.append(bmax.z)
            payloads.append(UInt32(i))

        self.tree = GpuBoundsBvh[Self.width](ctx, leaf_bounds, payloads)
        _ = self.tree.build(ctx)

        self.leaf_vertices = ctx.enqueue_create_buffer[DType.float32](
            self.tree.max_leaf_blocks * Self.width * GPU_TRI_LEAF_VERTEX_STRIDE
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
        ctx.enqueue_function[pack_triangle_leaf_blocks_kernel[Self.width]](
            self.vertices.unsafe_ptr(),
            self.tree.leaf_block_indices.unsafe_ptr(),
            self.leaf_vertices.unsafe_ptr(),
            self.leaf_prims.unsafe_ptr(),
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
        ctx.enqueue_function[trace_gpu_triangle_bvh_primary_kernel[Self.width]](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.leaf_vertices.unsafe_ptr(),
            self.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            d_rays.unsafe_ptr(),
            d_hits_f32.unsafe_ptr(),
            d_hits_u32.unsafe_ptr(),
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
        ctx.enqueue_function[trace_gpu_triangle_bvh_shadow_kernel[Self.width]](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.leaf_vertices.unsafe_ptr(),
            self.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            d_rays.unsafe_ptr(),
            d_flags.unsafe_ptr(),
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


def _flatten_vertices(verts: List[Vec3f32]) -> List[Float32]:
    var out = List[Float32](capacity=max(len(verts), 1) * 3)
    for i in range(len(verts)):
        out.append(verts[i].x)
        out.append(verts[i].y)
        out.append(verts[i].z)
    return out^


def pack_triangle_leaf_blocks_kernel[
    width: Int,
](
    vertices: UnsafePointer[Float32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
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

        var out_base = idx * GPU_TRI_LEAF_VERTEX_STRIDE
        if prim == GPU_WIDE_EMPTY_LANE:
            for k in range(GPU_TRI_LEAF_VERTEX_STRIDE):
                leaf_vertices[out_base + k] = 0.0
        else:
            var in_base = Int(prim) * GPU_TRI_LEAF_VERTEX_STRIDE
            for k in range(GPU_TRI_LEAF_VERTEX_STRIDE):
                leaf_vertices[out_base + k] = vertices[in_base + k]


def trace_gpu_triangle_bvh_primary_kernel[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
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
    var hit = trace_gpu_wide_triangle_ray[width, TRACE_PRIMARY_FULL](
        wide_bounds,
        wide_data,
        wide_counts,
        leaf_vertices,
        leaf_prims,
        root_idx,
        ray,
    )

    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = hit.t
    hits_f32[hit_base + 1] = hit.u
    hits_f32[hit_base + 2] = hit.v
    hits_u32[ray_idx] = hit.prim


def trace_gpu_triangle_bvh_shadow_kernel[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
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
    var hit = trace_gpu_wide_triangle_ray[width, TRACE_SHADOW](
        wide_bounds,
        wide_data,
        wide_counts,
        leaf_vertices,
        leaf_prims,
        root_idx,
        ray,
    )

    flags[ray_idx] = hit.occluded


@always_inline
def trace_gpu_wide_triangle_ray[
    width: Int,
    mode: String = TRACE_PRIMARY_FULL,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
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
                        var leaf_hit = _intersect_triangle_leaf_block[
                            width, mode
                        ](
                            leaf_vertices,
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
def _intersect_triangle_leaf_block[
    width: Int,
    mode: String,
](
    leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
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
                var base = idx * GPU_TRI_LEAF_VERTEX_STRIDE
                var v0 = Vec3f32.load(leaf_vertices, base + 0)
                var v1 = Vec3f32.load(leaf_vertices, base + 3)
                var v2 = Vec3f32.load(leaf_vertices, base + 6)

                var tri_hit = intersect_ray_tri(
                    ray.o,
                    ray.d,
                    v0,
                    v1,
                    v2,
                    best_t,
                )

                if tri_hit.mask:
                    comptime if mode == TRACE_SHADOW:
                        return True
                    else:
                        hit_any = True
                        best_t = tri_hit.t
                        comptime if mode == TRACE_PRIMARY_FULL:
                            best_u = tri_hit.u
                            best_v = tri_hit.v
                            best_prim = prim

    return hit_any
