from std.math import ceildiv, max
from std.time import perf_counter_ns
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.bvh.camera import Camera
from bajo.bvh.gpu.utils import (
    GpuBuildTimings,
    upload_vertices,
    upload_list,
)
from bajo.core import AABB, Vec3f32, vmin, vmax, normalize, cross, Vec3
from bajo.bvh.types import Ray, Hit, BlasSet, TriangleLeafBlock
from bajo.bvh.constants import (
    EMPTY_LANE,
    TRACE,
    TRI_LEAF_VERTEX_STRIDE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    f32_max,
)
from bajo.bvh.gpu.bounds_bvh import GpuBoundsBvh
from bajo.core.intersect import intersect_ray_tri
from bajo.bvh.gpu.trace import trace_bounds_bvh
from bajo.core.utils import min_argmin


def build_triangle_blas_set[
    width: SIMDSize
](mut ctx: DeviceContext, vertex_sets: List[List[Vec3f32]]) raises -> BlasSet[
    width
]:
    if len(vertex_sets) == 0:
        var descs = List[UInt32](length=BlasSet.STRIDE, fill=0)
        var dummy_f32 = [Float32(0.0)]
        var dummy_u32 = [EMPTY_LANE]

        return BlasSet[width](
            upload_list(ctx, descs),
            upload_list(ctx, dummy_f32),
            upload_list(ctx, dummy_u32),
            upload_list(ctx, dummy_u32),
            upload_list(ctx, dummy_f32),
            upload_list(ctx, dummy_u32),
            0,
        )

    var descs = List[UInt32](capacity=len(vertex_sets) * BlasSet.STRIDE)

    var total_wide_bounds = 0
    var total_wide_lanes = 0
    var total_leaf_vertices = 0
    var total_leaf_prims = 0

    # First pass: compute final packed offsets without building/downloading.
    for blas_idx in range(len(vertex_sets)):
        var tri_count = len(vertex_sets[blas_idx]) / 3
        var internal_count = max(tri_count - 1, 0)
        var max_wide_nodes = max(internal_count, 1)
        var max_leaf_blocks = max(internal_count * width, 1)

        var wide_bounds_base = UInt32(total_wide_bounds)
        var wide_lane_base = UInt32(total_wide_lanes)
        var leaf_f32_base = UInt32(total_leaf_vertices)
        var leaf_u32_base = UInt32(total_leaf_prims)

        descs.append(wide_bounds_base)
        descs.append(wide_lane_base)
        descs.append(leaf_f32_base)
        descs.append(leaf_u32_base)

        # Filled after the actual GPU BLAS build.
        descs.append(UInt32(0))  # BlasSet.ROOT_IDX

        descs.append(UInt32(max_wide_nodes))
        descs.append(UInt32(max_leaf_blocks))
        descs.append(UInt32(tri_count))

        total_wide_bounds += max_wide_nodes * width * AABB.STRIDE
        total_wide_lanes += max_wide_nodes * width
        total_leaf_vertices += max_leaf_blocks * width * TRI_LEAF_VERTEX_STRIDE
        total_leaf_prims += max_leaf_blocks * width

    var packed_wide_bounds = ctx.enqueue_create_buffer[DType.float32](
        max(total_wide_bounds, 1)
    )
    var packed_wide_data = ctx.enqueue_create_buffer[DType.uint32](
        max(total_wide_lanes, 1)
    )
    var packed_wide_counts = ctx.enqueue_create_buffer[DType.uint32](
        max(total_wide_lanes, 1)
    )
    var packed_leaf_vertices = ctx.enqueue_create_buffer[DType.float32](
        max(total_leaf_vertices, 1)
    )
    var packed_leaf_prims = ctx.enqueue_create_buffer[DType.uint32](
        max(total_leaf_prims, 1)
    )

    # Second pass: build each BLAS, then copy its device buffers into the
    # final packed device buffers. We synchronize inside the loop so the
    # temporary BLAS buffers stay alive until their copy kernels finish.
    for blas_idx in range(len(vertex_sets)):
        var d_vertices = upload_vertices(ctx, vertex_sets[blas_idx])
        var blas = GpuTriangleBvh[width](ctx, d_vertices)

        var desc_base = blas_idx * BlasSet.STRIDE

        descs[desc_base + BlasSet.ROOT_IDX] = blas.tree.root_idx

        var wide_bounds_base = Int(descs[desc_base + BlasSet.WIDE_BOUNDS_BASE])
        var wide_lane_base = Int(descs[desc_base + BlasSet.WIDE_LANE_BASE])
        var leaf_f32_base = Int(descs[desc_base + BlasSet.LEAF_F32_BASE])
        var leaf_u32_base = Int(descs[desc_base + BlasSet.LEAF_U32_BASE])

        blas.tree.wide_bounds.enqueue_copy_to(
            packed_wide_bounds.unsafe_ptr() + wide_bounds_base
        )
        blas.tree.wide_data.enqueue_copy_to(
            packed_wide_data.unsafe_ptr() + wide_lane_base
        )
        blas.tree.wide_counts.enqueue_copy_to(
            packed_wide_counts.unsafe_ptr() + wide_lane_base
        )
        blas.leaf_vertices.enqueue_copy_to(
            packed_leaf_vertices.unsafe_ptr() + leaf_f32_base
        )
        blas.leaf_prims.enqueue_copy_to(
            packed_leaf_prims.unsafe_ptr() + leaf_u32_base
        )

        ctx.synchronize()

    return BlasSet[width](
        upload_list(ctx, descs),
        packed_wide_bounds,
        packed_wide_data,
        packed_wide_counts,
        packed_leaf_vertices,
        packed_leaf_prims,
        len(vertex_sets),
    )


struct GpuTriangleBvh[width: SIMDSize](Movable):
    var tree: GpuBoundsBvh[Self.width]
    var vertices: DeviceBuffer[DType.float32]
    var leaf_vertices: DeviceBuffer[DType.float32]
    var leaf_prims: DeviceBuffer[DType.uint32]
    var tri_count: Int
    var timings: GpuBuildTimings

    def __init__(
        out self,
        mut ctx: DeviceContext,
        vertices: DeviceBuffer[DType.float32],
    ) raises:
        self.vertices = vertices
        self.tri_count = len(vertices) / 9

        var leaf_bounds = ctx.enqueue_create_buffer[DType.float32](
            self.tri_count * AABB.STRIDE
        )
        var payloads = ctx.enqueue_create_buffer[DType.uint32](self.tri_count)

        var start = perf_counter_ns()
        var blocks = ceildiv(
            max(self.tri_count, 1),
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        ctx.enqueue_function[compute_triangle_bounds_kernel](
            self.vertices,
            leaf_bounds,
            payloads,
            self.tri_count,
            grid_dim=blocks,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        var bounds_pack_ns = Int(perf_counter_ns() - start)

        self.tree = GpuBoundsBvh[Self.width](ctx, leaf_bounds, payloads)
        self.timings = self.tree.build(ctx)
        self.timings.bounds_pack_ns = bounds_pack_ns

        var leaf_block_capacity = max(self.tree.leaf_block_count, 1)
        self.leaf_vertices = ctx.enqueue_create_buffer[DType.float32](
            leaf_block_capacity * Self.width * TRI_LEAF_VERTEX_STRIDE
        )
        self.leaf_prims = ctx.enqueue_create_buffer[DType.uint32](
            leaf_block_capacity * Self.width
        )

        self._pack_leaf_blocks(ctx)

        # print(t"tri_count = {self.tri_count}")
        # print(t"leaf_block_count = {self.tree.leaf_block_count}")
        # print(t"max_leaf_blocks = {self.tree.max_leaf_blocks}")
        # print(t"packed leaf lanes = {self.tree.leaf_block_count * Self.width}")
        # print(
        #     t"leaf lanes / triangles = "
        #     t"{Float64(self.tree.leaf_block_count * Self.width) / Float64(self.tri_count)}"
        # )

    def _pack_leaf_blocks(
        mut self,
        ctx: DeviceContext,
    ) raises:
        var start = perf_counter_ns()
        var leaf_lane_count = max(self.tree.leaf_block_count * Self.width, 1)
        var blocks = ceildiv(
            leaf_lane_count,
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.enqueue_function[pack_triangle_leaf_lanes_kernel[Self.width]](
            self.vertices,
            self.tree.leaf_block_indices,
            self.leaf_vertices,
            self.leaf_prims,
            leaf_lane_count,
            grid_dim=blocks,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        self.timings.leaf_pack_ns = Int(perf_counter_ns() - start)

    def launch_camera(
        self,
        ctx: DeviceContext,
        d_camera_params: DeviceBuffer[DType.float32],
        d_hits_f32: DeviceBuffer[DType.float32],
        d_hits_u32: DeviceBuffer[DType.uint32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        ctx.enqueue_function[trace_triangle_bvh_camera_kernel[Self.width]](
            self.tree.wide_bounds,
            self.tree.wide_data,
            self.tree.wide_counts,
            self.leaf_vertices,
            self.leaf_prims,
            self.tree.root_idx,
            d_camera_params,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            cwidth,
            cheight,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


def compute_triangle_bounds_kernel(
    vertices: UnsafePointer[Float32, MutAnyOrigin],
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    payloads: UnsafePointer[UInt32, MutAnyOrigin],
    tri_count: Int,
):
    var tri_idx = global_idx.x
    if tri_idx >= tri_count:
        return

    var vbase = tri_idx * TRI_LEAF_VERTEX_STRIDE

    var v0 = Vec3f32.load(vertices, vbase + 0)
    var v1 = Vec3f32.load(vertices, vbase + 3)
    var v2 = Vec3f32.load(vertices, vbase + 6)

    var bmin = vmin(vmin(v0, v1), v2)
    var bmax = vmax(vmax(v0, v1), v2)

    var bbase = tri_idx * AABB.STRIDE

    leaf_bounds[bbase + 0] = bmin.x
    leaf_bounds[bbase + 1] = bmin.y
    leaf_bounds[bbase + 2] = bmin.z
    leaf_bounds[bbase + 3] = bmax.x
    leaf_bounds[bbase + 4] = bmax.y
    leaf_bounds[bbase + 5] = bmax.z

    payloads[tri_idx] = UInt32(tri_idx)


def pack_triangle_leaf_lanes_kernel[
    width: SIMDSize,
](
    vertices: UnsafePointer[Float32, ImmutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, ImmutAnyOrigin],
    leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_lane_count: Int,
):
    var lane_idx = global_idx.x
    if lane_idx >= leaf_lane_count:
        return

    var prim = leaf_block_indices[lane_idx]
    leaf_prims[lane_idx] = prim

    # traversal checks prim != EMPTY_LANE
    if prim == EMPTY_LANE:
        return

    # AoSoA : [block][field][lane]
    var lane = lane_idx % width
    var leaf_block_idx = lane_idx / width
    var in_base = Int(prim) * TRI_LEAF_VERTEX_STRIDE
    var out_base = leaf_block_idx * TRI_LEAF_VERTEX_STRIDE * width

    comptime for field in range(TRI_LEAF_VERTEX_STRIDE):
        leaf_vertices[out_base + field * width + lane] = vertices[
            in_base + field
        ]


def trace_triangle_bvh_camera_kernel[
    width: SIMDSize,
](
    wide_bounds: UnsafePointer[Float32, ImmutAnyOrigin],
    wide_data: UnsafePointer[UInt32, ImmutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, ImmutAnyOrigin],
    leaf_vertices: UnsafePointer[Float32, ImmutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, ImmutAnyOrigin],
    root_idx: UInt32,
    camera_params: UnsafePointer[Float32, ImmutAnyOrigin],
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
    width_px: Int,
    height_px: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var pixels_per_view = width_px * height_px
    var view_idx = ray_idx / pixels_per_view
    var local_idx = ray_idx - view_idx * pixels_per_view
    var px_i = local_idx % width_px
    var py_i = local_idx / width_px

    var camera = Camera(camera_params, view_idx * Camera.STRIDE)
    var ray = camera.make_ray(px_i, py_i, width_px, height_px)

    var hit = trace_bounds_bvh[
        width,
        TRACE.CLOSEST_HIT,
        _intersect_triangle_leaf[
            width,
            TRACE.CLOSEST_HIT,
        ],
    ](
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


# AoSoA :[block][field][lane]
def _intersect_triangle_leaf[
    width: SIMDSize,
    mode: TRACE,
](
    leaf_vertices: UnsafePointer[Float32, ImmutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, ImmutAnyOrigin],
    leaf_block_idx: UInt32,
    ray: Ray,
    mut hit: Hit,
) capturing -> Bool:
    var any_hit = False
    var block_base = Int(leaf_block_idx) * TRI_LEAF_VERTEX_STRIDE * width
    var prim_base = Int(leaf_block_idx) * width
    var prims = leaf_prims.load[width=width](prim_base)

    comptime for lane in range(width):
        if prims[lane] != EMPTY_LANE:
            var v0 = Vec3f32(
                leaf_vertices[block_base + 0 * width + lane],
                leaf_vertices[block_base + 1 * width + lane],
                leaf_vertices[block_base + 2 * width + lane],
            )
            var v1 = Vec3f32(
                leaf_vertices[block_base + 3 * width + lane],
                leaf_vertices[block_base + 4 * width + lane],
                leaf_vertices[block_base + 5 * width + lane],
            )
            var v2 = Vec3f32(
                leaf_vertices[block_base + 6 * width + lane],
                leaf_vertices[block_base + 7 * width + lane],
                leaf_vertices[block_base + 8 * width + lane],
            )

            var tri_hit = intersect_ray_tri(
                ray.o,
                ray.d,
                v0,
                v1,
                v2,
                hit.t,
                ray.t_min,
            )

            if tri_hit.mask and tri_hit.t < hit.t:
                hit.t = tri_hit.t
                comptime if mode == TRACE.ANY_HIT:
                    return True
                else:
                    hit.u = tri_hit.u
                    hit.v = tri_hit.v
                    hit.prim = prims[lane]
                    hit.inst = EMPTY_LANE
                    hit.normal = normalize(cross(v1 - v0, v2 - v0))
                    any_hit = True
    return any_hit


# # it works now, but it is slower
# def _intersect_triangle_leaf[
#     width: SIMDSize,
#     mode: TRACE,
# ](
#     leaf_vertices: UnsafePointer[Float32, ImmutAnyOrigin],
#     leaf_prims: UnsafePointer[UInt32, ImmutAnyOrigin],
#     leaf_block_idx: UInt32,
#     item_count: UInt32,
#     ray: Ray,
#     mut hit: Hit,
# ) capturing -> Bool:
#     var block_base = Int(leaf_block_idx) * TRI_LEAF_VERTEX_STRIDE * width
#     var prim_base = Int(leaf_block_idx) * width

#     var prim_indices = leaf_prims.load[width=width](prim_base)

#     var valid_lane = prim_indices.ne(EMPTY_LANE)

#     comptime for lane in range(width):
#         if lane >= Int(item_count):
#             valid_lane[lane] = False

#     if not valid_lane.reduce_or():
#         return False

#     # AoSoA layout:
#     #   [block][field][lane]
#     #
#     # fields:
#     #   0..2 = v0.xyz
#     #   3..5 = v1.xyz
#     #   6..8 = v2.xyz
#     var v0 = Vec3[DType.float32, width](
#         leaf_vertices.load[width=width](block_base + 0 * width),
#         leaf_vertices.load[width=width](block_base + 1 * width),
#         leaf_vertices.load[width=width](block_base + 2 * width),
#     )

#     var v1 = Vec3[DType.float32, width](
#         leaf_vertices.load[width=width](block_base + 3 * width),
#         leaf_vertices.load[width=width](block_base + 4 * width),
#         leaf_vertices.load[width=width](block_base + 5 * width),
#     )

#     var v2 = Vec3[DType.float32, width](
#         leaf_vertices.load[width=width](block_base + 6 * width),
#         leaf_vertices.load[width=width](block_base + 7 * width),
#         leaf_vertices.load[width=width](block_base + 8 * width),
#     )

#     var O = Vec3[DType.float32, width](ray.o.x, ray.o.y, ray.o.z)
#     var D = Vec3[DType.float32, width](ray.d.x, ray.d.y, ray.d.z)

#     var h = intersect_ray_tri(
#         O,
#         D,
#         v0,
#         v1,
#         v2,
#         hit.t,
#     )

#     var hit_mask = h.mask & valid_lane

#     if not hit_mask.reduce_or():
#         return False

#     comptime if mode == TRACE.CLOSEST_HIT:
#         var lane_t = hit_mask.select(h.t, f32_max)
#         var min_t = lane_t.reduce_min()

#         if min_t >= hit.t:
#             return False

#         hit.t = min_t

#         comptime for lane in range(width):
#             if hit_mask[lane] and h.t[lane] == min_t:
#                 hit.u = h.u[lane]
#                 hit.v = h.v[lane]
#                 hit.prim = prim_indices[lane]
#                 hit.inst = EMPTY_LANE

#                 var sv0 = Vec3f32(v0.x[lane], v0.y[lane], v0.z[lane])
#                 var sv1 = Vec3f32(v1.x[lane], v1.y[lane], v1.z[lane])
#                 var sv2 = Vec3f32(v2.x[lane], v2.y[lane], v2.z[lane])
#                 hit.normal = normalize(cross(sv1 - sv0, sv2 - sv0))

#                 return True

#         return True
