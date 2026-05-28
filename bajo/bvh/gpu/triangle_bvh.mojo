from std.math import ceildiv, max
from std.time import perf_counter_ns
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.bvh.camera import Camera
from bajo.bvh.gpu.utils import (
    GpuBuildTimings,
    upload_vertices,
    upload_list,
)
from bajo.core.vec import Vec3f32, vmin, vmax, normalize, cross
from bajo.bvh.types import Ray, Hit, BlasSet
from bajo.bvh.constants import (
    EMPTY_LANE,
    TRACE,
    TRI_LEAF_VERTEX_STRIDE,
    BLAS_DESC_STRIDE,
    BLAS_DESC_ROOT_IDX,
    BLAS_DESC_WIDE_BOUNDS_BASE,
    BLAS_DESC_WIDE_LANE_BASE,
    BLAS_DESC_LEAF_F32_BASE,
    BLAS_DESC_LEAF_U32_BASE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    BOUNDS_STRIDE,
)
from bajo.bvh.gpu.bounds_bvh import GpuBoundsBvh
from bajo.core.intersect import intersect_ray_tri
from bajo.bvh.gpu.trace import trace_bounds_bvh


struct GpuTriangleBlasSetBuilder[width: Int]:
    """Builds each BLAS, then pack them."""

    var vertex_sets: List[List[Vec3f32]]

    def __init__(out self):
        self.vertex_sets = List[List[Vec3f32]]()

    def add(mut self, vertices: List[Vec3f32]) -> UInt32:
        var idx = UInt32(len(self.vertex_sets))
        self.vertex_sets.append(vertices.copy())
        return idx

    def build(mut self, mut ctx: DeviceContext) raises -> BlasSet[Self.width]:
        if len(self.vertex_sets) == 0:
            var descs = List[UInt32](length=BLAS_DESC_STRIDE, fill=0)
            var dummy_f32 = [Float32(0.0)]
            var dummy_u32 = [EMPTY_LANE]

            return BlasSet[Self.width](
                upload_list(ctx, descs),
                upload_list(ctx, dummy_f32),
                upload_list(ctx, dummy_u32),
                upload_list(ctx, dummy_u32),
                upload_list(ctx, dummy_f32),
                upload_list(ctx, dummy_u32),
                0,
            )

        var descs = List[UInt32](
            capacity=len(self.vertex_sets) * BLAS_DESC_STRIDE
        )

        var total_wide_bounds = 0
        var total_wide_lanes = 0
        var total_leaf_vertices = 0
        var total_leaf_prims = 0

        # First pass: compute final packed offsets without building/downloading.
        for blas_idx in range(len(self.vertex_sets)):
            var tri_count = len(self.vertex_sets[blas_idx]) / 3
            var internal_count = max(tri_count - 1, 0)
            var max_wide_nodes = max(internal_count, 1)
            var max_leaf_blocks = max(internal_count * Self.width, 1)

            var wide_bounds_base = UInt32(total_wide_bounds)
            var wide_lane_base = UInt32(total_wide_lanes)
            var leaf_f32_base = UInt32(total_leaf_vertices)
            var leaf_u32_base = UInt32(total_leaf_prims)

            descs.append(wide_bounds_base)
            descs.append(wide_lane_base)
            descs.append(leaf_f32_base)
            descs.append(leaf_u32_base)

            # Filled after the actual GPU BLAS build.
            descs.append(UInt32(0))  # BLAS_DESC_ROOT_IDX

            descs.append(UInt32(max_wide_nodes))
            descs.append(UInt32(max_leaf_blocks))
            descs.append(UInt32(tri_count))

            total_wide_bounds += max_wide_nodes * Self.width * BOUNDS_STRIDE
            total_wide_lanes += max_wide_nodes * Self.width
            total_leaf_vertices += (
                max_leaf_blocks * Self.width * TRI_LEAF_VERTEX_STRIDE
            )
            total_leaf_prims += max_leaf_blocks * Self.width

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
        for blas_idx in range(len(self.vertex_sets)):
            var d_vertices = upload_vertices(ctx, self.vertex_sets[blas_idx])
            var blas = GpuTriangleBvh[Self.width](ctx, d_vertices)

            var desc_base = blas_idx * BLAS_DESC_STRIDE

            descs[desc_base + BLAS_DESC_ROOT_IDX] = blas.tree.root_idx

            var wide_bounds_base = Int(
                descs[desc_base + BLAS_DESC_WIDE_BOUNDS_BASE]
            )
            var wide_lane_base = Int(
                descs[desc_base + BLAS_DESC_WIDE_LANE_BASE]
            )
            var leaf_f32_base = Int(descs[desc_base + BLAS_DESC_LEAF_F32_BASE])
            var leaf_u32_base = Int(descs[desc_base + BLAS_DESC_LEAF_U32_BASE])

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

        return BlasSet[Self.width](
            upload_list(ctx, descs),
            packed_wide_bounds,
            packed_wide_data,
            packed_wide_counts,
            packed_leaf_vertices,
            packed_leaf_prims,
            len(self.vertex_sets),
        )


struct GpuTriangleBvh[width: Int]:
    var tree: GpuBoundsBvh[Self.width]
    var vertices: DeviceBuffer[DType.float32]
    var leaf_vertices: DeviceBuffer[DType.float32]
    var leaf_prims: DeviceBuffer[DType.uint32]
    var tri_count: Int
    var leaf_pack_ns: Int
    var bounds_pack_ns: Int
    var timings: GpuBuildTimings

    def __init__(
        out self,
        mut ctx: DeviceContext,
        vertices: DeviceBuffer[DType.float32],
    ) raises:
        self.vertices = vertices
        self.tri_count = len(vertices) / 9
        self.leaf_pack_ns = 0
        self.bounds_pack_ns = 0

        var leaf_bounds = ctx.enqueue_create_buffer[DType.float32](
            self.tri_count * BOUNDS_STRIDE
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
        self.bounds_pack_ns = Int(perf_counter_ns() - start)

        self.tree = GpuBoundsBvh[Self.width](ctx, leaf_bounds, payloads)
        self.timings = self.tree.build(ctx)

        self.leaf_vertices = ctx.enqueue_create_buffer[DType.float32](
            self.tree.max_leaf_blocks * Self.width * TRI_LEAF_VERTEX_STRIDE
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
            max(self.tree.leaf_block_count, 1),
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.enqueue_function[pack_triangle_leaf_blocks_kernel[Self.width]](
            self.vertices,
            self.tree.leaf_block_indices,
            self.leaf_vertices,
            self.leaf_prims,
            self.tree.leaf_block_count,
            grid_dim=blocks,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        self.leaf_pack_ns = Int(perf_counter_ns() - start)

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

    var bbase = tri_idx * BOUNDS_STRIDE

    leaf_bounds[bbase + 0] = bmin.x
    leaf_bounds[bbase + 1] = bmin.y
    leaf_bounds[bbase + 2] = bmin.z
    leaf_bounds[bbase + 3] = bmax.x
    leaf_bounds[bbase + 4] = bmax.y
    leaf_bounds[bbase + 5] = bmax.z

    payloads[tri_idx] = UInt32(tri_idx)


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

        var out_base = idx * TRI_LEAF_VERTEX_STRIDE
        if prim == EMPTY_LANE:
            for k in range(TRI_LEAF_VERTEX_STRIDE):
                leaf_vertices[out_base + k] = 0.0
        else:
            var in_base = Int(prim) * TRI_LEAF_VERTEX_STRIDE
            for k in range(TRI_LEAF_VERTEX_STRIDE):
                leaf_vertices[out_base + k] = vertices[in_base + k]


def trace_triangle_bvh_camera_kernel[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    root_idx: UInt32,
    camera_params: UnsafePointer[Float32, MutAnyOrigin],
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


# this version works !!!!
def _intersect_triangle_leaf[
    width: Int,
    mode: TRACE,
](
    leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: Ray,
    mut hit: Hit,
) capturing -> Bool:
    var any_hit = False

    comptime for lane in range(width):
        if lane < Int(item_count):
            var idx = Int(leaf_block_idx) * width + lane
            var prim = UInt32(leaf_prims[idx])

            if prim != EMPTY_LANE:
                var base = idx * TRI_LEAF_VERTEX_STRIDE

                var v0 = Vec3f32.load(leaf_vertices, base)
                var v1 = Vec3f32.load(leaf_vertices, base + 3)
                var v2 = Vec3f32.load(leaf_vertices, base + 6)

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
                        hit.prim = prim
                        hit.inst = EMPTY_LANE
                        hit.normal = normalize(cross(v1 - v0, v2 - v0))
                        any_hit = True

    return any_hit


# I dont know why but this version doesnt work :((((
# def _load_triangle_leaf[
#     width: Int,
# ](
#     leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
#     leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
#     leaf_block_idx: UInt32,
#     item_count: UInt32,
# ) -> TriangleLeafBlock[width]:
#     var block = TriangleLeafBlock[width]()

#     comptime for lane in range(width):
#         if lane < Int(item_count):
#             var idx = Int(leaf_block_idx) * width + lane
#             var prim = UInt32(leaf_prims[idx])

#             if prim != EMPTY_LANE:
#                 var base = idx * TRI_LEAF_VERTEX_STRIDE

#                 block.v0.x[lane] = leaf_vertices[base + 0]
#                 block.v0.y[lane] = leaf_vertices[base + 1]
#                 block.v0.z[lane] = leaf_vertices[base + 2]

#                 block.v1.x[lane] = leaf_vertices[base + 3]
#                 block.v1.y[lane] = leaf_vertices[base + 4]
#                 block.v1.z[lane] = leaf_vertices[base + 5]

#                 block.v2.x[lane] = leaf_vertices[base + 6]
#                 block.v2.y[lane] = leaf_vertices[base + 7]
#                 block.v2.z[lane] = leaf_vertices[base + 8]

#                 block.prim_indices[lane] = prim
#                 block.valid_lane[lane] = True

#     return block^


# def _intersect_triangle_leaf[
#     width: Int,
#     mode: String,
# ](
#     leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
#     leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
#     leaf_block_idx: UInt32,
#     item_count: UInt32,
#     ray: Ray,
#     mut hit: Hit,
# ) capturing -> Bool:
#     comptime assert mode in [TRACE_CLOSEST_HIT, TRACE_ANY_HIT]

#     var block = _load_triangle_leaf[width](
#         leaf_vertices,
#         leaf_prims,
#         leaf_block_idx,
#         item_count,
#     )

#     var O = Vec3[DType.float32, width](ray.o.x, ray.o.y, ray.o.z)
#     var D = Vec3[DType.float32, width](ray.d.x, ray.d.y, ray.d.z)

#     var h = intersect_ray_tri(
#         O,
#         D,
#         block.v0,
#         block.v1,
#         block.v2,
#         hit.t,
#     )

#     var t_min_mask = h.t.ge(ray.t_min)
#     var hit_mask = h.mask & block.valid_lane & t_min_mask

#     if not hit_mask.reduce_or():
#         return False

#     comptime if mode == TRACE_ANY_HIT:
#         return True
#     else:
#         var min_t = hit_mask.select(h.t, f32_max).reduce_min()

#         if min_t < hit.t:
#             hit.t = min_t

#             comptime if mode == TRACE_CLOSEST_HIT:
#                 comptime for lane in range(width):
#                     if hit_mask[lane] and h.t[lane] == min_t:
#                         hit.u = h.u[lane]
#                         hit.v = h.v[lane]
#                         hit.prim = block.prim_indices[lane]
#                         hit.inst = EMPTY_LANE
#         return True
