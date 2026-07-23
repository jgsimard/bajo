from std.math import ceildiv, max
from std.time import perf_counter_ns
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.bvh.camera import Camera
from bajo.bvh.gpu.utils import (
    GpuBuildTimings,
    upload_vertices,
    upload_list,
)
from bajo.core import (
    AABB,
    Vec3f32,
    vmin,
    vmax,
    normalize,
    cross,
    Vec3,
    Point3f32,
    Frame,
    GeoKind,
    Rayf32,
)
from bajo.bvh.types import Hit, BlasSet, TriangleLeafBlock
from bajo.bvh.constants import (
    EMPTY_LANE,
    TRACE,
    TRI_LEAF_VERTEX_STRIDE,
    TRI_LEAF_PACKED_STRIDE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    f32_max,
    WideNode,
)
from bajo.bvh.gpu.bounds_bvh import GpuBoundsBvh
from bajo.core.intersect import intersect_ray_tri
from bajo.bvh.gpu.trace import trace_bounds_bvh
from bajo.core.utils import min_argmin


def build_triangle_blas_set[
    width: SIMDLength
](
    mut ctx: DeviceContext,
    vertex_sets: List[List[Point3f32[Frame.LOCAL]]],
) raises -> BlasSet[width]:
    debug_assert["safe"](len(vertex_sets) > 0)

    var descs = List[UInt32](capacity=len(vertex_sets) * BlasSet.STRIDE)

    var total_wide_nodes = 0
    var total_leaf_vertices = 0

    # First pass: compute final packed offsets without building/downloading.
    for blas_idx in range(len(vertex_sets)):
        var tri_count = len(vertex_sets[blas_idx]) / 3
        debug_assert["safe"](tri_count > 0)

        var internal_count = tri_count - 1
        var max_wide_nodes = max(internal_count, 1)
        var max_leaf_blocks = max(tri_count, 1)

        var wide_node_base = UInt32(total_wide_nodes)
        var leaf_f32_base = UInt32(total_leaf_vertices)
        descs.append(wide_node_base)
        descs.append(leaf_f32_base)

        # Filled after the actual GPU BLAS build.
        descs.append(UInt32(0))  # BlasSet.ROOT_IDX

        descs.append(UInt32(max_wide_nodes))
        descs.append(UInt32(max_leaf_blocks))
        descs.append(UInt32(tri_count))

        total_wide_nodes += max_wide_nodes * width * WideNode.CHILD_STRIDE
        total_leaf_vertices += max_leaf_blocks * width * TRI_LEAF_PACKED_STRIDE

    var wide_nodes = ctx.enqueue_create_buffer[DType.float32](total_wide_nodes)
    var leaf_vertices = ctx.enqueue_create_buffer[DType.float32](
        total_leaf_vertices
    )
    # Second pass: build each BLAS, then copy its device buffers into the
    # final packed device buffers. We synchronize inside the loop so the
    # temporary BLAS buffers stay alive until their copy kernels finish.
    for blas_idx in range(len(vertex_sets)):
        var d_vertices = upload_vertices(ctx, vertex_sets[blas_idx])
        var blas = GpuTriangleBvh[Frame.LOCAL, width](ctx, d_vertices)

        var desc_base = blas_idx * BlasSet.STRIDE

        descs[desc_base + BlasSet.ROOT_IDX] = blas.tree.root_idx
        descs[desc_base + BlasSet.NODE_COUNT] = UInt32(blas.tree.node_count)
        descs[desc_base + BlasSet.LEAF_BLOCK_COUNT] = UInt32(
            blas.tree.leaf_block_count
        )

        var wide_node_base = Int(descs[desc_base + BlasSet.WIDE_NODE_BASE])
        var leaf_f32_base = Int(descs[desc_base + BlasSet.LEAF_F32_BASE])
        blas.tree.wide_nodes.enqueue_copy_to(
            wide_nodes.unsafe_ptr() + wide_node_base
        )
        blas.leaf_vertices.enqueue_copy_to(
            leaf_vertices.unsafe_ptr() + leaf_f32_base
        )
        ctx.synchronize()

    return BlasSet[width](
        upload_list(ctx, descs),
        wide_nodes,
        leaf_vertices,
        len(vertex_sets),
    )


struct GpuTriangleBvh[frame: Frame, width: SIMDLength](Movable):
    var tree: GpuBoundsBvh[Self.width]
    var vertices: DeviceBuffer[DType.float32]
    var leaf_vertices: DeviceBuffer[DType.float32]
    var tri_count: Int
    var timings: GpuBuildTimings

    def __init__(
        out self,
        mut ctx: DeviceContext,
        vertices: DeviceBuffer[DType.float32],
        measure_build: Bool = False,
    ) raises:
        self.vertices = vertices
        self.tri_count = len(vertices) / 9

        var leaf_bounds = ctx.enqueue_create_buffer[DType.float32](
            self.tri_count * AABB[Self.frame].STRIDE
        )
        var payloads = ctx.enqueue_create_buffer[DType.uint32](self.tri_count)

        var bounds_pack_start = Int(0)
        if measure_build:
            ctx.synchronize()
            bounds_pack_start = perf_counter_ns()

        var blocks = ceildiv(
            max(self.tri_count, 1),
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        ctx.enqueue_function[compute_triangle_bounds_kernel[Self.frame]](
            self.vertices,
            leaf_bounds,
            payloads,
            self.tri_count,
            grid_dim=blocks,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        var bounds_pack_ns = Int(0)
        if measure_build:
            ctx.synchronize()
            bounds_pack_ns = Int(perf_counter_ns() - bounds_pack_start)

        self.tree = GpuBoundsBvh[Self.width](ctx, self.tri_count)
        self.timings = self.tree.build(
            ctx,
            leaf_bounds,
            payloads,
            measure_build=measure_build,
        )
        self.timings.bounds_pack_ns = bounds_pack_ns

        var leaf_block_capacity = max(self.tree.leaf_block_count, 1)
        self.leaf_vertices = ctx.enqueue_create_buffer[DType.float32](
            leaf_block_capacity * Self.width * TRI_LEAF_PACKED_STRIDE
        )

        self._pack_leaf_blocks(ctx, measure_build)

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
        measure_build: Bool,
    ) raises:
        var start = Int(0)
        if measure_build:
            start = perf_counter_ns()

        var leaf_lane_count = max(self.tree.leaf_block_count * Self.width, 1)
        var blocks = ceildiv(
            leaf_lane_count,
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.enqueue_function[pack_triangle_leaf_lanes_kernel[Self.width]](
            self.vertices,
            self.tree.leaf_block_indices,
            self.leaf_vertices,
            leaf_lane_count,
            grid_dim=blocks,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        if measure_build:
            ctx.synchronize()
            self.timings.leaf_pack_ns = Int(perf_counter_ns() - start)

    def launch_camera(
        self,
        ctx: DeviceContext,
        d_camera_params: DeviceBuffer[DType.float32],
        d_hits: DeviceBuffer[DType.float32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        comptime assert Self.frame == Frame.WORLD
        ctx.enqueue_function[trace_triangle_bvh_camera_kernel[Self.width]](
            self.tree.wide_nodes,
            self.leaf_vertices,
            self.tree.root_idx,
            d_camera_params,
            d_hits,
            ray_count,
            cwidth,
            cheight,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


def compute_triangle_bounds_kernel[
    frame: Frame,
](
    vertices: UnsafePointer[Float32, ImmutAnyOrigin],
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    payloads: UnsafePointer[UInt32, MutAnyOrigin],
    tri_count: Int,
):
    var tri_idx = global_idx.x
    if tri_idx >= tri_count:
        return

    var vbase = tri_idx * TRI_LEAF_VERTEX_STRIDE

    var v0 = Point3f32[frame].load(vertices, vbase + 0)
    var v1 = Point3f32[frame].load(vertices, vbase + 3)
    var v2 = Point3f32[frame].load(vertices, vbase + 6)

    var bmin = vmin(vmin(v0, v1), v2)
    var bmax = vmax(vmax(v0, v1), v2)

    var bbase = tri_idx * AABB[frame].STRIDE

    leaf_bounds[bbase + 0] = bmin.x
    leaf_bounds[bbase + 1] = bmin.y
    leaf_bounds[bbase + 2] = bmin.z
    leaf_bounds[bbase + 3] = bmax.x
    leaf_bounds[bbase + 4] = bmax.y
    leaf_bounds[bbase + 5] = bmax.z

    payloads[tri_idx] = UInt32(tri_idx)


def pack_triangle_leaf_lanes_kernel[
    width: SIMDLength,
](
    vertices: UnsafePointer[Float32, ImmutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, ImmutAnyOrigin],
    leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    leaf_lane_count: Int,
):
    var lane_idx = global_idx.x
    if lane_idx >= leaf_lane_count:
        return

    var prim = leaf_block_indices[lane_idx]

    # AoSoA : [block][field][lane]
    # Packed fields:
    #   0..2   = v0.xyz
    #   3      = prim id bits
    #   4..6   = v1.xyz
    #   7      = pad
    #   8..10  = v2.xyz
    #   11     = pad
    var lane = lane_idx % width
    var leaf_block_idx = lane_idx / width
    var out_base = leaf_block_idx * TRI_LEAF_PACKED_STRIDE * width
    var leaf_vertices_u32 = leaf_vertices.bitcast[UInt32]()

    leaf_vertices_u32[out_base + 3 * width + lane] = prim

    # traversal checks packed prim != EMPTY_LANE
    if prim == EMPTY_LANE:
        return

    var in_base = Int(prim) * TRI_LEAF_VERTEX_STRIDE

    leaf_vertices[out_base + 0 * width + lane] = vertices[in_base + 0]
    leaf_vertices[out_base + 1 * width + lane] = vertices[in_base + 1]
    leaf_vertices[out_base + 2 * width + lane] = vertices[in_base + 2]
    leaf_vertices[out_base + 4 * width + lane] = vertices[in_base + 3]
    leaf_vertices[out_base + 5 * width + lane] = vertices[in_base + 4]
    leaf_vertices[out_base + 6 * width + lane] = vertices[in_base + 5]
    leaf_vertices[out_base + 7 * width + lane] = 0.0
    leaf_vertices[out_base + 8 * width + lane] = vertices[in_base + 6]
    leaf_vertices[out_base + 9 * width + lane] = vertices[in_base + 7]
    leaf_vertices[out_base + 10 * width + lane] = vertices[in_base + 8]
    leaf_vertices[out_base + 11 * width + lane] = 0.0


def trace_triangle_bvh_camera_kernel[
    width: SIMDLength,
](
    wide_nodes: UnsafePointer[Float32, ImmutAnyOrigin],
    leaf_vertices: UnsafePointer[Float32, ImmutAnyOrigin],
    root_idx: UInt32,
    camera_params: UnsafePointer[Float32, ImmutAnyOrigin],
    hits: UnsafePointer[Float32, MutAnyOrigin],
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
        Frame.WORLD,
        width,
        TRACE.CLOSEST_HIT,
        _intersect_triangle_leaf[
            Frame.WORLD,
            width,
            TRACE.CLOSEST_HIT,
        ],
    ](
        wide_nodes,
        leaf_vertices,
        root_idx,
        ray,
    )
    hit.store(hits, ray_idx)


# AoSoA :[block][field][lane]
def _intersect_triangle_leaf[
    frame: Frame,
    width: SIMDLength,
    mode: TRACE,
](
    leaf_vertices: UnsafePointer[mut=False, Float32, _],
    leaf_block_idx: UInt32,
    ray: Rayf32[frame],
    mut hit: Hit[frame],
) capturing -> Bool:
    var any_hit = False
    var block_base = Int(leaf_block_idx) * TRI_LEAF_PACKED_STRIDE * width
    var leaf_vertices_u32 = leaf_vertices.bitcast[UInt32]()

    comptime for lane in range(width):
        var prim = leaf_vertices_u32[block_base + 3 * width + lane]
        if prim == EMPTY_LANE:
            continue
        var v0 = Point3f32[frame](
            leaf_vertices[block_base + 0 * width + lane],
            leaf_vertices[block_base + 1 * width + lane],
            leaf_vertices[block_base + 2 * width + lane],
        )
        var v1 = Point3f32[frame](
            leaf_vertices[block_base + 4 * width + lane],
            leaf_vertices[block_base + 5 * width + lane],
            leaf_vertices[block_base + 6 * width + lane],
        )
        var v2 = Point3f32[frame](
            leaf_vertices[block_base + 8 * width + lane],
            leaf_vertices[block_base + 9 * width + lane],
            leaf_vertices[block_base + 10 * width + lane],
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
                hit.prim = prim
                hit.inst = EMPTY_LANE
                hit.normal = normalize(cross(v1 - v0, v2 - v0)).unsafe_convert[
                    new_kind=GeoKind.NORMAL
                ]()
                any_hit = True
    return any_hit
