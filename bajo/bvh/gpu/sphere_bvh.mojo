from std.math import ceildiv, max
from std.time import perf_counter_ns
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.bvh.camera import Camera
from bajo.bvh.constants import (
    EMPTY_LANE,
    TRACE,
    f32_max,
    SPHERE_LEAF_PACKED_STRIDE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    WideNode,
)
from bajo.core.utils import min_argmin
from bajo.core import AABB, Vec3, Point3
from bajo.core.intersect import intersect_ray_sphere
from bajo.bvh.types import Sphere, Ray, Hit, BlasSet
from bajo.bvh.gpu.bounds_bvh import GpuBoundsBvh
from bajo.bvh.gpu.trace import trace_bounds_bvh
from bajo.bvh.gpu.utils import GpuBuildTimings, upload_list


def build_sphere_blas_set[
    width: SIMDSize
](mut ctx: DeviceContext, sphere_sets: List[List[Sphere]]) raises -> BlasSet[
    width
]:
    debug_assert["safe"](len(sphere_sets) > 0)

    var descs = List[UInt32](capacity=len(sphere_sets) * BlasSet.STRIDE)

    var total_wide_nodes = 0
    var total_leaf_spheres = 0

    # First pass: compute final packed offsets without building/downloading.
    for blas_idx in range(len(sphere_sets)):
        var sphere_count = len(sphere_sets[blas_idx])
        debug_assert["safe"](sphere_count > 0)

        var internal_count = sphere_count - 1
        var max_wide_nodes = max(internal_count, 1)
        var max_leaf_blocks = max(sphere_count, 1)

        var wide_node_base = UInt32(total_wide_nodes)
        var leaf_f32_base = UInt32(total_leaf_spheres)

        descs.append(wide_node_base)
        descs.append(leaf_f32_base)

        # Filled after the actual GPU BLAS build.
        descs.append(UInt32(0))  # BlasSet.ROOT_IDX

        descs.append(UInt32(max_wide_nodes))
        descs.append(UInt32(max_leaf_blocks))
        descs.append(UInt32(sphere_count))

        total_wide_nodes += max_wide_nodes * width * WideNode.CHILD_STRIDE
        total_leaf_spheres += (
            max_leaf_blocks * width * SPHERE_LEAF_PACKED_STRIDE
        )

    var wide_nodes = ctx.enqueue_create_buffer[DType.float32](total_wide_nodes)
    var leaf_spheres = ctx.enqueue_create_buffer[DType.float32](
        total_leaf_spheres
    )

    # Second pass: build each BLAS, then copy its device buffers into the
    # final packed device buffers.
    for blas_idx in range(len(sphere_sets)):
        var blas = GpuSphereBvh[width](ctx, sphere_sets[blas_idx])

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
        blas.leaf_spheres.enqueue_copy_to(
            leaf_spheres.unsafe_ptr() + leaf_f32_base
        )

        ctx.synchronize()

    return BlasSet[width](
        upload_list(ctx, descs),
        wide_nodes,
        leaf_spheres,
        len(sphere_sets),
    )


struct GpuSphereBvh[width: SIMDSize]:
    var tree: GpuBoundsBvh[Self.width]
    var spheres: DeviceBuffer[DType.float32]
    var leaf_spheres: DeviceBuffer[DType.float32]
    var sphere_count: Int
    var timings: GpuBuildTimings

    def __init__(
        out self,
        mut ctx: DeviceContext,
        spheres: List[Sphere],
    ) raises:
        self.sphere_count = len(spheres)

        var flat_spheres = _flatten_spheres(spheres)
        self.spheres = upload_list(ctx, flat_spheres)

        var leaf_bounds = List[Float32](
            capacity=max(self.sphere_count, 1) * AABB.STRIDE
        )
        var payloads = List[UInt32](capacity=max(self.sphere_count, 1))

        for i, s in enumerate(spheres):
            leaf_bounds.append(s.center.x - s.radius)
            leaf_bounds.append(s.center.y - s.radius)
            leaf_bounds.append(s.center.z - s.radius)
            leaf_bounds.append(s.center.x + s.radius)
            leaf_bounds.append(s.center.y + s.radius)
            leaf_bounds.append(s.center.z + s.radius)
            payloads.append(UInt32(i))

        var d_payloads = upload_list(ctx, payloads)
        var d_leaf_bounds = upload_list(ctx, leaf_bounds)

        self.tree = GpuBoundsBvh[Self.width](ctx, self.sphere_count)
        self.timings = self.tree.build(ctx, d_leaf_bounds, d_payloads)

        var leaf_block_capacity = max(self.tree.leaf_block_count, 1)
        self.leaf_spheres = ctx.enqueue_create_buffer[DType.float32](
            leaf_block_capacity * Self.width * SPHERE_LEAF_PACKED_STRIDE
        )
        self._pack_leaf_blocks(ctx)

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
        ctx.enqueue_function[pack_sphere_leaf_lanes_kernel[Self.width]](
            self.spheres,
            self.tree.leaf_block_indices,
            self.leaf_spheres,
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
        d_hits: DeviceBuffer[DType.float32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        ctx.enqueue_function[trace_sphere_bvh_camera_kernel[Self.width]](
            self.tree.wide_nodes,
            self.leaf_spheres,
            self.tree.root_idx,
            d_camera_params,
            d_hits,
            ray_count,
            cwidth,
            cheight,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


def trace_sphere_bvh_camera_kernel[
    width: SIMDSize,
](
    wide_nodes: UnsafePointer[Float32, ImmutAnyOrigin],
    leaf_spheres: UnsafePointer[Float32, ImmutAnyOrigin],
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
        width,
        TRACE.CLOSEST_HIT,
        _intersect_sphere_leaf[
            width,
            TRACE.CLOSEST_HIT,
        ],
    ](
        wide_nodes,
        leaf_spheres,
        root_idx,
        ray,
    )
    hit.store(hits, ray_idx)


def _intersect_sphere_leaf[
    width: SIMDSize,
    mode: TRACE,
](
    leaf_spheres: UnsafePointer[mut=False, Float32, _],
    leaf_block_idx: UInt32,
    ray: Ray,
    mut hit: Hit,
) capturing -> Bool:
    var block_base = Int(leaf_block_idx) * SPHERE_LEAF_PACKED_STRIDE * width
    var leaf_spheres_u32 = leaf_spheres.bitcast[UInt32]()

    var center = Point3(
        leaf_spheres.load[width=width](block_base + 0 * width),
        leaf_spheres.load[width=width](block_base + 1 * width),
        leaf_spheres.load[width=width](block_base + 2 * width),
    )
    var radius = leaf_spheres.load[width=width](block_base + 3 * width)
    var prim_indices = leaf_spheres_u32.load[width=width](
        block_base + 4 * width
    )

    var O = ray.origin[width]()
    var D = ray.direction[width]()

    var hit_sphere = intersect_ray_sphere(
        O, D, center, radius, hit.t, ray.t_min
    )
    var valid_lanes = prim_indices.ne(EMPTY_LANE)
    var hit_mask = hit_sphere.mask & valid_lanes

    if not hit_mask.reduce_or():
        return False

    comptime if mode == TRACE.CLOSEST_HIT:
        _t = hit_mask.select(hit_sphere.t, f32_max)
        min_t, lane = min_argmin(_t)

        hit.t = min_t
        hit.u = 0.0
        hit.v = 0.0
        hit.inst = EMPTY_LANE
        hit.prim = prim_indices[lane]

    return True


def pack_sphere_leaf_lanes_kernel[
    width: SIMDSize,
](
    spheres: UnsafePointer[Float32, ImmutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, ImmutAnyOrigin],
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_lane_count: Int,
):
    var lane_idx = global_idx.x
    if lane_idx >= leaf_lane_count:
        return

    var lane = lane_idx % width
    var block_idx = lane_idx / width

    var prim = UInt32(leaf_block_indices[lane_idx])
    var out_base = block_idx * SPHERE_LEAF_PACKED_STRIDE * width
    var leaf_spheres_u32 = leaf_spheres.bitcast[UInt32]()

    # AoSoA: [block][field][lane]
    # Packed fields:
    #   0..2 = center.xyz
    #   3    = radius
    #   4    = prim id bits
    leaf_spheres_u32[out_base + 4 * width + lane] = prim

    # traversal checks packed prim != EMPTY_LANE
    if prim == EMPTY_LANE:
        return

    var in_base = Int(prim) * Sphere.STRIDE

    leaf_spheres[out_base + 0 * width + lane] = spheres[in_base + 0]
    leaf_spheres[out_base + 1 * width + lane] = spheres[in_base + 1]
    leaf_spheres[out_base + 2 * width + lane] = spheres[in_base + 2]
    leaf_spheres[out_base + 3 * width + lane] = spheres[in_base + 3]


def _flatten_spheres(spheres: List[Sphere]) -> List[Float32]:
    var out = List[Float32](capacity=max(len(spheres), 1) * Sphere.STRIDE)
    for sphere in spheres:
        out.append(sphere.center.x)
        out.append(sphere.center.y)
        out.append(sphere.center.z)
        out.append(sphere.radius)
    return out^
