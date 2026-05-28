from std.math import ceildiv, max
from std.time import perf_counter_ns
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.bvh.camera import Camera
from bajo.bvh.constants import (
    EMPTY_LANE,
    TRACE,
    BOUNDS_STRIDE,
    f32_max,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)
from bajo.core.intersect import intersect_ray_sphere
from bajo.core.vec import Vec3
from bajo.bvh.types import Sphere, Ray, Hit, SphereLeafBlock, BlasSet
from bajo.bvh.gpu.bounds_bvh import GpuBoundsBvh
from bajo.bvh.gpu.trace import trace_bounds_bvh
from bajo.bvh.gpu.utils import GpuBuildTimings, upload_list


struct GpuSphereBlasSetBuilder[width: Int]:
    """Builds each sphere BLAS, then packs them."""

    var sphere_sets: List[List[Sphere]]

    def __init__(out self):
        self.sphere_sets = List[List[Sphere]]()

    def add(mut self, spheres: List[Sphere]) -> UInt32:
        var idx = UInt32(len(self.sphere_sets))
        self.sphere_sets.append(spheres.copy())
        return idx

    def build(mut self, mut ctx: DeviceContext) raises -> BlasSet[Self.width]:
        if len(self.sphere_sets) == 0:
            var descs = List[UInt32](length=BlasSet.STRIDE, fill=0)
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
            capacity=len(self.sphere_sets) * BlasSet.STRIDE
        )

        var total_wide_bounds = 0
        var total_wide_lanes = 0
        var total_leaf_spheres = 0
        var total_leaf_prims = 0

        # First pass: compute final packed offsets without building/downloading.
        for blas_idx in range(len(self.sphere_sets)):
            var sphere_count = len(self.sphere_sets[blas_idx])
            var internal_count = max(sphere_count - 1, 0)
            var max_wide_nodes = max(internal_count, 1)
            var max_leaf_blocks = max(internal_count * Self.width, 1)

            var wide_bounds_base = UInt32(total_wide_bounds)
            var wide_lane_base = UInt32(total_wide_lanes)
            var leaf_f32_base = UInt32(total_leaf_spheres)
            var leaf_u32_base = UInt32(total_leaf_prims)

            descs.append(wide_bounds_base)
            descs.append(wide_lane_base)
            descs.append(leaf_f32_base)
            descs.append(leaf_u32_base)

            # Filled after the actual GPU BLAS build.
            descs.append(UInt32(0))  # BlasSet.ROOT_IDX

            descs.append(UInt32(max_wide_nodes))
            descs.append(UInt32(max_leaf_blocks))
            descs.append(UInt32(sphere_count))

            total_wide_bounds += max_wide_nodes * Self.width * BOUNDS_STRIDE
            total_wide_lanes += max_wide_nodes * Self.width
            total_leaf_spheres += max_leaf_blocks * Self.width * Sphere.STRIDE
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
        var packed_leaf_spheres = ctx.enqueue_create_buffer[DType.float32](
            max(total_leaf_spheres, 1)
        )
        var packed_leaf_prims = ctx.enqueue_create_buffer[DType.uint32](
            max(total_leaf_prims, 1)
        )

        # Second pass: build each BLAS, then copy its device buffers into the
        # final packed device buffers.
        for blas_idx in range(len(self.sphere_sets)):
            var blas = GpuSphereBvh[Self.width](ctx, self.sphere_sets[blas_idx])

            var desc_base = blas_idx * BlasSet.STRIDE

            descs[desc_base + BlasSet.ROOT_IDX] = blas.tree.root_idx

            var wide_bounds_base = Int(
                descs[desc_base + BlasSet.WIDE_BOUNDS_BASE]
            )
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
            blas.leaf_spheres.enqueue_copy_to(
                packed_leaf_spheres.unsafe_ptr() + leaf_f32_base
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
            packed_leaf_spheres,
            packed_leaf_prims,
            len(self.sphere_sets),
        )


struct GpuSphereBvh[width: Int]:
    var tree: GpuBoundsBvh[Self.width]
    var spheres: DeviceBuffer[DType.float32]
    var leaf_spheres: DeviceBuffer[DType.float32]
    var leaf_prims: DeviceBuffer[DType.uint32]
    var sphere_count: Int
    var leaf_pack_ns: Int
    var timings: GpuBuildTimings

    def __init__(
        out self,
        mut ctx: DeviceContext,
        spheres: List[Sphere],
    ) raises:
        self.sphere_count = len(spheres)
        self.leaf_pack_ns = 0

        var flat_spheres = _flatten_spheres(spheres)
        self.spheres = upload_list(ctx, flat_spheres)

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

        var d_payloads = upload_list(ctx, payloads)
        var d_leaf_bounds = upload_list(ctx, leaf_bounds)

        self.tree = GpuBoundsBvh[Self.width](ctx, d_leaf_bounds, d_payloads)
        self.timings = self.tree.build(ctx)

        self.leaf_spheres = ctx.enqueue_create_buffer[DType.float32](
            self.tree.max_leaf_blocks * Self.width * Sphere.STRIDE
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
        ctx.enqueue_function[trace_sphere_bvh_camera_kernel[Self.width]](
            self.tree.wide_bounds,
            self.tree.wide_data,
            self.tree.wide_counts,
            self.leaf_spheres,
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


def trace_sphere_bvh_camera_kernel[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
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
        _intersect_sphere_leaf[
            width,
            TRACE.CLOSEST_HIT,
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
                var base = idx * Sphere.STRIDE

                block.center.x[lane] = leaf_spheres[base + 0]
                block.center.y[lane] = leaf_spheres[base + 1]
                block.center.z[lane] = leaf_spheres[base + 2]
                block.radius[lane] = leaf_spheres[base + 3]

                block.prim_indices[lane] = prim
                block.valid_lane[lane] = True

    return block^


def _intersect_sphere_leaf[
    width: Int,
    mode: TRACE,
](
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: Ray,
    mut hit: Hit,
) capturing -> Bool:
    var block = _load_sphere_leaf_packet[width](
        leaf_spheres,
        leaf_prims,
        leaf_block_idx,
        item_count,
    )

    var O = Vec3[DType.float32, width](ray.o.x, ray.o.y, ray.o.z)
    var D = Vec3[DType.float32, width](ray.d.x, ray.d.y, ray.d.z)

    var hit_sphere = intersect_ray_sphere(
        O, D, block.center, block.radius, hit.t, ray.t_min
    )
    var hit_mask = hit_sphere.mask & block.valid_lane

    if not hit_mask.reduce_or():
        return False

    var min_t = hit_mask.select(hit_sphere.t, f32_max).reduce_min()

    if min_t < hit.t:
        hit.t = min_t
        comptime if mode == TRACE.ANY_HIT:
            return True

        comptime if mode == TRACE.CLOSEST_HIT:
            hit.u = 0.0
            hit.v = 0.0
            hit.inst = EMPTY_LANE

            comptime for lane in range(width):
                if hit_mask[lane] and hit_sphere.t[lane] == min_t:
                    hit.prim = block.prim_indices[lane]

        return True
    return False


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

        var out_base = idx * Sphere.STRIDE
        if prim == EMPTY_LANE:
            for k in range(Sphere.STRIDE):
                leaf_spheres[out_base + k] = 0.0
        else:
            var in_base = Int(prim) * Sphere.STRIDE
            for k in range(Sphere.STRIDE):
                leaf_spheres[out_base + k] = spheres[in_base + k]


def _flatten_spheres(spheres: List[Sphere]) -> List[Float32]:
    var out = List[Float32](capacity=max(len(spheres), 1) * Sphere.STRIDE)
    for sphere in spheres:
        out.append(sphere.center.x)
        out.append(sphere.center.y)
        out.append(sphere.center.z)
        out.append(sphere.radius)
    return out^
