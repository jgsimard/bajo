from std.math import max
from std.time import perf_counter_ns
from std.gpu import DeviceBuffer
from std.gpu.host import DeviceContext

from bajo.core.vec import Vec3f32
from bajo.core.bvh import flatten_vertices, copy_list_to_device
from bajo.core.bvh.gpu.validate import (
    validate_sorted_keys,
    validate_topology,
    validate_refit_bounds,
)
from bajo.sort.gpu.radix_sort import device_radix_sort_pairs, RadixSortWorkspace
from bajo.core.bvh.gpu.kernels import (
    compute_morton_codes_kernel,
    init_lbvh_topology_kernel,
    init_lbvh_bounds_kernel,
    build_lbvh_topology_kernel,
    refit_lbvh_bounds_kernel,
    trace_lbvh_gpu_camera_kernel,
    trace_lbvh_gpu_primary_kernel,
    TRACE_PRIMARY_FULL,
    TRACE_PRIMARY_T,
    TRACE_SHADOW,
)


comptime GPU_LBVH_BLOCK_SIZE = 128
comptime GPU_LBVH_INF_NS = 9223372036854775807


@fieldwise_init
struct GpuLBVHBuildTimings(TrivialRegisterPassable):
    var morton_ns: Int
    var sort_ns: Int
    var topology_ns: Int
    var refit_ns: Int
    var total_ns: Int

    def __init__(out self):
        self.morton_ns = Int.MAX
        self.sort_ns = Int.MAX
        self.topology_ns = Int.MAX
        self.refit_ns = Int.MAX
        self.total_ns = Int.MAX

    def min(mut self, rhs: Self):
        self.morton_ns = min(self.morton_ns, rhs.morton_ns)
        self.sort_ns = min(self.sort_ns, rhs.sort_ns)
        self.topology_ns = min(self.topology_ns, rhs.topology_ns)
        self.refit_ns = min(self.refit_ns, rhs.refit_ns)
        self.total_ns = min(self.total_ns, rhs.total_ns)


@fieldwise_init
struct GpuLBVHValidation(TrivialRegisterPassable):
    var sorted_ok: Bool
    var values_ok: Bool
    var topology_ok: Bool
    var topology_root_count: UInt32
    var topology_root_idx: UInt32
    var bounds_ok: Bool
    var bounds_diff: Float64
    var refit_root_idx: UInt32
    var guard: UInt64


@always_inline
def gpu_lbvh_blocks_for(n: Int) -> Int:
    return (n + GPU_LBVH_BLOCK_SIZE - 1) // GPU_LBVH_BLOCK_SIZE


@always_inline
def gpu_lbvh_stage_sum(t: GpuLBVHBuildTimings) -> Int:
    return t.morton_ns + t.sort_ns + t.topology_ns + t.refit_ns


struct GpuLBVH:
    var tri_count: Int
    var internal_count: Int
    var root_idx: UInt32

    var blocks_leaves: Int
    var blocks_internal: Int
    var blocks_init: Int

    var d_vertices: DeviceBuffer[DType.float32]
    var d_keys: DeviceBuffer[DType.uint32]
    var d_values: DeviceBuffer[DType.uint32]
    var d_node_meta: DeviceBuffer[DType.uint32]
    var d_leaf_parent: DeviceBuffer[DType.uint32]
    var d_node_bounds: DeviceBuffer[DType.float32]
    var d_node_flags: DeviceBuffer[DType.uint32]
    var workspace: RadixSortWorkspace[DType.uint32, DType.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        tri_vertices: List[Vec3f32],
    ) raises:
        var vertices = flatten_vertices(tri_vertices)

        self.tri_count = len(tri_vertices) // 3
        self.internal_count = self.tri_count - 1
        self.root_idx = UInt32(0)

        self.blocks_leaves = gpu_lbvh_blocks_for(self.tri_count)
        self.blocks_internal = gpu_lbvh_blocks_for(self.internal_count)
        self.blocks_init = gpu_lbvh_blocks_for(
            max(self.tri_count, self.internal_count)
        )

        self.d_vertices = copy_list_to_device(ctx, vertices)
        self.d_keys = ctx.enqueue_create_buffer[DType.uint32](self.tri_count)
        self.d_values = ctx.enqueue_create_buffer[DType.uint32](self.tri_count)
        self.d_node_meta = ctx.enqueue_create_buffer[DType.uint32](
            self.internal_count * 4
        )
        self.d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](
            self.tri_count
        )
        self.d_node_bounds = ctx.enqueue_create_buffer[DType.float32](
            self.internal_count * 12
        )
        self.d_node_flags = ctx.enqueue_create_buffer[DType.uint32](
            self.internal_count
        )
        self.workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
            ctx, self.tri_count
        )

    def build(
        mut self,
        ctx: DeviceContext,
        centroid_min: Vec3f32,
        norm: Vec3f32,
    ) raises -> GpuLBVHBuildTimings:
        var start = perf_counter_ns()

        self.launch_morton(ctx, centroid_min, norm)
        ctx.synchronize()
        var m = perf_counter_ns()

        device_radix_sort_pairs[DType.uint32, DType.uint32](
            ctx, self.workspace, self.d_keys, self.d_values, self.tri_count
        )
        ctx.synchronize()
        var s = perf_counter_ns()

        self.launch_topology(ctx)
        ctx.synchronize()
        var t = perf_counter_ns()

        self.launch_refit(ctx)
        ctx.synchronize()
        var r = perf_counter_ns()

        return GpuLBVHBuildTimings(
            Int(m - start),
            Int(s - m),
            Int(t - s),
            Int(r - t),
            Int(r - start),
        )

    def best_build(
        mut self,
        ctx: DeviceContext,
        centroid_min: Vec3f32,
        norm: Vec3f32,
        repeats: Int,
    ) raises -> GpuLBVHBuildTimings:
        var out_t = GpuLBVHBuildTimings()
        for _ in range(repeats):
            var t = self.build(ctx, centroid_min, norm)
            out_t.min(t)

        return out_t

    def validate(
        mut self,
        scene_min: Vec3f32,
        scene_max: Vec3f32,
    ) raises -> GpuLBVHValidation:
        var sorted_validation = validate_sorted_keys(
            self.d_keys, self.d_values, self.tri_count
        )
        var topo_validation = validate_topology(
            self.d_node_meta, self.d_leaf_parent, self.tri_count
        )
        var refit_validation = validate_refit_bounds(
            self.d_node_bounds,
            self.d_node_flags,
            self.d_node_meta,
            self.tri_count,
            scene_min,
            scene_max,
        )

        self.root_idx = UInt32(refit_validation[2])
        var guard = (
            sorted_validation[6] + topo_validation[3] + refit_validation[3]
        )

        return GpuLBVHValidation(
            sorted_validation[0],
            sorted_validation[1],
            topo_validation[0],
            UInt32(topo_validation[1]),
            topo_validation[2],
            refit_validation[0],
            refit_validation[1],
            refit_validation[2],
            guard,
        )

    def launch_morton(
        self,
        ctx: DeviceContext,
        centroid_min: Vec3f32,
        norm: Vec3f32,
    ) raises:
        ctx.enqueue_function[
            compute_morton_codes_kernel, compute_morton_codes_kernel
        ](
            self.d_vertices.unsafe_ptr(),
            self.d_keys.unsafe_ptr(),
            self.d_values.unsafe_ptr(),
            self.tri_count,
            centroid_min.x(),
            centroid_min.y(),
            centroid_min.z(),
            norm.x(),
            norm.y(),
            norm.z(),
            grid_dim=self.blocks_leaves,
            block_dim=GPU_LBVH_BLOCK_SIZE,
        )

    def launch_topology(self, ctx: DeviceContext) raises:
        ctx.enqueue_function[
            init_lbvh_topology_kernel, init_lbvh_topology_kernel
        ](
            self.d_node_meta.unsafe_ptr(),
            self.d_leaf_parent.unsafe_ptr(),
            self.internal_count,
            self.tri_count,
            grid_dim=self.blocks_init,
            block_dim=GPU_LBVH_BLOCK_SIZE,
        )
        ctx.enqueue_function[
            build_lbvh_topology_kernel, build_lbvh_topology_kernel
        ](
            self.d_keys.unsafe_ptr(),
            self.d_node_meta.unsafe_ptr(),
            self.d_leaf_parent.unsafe_ptr(),
            self.tri_count,
            grid_dim=self.blocks_internal,
            block_dim=GPU_LBVH_BLOCK_SIZE,
        )

    def launch_refit(self, ctx: DeviceContext) raises:
        ctx.enqueue_function[init_lbvh_bounds_kernel, init_lbvh_bounds_kernel](
            self.d_node_bounds.unsafe_ptr(),
            self.d_node_flags.unsafe_ptr(),
            self.internal_count,
            grid_dim=self.blocks_internal,
            block_dim=GPU_LBVH_BLOCK_SIZE,
        )
        ctx.enqueue_function[
            refit_lbvh_bounds_kernel, refit_lbvh_bounds_kernel
        ](
            self.d_vertices.unsafe_ptr(),
            self.d_values.unsafe_ptr(),
            self.d_node_meta.unsafe_ptr(),
            self.d_leaf_parent.unsafe_ptr(),
            self.d_node_bounds.unsafe_ptr(),
            self.d_node_flags.unsafe_ptr(),
            self.tri_count,
            grid_dim=self.blocks_leaves,
            block_dim=GPU_LBVH_BLOCK_SIZE,
        )

    def launch_uploaded_primary(
        self,
        ctx: DeviceContext,
        d_rays: DeviceBuffer[DType.float32],
        d_hits_f32: DeviceBuffer[DType.float32],
        d_hits_u32: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[
            trace_lbvh_gpu_primary_kernel, trace_lbvh_gpu_primary_kernel
        ](
            self.d_vertices.unsafe_ptr(),
            self.d_values.unsafe_ptr(),
            self.d_node_meta.unsafe_ptr(),
            self.d_node_bounds.unsafe_ptr(),
            d_rays.unsafe_ptr(),
            d_hits_f32.unsafe_ptr(),
            d_hits_u32.unsafe_ptr(),
            ray_count,
            self.root_idx,
            grid_dim=gpu_lbvh_blocks_for(ray_count),
            block_dim=GPU_LBVH_BLOCK_SIZE,
        )

    def launch_camera_trace[
        mode: String
    ](
        self,
        ctx: DeviceContext,
        d_camera_params: DeviceBuffer[DType.float32],
        out_f32: DeviceBuffer[DType.float32],
        out_u32: DeviceBuffer[DType.uint32],
        ray_count: Int,
        width: Int,
        height: Int,
        views: Int,
    ) raises:
        ctx.enqueue_function[
            trace_lbvh_gpu_camera_kernel[mode],
            trace_lbvh_gpu_camera_kernel[mode],
        ](
            self.d_vertices.unsafe_ptr(),
            self.d_values.unsafe_ptr(),
            self.d_node_meta.unsafe_ptr(),
            self.d_node_bounds.unsafe_ptr(),
            d_camera_params.unsafe_ptr(),
            out_f32.unsafe_ptr(),
            out_u32.unsafe_ptr(),
            ray_count,
            width,
            height,
            views,
            self.root_idx,
            grid_dim=gpu_lbvh_blocks_for(ray_count),
            block_dim=GPU_LBVH_BLOCK_SIZE,
        )
