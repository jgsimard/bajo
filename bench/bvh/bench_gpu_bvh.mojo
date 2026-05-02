from std.benchmark import keep
from std.math import abs, max, round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu import DeviceContext, DeviceBuffer

from bajo.core.utils import (
    ns_to_ms,
    ns_to_mrays_per_s,
    pack_obj_triangles,
    print_vec3_rounded,
)
from bajo.sort.gpu.radix_sort import device_radix_sort_pairs, RadixSortWorkspace
from bajo.core.vec import Vec3f32, normalize
from bajo.core.bvh.gpu.validate import (
    validate_sorted_keys,
    validate_topology,
    validate_refit_bounds,
)
from bajo.core.bvh import (
    generate_primary_rays,
    trace_bvh_primary,
    trace_bvh_shadow,
    flatten_rays,
    flatten_vertices,
    copy_list_to_device,
    compute_bounds,
)
from bajo.core.bvh.cpu.binary_bvh import BinaryBvh, Ray
from bajo.core.bvh.gpu.kernels import (
    compute_centroid_bounds,
    generate_camera_params,
    compute_morton_codes_kernel,
    init_lbvh_topology_kernel,
    init_lbvh_bounds_kernel,
    build_lbvh_topology_kernel,
    refit_lbvh_bounds_kernel,
    trace_lbvh_gpu_camera_kernel,
    trace_lbvh_gpu_primary_kernel,
    reduce_hit_t_kernel,
    reduce_u32_flags_kernel,
    GPU_REDUCE_THREADS,
    TRACE_PRIMARY_FULL,
    TRACE_PRIMARY_T,
    TRACE_SHADOW,
)

# comptime DEFAULT_OBJ_PATH = "./assets/powerplant/powerplant.obj"
comptime DEFAULT_OBJ_PATH = "./assets/bunny/bunny.obj"
comptime GPU_BLOCK_SIZE = 128
comptime BENCH_REPEATS = 8
comptime PRIMARY_WIDTH = 640
comptime PRIMARY_HEIGHT = 360
comptime PRIMARY_VIEWS = 3
comptime INF_NS = 9223372036854775807

# Meaningful default suite:
# - Build/refit isolates Morton, radix sort, topology, and bottom-up bounds.
# - Direct uploaded-ray traversal shows the cost of sending rays every frame.
# - Camera full-download traversal shows the cost of writing/downloading full hits.
# - Camera reduce/shadow is the current best GPU path.
comptime RUN_DIRECT_RAY_UPLOAD_BENCH = True
comptime RUN_CAMERA_FULL_DOWNLOAD_BENCH = True
comptime RUN_CAMERA_REDUCE_AND_SHADOW_BENCH = True


@fieldwise_init
struct GpuBuildTimings(Copyable):
    var static_setup_ns: Int
    var morton_ns: Int
    var sort_ns: Int
    var topology_ns: Int
    var refit_ns: Int
    var total_ns: Int


@fieldwise_init
struct GpuBuildValidation(Copyable):
    var sorted_ok: Bool
    var values_ok: Bool
    var topology_ok: Bool
    var topology_root_count: Int
    var topology_root_idx: UInt32
    var bounds_ok: Bool
    var bounds_diff: Float64
    var root_idx: UInt32
    var guard: UInt64


@fieldwise_init
struct GpuBuildResult(Copyable):
    var timings: GpuBuildTimings
    var validation: GpuBuildValidation


@fieldwise_init
struct GpuDirectTraversalResult(Copyable):
    var upload_ns: Int
    var kernel_ns: Int
    var download_ns: Int
    var frame_ns: Int
    var checksum: Float64
    var hit_count: UInt32
    var diff: Float64


@fieldwise_init
struct GpuCameraFullResult(Copyable):
    var kernel_ns: Int
    var download_ns: Int
    var frame_ns: Int
    var checksum: Float64
    var hit_count: UInt32
    var diff: Float64


@fieldwise_init
struct GpuPrimaryReduceResult(Copyable):
    var kernel_ns: Int
    var reduce_ns: Int
    var download_ns: Int
    var frame_ns: Int
    var checksum: Float64
    var hit_count: UInt32
    var diff: Float64


@fieldwise_init
struct GpuShadowReduceResult(Copyable):
    var kernel_ns: Int
    var reduce_ns: Int
    var download_ns: Int
    var frame_ns: Int
    var occluded: UInt32
    var diff: Int


@fieldwise_init
struct GpuReduceAndShadowResult(Copyable):
    var primary: GpuPrimaryReduceResult
    var shadow: GpuShadowReduceResult


@fieldwise_init
struct GpuSuiteResult(Copyable):
    var build: GpuBuildResult
    var direct: GpuDirectTraversalResult
    var camera_full: GpuCameraFullResult
    var reduce_shadow: GpuReduceAndShadowResult


@fieldwise_init
struct CpuReferenceResult(Copyable):
    var build_ns: Int
    var trace_ns: Int
    var checksum: Float64
    var occluded: Int


@always_inline
def _min_ns(current: Int, candidate: Int) -> Int:
    if candidate < current:
        return candidate
    return current


@always_inline
def _abs_i32(v: Int) -> Int:
    if v < 0:
        return -v
    return v


@always_inline
def _blocks_for(n: Int) -> Int:
    return (n + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE


@always_inline
def _ms(ns: Int) -> Float64:
    return round(ns_to_ms(ns), 3)


@always_inline
def _mrays(ns: Int, ray_count: Int) -> Float64:
    return round(ns_to_mrays_per_s(ns, ray_count), 3)


@always_inline
def _build_stage_sum_ns(t: GpuBuildTimings) -> Int:
    return t.morton_ns + t.sort_ns + t.topology_ns + t.refit_ns


@always_inline
def _build_total_ns(t: GpuBuildTimings) -> Int:
    return t.total_ns


@always_inline
def _primary_reduce_frame_ns(r: GpuPrimaryReduceResult) -> Int:
    return r.frame_ns


@always_inline
def _shadow_reduce_frame_ns(r: GpuShadowReduceResult) -> Int:
    return r.frame_ns


def _launch_morton(
    ctx: DeviceContext,
    d_vertices: DeviceBuffer[DType.float32],
    d_keys: DeviceBuffer[DType.uint32],
    d_values: DeviceBuffer[DType.uint32],
    tri_count: Int,
    centroid_min: Vec3f32,
    norm: Vec3f32,
    blocks_leaves: Int,
) raises:
    ctx.enqueue_function[
        compute_morton_codes_kernel, compute_morton_codes_kernel
    ](
        d_vertices.unsafe_ptr(),
        d_keys.unsafe_ptr(),
        d_values.unsafe_ptr(),
        tri_count,
        centroid_min.x(),
        centroid_min.y(),
        centroid_min.z(),
        norm.x(),
        norm.y(),
        norm.z(),
        grid_dim=blocks_leaves,
        block_dim=GPU_BLOCK_SIZE,
    )


def _launch_topology(
    ctx: DeviceContext,
    d_keys: DeviceBuffer[DType.uint32],
    d_node_meta: DeviceBuffer[DType.uint32],
    d_leaf_parent: DeviceBuffer[DType.uint32],
    tri_count: Int,
    internal_count: Int,
    blocks_init: Int,
    blocks_internal: Int,
) raises:
    ctx.enqueue_function[init_lbvh_topology_kernel, init_lbvh_topology_kernel](
        d_node_meta.unsafe_ptr(),
        d_leaf_parent.unsafe_ptr(),
        internal_count,
        tri_count,
        grid_dim=blocks_init,
        block_dim=GPU_BLOCK_SIZE,
    )
    ctx.enqueue_function[
        build_lbvh_topology_kernel, build_lbvh_topology_kernel
    ](
        d_keys.unsafe_ptr(),
        d_node_meta.unsafe_ptr(),
        d_leaf_parent.unsafe_ptr(),
        tri_count,
        grid_dim=blocks_internal,
        block_dim=GPU_BLOCK_SIZE,
    )


def _launch_refit(
    ctx: DeviceContext,
    d_vertices: DeviceBuffer[DType.float32],
    d_values: DeviceBuffer[DType.uint32],
    d_node_meta: DeviceBuffer[DType.uint32],
    d_leaf_parent: DeviceBuffer[DType.uint32],
    d_node_bounds: DeviceBuffer[DType.float32],
    d_node_flags: DeviceBuffer[DType.uint32],
    tri_count: Int,
    internal_count: Int,
    blocks_internal: Int,
    blocks_leaves: Int,
) raises:
    ctx.enqueue_function[init_lbvh_bounds_kernel, init_lbvh_bounds_kernel](
        d_node_bounds.unsafe_ptr(),
        d_node_flags.unsafe_ptr(),
        internal_count,
        grid_dim=blocks_internal,
        block_dim=GPU_BLOCK_SIZE,
    )
    ctx.enqueue_function[refit_lbvh_bounds_kernel, refit_lbvh_bounds_kernel](
        d_vertices.unsafe_ptr(),
        d_values.unsafe_ptr(),
        d_node_meta.unsafe_ptr(),
        d_leaf_parent.unsafe_ptr(),
        d_node_bounds.unsafe_ptr(),
        d_node_flags.unsafe_ptr(),
        tri_count,
        grid_dim=blocks_leaves,
        block_dim=GPU_BLOCK_SIZE,
    )


def _time_build_once(
    ctx: DeviceContext,
    mut workspace: RadixSortWorkspace[DType.uint32, DType.uint32],
    d_vertices: DeviceBuffer[DType.float32],
    mut d_keys: DeviceBuffer[DType.uint32],
    mut d_values: DeviceBuffer[DType.uint32],
    d_node_meta: DeviceBuffer[DType.uint32],
    d_leaf_parent: DeviceBuffer[DType.uint32],
    d_node_bounds: DeviceBuffer[DType.float32],
    d_node_flags: DeviceBuffer[DType.uint32],
    tri_count: Int,
    internal_count: Int,
    centroid_min: Vec3f32,
    norm: Vec3f32,
    blocks_leaves: Int,
    blocks_internal: Int,
    blocks_init: Int,
) raises -> GpuBuildTimings:
    var m0 = perf_counter_ns()
    _launch_morton(
        ctx,
        d_vertices,
        d_keys,
        d_values,
        tri_count,
        centroid_min,
        norm,
        blocks_leaves,
    )
    ctx.synchronize()
    var m1 = perf_counter_ns()

    device_radix_sort_pairs[DType.uint32, DType.uint32](
        ctx, workspace, d_keys, d_values, tri_count
    )
    ctx.synchronize()
    var s1 = perf_counter_ns()

    _launch_topology(
        ctx,
        d_keys,
        d_node_meta,
        d_leaf_parent,
        tri_count,
        internal_count,
        blocks_init,
        blocks_internal,
    )
    ctx.synchronize()
    var t1 = perf_counter_ns()

    _launch_refit(
        ctx,
        d_vertices,
        d_values,
        d_node_meta,
        d_leaf_parent,
        d_node_bounds,
        d_node_flags,
        tri_count,
        internal_count,
        blocks_internal,
        blocks_leaves,
    )
    ctx.synchronize()
    var r1 = perf_counter_ns()

    return GpuBuildTimings(
        0,
        Int(m1 - m0),
        Int(s1 - m1),
        Int(t1 - s1),
        Int(r1 - t1),
        Int(r1 - m0),
    )


def _best_build_timings(
    ctx: DeviceContext,
    mut workspace: RadixSortWorkspace[DType.uint32, DType.uint32],
    d_vertices: DeviceBuffer[DType.float32],
    mut d_keys: DeviceBuffer[DType.uint32],
    mut d_values: DeviceBuffer[DType.uint32],
    d_node_meta: DeviceBuffer[DType.uint32],
    d_leaf_parent: DeviceBuffer[DType.uint32],
    d_node_bounds: DeviceBuffer[DType.float32],
    d_node_flags: DeviceBuffer[DType.uint32],
    tri_count: Int,
    internal_count: Int,
    centroid_min: Vec3f32,
    norm: Vec3f32,
    blocks_leaves: Int,
    blocks_internal: Int,
    blocks_init: Int,
    repeats: Int,
) raises -> GpuBuildTimings:
    var best_morton_ns = Int(INF_NS)
    var best_sort_ns = Int(INF_NS)
    var best_topology_ns = Int(INF_NS)
    var best_refit_ns = Int(INF_NS)
    var best_total_ns = Int(INF_NS)

    for _ in range(repeats):
        var t = _time_build_once(
            ctx,
            workspace,
            d_vertices,
            d_keys,
            d_values,
            d_node_meta,
            d_leaf_parent,
            d_node_bounds,
            d_node_flags,
            tri_count,
            internal_count,
            centroid_min,
            norm,
            blocks_leaves,
            blocks_internal,
            blocks_init,
        )
        best_morton_ns = _min_ns(best_morton_ns, t.morton_ns)
        best_sort_ns = _min_ns(best_sort_ns, t.sort_ns)
        best_topology_ns = _min_ns(best_topology_ns, t.topology_ns)
        best_refit_ns = _min_ns(best_refit_ns, t.refit_ns)
        best_total_ns = _min_ns(best_total_ns, t.total_ns)

    return GpuBuildTimings(
        0,
        best_morton_ns,
        best_sort_ns,
        best_topology_ns,
        best_refit_ns,
        best_total_ns,
    )


def _validate_current_lbvh(
    d_keys: DeviceBuffer[DType.uint32],
    d_values: DeviceBuffer[DType.uint32],
    d_node_meta: DeviceBuffer[DType.uint32],
    d_leaf_parent: DeviceBuffer[DType.uint32],
    d_node_bounds: DeviceBuffer[DType.float32],
    d_node_flags: DeviceBuffer[DType.uint32],
    tri_count: Int,
    scene_min: Vec3f32,
    scene_max: Vec3f32,
) raises -> GpuBuildValidation:
    var sorted_validation = validate_sorted_keys(d_keys, d_values, tri_count)
    var topo_validation = validate_topology(
        d_node_meta, d_leaf_parent, tri_count
    )
    var refit_validation = validate_refit_bounds(
        d_node_bounds,
        d_node_flags,
        d_node_meta,
        tri_count,
        scene_min,
        scene_max,
    )
    var guard = sorted_validation[6] + topo_validation[3] + refit_validation[3]
    return GpuBuildValidation(
        sorted_validation[0],
        sorted_validation[1],
        topo_validation[0],
        topo_validation[1],
        topo_validation[2],
        refit_validation[0],
        refit_validation[1],
        refit_validation[2],
        guard,
    )


def _upload_rays(
    ctx: DeviceContext,
    d_rays: DeviceBuffer[DType.float32],
    rays_flat: List[Float32],
) raises:
    with d_rays.map_to_host() as h:
        for i in range(len(rays_flat)):
            h[i] = rays_flat[i]
    ctx.synchronize()


def _download_full_hit_checksum(
    ctx: DeviceContext,
    d_hits_f32: DeviceBuffer[DType.float32],
    ray_count: Int,
) raises -> Tuple[Float64, UInt32]:
    var checksum = 0.0
    var hit_count = UInt32(0)
    with d_hits_f32.map_to_host() as h:
        for i in range(ray_count):
            var t = h[i * 3]
            if t < 1.0e20:
                checksum += Float64(t)
                hit_count += 1
    ctx.synchronize()
    return (checksum, hit_count)


def _download_reduced_hit_t(
    ctx: DeviceContext,
    d_partial_sums: DeviceBuffer[DType.float64],
    d_partial_counts: DeviceBuffer[DType.uint32],
) raises -> Tuple[Float64, UInt32]:
    var checksum = 0.0
    var hit_count = UInt32(0)
    with d_partial_sums.map_to_host() as sums:
        for i in range(GPU_REDUCE_THREADS):
            checksum += sums[i]
    with d_partial_counts.map_to_host() as counts:
        for i in range(GPU_REDUCE_THREADS):
            hit_count += counts[i]
    ctx.synchronize()
    return (checksum, hit_count)


def _download_reduced_u32_count(
    ctx: DeviceContext,
    d_partial_counts: DeviceBuffer[DType.uint32],
) raises -> UInt32:
    var total = UInt32(0)
    with d_partial_counts.map_to_host() as counts:
        for i in range(GPU_REDUCE_THREADS):
            total += counts[i]
    ctx.synchronize()
    return total


def _launch_direct_primary(
    ctx: DeviceContext,
    d_vertices: DeviceBuffer[DType.float32],
    d_values: DeviceBuffer[DType.uint32],
    d_node_meta: DeviceBuffer[DType.uint32],
    d_node_bounds: DeviceBuffer[DType.float32],
    d_rays: DeviceBuffer[DType.float32],
    d_hits_f32: DeviceBuffer[DType.float32],
    d_hits_u32: DeviceBuffer[DType.uint32],
    ray_count: Int,
    root_idx: UInt32,
    blocks_rays: Int,
) raises:
    ctx.enqueue_function[
        trace_lbvh_gpu_primary_kernel, trace_lbvh_gpu_primary_kernel
    ](
        d_vertices.unsafe_ptr(),
        d_values.unsafe_ptr(),
        d_node_meta.unsafe_ptr(),
        d_node_bounds.unsafe_ptr(),
        d_rays.unsafe_ptr(),
        d_hits_f32.unsafe_ptr(),
        d_hits_u32.unsafe_ptr(),
        ray_count,
        root_idx,
        grid_dim=blocks_rays,
        block_dim=GPU_BLOCK_SIZE,
    )


def _launch_camera_trace[
    mode: String
](
    ctx: DeviceContext,
    d_vertices: DeviceBuffer[DType.float32],
    d_values: DeviceBuffer[DType.uint32],
    d_node_meta: DeviceBuffer[DType.uint32],
    d_node_bounds: DeviceBuffer[DType.float32],
    d_camera_params: DeviceBuffer[DType.float32],
    out_f32: DeviceBuffer[DType.float32],
    out_u32: DeviceBuffer[DType.uint32],
    ray_count: Int,
    root_idx: UInt32,
    blocks_rays: Int,
) raises:
    ctx.enqueue_function[
        trace_lbvh_gpu_camera_kernel[mode],
        trace_lbvh_gpu_camera_kernel[mode],
    ](
        d_vertices.unsafe_ptr(),
        d_values.unsafe_ptr(),
        d_node_meta.unsafe_ptr(),
        d_node_bounds.unsafe_ptr(),
        d_camera_params.unsafe_ptr(),
        out_f32.unsafe_ptr(),
        out_u32.unsafe_ptr(),
        ray_count,
        PRIMARY_WIDTH,
        PRIMARY_HEIGHT,
        PRIMARY_VIEWS,
        root_idx,
        grid_dim=blocks_rays,
        block_dim=GPU_BLOCK_SIZE,
    )


def _benchmark_direct_uploaded_rays(
    ctx: DeviceContext,
    d_vertices: DeviceBuffer[DType.float32],
    d_values: DeviceBuffer[DType.uint32],
    d_node_meta: DeviceBuffer[DType.uint32],
    d_node_bounds: DeviceBuffer[DType.float32],
    d_rays: DeviceBuffer[DType.float32],
    d_hits_f32: DeviceBuffer[DType.float32],
    d_hits_u32: DeviceBuffer[DType.uint32],
    rays_flat: List[Float32],
    ray_count: Int,
    root_idx: UInt32,
    reference_checksum: Float64,
    blocks_rays: Int,
    repeats: Int,
) raises -> GpuDirectTraversalResult:
    _upload_rays(ctx, d_rays, rays_flat)
    _launch_direct_primary(
        ctx,
        d_vertices,
        d_values,
        d_node_meta,
        d_node_bounds,
        d_rays,
        d_hits_f32,
        d_hits_u32,
        ray_count,
        root_idx,
        blocks_rays,
    )
    ctx.synchronize()

    var best_upload_ns = Int(INF_NS)
    var best_kernel_ns = Int(INF_NS)
    var best_download_ns = Int(INF_NS)
    var best_frame_ns = Int(INF_NS)
    var checksum = 0.0
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var frame0 = perf_counter_ns()

        var u0 = perf_counter_ns()
        _upload_rays(ctx, d_rays, rays_flat)
        var u1 = perf_counter_ns()
        best_upload_ns = _min_ns(best_upload_ns, Int(u1 - u0))

        var k0 = perf_counter_ns()
        _launch_direct_primary(
            ctx,
            d_vertices,
            d_values,
            d_node_meta,
            d_node_bounds,
            d_rays,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            root_idx,
            blocks_rays,
        )
        ctx.synchronize()
        var k1 = perf_counter_ns()
        best_kernel_ns = _min_ns(best_kernel_ns, Int(k1 - k0))

        var d0 = perf_counter_ns()
        var downloaded = _download_full_hit_checksum(ctx, d_hits_f32, ray_count)
        checksum = downloaded[0]
        hit_count = downloaded[1]
        var d1 = perf_counter_ns()
        best_download_ns = _min_ns(best_download_ns, Int(d1 - d0))

        var frame1 = perf_counter_ns()
        best_frame_ns = _min_ns(best_frame_ns, Int(frame1 - frame0))

    return GpuDirectTraversalResult(
        best_upload_ns,
        best_kernel_ns,
        best_download_ns,
        best_frame_ns,
        checksum,
        hit_count,
        abs(checksum - reference_checksum),
    )


def _benchmark_camera_full_download(
    ctx: DeviceContext,
    d_vertices: DeviceBuffer[DType.float32],
    d_values: DeviceBuffer[DType.uint32],
    d_node_meta: DeviceBuffer[DType.uint32],
    d_node_bounds: DeviceBuffer[DType.float32],
    d_camera_params: DeviceBuffer[DType.float32],
    d_hits_f32: DeviceBuffer[DType.float32],
    d_hits_u32: DeviceBuffer[DType.uint32],
    ray_count: Int,
    root_idx: UInt32,
    reference_checksum: Float64,
    blocks_rays: Int,
    repeats: Int,
) raises -> GpuCameraFullResult:
    _launch_camera_trace[TRACE_PRIMARY_FULL](
        ctx,
        d_vertices,
        d_values,
        d_node_meta,
        d_node_bounds,
        d_camera_params,
        d_hits_f32,
        d_hits_u32,
        ray_count,
        root_idx,
        blocks_rays,
    )
    ctx.synchronize()

    var best_kernel_ns = Int(INF_NS)
    var best_download_ns = Int(INF_NS)
    var best_frame_ns = Int(INF_NS)
    var checksum = 0.0
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var frame0 = perf_counter_ns()

        var k0 = perf_counter_ns()
        _launch_camera_trace[TRACE_PRIMARY_FULL](
            ctx,
            d_vertices,
            d_values,
            d_node_meta,
            d_node_bounds,
            d_camera_params,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            root_idx,
            blocks_rays,
        )
        ctx.synchronize()
        var k1 = perf_counter_ns()
        best_kernel_ns = _min_ns(best_kernel_ns, Int(k1 - k0))

        var d0 = perf_counter_ns()
        var downloaded = _download_full_hit_checksum(ctx, d_hits_f32, ray_count)
        checksum = downloaded[0]
        hit_count = downloaded[1]
        var d1 = perf_counter_ns()
        best_download_ns = _min_ns(best_download_ns, Int(d1 - d0))

        var frame1 = perf_counter_ns()
        best_frame_ns = _min_ns(best_frame_ns, Int(frame1 - frame0))

    return GpuCameraFullResult(
        best_kernel_ns,
        best_download_ns,
        best_frame_ns,
        checksum,
        hit_count,
        abs(checksum - reference_checksum),
    )


def _benchmark_primary_reduce(
    ctx: DeviceContext,
    d_vertices: DeviceBuffer[DType.float32],
    d_values: DeviceBuffer[DType.uint32],
    d_node_meta: DeviceBuffer[DType.uint32],
    d_node_bounds: DeviceBuffer[DType.float32],
    d_camera_params: DeviceBuffer[DType.float32],
    d_hit_t: DeviceBuffer[DType.float32],
    d_occluded: DeviceBuffer[DType.uint32],
    d_partial_sums: DeviceBuffer[DType.float64],
    d_partial_counts: DeviceBuffer[DType.uint32],
    ray_count: Int,
    root_idx: UInt32,
    reference_checksum: Float64,
    blocks_rays: Int,
    reduce_blocks: Int,
    repeats: Int,
) raises -> GpuPrimaryReduceResult:
    var best_kernel_ns = Int(INF_NS)
    var best_reduce_ns = Int(INF_NS)
    var best_download_ns = Int(INF_NS)
    var best_frame_ns = Int(INF_NS)
    var checksum = 0.0
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var frame0 = perf_counter_ns()

        var k0 = perf_counter_ns()
        _launch_camera_trace[TRACE_PRIMARY_T](
            ctx,
            d_vertices,
            d_values,
            d_node_meta,
            d_node_bounds,
            d_camera_params,
            d_hit_t,
            d_occluded,  # unused out_u32 dummy
            ray_count,
            root_idx,
            blocks_rays,
        )
        ctx.synchronize()
        var k1 = perf_counter_ns()
        best_kernel_ns = _min_ns(best_kernel_ns, Int(k1 - k0))

        var r0 = perf_counter_ns()
        ctx.enqueue_function[reduce_hit_t_kernel, reduce_hit_t_kernel](
            d_hit_t.unsafe_ptr(),
            d_partial_sums.unsafe_ptr(),
            d_partial_counts.unsafe_ptr(),
            ray_count,
            GPU_REDUCE_THREADS,
            grid_dim=reduce_blocks,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var r1 = perf_counter_ns()
        best_reduce_ns = _min_ns(best_reduce_ns, Int(r1 - r0))

        var d0 = perf_counter_ns()
        var downloaded = _download_reduced_hit_t(
            ctx, d_partial_sums, d_partial_counts
        )
        checksum = downloaded[0]
        hit_count = downloaded[1]
        var d1 = perf_counter_ns()
        best_download_ns = _min_ns(best_download_ns, Int(d1 - d0))

        var frame1 = perf_counter_ns()
        best_frame_ns = _min_ns(best_frame_ns, Int(frame1 - frame0))

    return GpuPrimaryReduceResult(
        best_kernel_ns,
        best_reduce_ns,
        best_download_ns,
        best_frame_ns,
        checksum,
        hit_count,
        abs(checksum - reference_checksum),
    )


def _benchmark_shadow_reduce(
    ctx: DeviceContext,
    d_vertices: DeviceBuffer[DType.float32],
    d_values: DeviceBuffer[DType.uint32],
    d_node_meta: DeviceBuffer[DType.uint32],
    d_node_bounds: DeviceBuffer[DType.float32],
    d_camera_params: DeviceBuffer[DType.float32],
    d_hit_t: DeviceBuffer[DType.float32],
    d_occluded: DeviceBuffer[DType.uint32],
    d_partial_counts: DeviceBuffer[DType.uint32],
    ray_count: Int,
    root_idx: UInt32,
    reference_occluded: Int,
    blocks_rays: Int,
    reduce_blocks: Int,
    repeats: Int,
) raises -> GpuShadowReduceResult:
    var best_kernel_ns = Int(INF_NS)
    var best_reduce_ns = Int(INF_NS)
    var best_download_ns = Int(INF_NS)
    var best_frame_ns = Int(INF_NS)
    var occluded = UInt32(0)

    for _ in range(repeats):
        var frame0 = perf_counter_ns()

        var k0 = perf_counter_ns()
        _launch_camera_trace[TRACE_SHADOW](
            ctx,
            d_vertices,
            d_values,
            d_node_meta,
            d_node_bounds,
            d_camera_params,
            d_hit_t,  # unused out_f32 dummy
            d_occluded,
            ray_count,
            root_idx,
            blocks_rays,
        )
        ctx.synchronize()
        var k1 = perf_counter_ns()
        best_kernel_ns = _min_ns(best_kernel_ns, Int(k1 - k0))

        var r0 = perf_counter_ns()
        ctx.enqueue_function[reduce_u32_flags_kernel, reduce_u32_flags_kernel](
            d_occluded.unsafe_ptr(),
            d_partial_counts.unsafe_ptr(),
            ray_count,
            GPU_REDUCE_THREADS,
            grid_dim=reduce_blocks,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var r1 = perf_counter_ns()
        best_reduce_ns = _min_ns(best_reduce_ns, Int(r1 - r0))

        var d0 = perf_counter_ns()
        occluded = _download_reduced_u32_count(ctx, d_partial_counts)
        var d1 = perf_counter_ns()
        best_download_ns = _min_ns(best_download_ns, Int(d1 - d0))

        var frame1 = perf_counter_ns()
        best_frame_ns = _min_ns(best_frame_ns, Int(frame1 - frame0))

    return GpuShadowReduceResult(
        best_kernel_ns,
        best_reduce_ns,
        best_download_ns,
        best_frame_ns,
        occluded,
        _abs_i32(Int(occluded) - reference_occluded),
    )


def run_gpu_lbvh_benchmark_suite(
    tri_vertices: List[Vec3f32],
    centroid_min: Vec3f32,
    centroid_max: Vec3f32,
    scene_min: Vec3f32,
    scene_max: Vec3f32,
    camera_params: List[Float32],
    rays: List[Ray],
    reference_checksum: Float64,
    reference_occluded: Int,
    repeats: Int,
) raises -> GpuSuiteResult:
    var tri_count = len(tri_vertices) // 3
    var internal_count = tri_count - 1
    var ray_count = len(rays)
    var vertices = flatten_vertices(tri_vertices)
    var rays_flat = flatten_rays(rays)
    var norm = normalize(centroid_max - centroid_min)

    with DeviceContext() as ctx:
        var static_t0 = perf_counter_ns()
        var d_vertices = copy_list_to_device(ctx, vertices)
        var d_camera_params = copy_list_to_device(ctx, camera_params)
        var d_keys = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var d_values = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
            ctx, tri_count
        )
        var d_node_meta = ctx.enqueue_create_buffer[DType.uint32](
            internal_count * 4
        )
        var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var d_node_bounds = ctx.enqueue_create_buffer[DType.float32](
            internal_count * 12
        )
        var d_node_flags = ctx.enqueue_create_buffer[DType.uint32](
            internal_count
        )
        var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count)
        var d_hit_t = ctx.enqueue_create_buffer[DType.float32](ray_count)
        var d_occluded = ctx.enqueue_create_buffer[DType.uint32](ray_count)
        var d_partial_sums = ctx.enqueue_create_buffer[DType.float64](
            GPU_REDUCE_THREADS
        )
        var d_partial_counts = ctx.enqueue_create_buffer[DType.uint32](
            GPU_REDUCE_THREADS
        )
        ctx.synchronize()
        var static_t1 = perf_counter_ns()

        var blocks_leaves = _blocks_for(tri_count)
        var blocks_internal = _blocks_for(internal_count)
        var blocks_init = _blocks_for(max(tri_count, internal_count))
        var blocks_rays = _blocks_for(ray_count)
        var reduce_blocks = _blocks_for(GPU_REDUCE_THREADS)

        var best_build = _best_build_timings(
            ctx,
            workspace,
            d_vertices,
            d_keys,
            d_values,
            d_node_meta,
            d_leaf_parent,
            d_node_bounds,
            d_node_flags,
            tri_count,
            internal_count,
            centroid_min,
            norm,
            blocks_leaves,
            blocks_internal,
            blocks_init,
            repeats,
        )
        best_build.static_setup_ns = Int(static_t1 - static_t0)

        # Build one final valid tree for all traversal benchmarks and validation.
        _ = _time_build_once(
            ctx,
            workspace,
            d_vertices,
            d_keys,
            d_values,
            d_node_meta,
            d_leaf_parent,
            d_node_bounds,
            d_node_flags,
            tri_count,
            internal_count,
            centroid_min,
            norm,
            blocks_leaves,
            blocks_internal,
            blocks_init,
        )

        var validation = _validate_current_lbvh(
            d_keys,
            d_values,
            d_node_meta,
            d_leaf_parent,
            d_node_bounds,
            d_node_flags,
            tri_count,
            scene_min,
            scene_max,
        )
        var build_result = GpuBuildResult(best_build^, validation.copy())

        var direct_result: GpuDirectTraversalResult
        comptime if RUN_DIRECT_RAY_UPLOAD_BENCH:
            direct_result = _benchmark_direct_uploaded_rays(
                ctx,
                d_vertices,
                d_values,
                d_node_meta,
                d_node_bounds,
                d_rays,
                d_hits_f32,
                d_hits_u32,
                rays_flat,
                ray_count,
                validation.root_idx,
                reference_checksum,
                blocks_rays,
                repeats,
            )
        else:
            direct_result = GpuDirectTraversalResult(
                0, 0, 0, 0, 0.0, UInt32(0), 0.0
            )

        var camera_full_result: GpuCameraFullResult
        comptime if RUN_CAMERA_FULL_DOWNLOAD_BENCH:
            camera_full_result = _benchmark_camera_full_download(
                ctx,
                d_vertices,
                d_values,
                d_node_meta,
                d_node_bounds,
                d_camera_params,
                d_hits_f32,
                d_hits_u32,
                ray_count,
                validation.root_idx,
                reference_checksum,
                blocks_rays,
                repeats,
            )
        else:
            camera_full_result = GpuCameraFullResult(0, 0, 0, 0.0, 0, 0.0)

        var reduce_shadow_result: GpuReduceAndShadowResult
        comptime if RUN_CAMERA_REDUCE_AND_SHADOW_BENCH:
            var primary_reduce = _benchmark_primary_reduce(
                ctx,
                d_vertices,
                d_values,
                d_node_meta,
                d_node_bounds,
                d_camera_params,
                d_hit_t,
                d_occluded,
                d_partial_sums,
                d_partial_counts,
                ray_count,
                validation.root_idx,
                reference_checksum,
                blocks_rays,
                reduce_blocks,
                repeats,
            )
            var shadow_reduce = _benchmark_shadow_reduce(
                ctx,
                d_vertices,
                d_values,
                d_node_meta,
                d_node_bounds,
                d_camera_params,
                d_hit_t,
                d_occluded,
                d_partial_counts,
                ray_count,
                validation.root_idx,
                reference_occluded,
                blocks_rays,
                reduce_blocks,
                repeats,
            )
            reduce_shadow_result = GpuReduceAndShadowResult(
                primary_reduce^, shadow_reduce^
            )
        else:
            reduce_shadow_result = GpuReduceAndShadowResult(
                GpuPrimaryReduceResult(0, 0, 0, 0, 0.0, 0, 0.0),
                GpuShadowReduceResult(0, 0, 0, 0, 0, 0),
            )
        return GpuSuiteResult(
            build_result^,
            direct_result^,
            camera_full_result^,
            reduce_shadow_result^,
        )


def _build_cpu_reference(
    mut tri_vertices: List[Vec3f32],
    rays: List[Ray],
) raises -> CpuReferenceResult:
    var ref_build_t0 = perf_counter_ns()
    var ref_bvh = BinaryBvh(
        tri_vertices.unsafe_ptr(), UInt32(len(tri_vertices) // 3)
    )
    ref_bvh.build["sah", True]()
    var ref_build_t1 = perf_counter_ns()

    var ref_trace_t0 = perf_counter_ns()
    var ref_checksum = trace_bvh_primary(ref_bvh, rays)
    var ref_occluded = trace_bvh_shadow(ref_bvh, rays)
    var ref_trace_t1 = perf_counter_ns()

    return CpuReferenceResult(
        Int(ref_build_t1 - ref_build_t0),
        Int(ref_trace_t1 - ref_trace_t0),
        ref_checksum,
        ref_occluded,
    )


def _print_scene_summary(
    tri_vertices: List[Vec3f32],
    bmin: Vec3f32,
    bmax: Vec3f32,
    cmin: Vec3f32,
    cmax: Vec3f32,
    load_ns: Int,
):
    var tri_count = len(tri_vertices) // 3
    print(t"Packed vertices: {len(tri_vertices)}")
    print(t"Triangles: {tri_count}")
    print(t"Internal nodes: {tri_count - 1}")
    print(t"Load+pack ms: {_ms(load_ns)}")
    print_vec3_rounded("Bounds min:", bmin)
    print_vec3_rounded("Bounds max:", bmax)
    print_vec3_rounded("Centroid min:", cmin)
    print_vec3_rounded("Centroid max:", cmax)


def _print_cpu_reference(reference: CpuReferenceResult):
    print("\nCPU reference")
    print("-------------")
    print(t"SAH MT build:       {_ms(reference.build_ns)} ms")
    print(
        t"reference queries:  {_ms(reference.trace_ns)} ms | checksum:"
        t" {round(reference.checksum, 3)} | occluded: {reference.occluded}"
    )


def _print_build_result(build: GpuBuildResult):
    var build_ns = _build_total_ns(build.timings)
    var staged_ns = _build_stage_sum_ns(build.timings)
    print("\nGPU LBVH build/refit")
    print("-------------------")
    print(t"static setup once:  { _ms(build.timings.static_setup_ns) } ms")
    print(
        t"valid:              sorted={build.validation.sorted_ok} |"
        t" values={build.validation.values_ok} |"
        t" topology={build.validation.topology_ok} |"
        t" bounds={build.validation.bounds_ok} |"
        t" root={build.validation.root_idx}"
    )
    print(t"morton generation:  {_ms(build.timings.morton_ns)} ms")
    print(t"radix sort pairs:   {_ms(build.timings.sort_ns)} ms")
    print(t"topology build:     {_ms(build.timings.topology_ns)} ms")
    print(t"bounds refit:       {_ms(build.timings.refit_ns)} ms")
    print(t"build total:        {_ms(build_ns)} ms")
    print(t"stage-sum total:    {_ms(staged_ns)} ms")
    print(
        t"validation detail: "
        t" topology_roots={build.validation.topology_root_count} |"
        t" topology_root={build.validation.topology_root_idx} |"
        t" refit_root={build.validation.root_idx} |"
        t" bounds_diff={round(build.validation.bounds_diff, 6)} |"
        t" guard={build.validation.guard}"
    )


def _print_direct_result(
    r: GpuDirectTraversalResult,
    build_ns: Int,
    ray_count: Int,
):
    comptime if RUN_DIRECT_RAY_UPLOAD_BENCH:
        var total_ns = build_ns + r.frame_ns
        print("\nUploaded primary rays + full hit download")
        print("-----------------------------------------")
        print(t"ray upload:         {_ms(r.upload_ns)} ms")
        print(
            t"traversal kernel:   {_ms(r.kernel_ns)} ms"
            t" | {_mrays(r.kernel_ns, ray_count)} MRays/s"
        )
        print(t"full hit download:  {_ms(r.download_ns)} ms")
        print(
            t"query total:        {_ms(r.frame_ns)} ms"
            t" | {_mrays(r.frame_ns, ray_count)} MRays/s"
        )
        print(
            t"build + query:      {_ms(total_ns)} ms"
            t" | {_mrays(total_ns, ray_count)} MRays/s"
        )
        print(
            t"validation diff:    {round(r.diff, 3)} |"
            t" checksum: {round(r.checksum, 3)} | hits: {r.hit_count}"
        )


def _print_camera_full_result(
    r: GpuCameraFullResult,
    build_ns: Int,
    ray_count: Int,
):
    comptime if RUN_CAMERA_FULL_DOWNLOAD_BENCH:
        var total_ns = build_ns + r.frame_ns
        print("\nGenerated primary rays + full hit download")
        print("------------------------------------------")
        print(
            t"camera kernel:      {_ms(r.kernel_ns)} ms"
            t" | {_mrays(r.kernel_ns, ray_count)} MRays/s"
        )
        print(t"full hit download:  {_ms(r.download_ns)} ms")
        print(
            t"query total:        {_ms(r.frame_ns)} ms"
            t" | {_mrays(r.frame_ns, ray_count)} MRays/s"
        )
        print(
            t"build + query:      {_ms(total_ns)} ms"
            t" | {_mrays(total_ns, ray_count)} MRays/s"
        )
        print(
            t"validation diff:    {round(r.diff, 3)} |"
            t" checksum: {round(r.checksum, 3)} | hits: {r.hit_count}"
        )


def _print_reduce_shadow_result(
    r: GpuReduceAndShadowResult,
    build_ns: Int,
    ray_count: Int,
    reference_occluded: Int,
):
    comptime if RUN_CAMERA_REDUCE_AND_SHADOW_BENCH:
        var primary_frame_ns = _primary_reduce_frame_ns(r.primary)
        var primary_total_ns = build_ns + primary_frame_ns
        var shadow_frame_ns = _shadow_reduce_frame_ns(r.shadow)
        var shadow_total_ns = build_ns + shadow_frame_ns

        print("\nGenerated primary rays + t-only GPU checksum reduction")
        print("------------------------------------------------------")
        print(
            t"camera kernel:      {_ms(r.primary.kernel_ns)} ms"
            t" | {_mrays(r.primary.kernel_ns, ray_count)} MRays/s"
        )
        print(t"checksum reduction: {_ms(r.primary.reduce_ns)} ms")
        print(t"partial download:   {_ms(r.primary.download_ns)} ms")
        print(
            t"query total:        {_ms(primary_frame_ns)} ms"
            t" | {_mrays(primary_frame_ns, ray_count)} MRays/s"
        )
        print(
            t"build + query:      {_ms(primary_total_ns)} ms"
            t" | {_mrays(primary_total_ns, ray_count)} MRays/s"
        )
        print(
            t"validation diff:    {round(r.primary.diff, 3)} | checksum:"
            t" {round(r.primary.checksum, 3)} | hits: {r.primary.hit_count}"
        )

        print("\nGenerated shadow rays + GPU occlusion reduction")
        print("-----------------------------------------------")
        print(
            t"shadow kernel:      {_ms(r.shadow.kernel_ns)} ms"
            t" | {_mrays(r.shadow.kernel_ns, ray_count)} MRays/s"
        )
        print(t"occlusion reduce:   {_ms(r.shadow.reduce_ns)} ms")
        print(t"partial download:   {_ms(r.shadow.download_ns)} ms")
        print(
            t"query total:        {_ms(shadow_frame_ns)} ms"
            t" | {_mrays(shadow_frame_ns, ray_count)} MRays/s"
        )
        print(
            t"build + query:      {_ms(shadow_total_ns)} ms"
            t" | {_mrays(shadow_total_ns, ray_count)} MRays/s"
        )
        print(
            t"validation diff:    {r.shadow.diff} |"
            t" occluded: {r.shadow.occluded} | ref: {reference_occluded}"
        )


def _print_suite_result(
    result: GpuSuiteResult,
    ray_count: Int,
    reference_occluded: Int,
):
    var build_ns = _build_total_ns(result.build.timings)
    _print_build_result(result.build)
    _print_direct_result(result.direct, build_ns, ray_count)
    _print_camera_full_result(result.camera_full, build_ns, ray_count)
    _print_reduce_shadow_result(
        result.reduce_shadow, build_ns, ray_count, reference_occluded
    )


def main() raises:
    print("GPU LBVH benchmark suite")
    print(t"Path: {DEFAULT_OBJ_PATH}")
    print(t"Image rays: {PRIMARY_WIDTH} x {PRIMARY_HEIGHT} x {PRIMARY_VIEWS}")
    print(t"Repeats: {BENCH_REPEATS}")

    print("\nLoading + packing OBJ...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles(DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var bounds = compute_bounds(tri_vertices)
    var bmin = bounds[0].copy()
    var bmax = bounds[1].copy()
    var cbounds = compute_centroid_bounds(tri_vertices)
    var cmin = cbounds[0].copy()
    var cmax = cbounds[1].copy()
    _print_scene_summary(
        tri_vertices,
        bmin,
        bmax,
        cmin,
        cmax,
        Int(load_t1 - load_t0),
    )

    print("\nGenerating reference rays and camera parameters...")
    var rays = generate_primary_rays(
        bmin, bmax, PRIMARY_WIDTH, PRIMARY_HEIGHT, PRIMARY_VIEWS
    )
    var camera_params = generate_camera_params(bmin, bmax, PRIMARY_VIEWS)
    print(t"Rays: {len(rays)}")
    print(t"Camera params floats: {len(camera_params)}")

    var cpu_ref = _build_cpu_reference(tri_vertices, rays)
    _print_cpu_reference(cpu_ref)

    print("\nGPU benchmark suite")
    print("-------------------")
    comptime if has_accelerator():
        var result = run_gpu_lbvh_benchmark_suite(
            tri_vertices,
            cmin,
            cmax,
            bmin,
            bmax,
            camera_params,
            rays,
            cpu_ref.checksum,
            cpu_ref.occluded,
            BENCH_REPEATS,
        )
        _print_suite_result(result, len(rays), cpu_ref.occluded)
        keep(result.build.validation.guard)
        keep(result.direct.hit_count)
        keep(result.camera_full.hit_count)
        keep(result.reduce_shadow.primary.hit_count)
        keep(result.reduce_shadow.shadow.occluded)
    else:
        print("No compatible GPU found; skipped Mojo GPU benchmark.")
