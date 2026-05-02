from std.benchmark import keep
from std.math import abs, round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu import DeviceContext, DeviceBuffer

from bajo.core.utils import (
    ns_to_ms,
    ns_to_mrays_per_s,
    pack_obj_triangles,
    print_vec3_rounded,
)
from bajo.core.vec import Vec3f32, normalize
from bajo.core.bvh import (
    generate_primary_rays,
    trace_bvh_primary,
    trace_bvh_shadow,
    flatten_rays,
    copy_list_to_device,
    compute_bounds,
)
from bajo.core.bvh.cpu.binary_bvh import BinaryBvh, Ray
from bajo.core.bvh.gpu.kernels import (
    compute_centroid_bounds,
    generate_camera_params,
    reduce_hit_t_kernel,
    reduce_u32_flags_kernel,
    GPU_REDUCE_THREADS,
    TRACE_PRIMARY_FULL,
    TRACE_PRIMARY_T,
    TRACE_SHADOW,
)
from bajo.core.bvh.gpu.lbvh import (
    GpuLBVH,
    gpu_lbvh_blocks_for,
    GPU_LBVH_BLOCK_SIZE,
)

from bajo.core.bvh.gpu.utils import (
    GpuBuildTimings,
    GpuLBVHValidation,
    GpuDirectTraversalResult,
    GpuPrimaryReduceResult,
    GpuCameraFullResult,
    GpuShadowReduceResult,
    GpuSuiteResult,
    GpuBuildResult,
    GpuReduceAndShadowResult,
    CpuReferenceResult,
    _download_full_hit_checksum,
    _download_reduced_hit_t,
    _download_reduced_u32_count,
)


# comptime DEFAULT_OBJ_PATH = "./assets/powerplant/powerplant.obj"
comptime DEFAULT_OBJ_PATH = "./assets/bunny/bunny.obj"
comptime BENCH_REPEATS = 8
comptime PRIMARY_WIDTH = 640
comptime PRIMARY_HEIGHT = 360
comptime PRIMARY_VIEWS = 3

comptime RUN_DIRECT_RAY_UPLOAD_BENCH = True
comptime RUN_CAMERA_FULL_DOWNLOAD_BENCH = True
comptime RUN_CAMERA_REDUCE_AND_SHADOW_BENCH = True


@always_inline
def _min_ns(current: Int, candidate: Int) -> Int:
    if candidate < current:
        return candidate
    return current


@always_inline
def _ms(ns: Int) -> Float64:
    return round(ns_to_ms(ns), 3)


@always_inline
def _mrays(ns: Int, ray_count: Int) -> Float64:
    return round(ns_to_mrays_per_s(ns, ray_count), 3)


@always_inline
def _stage_primary_reduce_ns(r: GpuPrimaryReduceResult) -> Int:
    return r.kernel_ns + r.reduce_ns + r.download_ns


@always_inline
def _stage_shadow_reduce_ns(r: GpuShadowReduceResult) -> Int:
    return r.kernel_ns + r.reduce_ns + r.download_ns


def _upload_rays(
    ctx: DeviceContext,
    d_rays: DeviceBuffer[DType.float32],
    rays_flat: List[Float32],
) raises:
    with d_rays.map_to_host() as h:
        for i in range(len(rays_flat)):
            h[i] = rays_flat[i]
    ctx.synchronize()


def _benchmark_direct_uploaded_rays(
    ctx: DeviceContext,
    mut lbvh: GpuLBVH,
    d_rays: DeviceBuffer[DType.float32],
    d_hits_f32: DeviceBuffer[DType.float32],
    d_hits_u32: DeviceBuffer[DType.uint32],
    rays_flat: List[Float32],
    ray_count: Int,
    reference_checksum: Float64,
    repeats: Int,
) raises -> GpuDirectTraversalResult:
    _upload_rays(ctx, d_rays, rays_flat)
    lbvh.launch_uploaded_primary(ctx, d_rays, d_hits_f32, d_hits_u32, ray_count)
    ctx.synchronize()

    var best_upload_ns = Int.MAX
    var best_kernel_ns = Int.MAX
    var best_download_ns = Int.MAX
    var best_frame_ns = Int.MAX
    var checksum = 0.0
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var frame0 = perf_counter_ns()

        var u0 = perf_counter_ns()
        _upload_rays(ctx, d_rays, rays_flat)
        var u1 = perf_counter_ns()
        best_upload_ns = _min_ns(best_upload_ns, Int(u1 - u0))

        var k0 = perf_counter_ns()
        lbvh.launch_uploaded_primary(
            ctx, d_rays, d_hits_f32, d_hits_u32, ray_count
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
    mut lbvh: GpuLBVH,
    d_camera_params: DeviceBuffer[DType.float32],
    d_hits_f32: DeviceBuffer[DType.float32],
    d_hits_u32: DeviceBuffer[DType.uint32],
    ray_count: Int,
    reference_checksum: Float64,
    repeats: Int,
) raises -> GpuCameraFullResult:
    lbvh.launch_camera_trace[TRACE_PRIMARY_FULL](
        ctx,
        d_camera_params,
        d_hits_f32,
        d_hits_u32,
        ray_count,
        PRIMARY_WIDTH,
        PRIMARY_HEIGHT,
        PRIMARY_VIEWS,
    )
    ctx.synchronize()

    var best_kernel_ns = Int.MAX
    var best_download_ns = Int.MAX
    var best_frame_ns = Int.MAX
    var checksum = 0.0
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var frame0 = perf_counter_ns()

        var k0 = perf_counter_ns()
        lbvh.launch_camera_trace[TRACE_PRIMARY_FULL](
            ctx,
            d_camera_params,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            PRIMARY_VIEWS,
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
    mut lbvh: GpuLBVH,
    d_camera_params: DeviceBuffer[DType.float32],
    d_hit_t: DeviceBuffer[DType.float32],
    d_occluded: DeviceBuffer[DType.uint32],
    d_partial_sums: DeviceBuffer[DType.float64],
    d_partial_counts: DeviceBuffer[DType.uint32],
    ray_count: Int,
    reference_checksum: Float64,
    repeats: Int,
) raises -> GpuPrimaryReduceResult:
    var reduce_blocks = gpu_lbvh_blocks_for(GPU_REDUCE_THREADS)
    var best_kernel_ns = Int.MAX
    var best_reduce_ns = Int.MAX
    var best_download_ns = Int.MAX
    var best_frame_ns = Int.MAX
    var checksum = 0.0
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var frame0 = perf_counter_ns()

        var k0 = perf_counter_ns()
        lbvh.launch_camera_trace[TRACE_PRIMARY_T](
            ctx,
            d_camera_params,
            d_hit_t,
            d_occluded,
            ray_count,
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            PRIMARY_VIEWS,
        )
        ctx.synchronize()
        var k1 = perf_counter_ns()
        best_kernel_ns = _min_ns(best_kernel_ns, Int(k1 - k0))

        var r0 = perf_counter_ns()
        ctx.enqueue_function[reduce_hit_t_kernel, reduce_hit_t_kernel](
            d_hit_t,
            d_partial_sums,
            d_partial_counts,
            ray_count,
            GPU_REDUCE_THREADS,
            grid_dim=reduce_blocks,
            block_dim=GPU_LBVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        var r1 = perf_counter_ns()
        best_reduce_ns = _min_ns(best_reduce_ns, Int(r1 - r0))

        var d0 = perf_counter_ns()
        var downloaded = _download_reduced_hit_t[GPU_REDUCE_THREADS](
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
    mut lbvh: GpuLBVH,
    d_camera_params: DeviceBuffer[DType.float32],
    d_hit_t: DeviceBuffer[DType.float32],
    d_occluded: DeviceBuffer[DType.uint32],
    d_partial_counts: DeviceBuffer[DType.uint32],
    ray_count: Int,
    reference_occluded: Int,
    repeats: Int,
) raises -> GpuShadowReduceResult:
    var reduce_blocks = gpu_lbvh_blocks_for(GPU_REDUCE_THREADS)
    var best_kernel_ns = Int.MAX
    var best_reduce_ns = Int.MAX
    var best_download_ns = Int.MAX
    var best_frame_ns = Int.MAX
    var occluded = UInt32(0)

    for _ in range(repeats):
        var frame0 = perf_counter_ns()

        var k0 = perf_counter_ns()
        lbvh.launch_camera_trace[TRACE_SHADOW](
            ctx,
            d_camera_params,
            d_hit_t,
            d_occluded,
            ray_count,
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            PRIMARY_VIEWS,
        )
        ctx.synchronize()
        var k1 = perf_counter_ns()
        best_kernel_ns = _min_ns(best_kernel_ns, Int(k1 - k0))

        var r0 = perf_counter_ns()
        ctx.enqueue_function[reduce_u32_flags_kernel, reduce_u32_flags_kernel](
            d_occluded,
            d_partial_counts,
            ray_count,
            GPU_REDUCE_THREADS,
            grid_dim=reduce_blocks,
            block_dim=GPU_LBVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        var r1 = perf_counter_ns()
        best_reduce_ns = _min_ns(best_reduce_ns, Int(r1 - r0))

        var d0 = perf_counter_ns()
        occluded = _download_reduced_u32_count[GPU_REDUCE_THREADS](
            ctx, d_partial_counts
        )
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
        abs(Int(occluded) - reference_occluded),
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
    var ray_count = len(rays)
    var rays_flat = flatten_rays(rays)
    var norm = normalize(centroid_max - centroid_min)

    with DeviceContext() as ctx:
        var setup0 = perf_counter_ns()
        var lbvh = GpuLBVH(ctx, tri_vertices)
        var d_camera_params = copy_list_to_device(ctx, camera_params)
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
        var setup1 = perf_counter_ns()

        var best_build = lbvh.best_build(ctx, centroid_min, norm, repeats)

        # Build one final valid tree for validation and traversal.
        _ = lbvh.build(ctx, centroid_min, norm)
        var validation = lbvh.validate(scene_min, scene_max)
        var build_result = GpuBuildResult(
            Int(setup1 - setup0), best_build, validation.copy()
        )

        var direct_result: GpuDirectTraversalResult
        comptime if RUN_DIRECT_RAY_UPLOAD_BENCH:
            direct_result = _benchmark_direct_uploaded_rays(
                ctx,
                lbvh,
                d_rays,
                d_hits_f32,
                d_hits_u32,
                rays_flat,
                ray_count,
                reference_checksum,
                repeats,
            )
        else:
            direct_result = GpuDirectTraversalResult(0, 0, 0, 0, 0.0, 0, 0.0)

        var camera_full_result: GpuCameraFullResult
        comptime if RUN_CAMERA_FULL_DOWNLOAD_BENCH:
            camera_full_result = _benchmark_camera_full_download(
                ctx,
                lbvh,
                d_camera_params,
                d_hits_f32,
                d_hits_u32,
                ray_count,
                reference_checksum,
                repeats,
            )
        else:
            camera_full_result = GpuCameraFullResult(0, 0, 0, 0.0, 0, 0.0)

        var reduce_shadow_result: GpuReduceAndShadowResult
        comptime if RUN_CAMERA_REDUCE_AND_SHADOW_BENCH:
            var primary_reduce = _benchmark_primary_reduce(
                ctx,
                lbvh,
                d_camera_params,
                d_hit_t,
                d_occluded,
                d_partial_sums,
                d_partial_counts,
                ray_count,
                reference_checksum,
                repeats,
            )
            var shadow_reduce = _benchmark_shadow_reduce(
                ctx,
                lbvh,
                d_camera_params,
                d_hit_t,
                d_occluded,
                d_partial_counts,
                ray_count,
                reference_occluded,
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
    var stage_sum_ns = build.timings.sum()
    print("\nGPU LBVH build/refit")
    print("-------------------")
    print(t"static setup once:  {_ms(build.static_setup_ns)} ms")
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
    print(t"build total:        {_ms(build.timings.total_ns)} ms")
    print(t"stage-sum total:    {_ms(stage_sum_ns)} ms")
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
        print(
            t"\nUploaded primary rays + full hit download\n"
            t"-----------------------------------------\n"
            t"ray upload:         {_ms(r.upload_ns)} ms\n"
            t"traversal kernel:   {_ms(r.kernel_ns)} ms"
            t" | {_mrays(r.kernel_ns, ray_count)} MRays/s\n"
            t"full hit download:  {_ms(r.download_ns)} ms\n"
            t"query total:        {_ms(r.frame_ns)} ms"
            t" | {_mrays(r.frame_ns, ray_count)} MRays/s\n"
            t"build + query:      {_ms(total_ns)} ms"
            t" | {_mrays(total_ns, ray_count)} MRays/s\n"
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
        print(
            t"\nGenerated primary rays + full hit download\n"
            t"------------------------------------------\n"
            t"camera kernel:      {_ms(r.kernel_ns)} ms"
            t" | {_mrays(r.kernel_ns, ray_count)} MRays/s\n"
            t"full hit download:  {_ms(r.download_ns)} ms\n"
            t"query total:        {_ms(r.frame_ns)} ms"
            t" | {_mrays(r.frame_ns, ray_count)} MRays/s\n"
            t"build + query:      {_ms(total_ns)} ms"
            t" | {_mrays(total_ns, ray_count)} MRays/s\n"
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
        var primary_total_ns = build_ns + r.primary.frame_ns
        var shadow_total_ns = build_ns + r.shadow.frame_ns

        print("\nGenerated primary rays + t-only GPU checksum reduction")
        print("------------------------------------------------------")
        print(
            t"camera kernel:      {_ms(r.primary.kernel_ns)} ms"
            t" | {_mrays(r.primary.kernel_ns, ray_count)} MRays/s"
        )
        print(t"checksum reduction: {_ms(r.primary.reduce_ns)} ms")
        print(t"partial download:   {_ms(r.primary.download_ns)} ms")
        print(
            t"query total:        {_ms(r.primary.frame_ns)} ms"
            t" | {_mrays(r.primary.frame_ns, ray_count)} MRays/s"
        )
        print(
            t"stage-sum query:    {_ms(_stage_primary_reduce_ns(r.primary))} ms"
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
            t"query total:        {_ms(r.shadow.frame_ns)} ms"
            t" | {_mrays(r.shadow.frame_ns, ray_count)} MRays/s"
        )
        print(
            t"stage-sum query:    {_ms(_stage_shadow_reduce_ns(r.shadow))} ms"
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
    var build_ns = result.build.timings.total_ns
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
        tri_vertices, bmin, bmax, cmin, cmax, Int(load_t1 - load_t0)
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
