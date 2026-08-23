from std.gpu import global_idx
from std.math import abs, ceildiv, round, min, max
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from bajo.core import AABB, Vec3f32, Point3f32, Rayf32, SegmentOffsets
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.bvh.host_utils import compute_bounds, sphere_bounds
from bajo.bvh.constants import (
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    TraceMode,
    TRI_LEAF_VERTEX_STRIDE,
    f32_max,
)
from bajo.bvh.cpu import CpuBlasSet
from bajo.bvh.types import Hit, Sphere
from bajo.bvh.cpu.blas_set import (
    build_cpu_sphere_blas_set,
    build_cpu_triangle_blas_set,
    trace_blas_set,
)
from bajo.bvh.gpu.sphere_bvh import (
    GpuSphereBvh,
    build_gpu_sphere_bvh,
    build_gpu_sphere_bvh_measured,
)
from bajo.bvh.gpu.triangle_bvh import (
    GpuTriangleBvh,
    build_gpu_triangle_bvh,
    build_gpu_triangle_bvh_measured,
    compute_triangle_bounds_kernel,
    trace_cwbvh8_triangles,
)
from bajo.bvh.gpu.builder.segmented_build import enqueue_segmented_wide_build
from bajo.bvh.gpu.compressed_bounds_bvh import (
    CWBVH_NODE_WORDS,
    CWBVH_TRIANGLE_WORDS,
    enqueue_segmented_cwbvh8_representation,
)
from bajo.bvh.gpu.camera_launch import validate_camera_launch
from bajo.bvh.gpu.ray_launch import (
    _load_packed_ray,
    _store_packed_hit,
    validate_ray_launch,
)
from bajo.bvh.gpu.utils import (
    GpuBuildTimings,
    _download_full_hit_checksum,
    _device_span,
    upload_list,
    upload_rays,
    upload_vertices,
)
from bajo.benchmark.bvh_reporting import (
    GpuBenchResult,
    print_transposed_header,
    _print_gpu_result_trace_rows,
    _print_gpu_result_validation_rows,
)
from bajo.benchmark.bvh_fixtures import make_camera_rays_and_params
from bajo.benchmark.gpu_bvh_fixtures import (
    Cwbvh8BenchBvh,
    build_cwbvh8_bench_bvh,
    trace_cwbvh8_camera_kernel,
)
from bajo.parser.obj.pack import pack_obj_triangles


# comptime DEFAULT_OBJ_PATH = "./assets/bunny/bunny.obj"
# comptime DEFAULT_OBJ_PATH = "./assets/dragon/dragon.obj"
# comptime DEFAULT_OBJ_PATH = "./assets/rungholt/rungholt.obj"
comptime DEFAULT_OBJ_PATH = "./assets/sponza/sponza.obj"
comptime PRIMARY_WIDTH = 1280
comptime PRIMARY_HEIGHT = 640
comptime FOV_SCALE = 0.2
comptime PRIMARY_VIEWS = 3
comptime BENCH_REPEATS = 8
comptime SPHERE_GRID_X = 64
comptime SPHERE_GRID_Y = 64
comptime SPHERE_RAY_WIDTH = 1280
comptime SPHERE_RAY_HEIGHT = 640
comptime SPHERE_RAY_VIEWS = 3

comptime TRIANGLE_HIT_REL_EPS = 0.0
comptime SPHERE_HIT_REL_EPS = 1.0e-3


@fieldwise_init
struct Cwbvh8BuildTimings:
    var total_ns: Int
    var wide: GpuBuildTimings
    var encode_pack_ns: Int


def _build_cwbvh8_measured(
    mut ctx: DeviceContext,
    d_vertices: DeviceBuffer[.float32],
    mut measured: Cwbvh8BuildTimings,
) raises -> Cwbvh8BenchBvh:
    """Build with stage barriers for attribution, not headline timing."""
    var tri_count = len(d_vertices) / TRI_LEAF_VERTEX_STRIDE
    ctx.synchronize()
    var total_start = perf_counter_ns()

    var leaf_bounds = ctx.enqueue_create_buffer[.float32](
        tri_count * AABB[.WORLD].STRIDE
    )
    var payloads = ctx.enqueue_create_buffer[.uint32](tri_count)
    ctx.synchronize()

    var bounds_start = perf_counter_ns()
    ctx.enqueue_function[compute_triangle_bounds_kernel[.WORLD]](
        _device_span[mut=False](d_vertices),
        _device_span[mut=True](leaf_bounds),
        _device_span[mut=True](payloads),
        grid_dim=ceildiv(tri_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    ctx.synchronize()
    var bounds_ns = Int(perf_counter_ns() - bounds_start)

    var build = enqueue_segmented_wide_build[
        8, 4, 3, .HPLOC, True, True
    ](
        ctx,
        SegmentOffsets.single(tri_count),
        leaf_bounds^,
        payloads^,
        True,
    )
    build.wait(ctx)
    var wide = build.timings
    wide.bounds_pack_ns = bounds_ns

    var nodes = ctx.enqueue_create_buffer[.float32](
        build.wide.node_segments.item_count() * CWBVH_NODE_WORDS
    )
    var triangles = ctx.enqueue_create_buffer[.float32](
        max(tri_count, 1) * CWBVH_TRIANGLE_WORDS
    )
    ctx.synchronize()

    var encode_pack_start = perf_counter_ns()
    var encoded_counts = enqueue_segmented_cwbvh8_representation[4](
        ctx,
        build.wide.wide_nodes,
        build.wide.leaf_block_indices,
        build.wide.node_segment_offsets,
        build.wide.leaf_block_segment_offsets,
        build.binary.segment_offsets,
        build.wide.node_counts,
        d_vertices,
        nodes,
        triangles,
    )
    ctx.synchronize()
    var encode_pack_ns = Int(perf_counter_ns() - encode_pack_start)
    with encoded_counts.map_to_host() as counts:
        if counts[0] != UInt32(tri_count):
            raise "unified measured CWBVH8 encoding lost triangles"

    measured = Cwbvh8BuildTimings(
        Int(perf_counter_ns() - total_start),
        wide,
        encode_pack_ns,
    )
    return Cwbvh8BenchBvh(nodes^, triangles^, UInt32(0))


def _print_cwbvh8_build_breakdown(
    hot_total_ns: Int, measured: Cwbvh8BuildTimings
):
    var tracked_ns = measured.wide.total() + measured.encode_pack_ns
    var overhead_ns = max(measured.total_ns - tracked_ns, 0)
    print("\nH-PLOC CWBVH8 instrumented build breakdown")
    print(
        t"production total: {round(ns_to_ms(hot_total_ns), 3)} ms; "
        t"instrumented total: {round(ns_to_ms(measured.total_ns), 3)} ms"
    )
    print("stage                    ms    % instrumented")
    print("-------------------- ------ ----------------")
    _print_cwbvh8_stage(
        "bounds packing", measured.wide.bounds_pack_ns, measured.total_ns
    )
    _print_cwbvh8_stage(
        "Morton keys", measured.wide.morton_ns, measured.total_ns
    )
    _print_cwbvh8_stage("radix sort", measured.wide.sort_ns, measured.total_ns)
    _print_cwbvh8_stage(
        "H-PLOC topology", measured.wide.topology_ns, measured.total_ns
    )
    _print_cwbvh8_stage(
        "binary refit", measured.wide.refit_ns, measured.total_ns
    )
    _print_cwbvh8_stage(
        "wide collapse", measured.wide.collapse_ns, measured.total_ns
    )
    _print_cwbvh8_stage(
        "CWBVH encode+pack", measured.encode_pack_ns, measured.total_ns
    )
    _print_cwbvh8_stage("allocation + host", overhead_ns, measured.total_ns)


def _print_cwbvh8_stage(label: String, stage_ns: Int, total_ns: Int):
    var percent = Float64(stage_ns) / Float64(total_ns) * 100.0
    print(
        label.ascii_ljust(20),
        String(round(ns_to_ms(stage_ns), 3)).ascii_rjust(6),
        String(round(percent, 1)).ascii_rjust(16),
    )


def _trace_cwbvh8_rays_kernel[
    mode: TraceMode,
](
    nodes: Pointer[Float32, ImmutAnyOrigin],
    triangles: Pointer[Float32, ImmutAnyOrigin],
    root_idx: UInt32,
    rays: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    ray_count: Int32,
):
    var ray_count_int = Int(ray_count)
    var ray_idx = global_idx.x
    if ray_idx >= ray_count_int:
        return
    var ray = _load_packed_ray[.WORLD](rays, ray_count_int, ray_idx)
    var hit = trace_cwbvh8_triangles[.WORLD, mode](
        nodes, triangles, root_idx, ray
    )
    _store_packed_hit[.WORLD](hit, hits, ray_count_int, ray_idx)


def _trace_cpu_triangle_bvh[
    width: SIMDLength
](
    bvh: CpuBlasSet[.TRIANGLE, width],
    rays: List[Rayf32[.WORLD]],
) -> Tuple[Float64, UInt32]:
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for ray in rays:
        var hit = trace_blas_set[width, width, .CLOSEST_HIT, .WORLD](
            bvh, UInt32(0), ray
        )
        if hit.t < f32_max:
            checksum += Float64(hit.t)
            hit_count += 1

    return (checksum, hit_count)


def _trace_cpu_sphere_bvh[
    width: SIMDLength
](
    bvh: CpuBlasSet[.SPHERE, width],
    rays: List[Rayf32[.WORLD]],
) -> Tuple[Float64, UInt32]:
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for ray in rays:
        var hit = trace_blas_set[width, width, .CLOSEST_HIT, .WORLD](
            bvh, UInt32(0), ray
        )
        if hit.t < f32_max:
            checksum += Float64(hit.t)
            hit_count += 1

    return (checksum, hit_count)


def _print_cpu_ref_header():
    var c0 = String("case").ascii_ljust(22)
    var c1 = String("trace").ascii_rjust(8)
    var c2 = String("hits").ascii_rjust(8)
    var c3 = String("checksum").ascii_rjust(12)

    print(t"{c0} {c1} {c2} {c3}")
    print("---------------------- -------- -------- ------------")


def _print_cpu_ref_row(
    label: String,
    traversal_ns: Int,
    hit_count: UInt32,
    checksum: Float64,
):
    var trace_ms = round(ns_to_ms(traversal_ns), 3)
    var checksum_r = round(checksum, 3)

    var c0 = label.ascii_ljust(22)
    var c1 = String(t"{trace_ms}").ascii_rjust(8)
    var c2 = String(t"{hit_count}").ascii_rjust(8)
    var c3 = String(t"{checksum_r}").ascii_rjust(12)

    print(t"{c0} {c1} {c2} {c3}")


def _print_cpu_triangle_reference[
    width: SIMDLength
](
    label: String,
    vertices: List[Point3f32[.WORLD]],
    rays: List[Rayf32[.WORLD]],
) -> Tuple[Float64, UInt32]:
    var bvh = build_cpu_triangle_blas_set[
        width, width, .LBVH, .WORLD
    ]([vertices.copy()])
    var t0 = perf_counter_ns()
    var result = _trace_cpu_triangle_bvh[width](bvh, rays)
    var t1 = perf_counter_ns()

    _print_cpu_ref_row(label, Int(t1 - t0), result[1], result[0])
    return result


def _print_cpu_sphere_reference[
    width: SIMDLength
](
    label: String,
    spheres: List[Sphere[.WORLD]],
    rays: List[Rayf32[.WORLD]],
) -> Tuple[Float64, UInt32]:
    var bvh = build_cpu_sphere_blas_set[width, .LBVH, .WORLD](
        [spheres.copy()]
    )
    var t0 = perf_counter_ns()
    var result = _trace_cpu_sphere_bvh[width](bvh, rays)
    var t1 = perf_counter_ns()

    _print_cpu_ref_row(label, Int(t1 - t0), result[1], result[0])
    return result


def _print_gpu_results_transposed(
    row0: GpuBenchResult,
    row1: GpuBenchResult,
    row2: GpuBenchResult,
):
    var value_width = 15

    print_transposed_header(
        value_width,
        row0.label,
        row1.label,
        row2.label,
    )
    _print_gpu_result_trace_rows(row0, row1, row2, value_width)
    _print_gpu_result_validation_rows(row0, row1, row2, value_width)


def _bench_camera_primary_triangle[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
](
    ctx: DeviceContext,
    mut bvh: GpuTriangleBvh[.WORLD, node_width, leaf_width],
    d_camera_params: DeviceBuffer[.float32],
    d_hits: DeviceBuffer[.float32],
    ray_count: Int,
    image_width: Int,
    image_height: Int,
    reference_checksum: Float64,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32, Float64]:
    bvh.launch_camera(
        ctx,
        d_camera_params,
        d_hits,
        ray_count,
        image_width,
        image_height,
    )
    ctx.synchronize()

    var best_kernel_ns = Int.MAX
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        bvh.launch_camera(
            ctx,
            d_camera_params,
            d_hits,
            ray_count,
            image_width,
            image_height,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_kernel_ns = min(best_kernel_ns, Int(t1 - t0))

        var downloaded = _download_full_hit_checksum(ctx, d_hits, ray_count)
        checksum = downloaded[0]
        hit_count = downloaded[1]

    return (
        best_kernel_ns,
        checksum,
        hit_count,
        abs(checksum - reference_checksum),
    )


def _bench_any_hit_triangle[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
](
    mut ctx: DeviceContext,
    mut bvh: GpuTriangleBvh[.WORLD, node_width, leaf_width],
    d_rays: DeviceBuffer[.float32],
    d_hits: DeviceBuffer[.float32],
    ray_count: Int,
    repeats: Int,
) raises -> Tuple[Int, UInt32]:
    bvh.launch_rays[mode=.ANY_HIT](ctx, d_rays, d_hits, ray_count)
    ctx.synchronize()

    var best_kernel_ns = Int.MAX
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        bvh.launch_rays[mode=.ANY_HIT](ctx, d_rays, d_hits, ray_count)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_kernel_ns = min(best_kernel_ns, Int(t1 - t0))

        var downloaded = _download_full_hit_checksum(ctx, d_hits, ray_count)
        hit_count = downloaded[1]

    return (best_kernel_ns, hit_count)


def _bench_camera_primary_cwbvh8(
    mut ctx: DeviceContext,
    bvh: Cwbvh8BenchBvh,
    d_camera_params: DeviceBuffer[.float32],
    d_hits: DeviceBuffer[.float32],
    ray_count: Int,
    image_width: Int,
    image_height: Int,
    reference_checksum: Float64,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32, Float64]:
    validate_camera_launch(
        d_camera_params, d_hits, ray_count, image_width, image_height
    )

    var best_kernel_ns = Int.MAX
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)
    for iteration in range(repeats + 1):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[trace_cwbvh8_camera_kernel](
            bvh.nodes,
            bvh.triangles,
            bvh.root_idx,
            d_camera_params,
            d_hits,
            Int32(ray_count),
            Int32(image_width),
            Int32(image_height),
            Float32(1.0) / Float32(image_height),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        if iteration > 0:
            best_kernel_ns = min(best_kernel_ns, Int(t1 - t0))
            var downloaded = _download_full_hit_checksum(ctx, d_hits, ray_count)
            checksum = downloaded[0]
            hit_count = downloaded[1]

    return (
        best_kernel_ns,
        checksum,
        hit_count,
        abs(checksum - reference_checksum),
    )


def _bench_any_hit_cwbvh8(
    mut ctx: DeviceContext,
    bvh: Cwbvh8BenchBvh,
    d_rays: DeviceBuffer[.float32],
    d_hits: DeviceBuffer[.float32],
    ray_count: Int,
    repeats: Int,
) raises -> Tuple[Int, UInt32]:
    validate_ray_launch(d_rays, d_hits, ray_count)

    var best_kernel_ns = Int.MAX
    var hit_count = UInt32(0)
    for iteration in range(repeats + 1):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[_trace_cwbvh8_rays_kernel[.ANY_HIT]](
            bvh.nodes,
            bvh.triangles,
            bvh.root_idx,
            d_rays,
            d_hits,
            Int32(ray_count),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        if iteration > 0:
            best_kernel_ns = min(best_kernel_ns, Int(t1 - t0))
            var downloaded = _download_full_hit_checksum(ctx, d_hits, ray_count)
            hit_count = downloaded[1]

    return (best_kernel_ns, hit_count)


def _run_cwbvh8(
    mut ctx: DeviceContext,
    d_vertices: DeviceBuffer[.float32],
    d_camera_params: DeviceBuffer[.float32],
    ray_count: Int,
    image_width: Int,
    image_height: Int,
    reference_checksum: Float64,
    reference_hit_count: UInt32,
    repeats: Int,
) raises -> Tuple[GpuBenchResult, Cwbvh8BenchBvh]:
    _ = build_cwbvh8_bench_bvh(ctx, d_vertices)
    var build0 = perf_counter_ns()
    var bvh = build_cwbvh8_bench_bvh(ctx, d_vertices)
    var build1 = perf_counter_ns()
    var d_hits = ctx.enqueue_create_buffer[.float32](
        ray_count * Hit[.WORLD].STRIDE
    )
    var trace = _bench_camera_primary_cwbvh8(
        ctx,
        bvh,
        d_camera_params,
        d_hits,
        ray_count,
        image_width,
        image_height,
        reference_checksum,
        repeats,
    )
    var measured = Cwbvh8BuildTimings(
        0, GpuBuildTimings(0, 0, 0, 0, 0, 0, 0), 0
    )
    # The measured path has different staged specializations from production.
    # Warm those kernels before recording the attribution run.
    _ = _build_cwbvh8_measured(ctx, d_vertices, measured)
    _ = _build_cwbvh8_measured(ctx, d_vertices, measured)
    _print_cwbvh8_build_breakdown(Int(build1 - build0), measured)
    return (
        GpuBenchResult(
            String("tri H-CWBVH8"),
            Int(build1 - build0),
            GpuBuildTimings(0, 0, 0, 0, 0, 0, 0),
            trace[0],
            ray_count,
            trace[1],
            trace[3],
            trace[2],
            reference_checksum,
            reference_hit_count,
            TRIANGLE_HIT_REL_EPS,
        ),
        bvh^,
    )


def _run_width[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
](
    mut ctx: DeviceContext,
    d_vertices: DeviceBuffer[.float32],
    d_camera_params: DeviceBuffer[.float32],
    ray_count: Int,
    image_width: Int,
    image_height: Int,
    reference_checksum: Float64,
    reference_hit_count: UInt32,
    repeats: Int,
) raises -> GpuBenchResult:
    _ = build_gpu_triangle_bvh[.WORLD, node_width, leaf_width](ctx, d_vertices)
    ctx.synchronize()

    var build0 = perf_counter_ns()
    var timings = GpuBuildTimings(0, 0, 0, 0, 0, 0, 0)
    var bvh = build_gpu_triangle_bvh_measured[.WORLD, node_width, leaf_width](
        ctx, d_vertices, timings
    )
    ctx.synchronize()
    var build1 = perf_counter_ns()

    var d_hits = ctx.enqueue_create_buffer[.float32](
        ray_count * Hit[.WORLD].STRIDE
    )

    var res = _bench_camera_primary_triangle[node_width, leaf_width](
        ctx,
        bvh,
        d_camera_params,
        d_hits,
        ray_count,
        image_width,
        image_height,
        reference_checksum,
        repeats,
    )

    return GpuBenchResult(
        String(t"tri n{Int(node_width)}/l{Int(leaf_width)}"),
        Int(build1 - build0),
        timings,
        res[0],
        ray_count,
        res[1],
        res[3],
        res[2],
        reference_checksum,
        reference_hit_count,
        TRIANGLE_HIT_REL_EPS,
    )


def _make_sphere_grid_sized(
    grid_x: Int, grid_y: Int
) -> List[Sphere[.WORLD]]:
    var spheres = List[Sphere[.WORLD]](capacity=grid_x * grid_y)

    for y in range(grid_y):
        for x in range(grid_x):
            var fx = Float32(x) - Float32(grid_x) * 0.5
            var fy = Float32(y) - Float32(grid_y) * 0.5
            var z = Float32(4 + ((x + y) % 8))
            spheres.append(
                Sphere[.WORLD](
                    Point3f32[.WORLD](fx * 2.5, fy * 2.5, z), 0.75
                )
            )

    return spheres^


def _make_sphere_grid() -> List[Sphere[.WORLD]]:
    return _make_sphere_grid_sized(SPHERE_GRID_X, SPHERE_GRID_Y)


def _bench_camera_primary_sphere[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
](
    ctx: DeviceContext,
    mut bvh: GpuSphereBvh[.WORLD, node_width, leaf_width],
    d_camera_params: DeviceBuffer[.float32],
    d_hits: DeviceBuffer[.float32],
    ray_count: Int,
    image_width: Int,
    image_height: Int,
    reference_checksum: Float64,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32, Float64]:
    bvh.launch_camera(
        ctx,
        d_camera_params,
        d_hits,
        ray_count,
        image_width,
        image_height,
    )
    ctx.synchronize()

    var best_kernel_ns = Int.MAX
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        bvh.launch_camera(
            ctx,
            d_camera_params,
            d_hits,
            ray_count,
            image_width,
            image_height,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_kernel_ns = min(best_kernel_ns, Int(t1 - t0))

        var downloaded = _download_full_hit_checksum(ctx, d_hits, ray_count)
        checksum = downloaded[0]
        hit_count = downloaded[1]

    return (
        best_kernel_ns,
        checksum,
        hit_count,
        abs(checksum - reference_checksum),
    )


def _run_sphere_width[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
](
    mut ctx: DeviceContext,
    spheres: List[Sphere[.WORLD]],
    ray_count: Int,
    d_camera_params: DeviceBuffer[.float32],
    image_width: Int,
    image_height: Int,
    reference_checksum: Float64,
    reference_hit_count: UInt32,
    repeats: Int,
) raises -> GpuBenchResult:
    _ = build_gpu_sphere_bvh[.WORLD, node_width, leaf_width](ctx, spheres)
    ctx.synchronize()

    var build0 = perf_counter_ns()
    var timings = GpuBuildTimings(0, 0, 0, 0, 0, 0, 0)
    var bvh = build_gpu_sphere_bvh_measured[.WORLD, node_width, leaf_width](
        ctx, spheres, timings
    )
    ctx.synchronize()
    var build1 = perf_counter_ns()

    var d_hits = ctx.enqueue_create_buffer[.float32](
        ray_count * Hit[.WORLD].STRIDE
    )

    var res = _bench_camera_primary_sphere[node_width, leaf_width](
        ctx,
        bvh,
        d_camera_params,
        d_hits,
        ray_count,
        image_width,
        image_height,
        reference_checksum,
        repeats,
    )

    return GpuBenchResult(
        String(t"sph n{Int(node_width)}/l{Int(leaf_width)}"),
        Int(build1 - build0),
        timings,
        res[0],
        ray_count,
        res[1],
        res[3],
        res[2],
        reference_checksum,
        reference_hit_count,
        SPHERE_HIT_REL_EPS,
    )


def run_benchmark() raises:
    print("GPU BoundsBvh benchmark")
    print("")
    print("Run configuration")
    print(t"OBJ path : {DEFAULT_OBJ_PATH}")
    print(
        t"triangle camera rays : {PRIMARY_WIDTH} x {PRIMARY_HEIGHT} x"
        t" {PRIMARY_VIEWS}"
    )
    print(t"repeats : {BENCH_REPEATS}")

    print("\nLoading + packing OBJ...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles[.WORLD](DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var bounds = compute_bounds(tri_vertices)
    var tri_count = len(tri_vertices) / 3
    print(t"triangles: {tri_count}")
    print(t"load+pack ms: {round(ns_to_ms(Int(load_t1 - load_t0)), 3)}")
    print("Bounds min:", round(bounds._min, 3))
    print("Bounds max:", round(bounds._max, 3))

    print("\nGenerating CPU reference camera rays...")
    var camera = make_camera_rays_and_params(
        bounds.unsafe_convert_frame[.WORLD](),
        PRIMARY_WIDTH,
        PRIMARY_HEIGHT,
        PRIMARY_VIEWS,
        FOV_SCALE,
    )
    var rays = camera[0].copy()
    var camera_params = camera[1].copy()
    print(t"rays : {len(rays)}")

    print("\nGPU triangle BLAS[width]")
    print("----------------------")
    print("\nCPU reference")
    _print_cpu_ref_header()
    var reference = _print_cpu_triangle_reference[8](
        String("CpuBlasSet[8] lbvh"),
        tri_vertices,
        rays,
    )
    var reference_checksum = reference[0]
    var reference_hit_count = reference[1]
    print("")

    comptime if not has_accelerator():
        raise "No compatible GPU found; skipped Mojo GPU BoundsBvh benchmark."

    with DeviceContext() as ctx:
        var d_vertices = upload_vertices(ctx, tri_vertices)
        var d_camera_params = upload_list(ctx, camera_params)
        ctx.synchronize()

        var tri2 = _run_width[2](
            ctx,
            d_vertices,
            d_camera_params,
            len(rays),
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            reference_checksum,
            reference_hit_count,
            BENCH_REPEATS,
        )
        var tri4 = _run_width[4](
            ctx,
            d_vertices,
            d_camera_params,
            len(rays),
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            reference_checksum,
            reference_hit_count,
            BENCH_REPEATS,
        )
        var tri8 = _run_width[8](
            ctx,
            d_vertices,
            d_camera_params,
            len(rays),
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            reference_checksum,
            reference_hit_count,
            BENCH_REPEATS,
        )

        _print_gpu_results_transposed(tri2, tri4, tri8)

        var tri2_leaf4 = _run_width[2, 4](
            ctx,
            d_vertices,
            d_camera_params,
            len(rays),
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            reference_checksum,
            reference_hit_count,
            BENCH_REPEATS,
        )
        print("\nIndependent node/leaf width comparison")
        _print_gpu_results_transposed(tri2, tri2_leaf4, tri4)

        var tri8_leaf4 = _run_width[8, 4](
            ctx,
            d_vertices,
            d_camera_params,
            len(rays),
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            reference_checksum,
            reference_hit_count,
            BENCH_REPEATS,
        )
        print("\nAdditional node/leaf width comparison")
        _print_gpu_results_transposed(tri2_leaf4, tri4, tri8_leaf4)

        var cwbvh8_result = _run_cwbvh8(
            ctx,
            d_vertices,
            d_camera_params,
            len(rays),
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            reference_checksum,
            reference_hit_count,
            BENCH_REPEATS,
        )
        var cwbvh8_row = cwbvh8_result[0].copy()
        var cwbvh8 = cwbvh8_result[1].copy()
        print("\nPreferred H-PLOC CWBVH8 comparison")
        print("CWBVH8 build stages are reported together as '- other'.")
        _print_gpu_results_transposed(tri2_leaf4, tri8_leaf4, cwbvh8_row)

        print("\nGPU any-hit traversal (packed rays)")
        print("---------------------------------")
        var d_trace_rays = upload_rays[.WORLD](ctx, rays)
        var d_any_hits = ctx.enqueue_create_buffer[.float32](
            len(rays) * Hit[.WORLD].STRIDE
        )

        var any_bvh2 = build_gpu_triangle_bvh[.WORLD, 2, 2](ctx, d_vertices)
        ctx.synchronize()
        var any2 = _bench_any_hit_triangle[2, 2](
            ctx, any_bvh2, d_trace_rays, d_any_hits, len(rays), BENCH_REPEATS
        )

        var any_bvh2_leaf4 = build_gpu_triangle_bvh[.WORLD, 2, 4](
            ctx, d_vertices
        )
        ctx.synchronize()
        var any2_leaf4 = _bench_any_hit_triangle[2, 4](
            ctx,
            any_bvh2_leaf4,
            d_trace_rays,
            d_any_hits,
            len(rays),
            BENCH_REPEATS,
        )
        var any_cwbvh8 = _bench_any_hit_cwbvh8(
            ctx,
            cwbvh8,
            d_trace_rays,
            d_any_hits,
            len(rays),
            BENCH_REPEATS,
        )

        print("layout                 best ms       MRay/s       hits")
        print(
            t"n2/l2                  {round(ns_to_ms(any2[0]), 3)}"
            t"     {round(ns_to_mrays_per_s(any2[0], len(rays)), 1)}"
            t"     {any2[1]}"
        )
        print(
            t"n2/l4                  {round(ns_to_ms(any2_leaf4[0]), 3)}"
            t"     {round(ns_to_mrays_per_s(any2_leaf4[0], len(rays)), 1)}"
            t"     {any2_leaf4[1]}"
        )
        print(
            t"H-PLOC CWBVH8          {round(ns_to_ms(any_cwbvh8[0]), 3)}"
            t"     {round(ns_to_mrays_per_s(any_cwbvh8[0], len(rays)), 1)}"
            t"     {any_cwbvh8[1]}"
        )

        print("\nGPU sphere BLAS[width]")
        print("--------------------")
        var spheres = _make_sphere_grid()
        var sphere_bounds = sphere_bounds(spheres)
        var sphere_camera = make_camera_rays_and_params(
            sphere_bounds,
            SPHERE_RAY_WIDTH,
            SPHERE_RAY_HEIGHT,
            SPHERE_RAY_VIEWS,
            FOV_SCALE,
        )
        var sphere_rays = sphere_camera[0].copy()
        var sphere_camera_params = sphere_camera[1].copy()
        var d_sphere_camera_params = upload_list(
            ctx,
            sphere_camera_params,
        )
        ctx.synchronize()

        print(t"spheres : {len(spheres)}")
        print(t"sphere rays : {len(sphere_rays)}")

        print("\nCPU sphere reference")
        _print_cpu_ref_header()

        var sphere_reference2 = _print_cpu_sphere_reference[2](
            String("CpuBlasSet[2] lbvh"),
            spheres,
            sphere_rays,
        )
        var sphere_reference4 = _print_cpu_sphere_reference[4](
            String("CpuBlasSet[4] lbvh"),
            spheres,
            sphere_rays,
        )
        var sphere_reference8 = _print_cpu_sphere_reference[8](
            String("CpuBlasSet[8] lbvh"),
            spheres,
            sphere_rays,
        )

        print("\n")
        var sph2 = _run_sphere_width[2](
            ctx,
            spheres,
            len(sphere_rays),
            d_sphere_camera_params,
            SPHERE_RAY_WIDTH,
            SPHERE_RAY_HEIGHT,
            sphere_reference2[0],
            sphere_reference2[1],
            BENCH_REPEATS,
        )
        var sph4 = _run_sphere_width[4](
            ctx,
            spheres,
            len(sphere_rays),
            d_sphere_camera_params,
            SPHERE_RAY_WIDTH,
            SPHERE_RAY_HEIGHT,
            sphere_reference4[0],
            sphere_reference4[1],
            BENCH_REPEATS,
        )
        var sph8 = _run_sphere_width[8](
            ctx,
            spheres,
            len(sphere_rays),
            d_sphere_camera_params,
            SPHERE_RAY_WIDTH,
            SPHERE_RAY_HEIGHT,
            sphere_reference8[0],
            sphere_reference8[1],
            BENCH_REPEATS,
        )
        _print_gpu_results_transposed(sph2, sph4, sph8)

        var sph2_leaf4 = _run_sphere_width[2, 4](
            ctx,
            spheres,
            len(sphere_rays),
            d_sphere_camera_params,
            SPHERE_RAY_WIDTH,
            SPHERE_RAY_HEIGHT,
            sphere_reference4[0],
            sphere_reference4[1],
            BENCH_REPEATS,
        )
        print("\nIndependent node/leaf width comparison")
        _print_gpu_results_transposed(sph2, sph2_leaf4, sph4)


def main() raises:
    run_benchmark()
