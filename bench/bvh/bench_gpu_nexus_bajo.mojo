"""Build and closest-hit traversal comparison for Bajo versus NexusBVH."""

from std.math import ceildiv, max, round
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.builder.hploc_layout import HPLOC_MERGING_THRESHOLD
from bajo.bvh.constants import GPU_BOUNDS_BVH_BLOCK_SIZE
from bajo.bvh.gpu.triangle_bvh import build_gpu_triangle_bvh
from bajo.bvh.gpu.trace import GpuTraversalStats
from bajo.bvh.gpu.utils import (
    _download_full_hit_checksum,
    upload_list,
    upload_vertices,
)
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Hit
from bajo.benchmark.bvh_fixtures import make_camera_rays_and_params
from bajo.core import Frame
from bajo.core.utils import ns_to_mrays_per_s, ns_to_ms
from bajo.parser.obj.pack import pack_obj_triangles
from bajo.benchmark.gpu_bvh_fixtures import (
    Cwbvh8BenchBvh,
    build_cwbvh8_bench_bvh,
    create_cwbvh8_bench_arena,
    trace_cwbvh8_camera_kernel,
    trace_cwbvh8_indexed_camera_kernel,
    trace_cwbvh8_camera_stats_kernel,
)


comptime DRAGON_PATH = "./assets/dragon/dragon.obj"
comptime BENCH_REPEATS = 11
comptime RAY_WIDTH = 1024
comptime RAY_HEIGHT = 576
comptime FOV_SCALE = 0.2


def _median_ns(timings: List[Int]) -> Int:
    var values = timings.copy()
    sort(values)
    return values[(len(values) - 1) >> 1]


def _run_case[
    method: GpuBvhBuildMethod,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    mut ctx: DeviceContext,
    d_vertices: DeviceBuffer[.float32],
    d_camera: DeviceBuffer[.float32],
    triangle_count: Int,
    ray_count: Int,
) raises:
    var method_name: String
    var label_prefix: String
    comptime if method == GpuBvhBuildMethod.LBVH:
        method_name = "lbvh"
        label_prefix = "LBVH"
    else:
        method_name = "hploc"
        label_prefix = "H-PLOC"
    var label = String(t"{label_prefix}-n{Int(node_width)}-l{Int(leaf_width)}")

    var warm = build_gpu_triangle_bvh[
        Frame.WORLD, node_width, leaf_width, method
    ](ctx, d_vertices)
    ctx.synchronize()

    var build_timings = List[Int](capacity=BENCH_REPEATS)
    for _ in range(BENCH_REPEATS):
        var start = perf_counter_ns()
        var bvh = build_gpu_triangle_bvh[
            Frame.WORLD, node_width, leaf_width, method
        ](ctx, d_vertices)
        ctx.synchronize()
        build_timings.append(Int(perf_counter_ns() - start))

    var build_values = build_timings.copy()
    sort(build_values)
    var build_median_ms = ns_to_ms(_median_ns(build_timings))
    var build_min_ms = ns_to_ms(build_values[0])
    var build_max_ms = ns_to_ms(build_values[len(build_values) - 1])

    var d_hits = ctx.enqueue_create_buffer[.float32](ray_count * Hit.STRIDE)
    warm.launch_camera(ctx, d_camera, d_hits, ray_count, RAY_WIDTH, RAY_HEIGHT)
    ctx.synchronize()

    var trace_timings = List[Int](capacity=BENCH_REPEATS)
    for _ in range(BENCH_REPEATS):
        var start = perf_counter_ns()
        warm.launch_camera(
            ctx, d_camera, d_hits, ray_count, RAY_WIDTH, RAY_HEIGHT
        )
        ctx.synchronize()
        trace_timings.append(Int(perf_counter_ns() - start))

    var trace_values = trace_timings.copy()
    sort(trace_values)
    var trace_median_ns = _median_ns(trace_timings)
    var trace_median_ms = ns_to_ms(trace_median_ns)
    var trace_min_ms = ns_to_ms(trace_values[0])
    var trace_max_ms = ns_to_ms(trace_values[len(trace_values) - 1])
    var validation = _download_full_hit_checksum(ctx, d_hits, ray_count)

    print(
        label.ascii_ljust(18),
        String(round(build_median_ms, 3)).ascii_rjust(10),
        String(round(trace_median_ms, 3)).ascii_rjust(10),
        String(
            round(ns_to_mrays_per_s(trace_median_ns, ray_count), 1)
        ).ascii_rjust(12),
        String(validation[1]).ascii_rjust(8),
    )
    print(
        t"RESULT\tbajo\t{label}\t{method_name}\twide\t"
        t"{Int(node_width)}\t{Int(leaf_width)}\t{Int(leaf_width)}\t"
        t"{triangle_count}\t"
        t"{build_median_ms}\t{build_min_ms}\t{build_max_ms}\t"
        t"{ray_count}\t{trace_median_ms}\t{trace_min_ms}\t"
        t"{trace_max_ms}\t{validation[1]}\t{validation[0]}"
    )


def _run_cwbvh8_case[
    method: GpuBvhBuildMethod,
    max_leaf_size: Int = 3,
    indexed_triangles: Bool = False,
    direct_conversion: Bool = True,
    spatial_slots: Bool = True,
    merging_threshold: Int = HPLOC_MERGING_THRESHOLD,
](
    mut ctx: DeviceContext,
    d_vertices: DeviceBuffer[.float32],
    d_camera: DeviceBuffer[.float32],
    triangle_count: Int,
    ray_count: Int,
) raises:
    var method_name: String
    var label_prefix: String
    comptime if method == GpuBvhBuildMethod.LBVH:
        method_name = "lbvh"
        label_prefix = "LBVH"
    else:
        method_name = "hploc"
        label_prefix = "H-PLOC"
    var label = String(t"{label_prefix}-CWBVH8-n8-l4-m{max_leaf_size}")
    comptime if indexed_triangles:
        label += "-indexed"
    comptime if not direct_conversion:
        label += "-staged"
    comptime if not spatial_slots:
        label += "-frontier-slots"
    comptime if merging_threshold != HPLOC_MERGING_THRESHOLD:
        label += String(t"-mt{merging_threshold}")
    var warm: Cwbvh8BenchBvh
    var primitive_ids = ctx.enqueue_create_buffer[.uint32](
        max(triangle_count, 1)
    )
    var emitted_node_count = 0
    var build_timings = List[Int](capacity=BENCH_REPEATS)
    comptime if method == GpuBvhBuildMethod.HPLOC:
        var arena = create_cwbvh8_bench_arena[
            max_leaf_size,
            direct_conversion,
            indexed_triangles,
            spatial_slots,
            merging_threshold,
        ](ctx, d_vertices)
        ctx.synchronize()
        arena.finish_synchronized()
        for _ in range(BENCH_REPEATS):
            var start = perf_counter_ns()
            arena.enqueue_rebuild(ctx, d_vertices)
            ctx.synchronize()
            build_timings.append(Int(perf_counter_ns() - start))
        arena.finish_synchronized()
        with arena.wide.node_counts.map_to_host() as node_counts:
            emitted_node_count = Int(node_counts[0])
        warm = Cwbvh8BenchBvh(
            arena.nodes.copy(), arena.triangles.copy(), UInt32(0)
        )
        primitive_ids = (
            arena.representation_workspace.compact_primitive_ids.copy()
        )
    else:
        warm = build_cwbvh8_bench_bvh[method, max_leaf_size](ctx, d_vertices)
        ctx.synchronize()
        for _ in range(BENCH_REPEATS):
            var start = perf_counter_ns()
            var bvh = build_cwbvh8_bench_bvh[method, max_leaf_size](
                ctx, d_vertices
            )
            ctx.synchronize()
            build_timings.append(Int(perf_counter_ns() - start))

    var build_values = build_timings.copy()
    sort(build_values)
    var build_median_ms = ns_to_ms(_median_ns(build_timings))
    var build_min_ms = ns_to_ms(build_values[0])
    var build_max_ms = ns_to_ms(build_values[len(build_values) - 1])

    var d_hits = ctx.enqueue_create_buffer[.float32](ray_count * Hit.STRIDE)
    comptime if indexed_triangles:
        ctx.enqueue_function[
            trace_cwbvh8_indexed_camera_kernel[max_leaf_size, 8]
        ](
            warm.nodes,
            primitive_ids,
            d_vertices,
            warm.root_idx,
            d_camera,
            d_hits,
            Int32(ray_count),
            Int32(RAY_WIDTH),
            Int32(RAY_HEIGHT),
            Float32(1.0) / Float32(RAY_HEIGHT),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
    else:
        ctx.enqueue_function[trace_cwbvh8_camera_kernel[max_leaf_size, 8]](
            warm.nodes,
            warm.triangles,
            warm.root_idx,
            d_camera,
            d_hits,
            Int32(ray_count),
            Int32(RAY_WIDTH),
            Int32(RAY_HEIGHT),
            Float32(1.0) / Float32(RAY_HEIGHT),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
    ctx.synchronize()

    var trace_timings = List[Int](capacity=BENCH_REPEATS)
    for _ in range(BENCH_REPEATS):
        var start = perf_counter_ns()
        comptime if indexed_triangles:
            ctx.enqueue_function[
                trace_cwbvh8_indexed_camera_kernel[max_leaf_size, 8]
            ](
                warm.nodes,
                primitive_ids,
                d_vertices,
                warm.root_idx,
                d_camera,
                d_hits,
                Int32(ray_count),
                Int32(RAY_WIDTH),
                Int32(RAY_HEIGHT),
                Float32(1.0) / Float32(RAY_HEIGHT),
                grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
        else:
            ctx.enqueue_function[trace_cwbvh8_camera_kernel[max_leaf_size, 8]](
                warm.nodes,
                warm.triangles,
                warm.root_idx,
                d_camera,
                d_hits,
                Int32(ray_count),
                Int32(RAY_WIDTH),
                Int32(RAY_HEIGHT),
                Float32(1.0) / Float32(RAY_HEIGHT),
                grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
        ctx.synchronize()
        trace_timings.append(Int(perf_counter_ns() - start))

    var trace_values = trace_timings.copy()
    sort(trace_values)
    var trace_median_ns = _median_ns(trace_timings)
    var trace_median_ms = ns_to_ms(trace_median_ns)
    var trace_min_ms = ns_to_ms(trace_values[0])
    var trace_max_ms = ns_to_ms(trace_values[len(trace_values) - 1])
    var validation = _download_full_hit_checksum(ctx, d_hits, ray_count)

    var node_visits = UInt64(0)
    var leaf_groups = UInt64(0)
    var primitive_tests = UInt64(0)
    var max_stack = UInt32(0)
    comptime if not indexed_triangles:
        var d_stats = ctx.enqueue_create_buffer[.uint32](
            ray_count * GpuTraversalStats.STRIDE
        )
        ctx.enqueue_function[trace_cwbvh8_camera_stats_kernel](
            warm.nodes,
            warm.triangles,
            warm.root_idx,
            d_camera,
            d_hits,
            d_stats,
            Int32(ray_count),
            Int32(RAY_WIDTH),
            Int32(RAY_HEIGHT),
            Float32(1.0) / Float32(RAY_HEIGHT),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        with d_stats.map_to_host() as stats:
            for ray_idx in range(ray_count):
                var base = ray_idx * GpuTraversalStats.STRIDE
                node_visits += UInt64(
                    stats[base + GpuTraversalStats.NODE_VISITS]
                )
                leaf_groups += UInt64(
                    stats[base + GpuTraversalStats.LEAF_BLOCKS]
                )
                primitive_tests += UInt64(
                    stats[base + GpuTraversalStats.PRIMITIVE_TESTS]
                )
                max_stack = max(
                    max_stack,
                    stats[base + GpuTraversalStats.MAX_STACK_DEPTH],
                )

    print(
        label.ascii_ljust(25),
        String(round(build_median_ms, 3)).ascii_rjust(10),
        String(round(trace_median_ms, 3)).ascii_rjust(10),
        String(
            round(ns_to_mrays_per_s(trace_median_ns, ray_count), 1)
        ).ascii_rjust(12),
        String(validation[1]).ascii_rjust(8),
    )
    comptime if not indexed_triangles:
        print(
            t"PROFILE_TRAVERSAL\tbajo\t{label}\t{node_visits}\t"
            t"{leaf_groups}\t{primitive_tests}\t{max_stack}"
        )
    if emitted_node_count > 0:
        print(t"PROFILE_BUILD\tbajo\t{label}\t{emitted_node_count}")
    print(
        t"RESULT\tbajo\t{label}\t{method_name}\tcwbvh8\t8\t4\t"
        t"{max_leaf_size}\t"
        t"{triangle_count}\t{build_median_ms}\t{build_min_ms}\t"
        t"{build_max_ms}\t{ray_count}\t{trace_median_ms}\t"
        t"{trace_min_ms}\t{trace_max_ms}\t{validation[1]}\t"
        t"{validation[0]}"
    )


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"

    var args = argv()
    if len(args) > 2:
        raise (
            "usage: bench_gpu_nexus_bajo "
            "[hploc-cwbvh8-m1|hploc-cwbvh8-m1-indexed|"
            "hploc-cwbvh8-m1-staged|hploc-cwbvh8-m1-frontier-slots|"
            "hploc-cwbvh8-m1-mt4|hploc-cwbvh8-m1-mt8]"
        )
    var profile_hploc_cwbvh8_m1 = (
        len(args) == 2 and args[1] == "hploc-cwbvh8-m1"
    )
    var profile_hploc_cwbvh8_m1_indexed = (
        len(args) == 2 and args[1] == "hploc-cwbvh8-m1-indexed"
    )
    var profile_hploc_cwbvh8_m1_staged = (
        len(args) == 2 and args[1] == "hploc-cwbvh8-m1-staged"
    )
    var profile_hploc_cwbvh8_m1_frontier_slots = (
        len(args) == 2 and args[1] == "hploc-cwbvh8-m1-frontier-slots"
    )
    var profile_hploc_cwbvh8_m1_mt4 = (
        len(args) == 2 and args[1] == "hploc-cwbvh8-m1-mt4"
    )
    var profile_hploc_cwbvh8_m1_mt8 = (
        len(args) == 2 and args[1] == "hploc-cwbvh8-m1-mt8"
    )
    if (
        len(args) == 2
        and not profile_hploc_cwbvh8_m1
        and not profile_hploc_cwbvh8_m1_indexed
        and not profile_hploc_cwbvh8_m1_staged
        and not profile_hploc_cwbvh8_m1_frontier_slots
        and not profile_hploc_cwbvh8_m1_mt4
        and not profile_hploc_cwbvh8_m1_mt8
    ):
        raise "unknown benchmark configuration"

    var vertices = pack_obj_triangles[Frame.WORLD](DRAGON_PATH)
    var triangle_count = len(vertices) / 3
    var bounds = compute_bounds(vertices)
    var camera = make_camera_rays_and_params(
        bounds,
        RAY_WIDTH,
        RAY_HEIGHT,
        1,
        FOV_SCALE,
    )
    var ray_count = len(camera[0])

    print("Bajo GPU BVH parameter sweep")
    print(t"Triangles: {triangle_count}")
    print(t"Camera rays: {RAY_WIDTH}x{RAY_HEIGHT} ({ray_count})")
    print(t"Median of {BENCH_REPEATS} synchronized runs")
    print("Configuration       Build ms   Trace ms       MRay/s     Hits")
    print("------------------ ---------- ---------- ------------ --------")

    with DeviceContext() as ctx:
        var d_vertices = upload_vertices(ctx, vertices)
        var d_camera = upload_list(ctx, camera[1])
        ctx.synchronize()

        if profile_hploc_cwbvh8_m1:
            _run_cwbvh8_case[.HPLOC, 1](
                ctx, d_vertices, d_camera, triangle_count, ray_count
            )
            return
        if profile_hploc_cwbvh8_m1_indexed:
            _run_cwbvh8_case[.HPLOC, 1, True](
                ctx, d_vertices, d_camera, triangle_count, ray_count
            )
            return
        if profile_hploc_cwbvh8_m1_staged:
            _run_cwbvh8_case[.HPLOC, 1, False, False](
                ctx, d_vertices, d_camera, triangle_count, ray_count
            )
            return
        if profile_hploc_cwbvh8_m1_frontier_slots:
            _run_cwbvh8_case[.HPLOC, 1, False, True, False](
                ctx, d_vertices, d_camera, triangle_count, ray_count
            )
            return
        if profile_hploc_cwbvh8_m1_mt4:
            _run_cwbvh8_case[.HPLOC, 1, False, True, True, 4](
                ctx, d_vertices, d_camera, triangle_count, ray_count
            )
            return
        if profile_hploc_cwbvh8_m1_mt8:
            _run_cwbvh8_case[.HPLOC, 1, False, True, True, 8](
                ctx, d_vertices, d_camera, triangle_count, ray_count
            )
            return

        _run_case[.LBVH, 2, 2](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_case[.LBVH, 2, 4](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_case[.LBVH, 4, 2](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_case[.LBVH, 4, 4](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_case[.LBVH, 8, 4](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_case[.LBVH, 8, 8](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_case[.HPLOC, 2, 2](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_case[.HPLOC, 2, 4](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_case[.HPLOC, 4, 2](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_case[.HPLOC, 4, 4](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_case[.HPLOC, 8, 4](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_case[.HPLOC, 8, 8](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_case[.HPLOC, 8, 1](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_cwbvh8_case[.LBVH](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_cwbvh8_case[.HPLOC](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
        _run_cwbvh8_case[.HPLOC, 1](
            ctx, d_vertices, d_camera, triangle_count, ray_count
        )
