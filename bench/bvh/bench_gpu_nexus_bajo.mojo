"""Build and closest-hit traversal comparison for Bajo versus NexusBVH."""

from std.math import ceildiv, round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.constants import GPU_BOUNDS_BVH_BLOCK_SIZE
from bajo.bvh.gpu.triangle_bvh import build_gpu_triangle_bvh
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
    build_cwbvh8_bench_bvh,
    trace_cwbvh8_camera_kernel,
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
    d_vertices: DeviceBuffer[DType.float32],
    d_camera: DeviceBuffer[DType.float32],
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
    var label = String(
        t"{label_prefix}-n{Int(node_width)}-l{Int(leaf_width)}"
    )

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

    var d_hits = ctx.enqueue_create_buffer[DType.float32](
        ray_count * Hit.STRIDE
    )
    warm.launch_camera(
        ctx, d_camera, d_hits, ray_count, RAY_WIDTH, RAY_HEIGHT
    )
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
](
    mut ctx: DeviceContext,
    d_vertices: DeviceBuffer[DType.float32],
    d_camera: DeviceBuffer[DType.float32],
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
    var label = String(
        t"{label_prefix}-CWBVH8-n8-l4-m{max_leaf_size}"
    )
    var warm = build_cwbvh8_bench_bvh[method, max_leaf_size](
        ctx, d_vertices
    )
    ctx.synchronize()

    var build_timings = List[Int](capacity=BENCH_REPEATS)
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

    var d_hits = ctx.enqueue_create_buffer[DType.float32](
        ray_count * Hit.STRIDE
    )
    ctx.enqueue_function[trace_cwbvh8_camera_kernel](
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
        ctx.enqueue_function[trace_cwbvh8_camera_kernel](
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

    print(
        label.ascii_ljust(25),
        String(round(build_median_ms, 3)).ascii_rjust(10),
        String(round(trace_median_ms, 3)).ascii_rjust(10),
        String(
            round(ns_to_mrays_per_s(trace_median_ns, ray_count), 1)
        ).ascii_rjust(12),
        String(validation[1]).ascii_rjust(8),
    )
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
