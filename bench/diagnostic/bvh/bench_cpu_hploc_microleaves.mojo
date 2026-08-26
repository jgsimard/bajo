"""CPU H-PLOC Morton-microleaf build/traversal experiment."""

from std.benchmark import keep
from std.math import min, round
from std.time import perf_counter_ns

from bajo.benchmark.bvh_fixtures import (
    make_camera_rays_and_params,
    make_grid_triangles,
    make_hit_and_miss_rays,
)
from bajo.benchmark.timing import summarize_timings
from bajo.bvh.constants import f32_max
from bajo.bvh.cpu import CpuBlasSet, CpuBvhBuildMethod
from bajo.bvh.cpu.blas_set import build_cpu_triangle_blas_set, trace_blas_set
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import BlasDesc
from bajo.core import Point3f32, Rayf32
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.parser.obj.pack import pack_obj_triangles


comptime OBJ_PATH = "./assets/dragon/dragon.obj"
comptime WIDTH = 16
comptime BUILD_REPEATS = 7
comptime TRACE_REPEATS = 7


@fieldwise_init
struct TraceResult(Copyable):
    var ns: Int
    var checksum: Float64
    var hits: Int


def _trace(
    bvh: CpuBlasSet[.TRIANGLE, WIDTH, WIDTH],
    rays: List[Rayf32[.WORLD]],
) -> Tuple[Float64, Int]:
    var checksum = 0.0
    var hits = 0
    for ray in rays:
        var hit = trace_blas_set[WIDTH, WIDTH, .CLOSEST_HIT, .WORLD](
            bvh, UInt32(0), ray
        )
        if hit.t < f32_max:
            checksum += (
                Float64(hit.t)
                + Float64(hit.u)
                + Float64(hit.v)
                + Float64(hit.normal.x)
                + Float64(hit.normal.y)
                + Float64(hit.normal.z)
                + Float64(hit.prim)
            )
            hits += 1
    return (checksum, hits)


def _benchmark_trace(
    bvh: CpuBlasSet[.TRIANGLE, WIDTH, WIDTH],
    rays: List[Rayf32[.WORLD]],
) -> TraceResult:
    var result = _trace(bvh, rays)
    var best_ns = Int.MAX
    for _ in range(TRACE_REPEATS):
        var start = perf_counter_ns()
        result = _trace(bvh, rays)
        best_ns = min(best_ns, Int(perf_counter_ns() - start))
    return TraceResult(best_ns, result[0], result[1])


def _benchmark_case[
    microleaf_size: Int,
](
    label: String,
    vertices: List[Point3f32[.WORLD]],
    rays: List[Rayf32[.WORLD]],
):
    var vertex_sets = [vertices.copy()]
    var bvh = build_cpu_triangle_blas_set[
        WIDTH,
        WIDTH,
        CpuBvhBuildMethod.HPLOC,
        .WORLD,
        microleaf_size,
    ](vertex_sets)
    var build_times = List[Int](capacity=BUILD_REPEATS)
    for _ in range(BUILD_REPEATS):
        var start = perf_counter_ns()
        bvh = build_cpu_triangle_blas_set[
            WIDTH,
            WIDTH,
            CpuBvhBuildMethod.HPLOC,
            .WORLD,
            microleaf_size,
        ](vertex_sets)
        build_times.append(Int(perf_counter_ns() - start))
    var build = summarize_timings(build_times)
    var trace = _benchmark_trace(bvh, rays)
    var desc = BlasDesc.load(bvh.descs.unsafe_ptr(), UInt32(0))
    keep(desc.node_count)
    print(
        t"{label}\t{microleaf_size}\t{round(ns_to_ms(build.median_ns), 3)}\t"
        t"{round(ns_to_ms(build.min_ns), 3)}\t"
        t"{round(ns_to_ms(trace.ns), 3)}\t"
        t"{round(ns_to_mrays_per_s(trace.ns, len(rays)), 3)}\t"
        t"{trace.hits}\t{round(trace.checksum, 3)}\t{desc.node_count}\t"
        t"{desc.leaf_block_count}"
    )


def _benchmark_scene(
    label: String,
    vertices: List[Point3f32[.WORLD]],
    rays: List[Rayf32[.WORLD]],
):
    print("")
    print(t"{label}: {len(vertices) // 3} triangles, {len(rays)} rays")
    print(
        "Scene\tMicroleaf\tBuild median ms\tBuild min ms\tTrace ms\tMRay/s\t"
        "Hits\tChecksum\tWide nodes\tLeaf blocks"
    )
    comptime for microleaf_size in [1, 2, 4, 8, 16]:
        _benchmark_case[microleaf_size](label, vertices, rays)


def main() raises:
    print("CPU H-PLOC Morton-microleaf experiment; BVH16 leaf16")
    var grid_vertices = make_grid_triangles()
    var grid_rays = make_hit_and_miss_rays()
    _benchmark_scene("grid", grid_vertices, grid_rays)

    var dragon_vertices = pack_obj_triangles[.WORLD](OBJ_PATH)
    var dragon_bounds = compute_bounds(dragon_vertices)
    var dragon_camera = make_camera_rays_and_params(
        dragon_bounds, 1024, 576, 1, 0.2
    )
    _benchmark_scene("dragon", dragon_vertices, dragon_camera[0].copy())
