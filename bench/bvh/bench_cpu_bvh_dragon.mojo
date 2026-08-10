from std.math import round
from std.time import perf_counter_ns

from bajo.bvh.constants import TRACE, f32_max
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.host_utils import compute_bounds
from bajo.obj.pack import pack_obj_triangles
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.core import Frame, Point3f32, Rayf32
from bench.bvh.fixtures import make_camera_rays_and_params
from bench.bvh.bench_printing import TablePrinter


comptime OBJ_PATH = "./assets/dragon/dragon.obj"
comptime RAY_WIDTH = 1024
comptime RAY_HEIGHT = 576
comptime FOV_SCALE = 0.2
comptime TRAVERSAL_REPEATS = 8


@fieldwise_init
struct SceneBenchResult(Copyable):
    var ns: Int
    var checksum: Float64
    var hits: Int


def trace_scene[
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
](
    bvh: TriangleBvh[Frame.WORLD, bounds_width, leaf_width],
    rays: List[Rayf32[Frame.WORLD]],
) -> Tuple[Float64, Int]:
    var checksum = 0.0
    var hits = 0

    for ray in rays:
        var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)

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


def benchmark_scene[
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
](
    bvh: TriangleBvh[Frame.WORLD, bounds_width, leaf_width],
    rays: List[Rayf32[Frame.WORLD]],
) -> SceneBenchResult:
    # Warm-up.
    var summary = trace_scene[bounds_width, leaf_width](bvh, rays)
    var best_ns = Int.MAX

    for _ in range(TRAVERSAL_REPEATS):
        var t0 = perf_counter_ns()
        summary = trace_scene[bounds_width, leaf_width](bvh, rays)
        var t1 = perf_counter_ns()

        var elapsed_ns = Int(t1 - t0)
        if elapsed_ns < best_ns:
            best_ns = elapsed_ns

    return SceneBenchResult(
        best_ns,
        summary[0],
        summary[1],
    )


def print_case_result(
    table: TablePrinter,
    split_method: String,
    bounds_width: Int,
    leaf_width: Int,
    build_ns: Int,
    result: SceneBenchResult,
    ray_count: Int,
) raises:
    table.result_line(
        split_method=split_method,
        bounds_width=String(bounds_width),
        leaf_width=String(leaf_width),
        build_ms=String(round(ns_to_ms(build_ns), 3)),
        trace_ms=String(round(ns_to_ms(result.ns), 3)),
        MRay_s=String(
            round(
                ns_to_mrays_per_s(result.ns, ray_count),
                3,
            )
        ),
        hits=String(result.hits),
        checksum=String(round(result.checksum, 3)),
    )


def benchmark_case[
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
    split_method: String,
](
    table: TablePrinter,
    vertices: List[Point3f32[Frame.WORLD]],
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    var t0 = perf_counter_ns()
    var bvh = TriangleBvh[
        Frame.WORLD,
        bounds_width,
        leaf_width,
    ].__init__[
        split_method
    ](vertices)
    var t1 = perf_counter_ns()

    var result = benchmark_scene[bounds_width, leaf_width](bvh, rays)

    print_case_result(
        table,
        split_method,
        bounds_width,
        leaf_width,
        Int(t1 - t0),
        result,
        len(rays),
    )


def benchmark_configurations[
    split_method: String
](
    table: TablePrinter,
    vertices: List[Point3f32[Frame.WORLD]],
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    comptime for bounds_width in [2, 4, 8]:
        benchmark_case[bounds_width, bounds_width, split_method](
            table,
            vertices,
            rays,
        )

    comptime for leaf_width in [2, 4, 8, 16]:
        benchmark_case[16, leaf_width, split_method](
            table,
            vertices,
            rays,
        )


def main() raises:
    print("Representative Dragon camera-ray benchmark")
    print(t"OBJ: {OBJ_PATH}")

    var vertices = pack_obj_triangles[Frame.WORLD](OBJ_PATH)
    var bounds = compute_bounds(vertices)
    var camera = make_camera_rays_and_params(
        bounds,
        RAY_WIDTH,
        RAY_HEIGHT,
        1,
        FOV_SCALE,
    )
    var rays = camera[0].copy()

    print(t"Triangles: {len(vertices) / 3}")
    print(t"Rays: {len(rays)}")

    var table = TablePrinter(
        split_method=12,
        bounds_width=12,
        leaf_width=10,
        build_ms=10,
        trace_ms=10,
        MRay_s=8,
        hits=6,
        checksum=15,
    )
    table.header()

    comptime for split_method in ["sah", "lbvh"]:
        benchmark_configurations[split_method](
            table,
            vertices,
            rays,
        )
