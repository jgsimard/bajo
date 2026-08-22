from std.math import round
from std.time import perf_counter_ns

from bajo.bvh.constants import TRACE, f32_max
from bajo.bvh.cpu.blas_set import build_triangle_blases, trace_triangle_blas_set
from bajo.bvh.types import BlasDesc, CpuBlasSet
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.core import Frame, Point3f32, Rayf32
from bajo.benchmark.bvh_fixtures import (
    PRIM_COUNT,
    RAY_COUNT,
    make_grid_triangles,
    make_hit_and_miss_rays,
)
from bajo.benchmark.bvh_reporting import TablePrinter


comptime TRAVERSAL_REPEATS = 8


@fieldwise_init
struct PrimaryBenchResult(Copyable):
    var ns: Int
    var checksum: Float64


def trace_triangle[
    width: SIMDLength
](bvh: CpuBlasSet[width], rays: List[Rayf32[Frame.WORLD]],) -> Float64:
    var checksum = 0.0

    for ray in rays:
        var hit = trace_triangle_blas_set[
            width, width, TRACE.CLOSEST_HIT, Frame.WORLD
        ](bvh, UInt32(0), ray)

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

    return checksum


def benchmark_triangle[
    width: SIMDLength
](
    bvh: CpuBlasSet[width],
    rays: List[Rayf32[Frame.WORLD]],
) -> PrimaryBenchResult:
    # Warm-up.
    var checksum = trace_triangle[width](bvh, rays)
    var best_ns = Int.MAX

    for _ in range(TRAVERSAL_REPEATS):
        var t0 = perf_counter_ns()
        checksum = trace_triangle[width](bvh, rays)
        var t1 = perf_counter_ns()

        var elapsed_ns = Int(t1 - t0)
        if elapsed_ns < best_ns:
            best_ns = elapsed_ns

    return PrimaryBenchResult(best_ns, checksum)


def print_case_result(
    table: TablePrinter,
    width: Int,
    split_method: String,
    build_ns: Int,
    nodes: Int,
    prims: Int,
    result: PrimaryBenchResult,
    ray_count: Int,
) raises:
    table.result_line(
        prim="tri",
        split_method=split_method,
        width=String(width),
        build=String(round(ns_to_ms(build_ns), 3)),
        nodes=String(nodes),
        prims=String(prims),
        primary=String(round(ns_to_ms(result.ns), 3)),
        MRay_s=String(
            round(
                ns_to_mrays_per_s(result.ns, ray_count),
                3,
            )
        ),
        checksum=String(round(result.checksum, 3)),
    )


def benchmark_case[
    width: SIMDLength,
    split_method: String,
](
    table: TablePrinter,
    vertices: List[Point3f32[Frame.WORLD]],
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    var t0 = perf_counter_ns()
    var bvh = build_triangle_blases[width, width, split_method, Frame.WORLD](
        [vertices.copy()]
    )
    var t1 = perf_counter_ns()

    var result = benchmark_triangle[width](bvh, rays)

    print_case_result(
        table,
        width,
        split_method,
        Int(t1 - t0),
        Int(BlasDesc.load(bvh.descs.unsafe_ptr(), UInt32(0)).node_count),
        len(vertices) / 3,
        result,
        len(rays),
    )


def benchmark_widths[
    split_method: String
](
    table: TablePrinter,
    vertices: List[Point3f32[Frame.WORLD]],
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    comptime for width in [2, 4, 8, 16]:
        benchmark_case[width, split_method](
            table,
            vertices,
            rays,
        )


def run_benchmark() raises:
    print("Primitive BoundsBvh benchmark")
    print(t"Primitives: {PRIM_COUNT}")
    print(t"Rays: {RAY_COUNT}")
    print(t"Traversal repeats: {TRAVERSAL_REPEATS}")

    print("\nGenerating primitives + rays...")
    var vertices = make_grid_triangles()
    var rays = make_hit_and_miss_rays()

    print(t"Triangle vertices: {len(vertices)}")
    print(t"Rays: {len(rays)}")

    print("\nResults")
    print("-------")

    var table = TablePrinter(
        prim=4,
        split_method=12,
        width=5,
        build=8,
        nodes=6,
        prims=6,
        primary=9,
        MRay_s=7,
        checksum=14,
    )
    table.header()

    comptime for split_method in ["median", "sah", "lbvh", "hploc"]:
        benchmark_widths[split_method](
            table,
            vertices,
            rays,
        )


def main() raises:
    run_benchmark()
