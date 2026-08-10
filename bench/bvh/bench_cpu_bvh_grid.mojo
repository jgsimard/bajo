from std.math import round
from std.time import perf_counter_ns

from bajo.bvh.constants import TRACE, f32_max
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.core import Frame, Vec3f32, Point3f32, Rayf32
from bench.bvh.bench_printing import TablePrinter


comptime GRID_SIDE = 256
comptime PRIM_COUNT = GRID_SIDE * GRID_SIDE
comptime RAY_REPEATS_PER_PRIM = 4
comptime RAY_COUNT = PRIM_COUNT * RAY_REPEATS_PER_PRIM
comptime TRAVERSAL_REPEATS = 8


def _grid_x(i: Int) -> Float32:
    return (Float32(i % GRID_SIDE) - Float32(GRID_SIDE) * 0.5) * 3.0


def _grid_y(i: Int) -> Float32:
    return (Float32(i / GRID_SIDE) - Float32(GRID_SIDE) * 0.5) * 3.0


def make_grid_triangles() -> List[Point3f32[Frame.WORLD]]:
    var vertices = List[Point3f32[Frame.WORLD]](capacity=PRIM_COUNT * 3)

    for i in range(PRIM_COUNT):
        var cx = _grid_x(i)
        var cy = _grid_y(i)

        vertices.append(
            Point3f32[Frame.WORLD](
                cx - 0.75,
                cy - 0.75,
                2.0,
            )
        )
        vertices.append(
            Point3f32[Frame.WORLD](
                cx + 0.75,
                cy - 0.75,
                2.0,
            )
        )
        vertices.append(
            Point3f32[Frame.WORLD](
                cx,
                cy + 0.75,
                2.0,
            )
        )

    return vertices^


def make_hit_and_miss_rays() -> List[Rayf32[Frame.WORLD]]:
    var rays = List[Rayf32[Frame.WORLD]](capacity=RAY_COUNT)

    for i in range(RAY_COUNT):
        var prim_idx = i % PRIM_COUNT

        if i % RAY_REPEATS_PER_PRIM == 0:
            # Deliberate miss.
            rays.append(
                Rayf32[Frame.WORLD](
                    Point3f32[Frame.WORLD](
                        10000.0 + Float32(i),
                        10000.0,
                        0.0,
                    ),
                    Vec3f32[Frame.WORLD](0.0, 0.0, 1.0),
                )
            )
        else:
            # Hit the corresponding grid primitive.
            rays.append(
                Rayf32[Frame.WORLD](
                    Point3f32[Frame.WORLD](
                        _grid_x(prim_idx),
                        _grid_y(prim_idx),
                        0.0,
                    ),
                    Vec3f32[Frame.WORLD](0.0, 0.0, 1.0),
                )
            )

    return rays^


@fieldwise_init
struct PrimaryBenchResult(Copyable):
    var ns: Int
    var checksum: Float64


def trace_triangle[
    width: SIMDLength
](
    bvh: TriangleBvh[Frame.WORLD, width],
    rays: List[Rayf32[Frame.WORLD]],
) -> Float64:
    var checksum = 0.0

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

    return checksum


def benchmark_triangle[
    width: SIMDLength
](
    bvh: TriangleBvh[Frame.WORLD, width],
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
    var bvh = TriangleBvh[
        Frame.WORLD,
        width,
    ].__init__[
        split_method
    ](vertices)
    var t1 = perf_counter_ns()

    var result = benchmark_triangle[width](bvh, rays)

    print_case_result(
        table,
        width,
        split_method,
        Int(t1 - t0),
        len(bvh.tree.nodes),
        bvh.tri_count,
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


def main() raises:
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

    comptime for split_method in ["median", "sah", "lbvh"]:
        benchmark_widths[split_method](
            table,
            vertices,
            rays,
        )
