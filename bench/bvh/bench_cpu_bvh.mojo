from std.math import round
from std.time import perf_counter_ns

from bajo.bvh.types import Sphere
from bajo.bvh.constants import TRACE, f32_max
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.sphere_bvh import SphereBvh
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
    var verts = List[Point3f32[Frame.WORLD]](capacity=PRIM_COUNT * 3)

    for i in range(PRIM_COUNT):
        var cx = _grid_x(i)
        var cy = _grid_y(i)

        verts.append(Point3f32[Frame.WORLD](cx - 0.75, cy - 0.75, 2.0))
        verts.append(Point3f32[Frame.WORLD](cx + 0.75, cy - 0.75, 2.0))
        verts.append(Point3f32[Frame.WORLD](cx, cy + 0.75, 2.0))

    return verts^


def make_grid_spheres() -> List[Sphere[Frame.WORLD]]:
    var spheres = List[Sphere[Frame.WORLD]](capacity=PRIM_COUNT)

    for i in range(PRIM_COUNT):
        var s = Sphere[Frame.WORLD](
            Point3f32[Frame.WORLD](_grid_x(i), _grid_y(i), 2.0), 0.75
        )
        spheres.append(s)

    return spheres^


def make_hit_and_miss_rays() -> List[Rayf32[Frame.WORLD]]:
    var rays = List[Rayf32[Frame.WORLD]](capacity=RAY_COUNT)

    for i in range(RAY_COUNT):
        var prim_idx = i % PRIM_COUNT

        if i % 4 == 0:
            # Deliberate miss.
            rays.append(
                Rayf32[Frame.WORLD](
                    Point3f32[Frame.WORLD](10000.0 + Float32(i), 10000.0, 0.0),
                    Vec3f32[Frame.WORLD](0.0, 0.0, 1.0),
                )
            )
        else:
            # Hit the corresponding grid primitive.
            rays.append(
                Rayf32[Frame.WORLD](
                    Point3f32[Frame.WORLD](
                        _grid_x(prim_idx), _grid_y(prim_idx), 0.0
                    ),
                    Vec3f32[Frame.WORLD](0.0, 0.0, 1.0),
                )
            )

    return rays^


@fieldwise_init
struct PrimaryBenchResult(Copyable):
    var ns: Int
    var checksum: Float64


def trace_triangle_primary[
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


def trace_sphere_primary[
    width: SIMDLength
](
    bvh: SphereBvh[Frame.WORLD, width],
    rays: List[Rayf32[Frame.WORLD]],
) -> Float64:
    var checksum = 0.0
    for ray in rays:
        var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)
        if hit.t < f32_max:
            checksum += Float64(hit.t)
    return checksum


def bench_triangle_primary[
    width: SIMDLength
](
    bvh: TriangleBvh[Frame.WORLD, width],
    rays: List[Rayf32[Frame.WORLD]],
) -> PrimaryBenchResult:
    var checksum = trace_triangle_primary[width](bvh, rays)
    var best_ns = Int.MAX

    for _ in range(TRAVERSAL_REPEATS):
        var t0 = perf_counter_ns()
        checksum = trace_triangle_primary[width](bvh, rays)
        var t1 = perf_counter_ns()

        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt
    return PrimaryBenchResult(best_ns, checksum)


def bench_sphere_primary[
    width: SIMDLength
](
    bvh: SphereBvh[Frame.WORLD, width],
    rays: List[Rayf32[Frame.WORLD]],
) -> PrimaryBenchResult:
    var checksum = trace_sphere_primary[width](bvh, rays)
    var best_ns = Int.MAX

    for _ in range(TRAVERSAL_REPEATS):
        var t0 = perf_counter_ns()
        checksum = trace_sphere_primary[width](bvh, rays)
        var t1 = perf_counter_ns()

        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt
    return PrimaryBenchResult(best_ns, checksum)


def print_case_result(
    table: TablePrinter,
    prim: String,
    width: Int,
    split_method: String,
    build_ns: Int,
    nodes: Int,
    prims: Int,
    primary: PrimaryBenchResult,
    ray_count: Int,
) raises:
    var build_ms = round(ns_to_ms(build_ns), 3)
    var primary_ms = round(ns_to_ms(primary.ns), 3)
    var primary_mrays = round(
        ns_to_mrays_per_s(primary.ns, ray_count),
        3,
    )
    var checksum = round(primary.checksum, 3)

    table.result_line(
        prim=prim,
        width=String(width),
        split_method=split_method,
        build=String(build_ms),
        nodes=String(nodes),
        prims=String(prims),
        primary=String(primary_ms),
        MRay_s=String(primary_mrays),
        checksum=String(checksum),
    )


def bench_triangle_case[
    width: SIMDLength,
    split_method: String,
](
    table: TablePrinter,
    vertices: List[Point3f32[Frame.WORLD]],
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    var t0 = perf_counter_ns()
    var bvh = TriangleBvh[Frame.WORLD, width].__init__[split_method](
        vertices.copy()
    )
    var t1 = perf_counter_ns()

    var build_ns = Int(t1 - t0)
    var primary = bench_triangle_primary[width](bvh, rays)

    print_case_result(
        table,
        "tri",
        width,
        split_method,
        build_ns,
        len(bvh.tree.nodes),
        bvh.tri_count,
        primary,
        len(rays),
    )


def bench_sphere_case[
    width: SIMDLength,
    split_method: String,
](
    table: TablePrinter,
    spheres: List[Sphere[Frame.WORLD]],
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    var t0 = perf_counter_ns()
    var bvh = SphereBvh[Frame.WORLD, width].__init__[split_method](
        spheres.copy()
    )
    var t1 = perf_counter_ns()

    var build_ns = Int(t1 - t0)
    var primary = bench_sphere_primary[width](bvh, rays)

    print_case_result(
        table,
        "sph",
        width,
        split_method,
        build_ns,
        len(bvh.tree.nodes),
        bvh.sphere_count,
        primary,
        len(rays),
    )


def bench_triangle_widths[
    split_method: String
](
    table: TablePrinter,
    vertices: List[Point3f32[Frame.WORLD]],
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    comptime for width in [2, 4, 8, 16]:
        bench_triangle_case[width, split_method](
            table,
            vertices,
            rays,
        )


def bench_sphere_widths[
    split_method: String
](
    table: TablePrinter,
    spheres: List[Sphere[Frame.WORLD]],
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    comptime for width in [2, 4, 8, 16]:
        bench_sphere_case[width, split_method](
            table,
            spheres,
            rays,
        )


def main() raises:
    print("Primitive BoundsBvh benchmark")
    print(t"Primitives: {PRIM_COUNT}")
    print(t"Rays: {RAY_COUNT}")
    print(t"Traversal repeats: {TRAVERSAL_REPEATS}")

    print("\nGenerating primitives + rays...")
    var tri_vertices = make_grid_triangles()
    var spheres = make_grid_spheres()
    var rays = make_hit_and_miss_rays()

    print(t"Triangle vertices: {len(tri_vertices)}")
    print(t"Spheres: {len(spheres)}")
    print(t"Rays: {len(rays)}")

    print("\nResults")
    print("-------")

    var table = TablePrinter(
        prim=4,
        width=5,
        split_method=12,
        build=8,
        nodes=6,
        prims=6,
        primary=9,
        MRay_s=7,
        checksum=14,
    )
    table.header()

    comptime for m in ["median", "sah", "lbvh"]:
        bench_triangle_widths[m](
            table,
            tri_vertices,
            rays,
        )

    comptime for m in ["median", "sah", "lbvh"]:
        bench_sphere_widths[m](
            table,
            spheres,
            rays,
        )
