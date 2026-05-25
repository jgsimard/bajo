from std.benchmark import keep
from std.math import round
from std.time import perf_counter_ns

from bajo.bvh.types import Ray, Sphere
from bajo.bvh.constants import TRACE
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.host_utils import hit_t_for_checksum
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.core.vec import Vec3f32


comptime GRID_SIDE = 256
comptime PRIM_COUNT = GRID_SIDE * GRID_SIDE
comptime RAY_REPEATS_PER_PRIM = 4
comptime RAY_COUNT = PRIM_COUNT * RAY_REPEATS_PER_PRIM
comptime TRAVERSAL_REPEATS = 8


def _grid_x(i: Int) -> Float32:
    return (Float32(i % GRID_SIDE) - Float32(GRID_SIDE) * 0.5) * 3.0


def _grid_y(i: Int) -> Float32:
    return (Float32(i / GRID_SIDE) - Float32(GRID_SIDE) * 0.5) * 3.0


def make_grid_triangles() -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=PRIM_COUNT * 3)

    for i in range(PRIM_COUNT):
        var cx = _grid_x(i)
        var cy = _grid_y(i)

        verts.append(Vec3f32(cx - 0.75, cy - 0.75, 2.0))
        verts.append(Vec3f32(cx + 0.75, cy - 0.75, 2.0))
        verts.append(Vec3f32(cx, cy + 0.75, 2.0))

    return verts^


def make_grid_spheres() -> List[Sphere]:
    var spheres = List[Sphere](capacity=PRIM_COUNT)

    for i in range(PRIM_COUNT):
        var s = Sphere(Vec3f32(_grid_x(i), _grid_y(i), 2.0), 0.75)
        spheres.append(s)

    return spheres^


def make_hit_and_miss_rays() -> List[Ray]:
    var rays = List[Ray](capacity=RAY_COUNT)

    for i in range(RAY_COUNT):
        var prim_idx = i % PRIM_COUNT

        if i % 4 == 0:
            # Deliberate miss.
            rays.append(
                Ray(
                    Vec3f32(10000.0 + Float32(i), 10000.0, 0.0),
                    Vec3f32(0.0, 0.0, 1.0),
                )
            )
        else:
            # Hit the corresponding grid primitive.
            rays.append(
                Ray(
                    Vec3f32(_grid_x(prim_idx), _grid_y(prim_idx), 0.0),
                    Vec3f32(0.0, 0.0, 1.0),
                )
            )

    return rays^


@fieldwise_init
struct PrimaryBenchResult(Copyable):
    var ns: Int
    var checksum: Float64


@fieldwise_init
struct ShadowBenchResult(Copyable):
    var ns: Int
    var occluded: Int


def print_result_legend():
    var c0 = String("case").ascii_ljust(12)
    var c1 = String("build").ascii_rjust(8)
    var c2 = String("nodes").ascii_rjust(6)
    var c3 = String("prims").ascii_rjust(6)
    var c4 = String("primary").ascii_rjust(9)
    var c5 = String("MRay/s").ascii_rjust(9)
    var c6 = String("checksum").ascii_rjust(9)

    print(t"{c0} {c1} {c2} {c3} {c4} {c5} {c6}")
    print("------------ -------- ------ ------ --------- --------- ---------")


def print_case_result(
    name: String,
    build_ns: Int,
    nodes: Int,
    prims: UInt32,
    primary: PrimaryBenchResult,
    ray_count: Int,
):
    var build_ms = round(ns_to_ms(build_ns), 3)

    var primary_ms = round(ns_to_ms(primary.ns), 3)
    var primary_mrays = round(ns_to_mrays_per_s(primary.ns, ray_count), 3)
    var checksum = round(primary.checksum, 3)

    var c0 = name.ascii_ljust(12)
    var c1 = String(t"{build_ms}").ascii_rjust(8)
    var c2 = String(t"{nodes}").ascii_rjust(6)
    var c3 = String(t"{prims}").ascii_rjust(6)
    var c4 = String(t"{primary_ms}").ascii_rjust(9)
    var c5 = String(t"{primary_mrays}").ascii_rjust(9)
    var c6 = String(t"{checksum}").ascii_rjust(9)

    print(t"{c0} {c1} {c2} {c3} {c4} {c5} {c6}")


def trace_triangle_primary[
    width: Int
](bvh: TriangleBvh[width], rays: List[Ray]) -> Float64:
    var checksum = Float64(0.0)

    for ray in rays:
        var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)
        checksum += hit_t_for_checksum(hit.t)

    return checksum


def trace_triangle_shadow[
    width: Int
](bvh: TriangleBvh[width], rays: List[Ray]) -> Int:
    var occluded = 0

    for ray in rays:
        if bvh.trace[TRACE.ANY_HIT](ray).is_occluded():
            occluded += 1

    return occluded


def trace_sphere_primary[
    width: Int
](bvh: SphereBvh[width], rays: List[Ray]) -> Float64:
    var checksum = Float64(0.0)

    for ray in rays:
        var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)
        checksum += hit_t_for_checksum(hit.t)

    return checksum


def trace_sphere_shadow[
    width: Int
](bvh: SphereBvh[width], rays: List[Ray]) -> Int:
    var occluded = 0

    for ray in rays:
        if bvh.trace[TRACE.ANY_HIT](ray).is_occluded():
            occluded += 1

    return occluded


def bench_triangle_primary[
    width: Int
](bvh: TriangleBvh[width], rays: List[Ray]) -> PrimaryBenchResult:
    var checksum = trace_triangle_primary[width](bvh, rays)
    var best_ns = Int.MAX

    for _ in range(TRAVERSAL_REPEATS):
        var t0 = perf_counter_ns()
        checksum = trace_triangle_primary[width](bvh, rays)
        var t1 = perf_counter_ns()

        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    keep(checksum)
    return PrimaryBenchResult(best_ns, checksum)


def bench_sphere_primary[
    width: Int
](bvh: SphereBvh[width], rays: List[Ray]) -> PrimaryBenchResult:
    var checksum = trace_sphere_primary[width](bvh, rays)
    var best_ns = Int.MAX

    for _ in range(TRAVERSAL_REPEATS):
        var t0 = perf_counter_ns()
        checksum = trace_sphere_primary[width](bvh, rays)
        var t1 = perf_counter_ns()

        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    keep(checksum)
    return PrimaryBenchResult(best_ns, checksum)


def _case_name[prim: String, width: Int, split_method: String]() -> String:
    name: String
    comptime if split_method == "median":
        name = "median "
    elif split_method == "sah":
        name = "sah    "
    elif split_method == "lbvh":
        name = "lbvh   "
    else:
        comptime assert False
    return String(t"{prim}{width} {name}")


def bench_triangle_case[
    width: Int,
    split_method: String,
](vertices: List[Vec3f32], rays: List[Ray],):
    var name = _case_name["tri", width, split_method]()

    var t0 = perf_counter_ns()
    var bvh = TriangleBvh[width].__init__[split_method](
        vertices.unsafe_ptr().unsafe_mut_cast[True](),
        UInt32(len(vertices) / 3),
    )
    var t1 = perf_counter_ns()

    var build_ns = Int(t1 - t0)
    var primary = bench_triangle_primary[width](bvh, rays)

    print_case_result(
        name,
        build_ns,
        len(bvh.tree.nodes),
        bvh.tri_count,
        primary,
        len(rays),
    )

    keep(len(bvh.tree.nodes))


def bench_sphere_case[
    width: Int,
    split_method: String,
](spheres: List[Sphere], rays: List[Ray],):
    var name = _case_name["sph", width, split_method]()

    var t0 = perf_counter_ns()
    var bvh = SphereBvh[width].__init__[split_method](
        spheres.unsafe_ptr().unsafe_mut_cast[True](),
        UInt32(len(spheres)),
    )
    var t1 = perf_counter_ns()

    var build_ns = Int(t1 - t0)
    var primary = bench_sphere_primary[width](bvh, rays)

    print_case_result(
        name,
        build_ns,
        len(bvh.tree.nodes),
        bvh.sphere_count,
        primary,
        len(rays),
    )

    keep(len(bvh.tree.nodes))


def bench_triangle_widths[
    split_method: String
](vertices: List[Vec3f32], rays: List[Ray],):
    bench_triangle_case[2, split_method](vertices, rays)
    bench_triangle_case[4, split_method](vertices, rays)
    bench_triangle_case[8, split_method](vertices, rays)


def bench_sphere_widths[
    split_method: String
](spheres: List[Sphere], rays: List[Ray],):
    bench_sphere_case[2, split_method](spheres, rays)
    bench_sphere_case[4, split_method](spheres, rays)
    bench_sphere_case[8, split_method](spheres, rays)


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
    print_result_legend()

    bench_triangle_widths["median"](tri_vertices, rays)
    bench_triangle_widths["sah"](tri_vertices, rays)
    bench_triangle_widths["lbvh"](tri_vertices, rays)

    bench_sphere_widths["median"](spheres, rays)
    bench_sphere_widths["sah"](spheres, rays)
    bench_sphere_widths["lbvh"](spheres, rays)

    # Keep owning lists alive until after all BVHs and traversals are done.
    keep(len(tri_vertices))
    keep(len(spheres))
    keep(len(rays))
