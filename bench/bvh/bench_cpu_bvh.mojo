from std.benchmark import keep
from std.math import round
from std.time import perf_counter_ns

from bajo.core.bvh.types import Ray
from bajo.core.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.core.bvh.cpu.sphere_bvh import SphereBvh, Sphere

from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.core.vec import Vec3f32


comptime GRID_SIDE = 64
comptime PRIM_COUNT = GRID_SIDE * GRID_SIDE
comptime RAY_REPEATS_PER_PRIM = 4
comptime RAY_COUNT = PRIM_COUNT * RAY_REPEATS_PER_PRIM
comptime TRAVERSAL_REPEATS = 8


@always_inline
def _grid_x(i: Int) -> Float32:
    return (Float32(i % GRID_SIDE) - Float32(GRID_SIDE) * 0.5) * 3.0


@always_inline
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


@always_inline
def _hit_t_for_checksum(t: Float32) -> Float64:
    if t < Float32(1e20):
        return Float64(t)

    return 0.0


def print_build_result(
    name: String,
    ns: Int,
    nodes: Int,
    prims: UInt32,
):
    var ms = round(ns_to_ms(ns), 3)
    print(t"{name} | {ms} ms | nodes: {nodes} | prims: {prims}")


def print_primary_result(
    name: String,
    best_ns: Int,
    ray_count: Int,
    checksum: Float64,
):
    var ms = round(ns_to_ms(best_ns), 3)
    var mrays = round(ns_to_mrays_per_s(best_ns, ray_count), 3)
    var csum = round(checksum, 3)
    print(t"{name} | {ms} ms | {mrays} MRays/s | checksum: {csum}")


def print_shadow_result(
    name: String,
    best_ns: Int,
    ray_count: Int,
    occluded: Int,
):
    var ms = round(ns_to_ms(best_ns), 3)
    var mrays = round(ns_to_mrays_per_s(best_ns, ray_count), 3)
    print(t"{name} | {ms} ms | {mrays} MRays/s | occluded: {occluded}")


def trace_triangle_primary[
    width: Int, split_method: String
](bvh: TriangleBvh[width, split_method], rays: List[Ray]) -> Float64:
    var checksum = Float64(0.0)

    for i in range(len(rays)):
        var ray = rays[i].copy()
        bvh.traverse(ray)
        checksum += _hit_t_for_checksum(ray.hit.t)

    return checksum


def trace_triangle_shadow[
    width: Int, split_method: String
](bvh: TriangleBvh[width, split_method], rays: List[Ray]) -> Int:
    var occluded = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        if bvh.is_occluded(ray):
            occluded += 1

    return occluded


def trace_sphere_primary[
    width: Int, split_method: String
](bvh: SphereBvh[width, split_method], rays: List[Ray]) -> Float64:
    var checksum = 0.0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        bvh.traverse(ray)
        checksum += _hit_t_for_checksum(ray.hit.t)

    return checksum


def trace_sphere_shadow[
    width: Int, split_method: String
](bvh: SphereBvh[width, split_method], rays: List[Ray]) -> Int:
    var occluded = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        if bvh.is_occluded(ray):
            occluded += 1

    return occluded


def bench_triangle_primary[
    width: Int, split_method: String
](name: String, bvh: TriangleBvh[width, split_method], rays: List[Ray],):
    var checksum = trace_triangle_primary[width, split_method](bvh, rays)
    var best_ns = Int.MAX

    for _ in range(TRAVERSAL_REPEATS):
        var t0 = perf_counter_ns()
        checksum = trace_triangle_primary[width](bvh, rays)
        var t1 = perf_counter_ns()

        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    keep(checksum)
    print_primary_result(name, best_ns, len(rays), checksum)


def bench_triangle_shadow[
    width: Int, split_method: String
](name: String, bvh: TriangleBvh[width, split_method], rays: List[Ray],):
    var occluded = trace_triangle_shadow[width, split_method](bvh, rays)
    var best_ns = Int.MAX

    for _ in range(TRAVERSAL_REPEATS):
        var t0 = perf_counter_ns()
        occluded = trace_triangle_shadow[width](bvh, rays)
        var t1 = perf_counter_ns()

        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    keep(occluded)
    print_shadow_result(name, best_ns, len(rays), occluded)


def bench_sphere_primary[
    width: Int, split_method: String
](name: String, bvh: SphereBvh[width, split_method], rays: List[Ray],):
    var checksum = trace_sphere_primary[width, split_method](bvh, rays)
    var best_ns = Int.MAX

    for _ in range(TRAVERSAL_REPEATS):
        var t0 = perf_counter_ns()
        checksum = trace_sphere_primary[width](bvh, rays)
        var t1 = perf_counter_ns()

        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    keep(checksum)
    print_primary_result(name, best_ns, len(rays), checksum)


def bench_sphere_shadow[
    width: Int, split_method: String
](name: String, bvh: SphereBvh[width, split_method], rays: List[Ray],):
    var occluded = trace_sphere_shadow[width](bvh, rays)
    var best_ns = Int.MAX

    for _ in range(TRAVERSAL_REPEATS):
        var t0 = perf_counter_ns()
        occluded = trace_sphere_shadow[width](bvh, rays)
        var t1 = perf_counter_ns()

        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    keep(occluded)
    print_shadow_result(name, best_ns, len(rays), occluded)


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

    print("\nBuild")
    print("-----")

    var t0 = perf_counter_ns()
    var tri2_median = TriangleBvh[2, "median"](
        tri_vertices.unsafe_ptr(),
        UInt32(len(tri_vertices) / 3),
    )
    var t1 = perf_counter_ns()
    print_build_result(
        "tri2 median ",
        Int(t1 - t0),
        len(tri2_median.tree.nodes),
        tri2_median.tri_count,
    )

    t0 = perf_counter_ns()
    var tri4_median = TriangleBvh[4, "median"](
        tri_vertices.unsafe_ptr(),
        UInt32(len(tri_vertices) / 3),
    )
    t1 = perf_counter_ns()
    print_build_result(
        "tri4 median ",
        Int(t1 - t0),
        len(tri4_median.tree.nodes),
        tri4_median.tri_count,
    )

    t0 = perf_counter_ns()
    var tri8_median = TriangleBvh[8, "median"](
        tri_vertices.unsafe_ptr(),
        UInt32(len(tri_vertices) / 3),
    )
    t1 = perf_counter_ns()
    print_build_result(
        "tri8 median ",
        Int(t1 - t0),
        len(tri8_median.tree.nodes),
        tri8_median.tri_count,
    )

    t0 = perf_counter_ns()
    var tri4_sah = TriangleBvh[4, "sah"](
        tri_vertices.unsafe_ptr(),
        UInt32(len(tri_vertices) / 3),
    )
    t1 = perf_counter_ns()
    print_build_result(
        "tri4 sah    ",
        Int(t1 - t0),
        len(tri4_sah.tree.nodes),
        tri4_sah.tri_count,
    )

    t0 = perf_counter_ns()
    var sph2_median = SphereBvh[2, "median"](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )
    t1 = perf_counter_ns()
    print_build_result(
        "sph2 median ",
        Int(t1 - t0),
        len(sph2_median.tree.nodes),
        sph2_median.sphere_count,
    )

    t0 = perf_counter_ns()
    var sph4_median = SphereBvh[4, "median"](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )
    t1 = perf_counter_ns()
    print_build_result(
        "sph4 median ",
        Int(t1 - t0),
        len(sph4_median.tree.nodes),
        sph4_median.sphere_count,
    )

    t0 = perf_counter_ns()
    var sph8_median = SphereBvh[8, "median"](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )
    t1 = perf_counter_ns()
    print_build_result(
        "sph8 median ",
        Int(t1 - t0),
        len(sph8_median.tree.nodes),
        sph8_median.sphere_count,
    )

    t0 = perf_counter_ns()
    var sph4_sah = SphereBvh[4, "sah"](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )
    t1 = perf_counter_ns()
    print_build_result(
        "sph4 sah    ",
        Int(t1 - t0),
        len(sph4_sah.tree.nodes),
        sph4_sah.sphere_count,
    )

    print("\nPrimary traversal")
    print("-----------------")
    bench_triangle_primary[2]("tri2 median ", tri2_median, rays)
    bench_triangle_primary[4]("tri4 median ", tri4_median, rays)
    bench_triangle_primary[8]("tri8 median ", tri8_median, rays)
    bench_triangle_primary[4]("tri4 sah    ", tri4_sah, rays)
    bench_sphere_primary[2]("sph2 median ", sph2_median, rays)
    bench_sphere_primary[4]("sph4 median ", sph4_median, rays)
    bench_sphere_primary[8]("sph8 median ", sph8_median, rays)
    bench_sphere_primary[4]("sph4 sah    ", sph4_sah, rays)

    print("\nShadow traversal")
    print("----------------")
    bench_triangle_shadow[2]("tri2 median ", tri2_median, rays)
    bench_triangle_shadow[4]("tri4 median ", tri4_median, rays)
    bench_triangle_shadow[8]("tri8 median ", tri8_median, rays)
    bench_triangle_shadow[4]("tri4 sah    ", tri4_sah, rays)
    bench_sphere_shadow[2]("sph2 median ", sph2_median, rays)
    bench_sphere_shadow[4]("sph4 median ", sph4_median, rays)
    bench_sphere_shadow[8]("sph8 median ", sph8_median, rays)
    bench_sphere_shadow[4]("sph4 sah    ", sph4_sah, rays)

    # Keep owning lists alive until after all BVHs and traversals are done.
    keep(len(tri_vertices))
    keep(len(spheres))
    keep(len(rays))
