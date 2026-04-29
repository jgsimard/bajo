from std.benchmark import keep
from std.math import sqrt
from std.time import perf_counter_ns

from bajo.obj import read_obj, triangulated_indices
from bajo.core.vec import Vec3f32, vmin, vmax
from bajo.core.bvh.tinybvh import BVH, Ray, WideBVH


comptime DEFAULT_OBJ_PATH = "./assets/bunny/bunny.obj"
comptime PRIMARY_WIDTH = 640
comptime PRIMARY_HEIGHT = 360
comptime PRIMARY_VIEWS = 3
comptime TRAVERSAL_REPEATS = 8


@always_inline
def ns_to_ms(ns: Int) -> Float64:
    return Float64(ns) / 1_000_000.0


@always_inline
def ns_to_mrays_per_s(ns: Int, ray_count: Int) -> Float64:
    var seconds = Float64(ns) * 1.0e-9
    if seconds <= 0.0:
        return 0.0
    return (Float64(ray_count) / seconds) / 1_000_000.0


@always_inline
def dot(a: Vec3f32, b: Vec3f32) -> Float32:
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z()


@always_inline
def cross(a: Vec3f32, b: Vec3f32) -> Vec3f32:
    return Vec3f32(
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x(),
    )


@always_inline
def normalize(v: Vec3f32) -> Vec3f32:
    var len2 = dot(v, v)
    if len2 <= 1.0e-20:
        return Vec3f32(0.0, 0.0, 0.0)
    return v * (1.0 / sqrt(len2))


@always_inline
def len_vec(v: Vec3f32) -> Float32:
    return sqrt(dot(v, v))


def pack_obj_triangles(path: String) raises -> List[Vec3f32]:
    var mesh = read_obj(path)
    var idx = triangulated_indices(mesh)

    var out = List[Vec3f32](capacity=len(idx))

    for i in range(len(idx)):
        var p = Int(idx[i].p)

        # OBJ vertex indices refer to vertices, but mesh.positions is flat xyz.
        var base = p * 3

        out.append(
            Vec3f32(
                mesh.positions[base + 0],
                mesh.positions[base + 1],
                mesh.positions[base + 2],
            )
        )

    return out^


def compute_bounds(verts: List[Vec3f32]) -> Tuple[Vec3f32, Vec3f32]:
    var bmin = Vec3f32(1.0e30, 1.0e30, 1.0e30)
    var bmax = Vec3f32(-1.0e30, -1.0e30, -1.0e30)

    for i in range(len(verts)):
        bmin = vmin(bmin, verts[i])
        bmax = vmax(bmax, verts[i])

    return (bmin^, bmax^)


def append_camera_rays(
    mut rays: List[Ray],
    origin: Vec3f32,
    target: Vec3f32,
    up_hint: Vec3f32,
    width: Int,
    height: Int,
):
    var forward = normalize(target - origin)
    var right = normalize(cross(forward, up_hint))
    var up = normalize(cross(right, forward))

    var aspect = Float32(width) / Float32(height)
    var fov_scale = Float32(0.75)

    for y in range(height):
        for x in range(width):
            var sx = ((Float32(x) + 0.5) / Float32(width)) * 2.0 - 1.0
            var sy = 1.0 - ((Float32(y) + 0.5) / Float32(height)) * 2.0
            var dir = normalize(
                forward
                + right * (sx * aspect * fov_scale)
                + up * (sy * fov_scale)
            )
            rays.append(Ray(origin, dir))


def generate_primary_rays(
    bounds_min: Vec3f32,
    bounds_max: Vec3f32,
    width: Int,
    height: Int,
    views: Int,
) -> List[Ray]:
    var rays = List[Ray](capacity=width * height * views)

    var center = (bounds_min + bounds_max) * 0.5
    var extent = bounds_max - bounds_min
    var radius = len_vec(extent) * 0.5
    if radius < 1.0:
        radius = 1.0
    var dist = radius * 2.8

    if views >= 1:
        append_camera_rays(
            rays,
            center + Vec3f32(0.0, 0.0, -dist),
            center,
            Vec3f32(0.0, 1.0, 0.0),
            width,
            height,
        )

    if views >= 2:
        append_camera_rays(
            rays,
            center + Vec3f32(-dist, 0.0, 0.0),
            center,
            Vec3f32(0.0, 1.0, 0.0),
            width,
            height,
        )

    if views >= 3:
        append_camera_rays(
            rays,
            center + Vec3f32(0.0, dist, 0.0),
            center,
            Vec3f32(0.0, 0.0, 1.0),
            width,
            height,
        )

    return rays^


@always_inline
def hit_t_for_checksum(t: Float32) -> Float64:
    if t < 1.0e20:
        return Float64(t)
    return 0.0


def trace_bvh_primary(bvh: BVH, rays: List[Ray]) -> Float64:
    var checksum = Float64(0.0)
    var hit_count = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        bvh.traverse(ray)
        checksum += hit_t_for_checksum(ray.hit.t)
        if ray.hit.t < 1.0e20:
            hit_count += 1

    keep(checksum)
    keep(hit_count)
    return checksum


def trace_bvh_shadow(bvh: BVH, rays: List[Ray]) -> Int:
    var occluded = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        if bvh.is_occluded(ray):
            occluded += 1

    keep(occluded)
    return occluded


def trace_wide_primary[
    width: Int
](wide: WideBVH[width], rays: List[Ray]) -> Float64:
    var checksum = Float64(0.0)
    var hit_count = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        wide.traverse(ray)
        checksum += hit_t_for_checksum(ray.hit.t)
        if ray.hit.t < 1.0e20:
            hit_count += 1

    keep(checksum)
    keep(hit_count)
    return checksum


def trace_wide_shadow[width: Int](wide: WideBVH[width], rays: List[Ray]) -> Int:
    var occluded = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        if wide.is_occluded(ray):
            occluded += 1

    keep(occluded)
    return occluded


def print_traversal_result(
    name: String,
    best_ns: Int,
    ray_count: Int,
    checksum: Float64,
):
    print(
        name,
        "|",
        ns_to_ms(best_ns),
        "ms |",
        ns_to_mrays_per_s(best_ns, ray_count),
        "MRays/s | checksum:",
        checksum,
    )


def print_shadow_result(
    name: String,
    best_ns: Int,
    ray_count: Int,
    occluded: Int,
):
    print(
        name,
        "|",
        ns_to_ms(best_ns),
        "ms |",
        ns_to_mrays_per_s(best_ns, ray_count),
        "MRays/s | occluded:",
        occluded,
    )


def bench_bvh_primary(name: String, bvh: BVH, rays: List[Ray], repeats: Int):
    # Warmup.
    var checksum = trace_bvh_primary(bvh, rays)
    var best_ns = Int(9223372036854775807)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        checksum = trace_bvh_primary(bvh, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_traversal_result(name, best_ns, len(rays), checksum)


def bench_bvh_shadow(name: String, bvh: BVH, rays: List[Ray], repeats: Int):
    # Warmup.
    var occluded = trace_bvh_shadow(bvh, rays)
    var best_ns = Int(9223372036854775807)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        occluded = trace_bvh_shadow(bvh, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_shadow_result(name, best_ns, len(rays), occluded)


def bench_wide_primary[
    width: Int
](name: String, wide: WideBVH[width], rays: List[Ray], repeats: Int):
    # Warmup.
    var checksum = trace_wide_primary[width](wide, rays)
    var best_ns = Int(9223372036854775807)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        checksum = trace_wide_primary[width](wide, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_traversal_result(name, best_ns, len(rays), checksum)


def bench_wide_shadow[
    width: Int
](name: String, wide: WideBVH[width], rays: List[Ray], repeats: Int):
    # Warmup.
    var occluded = trace_wide_shadow[width](wide, rays)
    var best_ns = Int(9223372036854775807)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        occluded = trace_wide_shadow[width](wide, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_shadow_result(name, best_ns, len(rays), occluded)


def main() raises:
    print("TinyBVH Mojo bunny speedtest")
    print("Path:", DEFAULT_OBJ_PATH)
    print("Image rays:", PRIMARY_WIDTH, "x", PRIMARY_HEIGHT, "x", PRIMARY_VIEWS)

    print("\nLoading + packing OBJ...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles(DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var tri_count = UInt32(len(tri_vertices) // 3)
    var bounds = compute_bounds(tri_vertices)
    var bmin = bounds[0].copy()
    var bmax = bounds[1].copy()

    print("Packed vertices:", len(tri_vertices))
    print("Triangles:", tri_count)
    print("Load+pack ms:", ns_to_ms(Int(load_t1 - load_t0)))
    print("Bounds min:", bmin)
    print("Bounds max:", bmax)

    print("\nGenerating rays...")
    var rays = generate_primary_rays(
        bmin, bmax, PRIMARY_WIDTH, PRIMARY_HEIGHT, PRIMARY_VIEWS
    )
    print("Rays:", len(rays))

    print("\nBuild")
    print("-----")

    var t0 = perf_counter_ns()
    var bvh_median = BVH(tri_vertices.unsafe_ptr(), tri_count)
    bvh_median.build["median", False]()
    var t1 = perf_counter_ns()
    print(
        "binary median ST |",
        ns_to_ms(Int(t1 - t0)),
        "ms | nodes:",
        bvh_median.nodes_used,
        "| quality:",
        bvh_median.tree_quality(),
    )

    t0 = perf_counter_ns()
    var bvh_sah = BVH(tri_vertices.unsafe_ptr(), tri_count)
    bvh_sah.build["sah", False]()
    t1 = perf_counter_ns()
    print(
        "binary sah ST    |",
        ns_to_ms(Int(t1 - t0)),
        "ms | nodes:",
        bvh_sah.nodes_used,
        "| quality:",
        bvh_sah.tree_quality(),
    )

    t0 = perf_counter_ns()
    var bvh_sah_mt = BVH(tri_vertices.unsafe_ptr(), tri_count)
    bvh_sah_mt.build["sah", True]()
    t1 = perf_counter_ns()
    print(
        "binary sah MT    |",
        ns_to_ms(Int(t1 - t0)),
        "ms | nodes:",
        bvh_sah_mt.nodes_used,
        "| quality:",
        bvh_sah_mt.tree_quality(),
    )

    t0 = perf_counter_ns()
    var wide4 = WideBVH[4](bvh_sah)
    t1 = perf_counter_ns()
    print(
        "wide4 collapse   |",
        ns_to_ms(Int(t1 - t0)),
        "ms | nodes:",
        len(wide4.nodes),
        "| leaves:",
        len(wide4.leaves),
    )

    t0 = perf_counter_ns()
    var wide8 = WideBVH[8](bvh_sah)
    t1 = perf_counter_ns()
    print(
        "wide8 collapse   |",
        ns_to_ms(Int(t1 - t0)),
        "ms | nodes:",
        len(wide8.nodes),
        "| leaves:",
        len(wide8.leaves),
    )

    print("\nPrimary traversal")
    print("-----------------")
    bench_bvh_primary("binary median", bvh_median, rays, TRAVERSAL_REPEATS)
    bench_bvh_primary("binary sah ST", bvh_sah, rays, TRAVERSAL_REPEATS)
    bench_bvh_primary("binary sah MT", bvh_sah_mt, rays, TRAVERSAL_REPEATS)
    bench_wide_primary[4]("wide4", wide4, rays, TRAVERSAL_REPEATS)
    bench_wide_primary[8]("wide8", wide8, rays, TRAVERSAL_REPEATS)

    print("\nShadow traversal")
    print("----------------")
    bench_bvh_shadow("binary median", bvh_median, rays, TRAVERSAL_REPEATS)
    bench_bvh_shadow("binary sah ST", bvh_sah, rays, TRAVERSAL_REPEATS)
    bench_bvh_shadow("binary sah MT", bvh_sah_mt, rays, TRAVERSAL_REPEATS)
    bench_wide_shadow[4]("wide4", wide4, rays, TRAVERSAL_REPEATS)
    bench_wide_shadow[8]("wide8", wide8, rays, TRAVERSAL_REPEATS)

    # Keep external vertex buffer alive until the end: BVH stores an UnsafePointer to it.
    keep(len(tri_vertices))
    keep(len(rays))
