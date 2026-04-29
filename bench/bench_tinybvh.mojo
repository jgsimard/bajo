from std.benchmark import keep
from std.math import abs, round, sqrt
from std.time import perf_counter_ns

from bajo.obj import read_obj, triangulated_indices
from bajo.core.vec import Vec3f32, vmin, vmax, cross, dot, length, normalize
from bajo.core.bvh.tinybvh import BVH, BVHGPU, Ray, WideBVH


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


def print_vec3_rounded(name: String, v: Vec3f32):
    var x = round(Float64(v.x()), 3)
    var y = round(Float64(v.y()), 3)
    var z = round(Float64(v.z()), 3)
    print(t"{name} ({x}, {y}, {z})")


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
    var radius = length(extent) * 0.5
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


def trace_gpu_primary(gpu: BVHGPU, rays: List[Ray]) -> Float64:
    var checksum = Float64(0.0)
    var hit_count = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        gpu.traverse(ray)
        checksum += hit_t_for_checksum(ray.hit.t)
        if ray.hit.t < 1.0e20:
            hit_count += 1

    keep(checksum)
    keep(hit_count)
    return checksum


def trace_gpu_shadow(gpu: BVHGPU, rays: List[Ray]) -> Int:
    var occluded = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        if gpu.is_occluded(ray):
            occluded += 1

    keep(occluded)
    return occluded


def print_build_bvh_result(
    name: String,
    ns: Int,
    nodes_used: UInt32,
    quality: Float32,
):
    var ms = round(ns_to_ms(ns), 3)
    var q = round(Float64(quality), 3)
    print(t"{name} | {ms} ms | nodes: {nodes_used} | quality: {q}")


def print_build_layout_result(
    name: String,
    ns: Int,
    nodes: Int,
    extra_label: String,
    extra_count: Int,
):
    var ms = round(ns_to_ms(ns), 3)
    print(t"{name} | {ms} ms | nodes: {nodes} | {extra_label}: {extra_count}")


def print_gpu_layout_result(
    ns: Int,
    nodes: Int,
    prims: Int,
    root_is_leaf: Bool,
):
    var ms = round(ns_to_ms(ns), 3)
    print(
        t"gpu layout       | {ms} ms | nodes: {nodes} | prims: {prims} | root"
        t" leaf: {root_is_leaf}"
    )


def print_traversal_result(
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


def print_primary_validation(
    name: String,
    reference_checksum: Float64,
    checksum: Float64,
):
    var diff = round(abs(checksum - reference_checksum), 3)
    if diff <= 0.001:
        print(t"{name} primary validation: OK | diff: {diff}")
    else:
        print(t"{name} primary validation: MISMATCH | diff: {diff}")


def print_shadow_validation(
    name: String,
    reference_occluded: Int,
    occluded: Int,
):
    if occluded == reference_occluded:
        print(t"{name} shadow validation:  OK | occluded: {occluded}")
    else:
        print(
            t"{name} shadow validation:  MISMATCH | ref: {reference_occluded} |"
            t" got: {occluded}"
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


def bench_gpu_primary(name: String, gpu: BVHGPU, rays: List[Ray], repeats: Int):
    # Warmup.
    var checksum = trace_gpu_primary(gpu, rays)
    var best_ns = Int(9223372036854775807)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        checksum = trace_gpu_primary(gpu, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_traversal_result(name, best_ns, len(rays), checksum)


def bench_gpu_shadow(name: String, gpu: BVHGPU, rays: List[Ray], repeats: Int):
    # Warmup.
    var occluded = trace_gpu_shadow(gpu, rays)
    var best_ns = Int(9223372036854775807)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        occluded = trace_gpu_shadow(gpu, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_shadow_result(name, best_ns, len(rays), occluded)


def main() raises:
    print("TinyBVH Mojo bunny speedtest")
    print(t"Path: {DEFAULT_OBJ_PATH}")
    print(t"Image rays: {PRIMARY_WIDTH} x {PRIMARY_HEIGHT} x {PRIMARY_VIEWS}")
    print(t"Traversal repeats: {TRAVERSAL_REPEATS}")

    print("\nLoading + packing OBJ...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles(DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var tri_count = UInt32(len(tri_vertices) // 3)
    var bounds = compute_bounds(tri_vertices)
    var bmin = bounds[0].copy()
    var bmax = bounds[1].copy()
    var load_ms = round(ns_to_ms(Int(load_t1 - load_t0)), 3)

    print(t"Packed vertices: {len(tri_vertices)}")
    print(t"Triangles: {tri_count}")
    print(t"Load+pack ms: {load_ms}")
    print_vec3_rounded("Bounds min:", bmin)
    print_vec3_rounded("Bounds max:", bmax)

    print("\nGenerating rays...")
    var rays = generate_primary_rays(
        bmin, bmax, PRIMARY_WIDTH, PRIMARY_HEIGHT, PRIMARY_VIEWS
    )
    print(t"Rays: {len(rays)}")

    print("\nBuild")
    print("-----")

    var t0 = perf_counter_ns()
    var bvh_median = BVH(tri_vertices.unsafe_ptr(), tri_count)
    bvh_median.build["median", False]()
    var t1 = perf_counter_ns()
    print_build_bvh_result(
        "binary median ST",
        Int(t1 - t0),
        bvh_median.nodes_used,
        bvh_median.tree_quality(),
    )

    t0 = perf_counter_ns()
    var bvh_sah = BVH(tri_vertices.unsafe_ptr(), tri_count)
    bvh_sah.build["sah", False]()
    t1 = perf_counter_ns()
    print_build_bvh_result(
        "binary sah ST   ",
        Int(t1 - t0),
        bvh_sah.nodes_used,
        bvh_sah.tree_quality(),
    )

    t0 = perf_counter_ns()
    var bvh_sah_mt = BVH(tri_vertices.unsafe_ptr(), tri_count)
    bvh_sah_mt.build["sah", True]()
    t1 = perf_counter_ns()
    print_build_bvh_result(
        "binary sah MT   ",
        Int(t1 - t0),
        bvh_sah_mt.nodes_used,
        bvh_sah_mt.tree_quality(),
    )

    t0 = perf_counter_ns()
    var wide4 = WideBVH[4](bvh_sah)
    t1 = perf_counter_ns()
    print_build_layout_result(
        "wide4 collapse  ",
        Int(t1 - t0),
        len(wide4.nodes),
        "leaves",
        len(wide4.leaves),
    )

    t0 = perf_counter_ns()
    var wide8 = WideBVH[8](bvh_sah)
    t1 = perf_counter_ns()
    print_build_layout_result(
        "wide8 collapse  ",
        Int(t1 - t0),
        len(wide8.nodes),
        "leaves",
        len(wide8.leaves),
    )

    t0 = perf_counter_ns()
    var gpu = BVHGPU(bvh_sah)
    t1 = perf_counter_ns()
    var gpu_root_is_leaf = False
    if len(gpu.nodes) > 0:
        gpu_root_is_leaf = gpu.nodes[0].is_leaf()
    print_gpu_layout_result(
        Int(t1 - t0), len(gpu.nodes), len(gpu.prim_indices), gpu_root_is_leaf
    )

    print("\nValidation")
    print("----------")
    var ref_checksum = trace_bvh_primary(bvh_sah, rays)
    var ref_occluded = trace_bvh_shadow(bvh_sah, rays)
    print_primary_validation(
        "binary median", ref_checksum, trace_bvh_primary(bvh_median, rays)
    )
    print_primary_validation(
        "binary sah MT", ref_checksum, trace_bvh_primary(bvh_sah_mt, rays)
    )
    print_primary_validation(
        "wide4", ref_checksum, trace_wide_primary[4](wide4, rays)
    )
    print_primary_validation(
        "wide8", ref_checksum, trace_wide_primary[8](wide8, rays)
    )
    print_primary_validation(
        "gpu layout CPU", ref_checksum, trace_gpu_primary(gpu, rays)
    )
    print_shadow_validation(
        "binary median", ref_occluded, trace_bvh_shadow(bvh_median, rays)
    )
    print_shadow_validation(
        "binary sah MT", ref_occluded, trace_bvh_shadow(bvh_sah_mt, rays)
    )
    print_shadow_validation(
        "wide4", ref_occluded, trace_wide_shadow[4](wide4, rays)
    )
    print_shadow_validation(
        "wide8", ref_occluded, trace_wide_shadow[8](wide8, rays)
    )
    print_shadow_validation(
        "gpu layout CPU", ref_occluded, trace_gpu_shadow(gpu, rays)
    )

    print("\nPrimary traversal")
    print("-----------------")
    bench_bvh_primary("binary median ", bvh_median, rays, TRAVERSAL_REPEATS)
    bench_bvh_primary("binary sah ST ", bvh_sah, rays, TRAVERSAL_REPEATS)
    bench_bvh_primary("binary sah MT ", bvh_sah_mt, rays, TRAVERSAL_REPEATS)
    bench_wide_primary[4]("wide4         ", wide4, rays, TRAVERSAL_REPEATS)
    bench_wide_primary[8]("wide8         ", wide8, rays, TRAVERSAL_REPEATS)
    bench_gpu_primary("gpu layout CPU", gpu, rays, TRAVERSAL_REPEATS)

    print("\nShadow traversal")
    print("----------------")
    bench_bvh_shadow("binary median ", bvh_median, rays, TRAVERSAL_REPEATS)
    bench_bvh_shadow("binary sah ST ", bvh_sah, rays, TRAVERSAL_REPEATS)
    bench_bvh_shadow("binary sah MT ", bvh_sah_mt, rays, TRAVERSAL_REPEATS)
    bench_wide_shadow[4]("wide4         ", wide4, rays, TRAVERSAL_REPEATS)
    bench_wide_shadow[8]("wide8         ", wide8, rays, TRAVERSAL_REPEATS)
    bench_gpu_shadow("gpu layout CPU", gpu, rays, TRAVERSAL_REPEATS)

    # Keep external vertex buffer alive until the end: BVH stores an UnsafePointer to it.
    keep(len(tri_vertices))
    keep(len(rays))
