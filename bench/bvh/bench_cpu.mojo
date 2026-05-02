from std.benchmark import keep
from std.math import abs, round
from std.time import perf_counter_ns

from bajo.core.bvh import (
    compute_bounds,
    hit_t_for_checksum,
    trace_bvh_shadow,
    generate_primary_rays,
)
from bajo.core.bvh.cpu.gpu_layout import BvhGpuLayout
from bajo.core.bvh.cpu.wide import WideBvh
from bajo.core.bvh.cpu.binary_bvh import BinaryBvh, Ray
from bajo.core.utils import (
    ns_to_ms,
    ns_to_mrays_per_s,
    print_vec3_rounded,
    pack_obj_triangles,
)

# comptime DEFAULT_OBJ_PATH = "./assets/powerplant/powerplant.obj"
comptime DEFAULT_OBJ_PATH = "./assets/bunny/bunny.obj"
comptime PRIMARY_WIDTH = 640
comptime PRIMARY_HEIGHT = 360
comptime PRIMARY_VIEWS = 3
comptime TRAVERSAL_REPEATS = 8


def trace_bvh_primary(bvh: BinaryBvh, rays: List[Ray]) -> Float64:
    var checksum = 0.0
    var hit_count = 0
    for i in range(len(rays)):
        var ray = rays[i].copy()
        bvh.traverse(ray)
        checksum += hit_t_for_checksum(ray.hit.t)
        if ray.hit.t < 1.0e20:
            hit_count += 1
    return checksum


def trace_wide_primary[
    width: Int
](wide: WideBvh[width], rays: List[Ray]) -> Float64:
    var checksum = 0.0
    var hit_count = 0
    for i in range(len(rays)):
        var ray = rays[i].copy()
        wide.traverse(ray)
        checksum += hit_t_for_checksum(ray.hit.t)
        if ray.hit.t < 1.0e20:
            hit_count += 1
    return checksum


def trace_wide_shadow[width: Int](wide: WideBvh[width], rays: List[Ray]) -> Int:
    var occluded = 0
    for i in range(len(rays)):
        var ray = rays[i].copy()
        if wide.is_occluded(ray):
            occluded += 1
    return occluded


def trace_gpu_primary(gpu: BvhGpuLayout, rays: List[Ray]) -> Float64:
    var checksum = 0.0
    var hit_count = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        gpu.traverse(ray)
        checksum += hit_t_for_checksum(ray.hit.t)
        if ray.hit.t < 1.0e20:
            hit_count += 1
    return checksum


def trace_gpu_shadow(gpu: BvhGpuLayout, rays: List[Ray]) -> Int:
    var occluded = 0

    for i in range(len(rays)):
        var ray = rays[i].copy()
        if gpu.is_occluded(ray):
            occluded += 1
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
    front = String(t"{name} primary").ascii_ljust(22)
    if diff <= 0.001:
        print(t"{front} : OK | diff: {diff}")
    else:
        print(t"{front} : MISMATCH | diff: {diff}")


def print_shadow_validation(
    name: String,
    reference_occluded: Int,
    occluded: Int,
):
    front = String(t"{name} shadow").ascii_ljust(22)
    if occluded == reference_occluded:
        print(t"{front} : OK | occluded: {occluded}")
    else:
        print(
            t"{front} : MISMATCH | ref: {reference_occluded} | got: {occluded}"
        )


def bench_bvh_primary(
    name: String, bvh: BinaryBvh, rays: List[Ray], repeats: Int
):
    # Warmup.
    var checksum = trace_bvh_primary(bvh, rays)
    var best_ns = Int.MAX

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        checksum = trace_bvh_primary(bvh, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_traversal_result(name, best_ns, len(rays), checksum)


def bench_bvh_shadow(
    name: String, bvh: BinaryBvh, rays: List[Ray], repeats: Int
):
    # Warmup.
    var occluded = trace_bvh_shadow(bvh, rays)
    var best_ns = Int.MAX

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
](name: String, wide: WideBvh[width], rays: List[Ray], repeats: Int):
    # Warmup.
    var checksum = trace_wide_primary[width](wide, rays)
    var best_ns = Int.MAX

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
](name: String, wide: WideBvh[width], rays: List[Ray], repeats: Int):
    # Warmup.
    var occluded = trace_wide_shadow[width](wide, rays)
    var best_ns = Int.MAX

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        occluded = trace_wide_shadow[width](wide, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_shadow_result(name, best_ns, len(rays), occluded)


def bench_gpu_primary(
    name: String, gpu: BvhGpuLayout, rays: List[Ray], repeats: Int
):
    # Warmup.
    var checksum = trace_gpu_primary(gpu, rays)
    var best_ns = Int.MAX

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        checksum = trace_gpu_primary(gpu, rays)
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < best_ns:
            best_ns = dt

    print_traversal_result(name, best_ns, len(rays), checksum)


def bench_gpu_shadow(
    name: String, gpu: BvhGpuLayout, rays: List[Ray], repeats: Int
):
    # Warmup.
    var occluded = trace_gpu_shadow(gpu, rays)
    var best_ns = Int.MAX

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
    var bvh_median = BinaryBvh(tri_vertices.unsafe_ptr(), tri_count)
    bvh_median.build["median", False]()
    var t1 = perf_counter_ns()
    print_build_bvh_result(
        "binary median ST",
        Int(t1 - t0),
        bvh_median.nodes_used,
        bvh_median.tree_quality(),
    )

    t0 = perf_counter_ns()
    var bvh_sah = BinaryBvh(tri_vertices.unsafe_ptr(), tri_count)
    bvh_sah.build["sah", False]()
    t1 = perf_counter_ns()
    print_build_bvh_result(
        "binary sah ST   ",
        Int(t1 - t0),
        bvh_sah.nodes_used,
        bvh_sah.tree_quality(),
    )

    t0 = perf_counter_ns()
    var bvh_sah_mt = BinaryBvh(tri_vertices.unsafe_ptr(), tri_count)
    bvh_sah_mt.build["sah", True]()
    t1 = perf_counter_ns()
    print_build_bvh_result(
        "binary sah MT   ",
        Int(t1 - t0),
        bvh_sah_mt.nodes_used,
        bvh_sah_mt.tree_quality(),
    )

    t0 = perf_counter_ns()
    var bvh_lbvh = BinaryBvh(tri_vertices.unsafe_ptr(), tri_count)
    bvh_lbvh.build["lbvh", False]()
    t1 = perf_counter_ns()
    print_build_bvh_result(
        "binary lbvh ST  ",
        Int(t1 - t0),
        bvh_lbvh.nodes_used,
        bvh_lbvh.tree_quality(),
    )

    t0 = perf_counter_ns()
    var wide4 = WideBvh[4](bvh_sah)
    t1 = perf_counter_ns()
    print_build_layout_result(
        "wide4 collapse  ",
        Int(t1 - t0),
        len(wide4.nodes),
        "leaves",
        len(wide4.leaves),
    )

    t0 = perf_counter_ns()
    var wide8 = WideBvh[8](bvh_sah)
    t1 = perf_counter_ns()
    print_build_layout_result(
        "wide8 collapse  ",
        Int(t1 - t0),
        len(wide8.nodes),
        "leaves",
        len(wide8.leaves),
    )

    t0 = perf_counter_ns()
    var wide8_lbvh = WideBvh[8](bvh_lbvh)
    t1 = perf_counter_ns()
    print_build_layout_result(
        "wide8 lbvh     ",
        Int(t1 - t0),
        len(wide8_lbvh.nodes),
        "leaves",
        len(wide8_lbvh.leaves),
    )

    t0 = perf_counter_ns()
    var gpu = BvhGpuLayout(bvh_sah)
    t1 = perf_counter_ns()
    var gpu_root_is_leaf = False
    if len(gpu.nodes) > 0:
        gpu_root_is_leaf = gpu.nodes[0].is_leaf()
    print_gpu_layout_result(
        Int(t1 - t0), len(gpu.nodes), len(gpu.prim_indices), gpu_root_is_leaf
    )

    t0 = perf_counter_ns()
    var gpu_lbvh = BvhGpuLayout(bvh_lbvh)
    t1 = perf_counter_ns()
    var gpu_lbvh_root_is_leaf = False
    if len(gpu_lbvh.nodes) > 0:
        gpu_lbvh_root_is_leaf = gpu_lbvh.nodes[0].is_leaf()
    print_gpu_layout_result(
        Int(t1 - t0),
        len(gpu_lbvh.nodes),
        len(gpu_lbvh.prim_indices),
        gpu_lbvh_root_is_leaf,
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
        "binary lbvh", ref_checksum, trace_bvh_primary(bvh_lbvh, rays)
    )
    print_primary_validation(
        "wide4", ref_checksum, trace_wide_primary[4](wide4, rays)
    )
    print_primary_validation(
        "wide8", ref_checksum, trace_wide_primary[8](wide8, rays)
    )
    print_primary_validation(
        "wide8 lbvh", ref_checksum, trace_wide_primary[8](wide8_lbvh, rays)
    )
    print_primary_validation(
        "gpu layout CPU", ref_checksum, trace_gpu_primary(gpu, rays)
    )
    print_primary_validation(
        "gpu lbvh CPU", ref_checksum, trace_gpu_primary(gpu_lbvh, rays)
    )
    print_shadow_validation(
        "binary median", ref_occluded, trace_bvh_shadow(bvh_median, rays)
    )
    print_shadow_validation(
        "binary sah MT", ref_occluded, trace_bvh_shadow(bvh_sah_mt, rays)
    )
    print_shadow_validation(
        "binary lbvh", ref_occluded, trace_bvh_shadow(bvh_lbvh, rays)
    )
    print_shadow_validation(
        "wide4", ref_occluded, trace_wide_shadow[4](wide4, rays)
    )
    print_shadow_validation(
        "wide8", ref_occluded, trace_wide_shadow[8](wide8, rays)
    )
    print_shadow_validation(
        "wide8 lbvh", ref_occluded, trace_wide_shadow[8](wide8_lbvh, rays)
    )
    print_shadow_validation(
        "gpu layout CPU", ref_occluded, trace_gpu_shadow(gpu, rays)
    )
    print_shadow_validation(
        "gpu lbvh CPU", ref_occluded, trace_gpu_shadow(gpu_lbvh, rays)
    )

    print("\nPrimary traversal")
    print("-----------------")
    bench_bvh_primary("binary median ", bvh_median, rays, TRAVERSAL_REPEATS)
    bench_bvh_primary("binary sah ST ", bvh_sah, rays, TRAVERSAL_REPEATS)
    bench_bvh_primary("binary sah MT ", bvh_sah_mt, rays, TRAVERSAL_REPEATS)
    bench_bvh_primary("binary lbvh   ", bvh_lbvh, rays, TRAVERSAL_REPEATS)
    bench_wide_primary[4]("wide4         ", wide4, rays, TRAVERSAL_REPEATS)
    bench_wide_primary[8]("wide8         ", wide8, rays, TRAVERSAL_REPEATS)
    bench_wide_primary[8]("wide8 lbvh    ", wide8_lbvh, rays, TRAVERSAL_REPEATS)
    bench_gpu_primary("gpu layout CPU", gpu, rays, TRAVERSAL_REPEATS)
    bench_gpu_primary("gpu lbvh CPU  ", gpu_lbvh, rays, TRAVERSAL_REPEATS)

    print("\nShadow traversal")
    print("----------------")
    bench_bvh_shadow("binary median ", bvh_median, rays, TRAVERSAL_REPEATS)
    bench_bvh_shadow("binary sah ST ", bvh_sah, rays, TRAVERSAL_REPEATS)
    bench_bvh_shadow("binary sah MT ", bvh_sah_mt, rays, TRAVERSAL_REPEATS)
    bench_bvh_shadow("binary lbvh   ", bvh_lbvh, rays, TRAVERSAL_REPEATS)
    bench_wide_shadow[4]("wide4         ", wide4, rays, TRAVERSAL_REPEATS)
    bench_wide_shadow[8]("wide8         ", wide8, rays, TRAVERSAL_REPEATS)
    bench_wide_shadow[8]("wide8 lbvh    ", wide8_lbvh, rays, TRAVERSAL_REPEATS)
    bench_gpu_shadow("gpu layout CPU", gpu, rays, TRAVERSAL_REPEATS)
    bench_gpu_shadow("gpu lbvh CPU  ", gpu_lbvh, rays, TRAVERSAL_REPEATS)

    # Keep the owning List alive after all traversals
    keep(len(tri_vertices))
    if len(tri_vertices) > 0:
        keep(tri_vertices[0].x())
        keep(tri_vertices[len(tri_vertices) - 1].x())
