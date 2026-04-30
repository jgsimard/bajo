from std.benchmark import keep
from std.math import abs, round
from std.sys import has_accelerator
from std.time import perf_counter_ns

from bajo.core.bvh.tinybvh import BVH
from bajo.core.bvh.gpu_lbvh import (
    BENCH_REPEATS,
    DEFAULT_OBJ_PATH,
    PRIMARY_HEIGHT,
    PRIMARY_VIEWS,
    PRIMARY_WIDTH,
    compute_bounds,
    compute_centroid_bounds,
    generate_camera_params,
    generate_primary_rays,
    ns_to_mrays_per_s,
    ns_to_ms,
    pack_obj_triangles,
    print_vec3_rounded,
    run_gpu_lbvh_camera_reduce_and_shadow_benchmark,
    trace_bvh_primary,
    trace_bvh_shadow,
)


def main() raises:
    print("Best GPU LBVH benchmark")
    print(t"Path: {DEFAULT_OBJ_PATH}")
    print(t"Image rays: {PRIMARY_WIDTH} x {PRIMARY_HEIGHT} x {PRIMARY_VIEWS}")
    print(t"Repeats: {BENCH_REPEATS}")

    print("\nLoading + packing OBJ...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles(DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var tri_count = len(tri_vertices) // 3
    var internal_count = tri_count - 1
    var bounds = compute_bounds(tri_vertices)
    var bmin = bounds[0].copy()
    var bmax = bounds[1].copy()
    var cbounds = compute_centroid_bounds(tri_vertices)
    var cmin = cbounds[0].copy()
    var cmax = cbounds[1].copy()
    var load_ms = round(ns_to_ms(Int(load_t1 - load_t0)), 3)

    print(t"Packed vertices: {len(tri_vertices)}")
    print(t"Triangles: {tri_count}")
    print(t"Internal nodes: {internal_count}")
    print(t"Load+pack ms: {load_ms}")
    print_vec3_rounded("Bounds min:", bmin)
    print_vec3_rounded("Bounds max:", bmax)
    print_vec3_rounded("Centroid min:", cmin)
    print_vec3_rounded("Centroid max:", cmax)

    print("\nGenerating reference rays and camera parameters...")
    var rays = generate_primary_rays(
        bmin, bmax, PRIMARY_WIDTH, PRIMARY_HEIGHT, PRIMARY_VIEWS
    )
    var camera_params = generate_camera_params(bmin, bmax, PRIMARY_VIEWS)
    print(t"Rays: {len(rays)}")
    print(t"Camera params floats: {len(camera_params)}")

    print("\nCPU reference")
    print("-------------")
    var ref_build_t0 = perf_counter_ns()
    var ref_bvh = BVH(tri_vertices.unsafe_ptr(), UInt32(tri_count))
    ref_bvh.build["sah", True]()
    var ref_build_t1 = perf_counter_ns()

    var ref_trace_t0 = perf_counter_ns()
    var ref_checksum = trace_bvh_primary(ref_bvh, rays)
    var ref_occluded = trace_bvh_shadow(ref_bvh, rays)
    var ref_trace_t1 = perf_counter_ns()

    print(
        t"SAH MT build:       "
        t" {round(ns_to_ms(Int(ref_build_t1 - ref_build_t0)), 3)} ms"
    )
    print(
        t"reference queries:  "
        t" {round(ns_to_ms(Int(ref_trace_t1 - ref_trace_t0)), 3)} ms |"
        t" checksum: {round(ref_checksum, 3)} | occluded: {ref_occluded}"
    )

    print("\nBest GPU LBVH path")
    print("------------------")
    comptime if has_accelerator():
        var gpu_result = run_gpu_lbvh_camera_reduce_and_shadow_benchmark(
            tri_vertices,
            cmin,
            cmax,
            bmin,
            bmax,
            camera_params,
            len(rays),
            ref_checksum,
            ref_occluded,
            BENCH_REPEATS,
        )

        var static_upload_ns = gpu_result[0]
        var morton_ns = gpu_result[1]
        var sort_ns = gpu_result[2]
        var topology_ns = gpu_result[3]
        var refit_ns = gpu_result[4]

        var primary_kernel_ns = gpu_result[5]
        var primary_reduce_ns = gpu_result[6]
        var primary_download_ns = gpu_result[7]
        var primary_checksum = gpu_result[8]
        var primary_hits = gpu_result[9]
        var primary_diff = gpu_result[10]

        var shadow_kernel_ns = gpu_result[11]
        var shadow_reduce_ns = gpu_result[12]
        var shadow_download_ns = gpu_result[13]
        var shadow_occluded = gpu_result[14]
        var shadow_diff = gpu_result[15]

        var sorted_ok = gpu_result[16]
        var topology_ok = gpu_result[17]
        var bounds_ok = gpu_result[18]
        var root_idx = gpu_result[19]
        var guard = gpu_result[20]

        var build_ns = morton_ns + sort_ns + topology_ns + refit_ns
        var primary_frame_ns = (
            primary_kernel_ns + primary_reduce_ns + primary_download_ns
        )
        var shadow_frame_ns = (
            shadow_kernel_ns + shadow_reduce_ns + shadow_download_ns
        )
        var primary_total_ns = build_ns + primary_frame_ns
        var shadow_total_ns = build_ns + shadow_frame_ns

        print(t"static upload once:  {round(ns_to_ms(static_upload_ns), 3)} ms")
        print(
            t"build valid:         sorted={sorted_ok} |"
            t" topology={topology_ok} | bounds={bounds_ok} | root={root_idx}"
        )

        print("\nGPU LBVH build")
        print(t"morton generation:   {round(ns_to_ms(morton_ns), 3)} ms")
        print(t"radix sort pairs:    {round(ns_to_ms(sort_ns), 3)} ms")
        print(t"topology build:      {round(ns_to_ms(topology_ns), 3)} ms")
        print(t"bounds refit:        {round(ns_to_ms(refit_ns), 3)} ms")
        print(t"build total:         {round(ns_to_ms(build_ns), 3)} ms")

        print("\nGenerated primary rays + GPU checksum")
        print(
            t"camera kernel:       {round(ns_to_ms(primary_kernel_ns), 3)} ms"
            t" | {round(ns_to_mrays_per_s(primary_kernel_ns, len(rays)), 3)} MRays/s"
        )
        print(
            t"checksum reduction:  {round(ns_to_ms(primary_reduce_ns), 3)} ms"
        )
        print(
            t"partial download:    {round(ns_to_ms(primary_download_ns), 3)} ms"
        )
        print(
            t"query total:         {round(ns_to_ms(primary_frame_ns), 3)} ms"
            t" | {round(ns_to_mrays_per_s(primary_frame_ns, len(rays)), 3)} MRays/s"
        )
        print(
            t"build + query total: {round(ns_to_ms(primary_total_ns), 3)} ms"
            t" | {round(ns_to_mrays_per_s(primary_total_ns, len(rays)), 3)} MRays/s"
        )
        print(
            t"validation diff:     {round(primary_diff, 3)}"
            t" | checksum: {round(primary_checksum, 3)} | hits: {primary_hits}"
        )

        print("\nGenerated shadow rays + GPU occlusion count")
        print(
            t"shadow kernel:       {round(ns_to_ms(shadow_kernel_ns), 3)} ms"
            t" | {round(ns_to_mrays_per_s(shadow_kernel_ns, len(rays)), 3)} MRays/s"
        )
        print(t"occlusion reduction: {round(ns_to_ms(shadow_reduce_ns), 3)} ms")
        print(
            t"partial download:    {round(ns_to_ms(shadow_download_ns), 3)} ms"
        )
        print(
            t"query total:         {round(ns_to_ms(shadow_frame_ns), 3)} ms"
            t" | {round(ns_to_mrays_per_s(shadow_frame_ns, len(rays)), 3)} MRays/s"
        )
        print(
            t"build + query total: {round(ns_to_ms(shadow_total_ns), 3)} ms"
            t" | {round(ns_to_mrays_per_s(shadow_total_ns, len(rays)), 3)} MRays/s"
        )
        print(
            t"validation diff:     {shadow_diff}"
            t" | occluded: {shadow_occluded} | ref: {ref_occluded}"
        )
        print(t"checksum guard:      {guard}")
    else:
        print("No compatible GPU found; skipped Mojo GPU benchmark.")

    keep(len(tri_vertices))
    keep(len(rays))
