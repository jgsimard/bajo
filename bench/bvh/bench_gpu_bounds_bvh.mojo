from std.math import abs, round, min, max
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu import DeviceContext, DeviceBuffer

from bajo.core import AABB, Vec3f32, dot
from bajo.core.intersect import intersect_ray_sphere
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.bvh.camera import Camera
from bajo.bvh.host_utils import compute_bounds, sphere_bounds
from bajo.bvh.constants import EMPTY_LANE, TRACE, f32_max
from bajo.bvh.types import Hit, Ray, Sphere
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.gpu.sphere_bvh import GpuSphereBvh
from bajo.bvh.gpu.triangle_bvh import GpuTriangleBvh
from bajo.bvh.gpu.utils import (
    GpuBuildTimings,
    _download_full_hit_checksum,
    upload_list,
    upload_vertices,
)
from bench.bvh.bench_printing import (
    GpuBenchResult,
    print_transposed_header,
    _print_gpu_result_trace_rows,
    _print_gpu_result_validation_rows,
)
from bajo.obj.pack import pack_obj_triangles


# comptime DEFAULT_OBJ_PATH = "./assets/bunny/bunny.obj"
comptime DEFAULT_OBJ_PATH = "./assets/dragon/dragon.obj"
comptime PRIMARY_WIDTH = 1280
comptime PRIMARY_HEIGHT = 640
comptime FOV_SCALE = 0.2
comptime PRIMARY_VIEWS = 3
comptime BENCH_REPEATS = 8
comptime SPHERE_GRID_X = 64
comptime SPHERE_GRID_Y = 64
comptime SPHERE_RAY_WIDTH = 1280
comptime SPHERE_RAY_HEIGHT = 640
comptime SPHERE_RAY_VIEWS = 3

comptime DEBUG_SPHERE_GRID_X = 16
comptime DEBUG_SPHERE_GRID_Y = 16
comptime DEBUG_SPHERE_RAY_WIDTH = 256
comptime DEBUG_SPHERE_RAY_HEIGHT = 128
comptime DEBUG_SPHERE_RAY_VIEWS = 1
comptime TRIANGLE_HIT_REL_EPS = 0.0
comptime SPHERE_HIT_REL_EPS = 1.0e-3
comptime DEBUG_BENCH_REPEATS = 3


def _make_camera_rays_and_params(
    bounds: AABB,
    width: Int,
    height: Int,
    views: Int,
    fov_scale: Float32,
) -> Tuple[List[Ray], List[Float32]]:
    var center = bounds.centroid()
    var extent = bounds.extent()

    var scene_w = max(max(extent.x, extent.y), extent.z)
    if scene_w < 1.0:
        scene_w = 1.0

    var rays = List[Ray](capacity=width * height * views)
    var params = List[Float32](capacity=views * Camera.STRIDE)

    for view in range(views):
        var view_offset = Float32(view) - Float32(views - 1) * 0.5
        var eye = center + Vec3f32(
            view_offset * scene_w * 0.30,
            extent.y * 0.20,
            -scene_w * 2.50,
        )
        var camera = Camera(
            eye,
            center,
            Vec3f32(0.0, 1.0, 0.0),
            fov_scale,
        )
        params.extend(camera.flatten())

        for py in range(height):
            for px in range(width):
                rays.append(camera.make_ray(px, py, width, height))

    return (rays^, params^)


def _trace_cpu_triangle_bvh[
    width: SIMDSize
](mut bvh: TriangleBvh[width], rays: List[Ray]) -> Tuple[Float64, UInt32]:
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for ray in rays:
        var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)
        if hit.t < f32_max:
            checksum += Float64(hit.t)
            hit_count += 1

    return (checksum, hit_count)


def _trace_cpu_sphere_bvh[
    width: SIMDSize
](mut bvh: SphereBvh[width], rays: List[Ray]) -> Tuple[Float64, UInt32]:
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for ray in rays:
        var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)
        if hit.t < f32_max:
            checksum += Float64(hit.t)
            hit_count += 1

    return (checksum, hit_count)


def _trace_cpu_sphere_bruteforce(
    spheres: List[Sphere],
    rays: List[Ray],
) -> Tuple[Float64, UInt32]:
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for ray in rays:
        var hit = _trace_cpu_sphere_bruteforce_one(spheres, ray)
        if hit.t < f32_max:
            checksum += Float64(hit.t)
            hit_count += 1

    return (checksum, hit_count)


def _trace_cpu_sphere_bruteforce_one(
    spheres: List[Sphere],
    ray: Ray,
) -> Hit:
    var hit = Hit.miss(ray.t_max)

    for prim_i, s in enumerate(spheres):
        var h = intersect_ray_sphere(
            ray.o,
            ray.d,
            s.center,
            s.radius,
            hit.t,
            ray.t_min,
        )

        if h.mask[0] and h.t[0] < hit.t:
            hit.t = h.t[0]
            hit.u = 0.0
            hit.v = 0.0
            hit.prim = UInt32(prim_i)
            hit.inst = EMPTY_LANE

    return hit


def _print_sphere_debug_mismatches(
    mut ctx: DeviceContext,
    spheres: List[Sphere],
    rays: List[Ray],
    d_hits_f32: DeviceBuffer[DType.float32],
    d_hits_u32: DeviceBuffer[DType.uint32],
    image_width: Int,
    image_height: Int,
    max_print: Int,
) raises:
    comptime T_DIFF_EPS = Float64(1.0e-3)

    var ray_count = len(rays)

    var h_hits_f32 = ctx.enqueue_create_host_buffer[DType.float32](
        ray_count * 3
    )
    var h_hits_u32 = ctx.enqueue_create_host_buffer[DType.uint32](ray_count)

    d_hits_f32.enqueue_copy_to(h_hits_f32)
    d_hits_u32.enqueue_copy_to(h_hits_u32)
    ctx.synchronize()

    var printed = 0
    var mismatch_count = 0
    var gpu_only = 0
    var cpu_only = 0
    var both_hit_diff_t = 0

    for ray_idx in range(ray_count):
        var cpu_hit = _trace_cpu_sphere_bruteforce_one(
            spheres,
            rays[ray_idx],
        )

        var gpu_t = h_hits_f32[ray_idx * 3 + 0]
        var gpu_prim = UInt32(h_hits_u32[ray_idx])

        var cpu_has_hit = cpu_hit.t < Float32(1.0e20)
        var gpu_has_hit = gpu_t < Float32(1.0e20)

        if cpu_has_hit != gpu_has_hit:
            mismatch_count += 1

            if gpu_has_hit:
                gpu_only += 1
            else:
                cpu_only += 1

            if printed < max_print:
                var pixels_per_view = image_width * image_height
                var view_idx = ray_idx / pixels_per_view
                var local_idx = ray_idx - view_idx * pixels_per_view
                var px_i = local_idx % image_width
                var py_i = local_idx / image_width

                print(
                    t"\nhit/miss ray {ray_idx} "
                    t"\nview={view_idx} px={px_i} py={py_i}: "
                    t"\ncpu_hit={cpu_has_hit} "
                    t"\ncpu_t={cpu_hit.t} "
                    t"\ncpu_prim={cpu_hit.prim} | "
                    t"\ngpu_hit={gpu_has_hit} "
                    t"\ngpu_t={gpu_t} "
                    t"\ngpu_prim={gpu_prim}"
                )

                var prim = cpu_hit.prim
                if gpu_has_hit:
                    prim = gpu_prim

                if prim != EMPTY_LANE:
                    var sphere = spheres[Int(prim)]
                    var ray = rays[ray_idx]
                    var oc = ray.o - sphere.center
                    var a = dot(ray.d, ray.d)
                    var half_b = dot(oc, ray.d)
                    var c = dot(oc, oc) - sphere.radius * sphere.radius
                    var det = half_b * half_b - a * c

                    print(
                        t"    sphere center={sphere.center} "
                        t"radius={sphere.radius} "
                        t"det={det} a={a}"
                    )

                printed += 1

        else:
            if cpu_has_hit and gpu_has_hit:
                var t_diff = abs(Float64(cpu_hit.t) - Float64(gpu_t))

                if t_diff > T_DIFF_EPS:
                    both_hit_diff_t += 1

    print(
        t"debug hit/miss mismatches: {mismatch_count} | "
        t"gpu_only={gpu_only} cpu_only={cpu_only} "
        t"both_hit_t_diff_gt_{T_DIFF_EPS}={both_hit_diff_t}"
    )


def _print_cpu_ref_header():
    var c0 = String("case").ascii_ljust(22)
    var c1 = String("trace").ascii_rjust(8)
    var c2 = String("hits").ascii_rjust(8)
    var c3 = String("checksum").ascii_rjust(12)

    print(t"{c0} {c1} {c2} {c3}")
    print("---------------------- -------- -------- ------------")


def _print_cpu_ref_row(
    label: String,
    traversal_ns: Int,
    hit_count: UInt32,
    checksum: Float64,
):
    var trace_ms = round(ns_to_ms(traversal_ns), 3)
    var checksum_r = round(checksum, 3)

    var c0 = label.ascii_ljust(22)
    var c1 = String(t"{trace_ms}").ascii_rjust(8)
    var c2 = String(t"{hit_count}").ascii_rjust(8)
    var c3 = String(t"{checksum_r}").ascii_rjust(12)

    print(t"{c0} {c1} {c2} {c3}")


def _print_cpu_triangle_reference[
    width: SIMDSize
](
    label: String,
    vertices: List[Vec3f32],
    rays: List[Ray],
) -> Tuple[
    Float64, UInt32
]:
    var bvh = TriangleBvh[width].__init__["lbvh"](vertices.copy())
    var t0 = perf_counter_ns()
    var result = _trace_cpu_triangle_bvh[width](bvh, rays)
    var t1 = perf_counter_ns()

    _print_cpu_ref_row(label, Int(t1 - t0), result[1], result[0])
    return result


def _print_cpu_sphere_reference[
    width: SIMDSize
](
    label: String,
    spheres: List[Sphere],
    rays: List[Ray],
) -> Tuple[
    Float64, UInt32
]:
    var bvh = SphereBvh[width].__init__["lbvh"](spheres.copy())
    var t0 = perf_counter_ns()
    var result = _trace_cpu_sphere_bvh[width](bvh, rays)
    var t1 = perf_counter_ns()

    _print_cpu_ref_row(label, Int(t1 - t0), result[1], result[0])
    return result


def _print_cpu_sphere_bruteforce_reference(
    label: String,
    spheres: List[Sphere],
    rays: List[Ray],
) -> Tuple[Float64, UInt32]:
    var t0 = perf_counter_ns()
    var result = _trace_cpu_sphere_bruteforce(spheres, rays)
    var t1 = perf_counter_ns()

    _print_cpu_ref_row(label, Int(t1 - t0), result[1], result[0])
    return result


def _print_gpu_results_transposed(
    row0: GpuBenchResult,
    row1: GpuBenchResult,
    row2: GpuBenchResult,
):
    var value_width = 15

    print_transposed_header(
        value_width,
        row0.label,
        row1.label,
        row2.label,
    )
    _print_gpu_result_trace_rows(row0, row1, row2, value_width)
    _print_gpu_result_validation_rows(row0, row1, row2, value_width)


def _bench_camera_primary_triangle[
    width: SIMDSize
](
    ctx: DeviceContext,
    mut bvh: GpuTriangleBvh[width],
    d_camera_params: DeviceBuffer[DType.float32],
    d_hits_f32: DeviceBuffer[DType.float32],
    d_hits_u32: DeviceBuffer[DType.uint32],
    ray_count: Int,
    image_width: Int,
    image_height: Int,
    reference_checksum: Float64,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32, Float64]:
    bvh.launch_camera(
        ctx,
        d_camera_params,
        d_hits_f32,
        d_hits_u32,
        ray_count,
        image_width,
        image_height,
    )
    ctx.synchronize()

    var best_kernel_ns = Int.MAX
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        bvh.launch_camera(
            ctx,
            d_camera_params,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            image_width,
            image_height,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_kernel_ns = min(best_kernel_ns, Int(t1 - t0))

        var downloaded = _download_full_hit_checksum(ctx, d_hits_f32, ray_count)
        checksum = downloaded[0]
        hit_count = downloaded[1]

    return (
        best_kernel_ns,
        checksum,
        hit_count,
        abs(checksum - reference_checksum),
    )


def _run_width[
    width: SIMDSize
](
    mut ctx: DeviceContext,
    d_vertices: DeviceBuffer[DType.float32],
    d_camera_params: DeviceBuffer[DType.float32],
    ray_count: Int,
    image_width: Int,
    image_height: Int,
    reference_checksum: Float64,
    reference_hit_count: UInt32,
    repeats: Int,
) raises -> GpuBenchResult:
    _ = GpuTriangleBvh[width](ctx, d_vertices)
    ctx.synchronize()

    var build0 = perf_counter_ns()
    var bvh = GpuTriangleBvh[width](ctx, d_vertices)
    ctx.synchronize()
    var build1 = perf_counter_ns()

    var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
    var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count)

    var res = _bench_camera_primary_triangle[width](
        ctx,
        bvh,
        d_camera_params,
        d_hits_f32,
        d_hits_u32,
        ray_count,
        image_width,
        image_height,
        reference_checksum,
        repeats,
    )

    return GpuBenchResult(
        String(t"tri{Int(width)}"),
        Int(build1 - build0),
        bvh.timings,
        res[0],
        ray_count,
        res[1],
        res[3],
        res[2],
        reference_checksum,
        reference_hit_count,
        TRIANGLE_HIT_REL_EPS,
    )


def _make_sphere_grid_sized(grid_x: Int, grid_y: Int) -> List[Sphere]:
    var spheres = List[Sphere](capacity=grid_x * grid_y)

    for y in range(grid_y):
        for x in range(grid_x):
            var fx = Float32(x) - Float32(grid_x) * 0.5
            var fy = Float32(y) - Float32(grid_y) * 0.5
            var z = Float32(4 + ((x + y) % 8))
            spheres.append(Sphere(Vec3f32(fx * 2.5, fy * 2.5, z), 0.75))

    return spheres^


def _make_sphere_grid() -> List[Sphere]:
    return _make_sphere_grid_sized(SPHERE_GRID_X, SPHERE_GRID_Y)


def _bench_camera_primary_sphere[
    width: SIMDSize
](
    ctx: DeviceContext,
    mut bvh: GpuSphereBvh[width],
    d_camera_params: DeviceBuffer[DType.float32],
    d_hits_f32: DeviceBuffer[DType.float32],
    d_hits_u32: DeviceBuffer[DType.uint32],
    ray_count: Int,
    image_width: Int,
    image_height: Int,
    reference_checksum: Float64,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32, Float64]:
    bvh.launch_camera(
        ctx,
        d_camera_params,
        d_hits_f32,
        d_hits_u32,
        ray_count,
        image_width,
        image_height,
    )
    ctx.synchronize()

    var best_kernel_ns = Int.MAX
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        bvh.launch_camera(
            ctx,
            d_camera_params,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            image_width,
            image_height,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_kernel_ns = min(best_kernel_ns, Int(t1 - t0))

        var downloaded = _download_full_hit_checksum(ctx, d_hits_f32, ray_count)
        checksum = downloaded[0]
        hit_count = downloaded[1]

    return (
        best_kernel_ns,
        checksum,
        hit_count,
        abs(checksum - reference_checksum),
    )


def _run_sphere_width[
    width: SIMDSize
](
    mut ctx: DeviceContext,
    spheres: List[Sphere],
    rays: List[Ray],
    d_camera_params: DeviceBuffer[DType.float32],
    image_width: Int,
    image_height: Int,
    reference_checksum: Float64,
    reference_hit_count: UInt32,
    repeats: Int,
    print_mismatches: Bool,
) raises -> GpuBenchResult:
    _ = GpuSphereBvh[width](ctx, spheres)
    ctx.synchronize()

    var build0 = perf_counter_ns()
    var bvh = GpuSphereBvh[width](ctx, spheres)
    ctx.synchronize()
    var build1 = perf_counter_ns()

    var ray_count = len(rays)
    var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
    var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count)

    var res = _bench_camera_primary_sphere[width](
        ctx,
        bvh,
        d_camera_params,
        d_hits_f32,
        d_hits_u32,
        ray_count,
        image_width,
        image_height,
        reference_checksum,
        repeats,
    )

    if print_mismatches:
        _print_sphere_debug_mismatches(
            ctx,
            spheres,
            rays,
            d_hits_f32,
            d_hits_u32,
            image_width,
            image_height,
            16,
        )

    return GpuBenchResult(
        String(t"sph{Int(width)}"),
        Int(build1 - build0),
        bvh.timings,
        res[0],
        ray_count,
        res[1],
        res[3],
        res[2],
        reference_checksum,
        reference_hit_count,
        SPHERE_HIT_REL_EPS,
    )


def main() raises:
    print("GPU BoundsBvh benchmark")
    print("")
    print("Run configuration")
    print(t"OBJ path : {DEFAULT_OBJ_PATH}")
    print(
        t"triangle camera rays : {PRIMARY_WIDTH} x {PRIMARY_HEIGHT} x"
        t" {PRIMARY_VIEWS}"
    )
    print(t"repeats : {BENCH_REPEATS}")

    print("\nLoading + packing OBJ...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles(DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var bounds = compute_bounds(tri_vertices)
    var tri_count = len(tri_vertices) / 3
    print(t"triangles: {tri_count}")
    print(t"load+pack ms: {round(ns_to_ms(Int(load_t1 - load_t0)), 3)}")
    print("Bounds min:", round(bounds._min, 3))
    print("Bounds max:", round(bounds._max, 3))

    print("\nGenerating CPU reference camera rays...")
    var camera = _make_camera_rays_and_params(
        bounds,
        PRIMARY_WIDTH,
        PRIMARY_HEIGHT,
        PRIMARY_VIEWS,
        FOV_SCALE,
    )
    var rays = camera[0].copy()
    var camera_params = camera[1].copy()
    print(t"rays : {len(rays)}")

    print("\nGPU TriangleBvh[width]")
    print("----------------------")
    print("\nCPU reference")
    _print_cpu_ref_header()
    var reference = _print_cpu_triangle_reference[8](
        String("TriangleBvh[8] lbvh"),
        tri_vertices,
        rays,
    )
    var reference_checksum = reference[0]
    var reference_hit_count = reference[1]
    print("")

    comptime if not has_accelerator():
        raise "No compatible GPU found; skipped Mojo GPU BoundsBvh benchmark."

    with DeviceContext() as ctx:
        var d_vertices = upload_vertices(ctx, tri_vertices)
        var d_camera_params = upload_list(ctx, camera_params)
        ctx.synchronize()

        var tri2 = _run_width[2](
            ctx,
            d_vertices,
            d_camera_params,
            len(rays),
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            reference_checksum,
            reference_hit_count,
            BENCH_REPEATS,
        )
        var tri4 = _run_width[4](
            ctx,
            d_vertices,
            d_camera_params,
            len(rays),
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            reference_checksum,
            reference_hit_count,
            BENCH_REPEATS,
        )
        var tri8 = _run_width[8](
            ctx,
            d_vertices,
            d_camera_params,
            len(rays),
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            reference_checksum,
            reference_hit_count,
            BENCH_REPEATS,
        )

        _print_gpu_results_transposed(tri2, tri4, tri8)

        print("\nGPU SphereBvh debug: small brute-force validation")
        print("------------------------------------------------")
        var debug_spheres = _make_sphere_grid_sized(
            DEBUG_SPHERE_GRID_X,
            DEBUG_SPHERE_GRID_Y,
        )
        var debug_sphere_bounds = sphere_bounds(debug_spheres)
        var debug_sphere_camera = _make_camera_rays_and_params(
            debug_sphere_bounds,
            DEBUG_SPHERE_RAY_WIDTH,
            DEBUG_SPHERE_RAY_HEIGHT,
            DEBUG_SPHERE_RAY_VIEWS,
            FOV_SCALE,
        )
        var debug_sphere_rays = debug_sphere_camera[0].copy()
        var debug_sphere_camera_params = debug_sphere_camera[1].copy()
        var d_debug_sphere_camera_params = upload_list(
            ctx,
            debug_sphere_camera_params,
        )
        ctx.synchronize()

        print(t"debug spheres : {len(debug_spheres)}")
        print(t"debug rays : {len(debug_sphere_rays)}")

        print("\nCPU debug sphere reference")
        _print_cpu_ref_header()

        # var debug_brute = _print_cpu_sphere_bruteforce_reference(
        #     String("Sphere brute force"),
        #     debug_spheres,
        #     debug_sphere_rays,
        # )
        _ = _print_cpu_sphere_reference[2](
            String("SphereBvh[2] lbvh"),
            debug_spheres,
            debug_sphere_rays,
        )
        _ = _print_cpu_sphere_reference[4](
            String("SphereBvh[4] lbvh"),
            debug_spheres,
            debug_sphere_rays,
        )
        _ = _print_cpu_sphere_reference[8](
            String("SphereBvh[8] lbvh"),
            debug_spheres,
            debug_sphere_rays,
        )

        # print("\nGPU debug sphere vs CPU brute force")
        # var debug_sph2 = _run_sphere_width[2](
        #     ctx,
        #     debug_spheres,
        #     debug_sphere_rays,
        #     d_debug_sphere_camera_params,
        #     DEBUG_SPHERE_RAY_WIDTH,
        #     DEBUG_SPHERE_RAY_HEIGHT,
        #     debug_brute[0],
        #     debug_brute[1],
        #     DEBUG_BENCH_REPEATS,
        #     False,
        # )
        # var debug_sph4 = _run_sphere_width[4](
        #     ctx,
        #     debug_spheres,
        #     debug_sphere_rays,
        #     d_debug_sphere_camera_params,
        #     DEBUG_SPHERE_RAY_WIDTH,
        #     DEBUG_SPHERE_RAY_HEIGHT,
        #     debug_brute[0],
        #     debug_brute[1],
        #     DEBUG_BENCH_REPEATS,
        #     False,
        # )
        # var debug_sph8 = _run_sphere_width[8](
        #     ctx,
        #     debug_spheres,
        #     debug_sphere_rays,
        #     d_debug_sphere_camera_params,
        #     DEBUG_SPHERE_RAY_WIDTH,
        #     DEBUG_SPHERE_RAY_HEIGHT,
        #     debug_brute[0],
        #     debug_brute[1],
        #     DEBUG_BENCH_REPEATS,
        #     True,
        # )
        # _print_gpu_results_transposed(debug_sph2, debug_sph4, debug_sph8)

        print("\nGPU SphereBvh[width]")
        print("--------------------")
        var spheres = _make_sphere_grid()
        var sphere_bounds = sphere_bounds(spheres)
        var sphere_camera = _make_camera_rays_and_params(
            sphere_bounds,
            SPHERE_RAY_WIDTH,
            SPHERE_RAY_HEIGHT,
            SPHERE_RAY_VIEWS,
            FOV_SCALE,
        )
        var sphere_rays = sphere_camera[0].copy()
        var sphere_camera_params = sphere_camera[1].copy()
        var d_sphere_camera_params = upload_list(
            ctx,
            sphere_camera_params,
        )
        ctx.synchronize()

        print(t"spheres : {len(spheres)}")
        print(t"sphere rays : {len(sphere_rays)}")

        print("\nCPU sphere reference")
        _print_cpu_ref_header()

        var sphere_reference2 = _print_cpu_sphere_reference[2](
            String("SphereBvh[2] lbvh"),
            spheres,
            sphere_rays,
        )
        var sphere_reference4 = _print_cpu_sphere_reference[4](
            String("SphereBvh[4] lbvh"),
            spheres,
            sphere_rays,
        )
        var sphere_reference8 = _print_cpu_sphere_reference[8](
            String("SphereBvh[8] lbvh"),
            spheres,
            sphere_rays,
        )

        print("\n")
        var sph2 = _run_sphere_width[2](
            ctx,
            spheres,
            sphere_rays,
            d_sphere_camera_params,
            SPHERE_RAY_WIDTH,
            SPHERE_RAY_HEIGHT,
            sphere_reference2[0],
            sphere_reference2[1],
            BENCH_REPEATS,
            False,
        )
        var sph4 = _run_sphere_width[4](
            ctx,
            spheres,
            sphere_rays,
            d_sphere_camera_params,
            SPHERE_RAY_WIDTH,
            SPHERE_RAY_HEIGHT,
            sphere_reference4[0],
            sphere_reference4[1],
            BENCH_REPEATS,
            False,
        )
        var sph8 = _run_sphere_width[8](
            ctx,
            spheres,
            sphere_rays,
            d_sphere_camera_params,
            SPHERE_RAY_WIDTH,
            SPHERE_RAY_HEIGHT,
            sphere_reference8[0],
            sphere_reference8[1],
            BENCH_REPEATS,
            False,
        )
        _print_gpu_results_transposed(sph2, sph4, sph8)
