from std.benchmark import keep
from std.math import abs, round, min, max
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu import DeviceContext, DeviceBuffer

from bajo.core.utils import (
    ns_to_ms,
    ns_to_mrays_per_s,
)
from bajo.core.aabb import AABB
from bajo.core.vec import Vec3, Vec3f32, dot
from bajo.bvh.camera import Camera, CAMERA_STRIDE
from bajo.bvh.host_utils import (
    compute_bounds,
    hit_t_for_checksum,
)
from bajo.bvh.gpu.utils import GpuBuildTimings, _download_full_hit_checksum
from bajo.bvh.constants import EMPTY_LANE, TRACE
from bajo.bvh.types import Hit, Ray, Sphere
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.gpu.sphere_bvh import GpuSphereBvh
from bajo.bvh.gpu.triangle_bvh import GpuTriangleBvh
from bajo.core.intersect import intersect_ray_sphere
from bajo.obj.pack import pack_obj_triangles
from bajo.bvh.gpu.utils import _upload_list, _upload_vertices


comptime DEFAULT_OBJ_PATH = "./assets/bunny/bunny.obj"
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
    var params = List[Float32](capacity=views * CAMERA_STRIDE)

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

        var flat = camera.flatten()
        for i in range(len(flat)):
            params.append(flat[i])

        for py in range(height):
            for px in range(width):
                rays.append(camera.make_ray(px, py, width, height))

    return (rays^, params^)


def _trace_cpu_triangle_bvh[
    width: Int
](mut bvh: TriangleBvh[width], rays: List[Ray]) -> Tuple[Float64, UInt32]:
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for ray in rays:
        var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)
        checksum += hit_t_for_checksum(hit.t)
        if hit.t < Float32(1.0e20):
            hit_count += 1

    return (checksum, hit_count)


def _trace_cpu_sphere_bvh[
    width: Int
](mut bvh: SphereBvh[width], rays: List[Ray]) -> Tuple[Float64, UInt32]:
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for ray in rays:
        var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)
        checksum += hit_t_for_checksum(hit.t)
        if hit.t < Float32(1.0e20):
            hit_count += 1

    return (checksum, hit_count)


def _trace_cpu_sphere_bruteforce(
    spheres: List[Sphere],
    rays: List[Ray],
) -> Tuple[Float64, UInt32]:
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for ray in rays:
        var hit = Hit.miss(ray.t_max)

        var O = Vec3[DType.float32, 1](ray.o.x, ray.o.y, ray.o.z)
        var D = Vec3[DType.float32, 1](ray.d.x, ray.d.y, ray.d.z)

        for prim_i in range(len(spheres)):
            var s = spheres[prim_i]
            var center = Vec3[DType.float32, 1](
                s.center.x,
                s.center.y,
                s.center.z,
            )
            var radius = SIMD[DType.float32, 1](s.radius)

            var h = intersect_ray_sphere(
                O,
                D,
                center,
                radius,
                SIMD[DType.float32, 1](hit.t),
                SIMD[DType.float32, 1](ray.t_min),
            )

            if h.mask[0] and h.t[0] < hit.t:
                hit.t = h.t[0]
                hit.u = 0.0
                hit.v = 0.0
                hit.prim = UInt32(prim_i)
                hit.inst = EMPTY_LANE

        checksum += hit_t_for_checksum(hit.t)
        if hit.t < Float32(1.0e20):
            hit_count += 1

    return (checksum, hit_count)


def _trace_cpu_sphere_bruteforce_one(
    spheres: List[Sphere],
    ray: Ray,
) -> Hit:
    var hit = Hit.miss(ray.t_max)

    var O = Vec3[DType.float32, 1](ray.o.x, ray.o.y, ray.o.z)
    var D = Vec3[DType.float32, 1](ray.d.x, ray.d.y, ray.d.z)

    for prim_i in range(len(spheres)):
        var s = spheres[prim_i]
        var center = Vec3[DType.float32, 1](
            s.center.x,
            s.center.y,
            s.center.z,
        )
        var radius = SIMD[DType.float32, 1](s.radius)

        var h = intersect_ray_sphere(
            O,
            D,
            center,
            radius,
            SIMD[DType.float32, 1](hit.t),
            SIMD[DType.float32, 1](ray.t_min),
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
                    t"hit/miss ray {ray_idx} "
                    t"view={view_idx} px={px_i} py={py_i}: "
                    t"cpu_hit={cpu_has_hit} "
                    t"cpu_t={cpu_hit.t} "
                    t"cpu_prim={cpu_hit.prim} | "
                    t"gpu_hit={gpu_has_hit} "
                    t"gpu_t={gpu_t} "
                    t"gpu_prim={gpu_prim}"
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


def _print_gpu_table_header(has_reference: Bool):
    var c0 = String("case").ascii_ljust(8)
    var c1 = String("build").ascii_rjust(8)
    var c2 = String("collapse").ascii_rjust(10)
    var c3 = String("pack").ascii_rjust(8)
    var c4 = String("camera").ascii_rjust(8)
    var c5 = String("MRay/s").ascii_rjust(9)
    var c6 = String("hits").ascii_rjust(8)
    var c7 = String("checksum").ascii_rjust(12)

    if has_reference:
        var c8 = String("diff").ascii_rjust(15)
        var c9 = String("dhit").ascii_rjust(6)
        var c10 = String("rel_dhit").ascii_rjust(10)
        var c11 = String("status").ascii_ljust(6)

        print(t"{c0} {c1} {c2} {c3} {c4} {c5} {c6} {c7} {c8} {c9} {c10} {c11}")
        print(
            "-------- -------- ---------- -------- -------- --------- --------"
            " ------------ --------------- ------ ---------- ------"
        )
    else:
        print(t"{c0} {c1} {c2} {c3} {c4} {c5} {c6} {c7}")
        print(
            "-------- -------- ---------- -------- -------- --------- --------"
            " ------------"
        )


def _print_gpu_row[
    width: Int
](
    label: String,
    build_ns: Int,
    timings: GpuBuildTimings,
    pack_ns: Int,
    kernel_ns: Int,
    ray_count: Int,
    checksum: Float64,
    diff: Float64,
    hit_count: UInt32,
    reference_hit_count: UInt32,
    hit_rel_eps: Float64,
):
    comptime CHECKSUM_ABS_EPS = 1.0e-3
    comptime CHECKSUM_PER_HIT_EPS = 1.0e-6

    var build_ms = round(ns_to_ms(build_ns), 3)
    var collapse_ms = round(ns_to_ms(timings.collapse_ns), 3)
    var pack_ms = round(ns_to_ms(pack_ns), 3)
    var kernel_ms = round(ns_to_ms(kernel_ns), 3)
    var mrays = round(ns_to_mrays_per_s(kernel_ns, ray_count), 3)
    var checksum_r = round(checksum, 3)
    var diff_r = round(diff, 6)

    var per_hit_diff = Float64(0.0)
    if hit_count > 0:
        per_hit_diff = diff / Float64(hit_count)

    var hit_diff = Int(hit_count) - Int(reference_hit_count)
    var abs_hit_diff = hit_diff
    if abs_hit_diff < 0:
        abs_hit_diff = 0 - abs_hit_diff

    var hit_rel_diff = Float64(0.0)
    if reference_hit_count > 0:
        hit_rel_diff = Float64(abs_hit_diff) / Float64(reference_hit_count)
    else:
        if hit_count > 0:
            hit_rel_diff = 1.0

    var hit_rel_diff_r = round(hit_rel_diff, 6)

    var status = String("OK")

    # Hit/miss disagreement is judged by relative hit-count error.
    # For strict rows, pass hit_rel_eps = 0.0.
    # For sphere rows, pass a small tolerance because silhouette/tangent rays
    # are not bit-stable between CPU and GPU ray reconstruction.
    if hit_rel_diff > hit_rel_eps:
        status = String("CHECK")
    else:
        if hit_diff == 0 and (
            diff > CHECKSUM_ABS_EPS and per_hit_diff > CHECKSUM_PER_HIT_EPS
        ):
            status = String("CHECK")

    var c0 = label.ascii_ljust(8)
    var c1 = String(t"{build_ms}").ascii_rjust(8)
    var c2 = String(t"{collapse_ms}").ascii_rjust(10)
    var c3 = String(t"{pack_ms}").ascii_rjust(8)
    var c4 = String(t"{kernel_ms}").ascii_rjust(8)
    var c5 = String(t"{mrays}").ascii_rjust(9)
    var c6 = String(t"{hit_count}").ascii_rjust(8)
    var c7 = String(t"{checksum_r}").ascii_rjust(12)
    var c8 = String(t"{diff_r}").ascii_rjust(15)
    var c9 = String(t"{hit_diff}").ascii_rjust(6)
    var c10 = String(t"{hit_rel_diff_r}").ascii_rjust(10)
    var c11 = status.ascii_ljust(6)

    print(t"{c0} {c1} {c2} {c3} {c4} {c5} {c6} {c7} {c8} {c9} {c10} {c11}")


def _bench_camera_primary_triangle[
    width: Int
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
    width: Int
](
    mut ctx: DeviceContext,
    d_vertices: DeviceBuffer[DType.float32],
    tri_count: Int,
    d_camera_params: DeviceBuffer[DType.float32],
    ray_count: Int,
    image_width: Int,
    image_height: Int,
    reference_checksum: Float64,
    reference_hit_count: UInt32,
    repeats: Int,
) raises:
    _ = GpuTriangleBvh[width](ctx, d_vertices, tri_count)
    ctx.synchronize()

    var build0 = perf_counter_ns()
    var bvh = GpuTriangleBvh[width](ctx, d_vertices, tri_count)
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

    _print_gpu_row[width](
        String(t"tri{width}"),
        Int(build1 - build0),
        bvh.timings,
        bvh.leaf_pack_ns,
        res[0],
        ray_count,
        res[1],
        res[3],
        res[2],
        reference_hit_count,
        TRIANGLE_HIT_REL_EPS,
    )

    keep(bvh.tree.leaf_block_count)
    keep(res[2])


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


def _sphere_bounds(spheres: List[Sphere]) -> AABB:
    var bounds = AABB.invalid()
    for s in spheres:
        var r = s.radius
        bounds.grow(Vec3f32(s.center.x - r, s.center.y - r, s.center.z - r))
        bounds.grow(Vec3f32(s.center.x + r, s.center.y + r, s.center.z + r))
    return bounds


def _bench_camera_primary_sphere[
    width: Int
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
    width: Int
](
    mut ctx: DeviceContext,
    spheres: List[Sphere],
    d_camera_params: DeviceBuffer[DType.float32],
    ray_count: Int,
    image_width: Int,
    image_height: Int,
    reference_checksum: Float64,
    reference_hit_count: UInt32,
    repeats: Int,
) raises:
    _ = GpuSphereBvh[width](ctx, spheres)
    ctx.synchronize()

    var build0 = perf_counter_ns()
    var bvh = GpuSphereBvh[width](ctx, spheres)
    ctx.synchronize()
    var build1 = perf_counter_ns()

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

    _print_gpu_row[width](
        String(t"sph{width}"),
        Int(build1 - build0),
        bvh.timings,
        bvh.leaf_pack_ns,
        res[0],
        ray_count,
        res[1],
        res[3],
        res[2],
        reference_hit_count,
        SPHERE_HIT_REL_EPS,
    )

    keep(bvh.tree.leaf_block_count)
    keep(res[2])


def _run_sphere_debug_width[
    width: Int
](
    mut ctx: DeviceContext,
    spheres: List[Sphere],
    rays: List[Ray],
    d_camera_params: DeviceBuffer[DType.float32],
    ray_count: Int,
    image_width: Int,
    image_height: Int,
    reference_checksum: Float64,
    reference_hit_count: UInt32,
    repeats: Int,
    print_mismatches: Bool,
) raises:
    _ = GpuSphereBvh[width](ctx, spheres)
    ctx.synchronize()

    var build0 = perf_counter_ns()
    var bvh = GpuSphereBvh[width](ctx, spheres)
    ctx.synchronize()
    var build1 = perf_counter_ns()

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

    _print_gpu_row[width](
        String(t"sph{width}"),
        Int(build1 - build0),
        bvh.timings,
        bvh.leaf_pack_ns,
        res[0],
        ray_count,
        res[1],
        res[3],
        res[2],
        reference_hit_count,
        SPHERE_HIT_REL_EPS,
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

    keep(bvh.tree.leaf_block_count)
    keep(res[2])


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
    var cpu_bvh = TriangleBvh[8].__init__["lbvh"](
        tri_vertices.unsafe_ptr().unsafe_mut_cast[True](),
        UInt32(tri_count),
    )
    var cpu_t0 = perf_counter_ns()
    var reference = _trace_cpu_triangle_bvh[8](cpu_bvh, rays)
    var cpu_t1 = perf_counter_ns()
    var reference_checksum = reference[0]
    var reference_hit_count = reference[1]
    _print_cpu_ref_header()
    _print_cpu_ref_row(
        String("TriangleBvh[8] lbvh"),
        Int(cpu_t1 - cpu_t0),
        reference_hit_count,
        reference_checksum,
    )
    print("")

    comptime if not has_accelerator():
        raise "No compatible GPU found; skipped Mojo GPU BoundsBvh benchmark."

    with DeviceContext() as ctx:
        var d_vertices = _upload_vertices(ctx, tri_vertices)
        var d_camera_params = _upload_list(ctx, camera_params)
        ctx.synchronize()

        _print_gpu_table_header(True)
        _run_width[2](
            ctx,
            d_vertices,
            tri_count,
            d_camera_params,
            len(rays),
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            reference_checksum,
            reference_hit_count,
            BENCH_REPEATS,
        )
        _run_width[4](
            ctx,
            d_vertices,
            tri_count,
            d_camera_params,
            len(rays),
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            reference_checksum,
            reference_hit_count,
            BENCH_REPEATS,
        )
        _run_width[8](
            ctx,
            d_vertices,
            tri_count,
            d_camera_params,
            len(rays),
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            reference_checksum,
            reference_hit_count,
            BENCH_REPEATS,
        )

        print("\nGPU SphereBvh debug: small brute-force validation")
        print("------------------------------------------------")
        var debug_spheres = _make_sphere_grid_sized(
            DEBUG_SPHERE_GRID_X,
            DEBUG_SPHERE_GRID_Y,
        )
        var debug_sphere_bounds = _sphere_bounds(debug_spheres)
        var debug_sphere_camera = _make_camera_rays_and_params(
            debug_sphere_bounds,
            DEBUG_SPHERE_RAY_WIDTH,
            DEBUG_SPHERE_RAY_HEIGHT,
            DEBUG_SPHERE_RAY_VIEWS,
            FOV_SCALE,
        )
        var debug_sphere_rays = debug_sphere_camera[0].copy()
        var debug_sphere_camera_params = debug_sphere_camera[1].copy()
        var d_debug_sphere_camera_params = _upload_list(
            ctx,
            debug_sphere_camera_params,
        )
        ctx.synchronize()

        print(t"debug spheres : {len(debug_spheres)}")
        print(t"debug rays : {len(debug_sphere_rays)}")

        print("\nCPU debug sphere reference")
        _print_cpu_ref_header()

        var debug_brute_t0 = perf_counter_ns()
        var debug_brute = _trace_cpu_sphere_bruteforce(
            debug_spheres,
            debug_sphere_rays,
        )
        var debug_brute_t1 = perf_counter_ns()
        _print_cpu_ref_row(
            String("Sphere brute force"),
            Int(debug_brute_t1 - debug_brute_t0),
            debug_brute[1],
            debug_brute[0],
        )

        var debug_cpu_sphere_bvh2 = SphereBvh[2].__init__["lbvh"](
            debug_spheres.unsafe_ptr().unsafe_mut_cast[True](),
            UInt32(len(debug_spheres)),
        )
        var debug_cpu_sphere_t20 = perf_counter_ns()
        var debug_sphere_reference2 = _trace_cpu_sphere_bvh[2](
            debug_cpu_sphere_bvh2,
            debug_sphere_rays,
        )
        var debug_cpu_sphere_t21 = perf_counter_ns()
        _print_cpu_ref_row(
            String("SphereBvh[2] lbvh"),
            Int(debug_cpu_sphere_t21 - debug_cpu_sphere_t20),
            debug_sphere_reference2[1],
            debug_sphere_reference2[0],
        )

        var debug_cpu_sphere_bvh4 = SphereBvh[4].__init__["lbvh"](
            debug_spheres.unsafe_ptr().unsafe_mut_cast[True](),
            UInt32(len(debug_spheres)),
        )
        var debug_cpu_sphere_t40 = perf_counter_ns()
        var debug_sphere_reference4 = _trace_cpu_sphere_bvh[4](
            debug_cpu_sphere_bvh4,
            debug_sphere_rays,
        )
        var debug_cpu_sphere_t41 = perf_counter_ns()
        _print_cpu_ref_row(
            String("SphereBvh[4] lbvh"),
            Int(debug_cpu_sphere_t41 - debug_cpu_sphere_t40),
            debug_sphere_reference4[1],
            debug_sphere_reference4[0],
        )

        var debug_cpu_sphere_bvh8 = SphereBvh[8].__init__["lbvh"](
            debug_spheres.unsafe_ptr().unsafe_mut_cast[True](),
            UInt32(len(debug_spheres)),
        )
        var debug_cpu_sphere_t80 = perf_counter_ns()
        var debug_sphere_reference8 = _trace_cpu_sphere_bvh[8](
            debug_cpu_sphere_bvh8,
            debug_sphere_rays,
        )
        var debug_cpu_sphere_t81 = perf_counter_ns()
        _print_cpu_ref_row(
            String("SphereBvh[8] lbvh"),
            Int(debug_cpu_sphere_t81 - debug_cpu_sphere_t80),
            debug_sphere_reference8[1],
            debug_sphere_reference8[0],
        )

        print("\nGPU debug sphere vs CPU brute force")
        _print_gpu_table_header(True)
        _run_sphere_debug_width[2](
            ctx,
            debug_spheres,
            debug_sphere_rays,
            d_debug_sphere_camera_params,
            len(debug_sphere_rays),
            DEBUG_SPHERE_RAY_WIDTH,
            DEBUG_SPHERE_RAY_HEIGHT,
            debug_brute[0],
            debug_brute[1],
            DEBUG_BENCH_REPEATS,
            False,
        )
        _run_sphere_debug_width[4](
            ctx,
            debug_spheres,
            debug_sphere_rays,
            d_debug_sphere_camera_params,
            len(debug_sphere_rays),
            DEBUG_SPHERE_RAY_WIDTH,
            DEBUG_SPHERE_RAY_HEIGHT,
            debug_brute[0],
            debug_brute[1],
            DEBUG_BENCH_REPEATS,
            False,
        )
        _run_sphere_debug_width[8](
            ctx,
            debug_spheres,
            debug_sphere_rays,
            d_debug_sphere_camera_params,
            len(debug_sphere_rays),
            DEBUG_SPHERE_RAY_WIDTH,
            DEBUG_SPHERE_RAY_HEIGHT,
            debug_brute[0],
            debug_brute[1],
            DEBUG_BENCH_REPEATS,
            True,
        )

        print("\nGPU SphereBvh[width]")
        print("--------------------")
        var spheres = _make_sphere_grid()
        var sphere_bounds = _sphere_bounds(spheres)
        var sphere_camera = _make_camera_rays_and_params(
            sphere_bounds,
            SPHERE_RAY_WIDTH,
            SPHERE_RAY_HEIGHT,
            SPHERE_RAY_VIEWS,
            FOV_SCALE,
        )
        var sphere_rays = sphere_camera[0].copy()
        var sphere_camera_params = sphere_camera[1].copy()
        var d_sphere_camera_params = _upload_list(
            ctx,
            sphere_camera_params,
        )
        ctx.synchronize()

        print(t"spheres : {len(spheres)}")
        print(t"sphere rays : {len(sphere_rays)}")

        print("\nCPU sphere reference")
        _print_cpu_ref_header()

        var cpu_sphere_bvh2 = SphereBvh[2].__init__["lbvh"](
            spheres.unsafe_ptr().unsafe_mut_cast[True](),
            UInt32(len(spheres)),
        )
        var cpu_sphere_t20 = perf_counter_ns()
        var sphere_reference2 = _trace_cpu_sphere_bvh[2](
            cpu_sphere_bvh2,
            sphere_rays,
        )
        var cpu_sphere_t21 = perf_counter_ns()
        _print_cpu_ref_row(
            String("SphereBvh[2] lbvh"),
            Int(cpu_sphere_t21 - cpu_sphere_t20),
            sphere_reference2[1],
            sphere_reference2[0],
        )

        var cpu_sphere_bvh4 = SphereBvh[4].__init__["lbvh"](
            spheres.unsafe_ptr().unsafe_mut_cast[True](),
            UInt32(len(spheres)),
        )
        var cpu_sphere_t40 = perf_counter_ns()
        var sphere_reference4 = _trace_cpu_sphere_bvh[4](
            cpu_sphere_bvh4,
            sphere_rays,
        )
        var cpu_sphere_t41 = perf_counter_ns()
        _print_cpu_ref_row(
            String("SphereBvh[4] lbvh"),
            Int(cpu_sphere_t41 - cpu_sphere_t40),
            sphere_reference4[1],
            sphere_reference4[0],
        )

        var cpu_sphere_bvh8 = SphereBvh[8].__init__["lbvh"](
            spheres.unsafe_ptr().unsafe_mut_cast[True](),
            UInt32(len(spheres)),
        )
        var cpu_sphere_t80 = perf_counter_ns()
        var sphere_reference8 = _trace_cpu_sphere_bvh[8](
            cpu_sphere_bvh8,
            sphere_rays,
        )
        var cpu_sphere_t81 = perf_counter_ns()
        _print_cpu_ref_row(
            String("SphereBvh[8] lbvh"),
            Int(cpu_sphere_t81 - cpu_sphere_t80),
            sphere_reference8[1],
            sphere_reference8[0],
        )

        print("")
        _print_gpu_table_header(True)

        _run_sphere_width[2](
            ctx,
            spheres,
            d_sphere_camera_params,
            len(sphere_rays),
            SPHERE_RAY_WIDTH,
            SPHERE_RAY_HEIGHT,
            sphere_reference2[0],
            sphere_reference2[1],
            BENCH_REPEATS,
        )
        _run_sphere_width[4](
            ctx,
            spheres,
            d_sphere_camera_params,
            len(sphere_rays),
            SPHERE_RAY_WIDTH,
            SPHERE_RAY_HEIGHT,
            sphere_reference4[0],
            sphere_reference4[1],
            BENCH_REPEATS,
        )
        _run_sphere_width[8](
            ctx,
            spheres,
            d_sphere_camera_params,
            len(sphere_rays),
            SPHERE_RAY_WIDTH,
            SPHERE_RAY_HEIGHT,
            sphere_reference8[0],
            sphere_reference8[1],
            BENCH_REPEATS,
        )
