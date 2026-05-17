from std.benchmark import keep
from std.math import abs, round, min
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu import DeviceContext, DeviceBuffer

from bajo.core.utils import (
    ns_to_ms,
    ns_to_mrays_per_s,
    pack_obj_triangles,
    print_vec3_rounded,
)
from bajo.core.aabb import AABB
from bajo.core.vec import Vec3f32
from bajo.core.bvh.host_utils import (
    compute_bounds,
    generate_primary_rays,
    flatten_rays,
    hit_t_for_checksum,
)
from bajo.core.bvh.types import Ray, Sphere
from bajo.core.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.core.bvh.gpu.sphere_bvh import GpuSphereBvh
from bajo.core.bvh.gpu.triangle_bvh import GpuTriangleBvh
from bajo.core.bvh.gpu.utils import _upload_rays, _download_full_hit_checksum


comptime DEFAULT_OBJ_PATH = "./assets/bunny/bunny.obj"
comptime PRIMARY_WIDTH = 640
comptime PRIMARY_HEIGHT = 360
comptime PRIMARY_VIEWS = 3
comptime BENCH_REPEATS = 8
comptime SPHERE_GRID_X = 64
comptime SPHERE_GRID_Y = 64
comptime SPHERE_RAY_WIDTH = 128
comptime SPHERE_RAY_HEIGHT = 64
comptime SPHERE_RAY_VIEWS = 2


def _trace_cpu_triangle_bvh[
    width: Int
](mut bvh: TriangleBvh[width], rays: List[Ray]) -> Float64:
    var checksum = Float64(0.0)
    for i in range(len(rays)):
        var ray = rays[i].copy()
        bvh.traverse(ray)
        checksum += hit_t_for_checksum(ray.hit.t)
    return checksum


@always_inline
def _stage_total_without_static(build_ns: Int, traversal_ns: Int) -> Int:
    return build_ns + traversal_ns


def _bench_uploaded_primary[
    width: Int
](
    ctx: DeviceContext,
    mut bvh: GpuTriangleBvh[width],
    d_rays: DeviceBuffer[DType.float32],
    d_hits_f32: DeviceBuffer[DType.float32],
    d_hits_u32: DeviceBuffer[DType.uint32],
    rays_flat: List[Float32],
    ray_count: Int,
    reference_checksum: Float64,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32, Float64]:
    _upload_rays(ctx, d_rays, rays_flat)
    bvh.launch_uploaded_primary(ctx, d_rays, d_hits_f32, d_hits_u32, ray_count)
    ctx.synchronize()

    var best_kernel_ns = Int.MAX
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        bvh.launch_uploaded_primary(
            ctx, d_rays, d_hits_f32, d_hits_u32, ray_count
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


def _print_gpu_table_header(has_reference: Bool):
    if has_reference:
        print(
            "| case | build ms | collapse ms | pack ms | trace ms | "
            "MRays/s | hits | checksum | diff | status |"
        )
    else:
        print(
            "| case | build ms | collapse ms | pack ms | trace ms | "
            "MRays/s | hits | checksum |"
        )


def _print_gpu_row[
    width: Int
](
    label: String,
    build_ns: Int,
    collapse_ns: Int,
    pack_ns: Int,
    kernel_ns: Int,
    ray_count: Int,
    checksum: Float64,
    diff: Float64,
    hit_count: UInt32,
):
    var build_ms = round(ns_to_ms(build_ns), 3)
    var collapse_ms = round(ns_to_ms(collapse_ns), 3)
    var pack_ms = round(ns_to_ms(pack_ns), 3)
    var kernel_ms = round(ns_to_ms(kernel_ns), 3)
    var mrays = round(ns_to_mrays_per_s(kernel_ns, ray_count), 3)
    var checksum_r = round(checksum, 3)
    var diff_r = round(diff, 6)

    var status = String("OK")
    if diff > 1.0e-5:
        status = String("CHECK")

    print(
        t"| {label} | {build_ms} | {collapse_ms} | {pack_ms} | "
        t"{kernel_ms} | {mrays} | {hit_count} | {checksum_r} | "
        t"{diff_r} | {status} |"
    )


def _print_gpu_row_no_ref[
    width: Int
](
    label: String,
    build_ns: Int,
    collapse_ns: Int,
    pack_ns: Int,
    kernel_ns: Int,
    ray_count: Int,
    checksum: Float64,
    hit_count: UInt32,
):
    var build_ms = round(ns_to_ms(build_ns), 3)
    var collapse_ms = round(ns_to_ms(collapse_ns), 3)
    var pack_ms = round(ns_to_ms(pack_ns), 3)
    var kernel_ms = round(ns_to_ms(kernel_ns), 3)
    var mrays = round(ns_to_mrays_per_s(kernel_ns, ray_count), 3)
    var checksum_r = round(checksum, 3)

    print(
        t"| {label} | {build_ms} | {collapse_ms} | {pack_ms} | "
        t"{kernel_ms} | {mrays} | {hit_count} | {checksum_r} |"
    )


def _run_width[
    width: Int
](
    mut ctx: DeviceContext,
    tri_vertices: List[Vec3f32],
    rays_flat: List[Float32],
    ray_count: Int,
    reference_checksum: Float64,
    repeats: Int,
) raises:
    _ = GpuTriangleBvh[width](ctx, tri_vertices)
    ctx.synchronize()

    var build0 = perf_counter_ns()
    var bvh = GpuTriangleBvh[width](ctx, tri_vertices)
    ctx.synchronize()
    var build1 = perf_counter_ns()

    var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
    var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
    var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count)

    var res = _bench_uploaded_primary[width](
        ctx,
        bvh,
        d_rays,
        d_hits_f32,
        d_hits_u32,
        rays_flat,
        ray_count,
        reference_checksum,
        repeats,
    )

    _print_gpu_row[width](
        String(t"tri{width}"),
        Int(build1 - build0),
        bvh.tree.collapse_ns,
        bvh.leaf_pack_ns,
        res[0],
        ray_count,
        res[1],
        res[3],
        res[2],
    )

    keep(bvh.tree.leaf_block_count)
    keep(res[2])


def _make_sphere_grid() -> List[Sphere]:
    var spheres = List[Sphere](capacity=SPHERE_GRID_X * SPHERE_GRID_Y)

    for y in range(SPHERE_GRID_Y):
        for x in range(SPHERE_GRID_X):
            var fx = Float32(x) - Float32(SPHERE_GRID_X) * 0.5
            var fy = Float32(y) - Float32(SPHERE_GRID_Y) * 0.5
            var z = Float32(4 + ((x + y) % 8))
            spheres.append(Sphere(Vec3f32(fx * 2.5, fy * 2.5, z), 0.75))

    return spheres^


def _sphere_bounds(spheres: List[Sphere]) -> AABB:
    var bounds = AABB.invalid()
    for i in range(len(spheres)):
        ref s = spheres[i]
        var r = s.radius
        bounds.grow(Vec3f32(s.center.x - r, s.center.y - r, s.center.z - r))
        bounds.grow(Vec3f32(s.center.x + r, s.center.y + r, s.center.z + r))
    return bounds


def _bench_uploaded_primary_sphere[
    width: Int
](
    ctx: DeviceContext,
    mut bvh: GpuSphereBvh[width],
    d_rays: DeviceBuffer[DType.float32],
    d_hits_f32: DeviceBuffer[DType.float32],
    d_hits_u32: DeviceBuffer[DType.uint32],
    rays_flat: List[Float32],
    ray_count: Int,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32]:
    _upload_rays(ctx, d_rays, rays_flat)
    bvh.launch_uploaded_primary(ctx, d_rays, d_hits_f32, d_hits_u32, ray_count)
    ctx.synchronize()

    var best_kernel_ns = Int.MAX
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        bvh.launch_uploaded_primary(
            ctx, d_rays, d_hits_f32, d_hits_u32, ray_count
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_kernel_ns = min(best_kernel_ns, Int(t1 - t0))

        var downloaded = _download_full_hit_checksum(ctx, d_hits_f32, ray_count)
        checksum = downloaded[0]
        hit_count = downloaded[1]

    return (best_kernel_ns, checksum, hit_count)


def _run_sphere_width[
    width: Int
](
    mut ctx: DeviceContext,
    spheres: List[Sphere],
    rays_flat: List[Float32],
    ray_count: Int,
    repeats: Int,
) raises:
    _ = GpuSphereBvh[width](ctx, spheres)
    ctx.synchronize()

    var build0 = perf_counter_ns()
    var bvh = GpuSphereBvh[width](ctx, spheres)
    ctx.synchronize()
    var build1 = perf_counter_ns()

    var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
    var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
    var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count)

    var res = _bench_uploaded_primary_sphere[width](
        ctx,
        bvh,
        d_rays,
        d_hits_f32,
        d_hits_u32,
        rays_flat,
        ray_count,
        repeats,
    )

    _print_gpu_row_no_ref[width](
        String(t"sph{width}"),
        Int(build1 - build0),
        bvh.tree.collapse_ns,
        bvh.leaf_pack_ns,
        res[0],
        ray_count,
        res[1],
        res[2],
    )

    keep(bvh.tree.leaf_block_count)
    keep(res[2])


def main() raises:
    print("GPU BoundsBvh benchmark")
    print("")
    print("Run configuration")
    print(t"OBJ path : {DEFAULT_OBJ_PATH}")
    print(
        t"triangle image rays : {PRIMARY_WIDTH} x {PRIMARY_HEIGHT} x"
        t" {PRIMARY_VIEWS}"
    )
    print(t"repeats : {BENCH_REPEATS}")

    print("\nLoading + packing OBJ...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles(DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var bounds = compute_bounds(tri_vertices)
    print(t"triangles: {len(tri_vertices) / 3}")
    print(t"load+pack ms: {round(ns_to_ms(Int(load_t1 - load_t0)), 3)}")
    print_vec3_rounded("Bounds min:", bounds._min)
    print_vec3_rounded("Bounds max:", bounds._max)

    print("\nGenerating rays...")
    var rays = generate_primary_rays(
        bounds, PRIMARY_WIDTH, PRIMARY_HEIGHT, PRIMARY_VIEWS
    )
    var rays_flat = flatten_rays(rays)
    print(t"rays : {len(rays)}")

    print("\nCPU reference")
    var cpu_bvh = TriangleBvh[8].__init__["lbvh"](
        tri_vertices.unsafe_ptr(), UInt32(len(tri_vertices) / 3)
    )
    var cpu_t0 = perf_counter_ns()
    var reference_checksum = _trace_cpu_triangle_bvh[8](cpu_bvh, rays)
    var cpu_t1 = perf_counter_ns()
    print("| case | traversal ms | checksum |")
    print(
        t"| TriangleBvh[8] lbvh | "
        t"{round(ns_to_ms(Int(cpu_t1 - cpu_t0)), 3)} | "
        t"{round(reference_checksum, 3)} |"
    )

    print("\nGPU TriangleBvh[width]")
    print("----------------------")
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            _print_gpu_table_header(True)
            _run_width[2](
                ctx,
                tri_vertices,
                rays_flat,
                len(rays),
                reference_checksum,
                BENCH_REPEATS,
            )
            _run_width[4](
                ctx,
                tri_vertices,
                rays_flat,
                len(rays),
                reference_checksum,
                BENCH_REPEATS,
            )
            _run_width[8](
                ctx,
                tri_vertices,
                rays_flat,
                len(rays),
                reference_checksum,
                BENCH_REPEATS,
            )

            print("\nGPU SphereBvh[width]")
            print("--------------------")
            var spheres = _make_sphere_grid()
            var sb = _sphere_bounds(spheres)
            var sphere_rays = generate_primary_rays(
                sb, SPHERE_RAY_WIDTH, SPHERE_RAY_HEIGHT, SPHERE_RAY_VIEWS
            )
            var sphere_rays_flat = flatten_rays(sphere_rays)
            print(t"spheres : {len(spheres)}")
            print(t"sphere rays : {len(sphere_rays)}")
            print("")
            _print_gpu_table_header(False)

            _run_sphere_width[2](
                ctx, spheres, sphere_rays_flat, len(sphere_rays), BENCH_REPEATS
            )
            _run_sphere_width[4](
                ctx, spheres, sphere_rays_flat, len(sphere_rays), BENCH_REPEATS
            )
            _run_sphere_width[8](
                ctx, spheres, sphere_rays_flat, len(sphere_rays), BENCH_REPEATS
            )
    else:
        print("No compatible GPU found; skipped Mojo GPU benchmark.")
