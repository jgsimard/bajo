from std.benchmark import keep
from std.math import abs, round, min
from std.time import perf_counter_ns
from std.gpu import DeviceContext, DeviceBuffer

from bajo.core.utils import (
    ns_to_ms,
    ns_to_mrays_per_s,
)
from bajo.core.aabb import AABB
from bajo.core.vec import Vec3f32
from bajo.bvh.host_utils import (
    compute_bounds,
    generate_primary_rays,
    flatten_rays,
    hit_t_for_checksum,
)
from bajo.bvh.constants import TRACE
from bajo.bvh.types import Ray, Sphere
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.gpu.sphere_bvh import GpuSphereBvh
from bajo.bvh.gpu.triangle_bvh import GpuTriangleBvh
from bajo.bvh.gpu.utils import _upload_rays, _download_full_hit_checksum
from bajo.obj.pack import pack_obj_triangles


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


def _trace_cpu_triangle_bvh[
    width: Int
](mut bvh: TriangleBvh[width], rays: List[Ray]) -> Float64:
    var checksum = Float64(0.0)
    for ray in rays:
        var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)
        checksum += hit_t_for_checksum(hit.t)
    return checksum


def _trace_cpu_sphere_bvh[
    width: Int
](mut bvh: SphereBvh[width], rays: List[Ray]) -> Float64:
    var checksum = Float64(0.0)
    for ray in rays:
        var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)
        checksum += hit_t_for_checksum(hit.t)
    return checksum


def _print_cpu_ref_header():
    var c0 = String("case").ascii_ljust(22)
    var c1 = String("trace").ascii_rjust(8)
    var c2 = String("checksum").ascii_rjust(12)

    print(t"{c0} {c1} {c2}")
    print("---------------------- -------- ------------")


def _print_cpu_ref_row(
    label: String,
    traversal_ns: Int,
    checksum: Float64,
):
    var trace_ms = round(ns_to_ms(traversal_ns), 3)
    var checksum_r = round(checksum, 3)

    var c0 = label.ascii_ljust(22)
    var c1 = String(t"{trace_ms}").ascii_rjust(8)
    var c2 = String(t"{checksum_r}").ascii_rjust(12)

    print(t"{c0} {c1} {c2}")


def _print_gpu_table_header(has_reference: Bool):
    var c0 = String("case").ascii_ljust(8)
    var c1 = String("build").ascii_rjust(8)
    var c2 = String("collapse").ascii_rjust(10)
    var c3 = String("pack").ascii_rjust(8)
    var c4 = String("trace").ascii_rjust(8)
    var c5 = String("MRay/s").ascii_rjust(9)
    var c6 = String("hits").ascii_rjust(8)
    var c7 = String("checksum").ascii_rjust(12)

    if has_reference:
        var c8 = String("diff").ascii_rjust(10)
        var c9 = String("status").ascii_ljust(6)

        print(t"{c0} {c1} {c2} {c3} {c4} {c5} {c6} {c7} {c8} {c9}")
        print(
            "-------- -------- ---------- -------- -------- --------- --------"
            " ------------ ---------- ------"
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
    collapse_ns: Int,
    pack_ns: Int,
    kernel_ns: Int,
    ray_count: Int,
    checksum: Float64,
    diff: Float64,
    hit_count: UInt32,
):
    comptime CHECKSUM_ABS_EPS = 1.0e-3
    comptime CHECKSUM_PER_HIT_EPS = 1.0e-6

    var build_ms = round(ns_to_ms(build_ns), 3)
    var collapse_ms = round(ns_to_ms(collapse_ns), 3)
    var pack_ms = round(ns_to_ms(pack_ns), 3)
    var kernel_ms = round(ns_to_ms(kernel_ns), 3)
    var mrays = round(ns_to_mrays_per_s(kernel_ns, ray_count), 3)
    var checksum_r = round(checksum, 3)
    var diff_r = round(diff, 6)

    var per_hit_diff = Float64(0.0)
    if hit_count > 0:
        per_hit_diff = diff / Float64(hit_count)

    var status = String("OK")
    if diff > CHECKSUM_ABS_EPS and per_hit_diff > CHECKSUM_PER_HIT_EPS:
        status = String("CHECK")

    var c0 = label.ascii_ljust(8)
    var c1 = String(t"{build_ms}").ascii_rjust(8)
    var c2 = String(t"{collapse_ms}").ascii_rjust(10)
    var c3 = String(t"{pack_ms}").ascii_rjust(8)
    var c4 = String(t"{kernel_ms}").ascii_rjust(8)
    var c5 = String(t"{mrays}").ascii_rjust(9)
    var c6 = String(t"{hit_count}").ascii_rjust(8)
    var c7 = String(t"{checksum_r}").ascii_rjust(12)
    var c8 = String(t"{diff_r}").ascii_rjust(10)
    var c9 = status.ascii_ljust(6)

    print(t"{c0} {c1} {c2} {c3} {c4} {c5} {c6} {c7} {c8} {c9}")


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

    var c0 = label.ascii_ljust(8)
    var c1 = String(t"{build_ms}").ascii_rjust(8)
    var c2 = String(t"{collapse_ms}").ascii_rjust(10)
    var c3 = String(t"{pack_ms}").ascii_rjust(8)
    var c4 = String(t"{kernel_ms}").ascii_rjust(8)
    var c5 = String(t"{mrays}").ascii_rjust(9)
    var c6 = String(t"{hit_count}").ascii_rjust(8)
    var c7 = String(t"{checksum_r}").ascii_rjust(12)

    print(t"{c0} {c1} {c2} {c3} {c4} {c5} {c6} {c7}")


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
    bvh.launch_uploaded(ctx, d_rays, d_hits_f32, d_hits_u32, ray_count)
    ctx.synchronize()

    var best_kernel_ns = Int.MAX
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        bvh.launch_uploaded(ctx, d_rays, d_hits_f32, d_hits_u32, ray_count)
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
    for s in spheres:
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
    reference_checksum: Float64,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32, Float64]:
    _upload_rays(ctx, d_rays, rays_flat)
    bvh.launch_uploaded(ctx, d_rays, d_hits_f32, d_hits_u32, ray_count)
    ctx.synchronize()

    var best_kernel_ns = Int.MAX
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        bvh.launch_uploaded(ctx, d_rays, d_hits_f32, d_hits_u32, ray_count)
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
    rays_flat: List[Float32],
    ray_count: Int,
    reference_checksum: Float64,
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
        reference_checksum,
        repeats,
    )

    _print_gpu_row[width](
        String(t"sph{width}"),
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
    print("Bounds min:", round(bounds._min, 3))
    print("Bounds max:", round(bounds._max, 3))

    print("\nGenerating rays...")
    var rays = generate_primary_rays(
        bounds,
        PRIMARY_WIDTH,
        PRIMARY_HEIGHT,
        PRIMARY_VIEWS,
        FOV_SCALE,
    )
    var rays_flat = flatten_rays(rays)
    print(t"rays : {len(rays)}")

    print("\nGPU TriangleBvh[width]")
    print("----------------------")
    print("\nCPU reference")
    var cpu_bvh = TriangleBvh[8].__init__["lbvh"](
        tri_vertices.unsafe_ptr().unsafe_mut_cast[True](),
        UInt32(len(tri_vertices) / 3),
    )
    var cpu_t0 = perf_counter_ns()
    var reference_checksum = _trace_cpu_triangle_bvh[8](cpu_bvh, rays)
    var cpu_t1 = perf_counter_ns()
    _print_cpu_ref_header()
    _print_cpu_ref_row(
        String("TriangleBvh[8] lbvh"),
        Int(cpu_t1 - cpu_t0),
        reference_checksum,
    )
    print("")

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
        var sphere_bounds = _sphere_bounds(spheres)
        var sphere_rays = generate_primary_rays(
            sphere_bounds,
            SPHERE_RAY_WIDTH,
            SPHERE_RAY_HEIGHT,
            SPHERE_RAY_VIEWS,
            FOV_SCALE,
        )
        var sphere_rays_flat = flatten_rays(sphere_rays)
        print(t"spheres : {len(spheres)}")
        print(t"sphere rays : {len(sphere_rays)}")

        print("\nCPU sphere reference")
        var cpu_sphere_bvh = SphereBvh[8].__init__["lbvh"](
            spheres.unsafe_ptr().unsafe_mut_cast[True](),
            UInt32(len(spheres)),
        )
        var cpu_sphere_t0 = perf_counter_ns()
        var sphere_reference_checksum = _trace_cpu_sphere_bvh[8](
            cpu_sphere_bvh,
            sphere_rays,
        )
        var cpu_sphere_t1 = perf_counter_ns()
        _print_cpu_ref_header()
        _print_cpu_ref_row(
            String("SphereBvh[8] lbvh"),
            Int(cpu_sphere_t1 - cpu_sphere_t0),
            sphere_reference_checksum,
        )

        print("")
        _print_gpu_table_header(True)

        _run_sphere_width[2](
            ctx,
            spheres,
            sphere_rays_flat,
            len(sphere_rays),
            sphere_reference_checksum,
            BENCH_REPEATS,
        )
        _run_sphere_width[4](
            ctx,
            spheres,
            sphere_rays_flat,
            len(sphere_rays),
            sphere_reference_checksum,
            BENCH_REPEATS,
        )
        _run_sphere_width[8](
            ctx,
            spheres,
            sphere_rays_flat,
            len(sphere_rays),
            sphere_reference_checksum,
            BENCH_REPEATS,
        )
