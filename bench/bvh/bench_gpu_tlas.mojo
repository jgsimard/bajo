from std.benchmark import keep
from std.math import abs, round, min, max
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
from bajo.core.mat import Mat44f32, _translation, _inv_translation
from bajo.core.bvh.types import Ray, Sphere, Instance
from bajo.core.bvh.host_utils import (
    compute_bounds,
    generate_primary_rays,
    flatten_rays,
    hit_t_for_checksum,
)
from bajo.core.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.core.bvh.cpu.tlas import Tlas


from bajo.core.bvh.gpu.tlas import GpuTlas
from bajo.core.bvh.gpu.sphere_bvh import GpuSphereBvh
from bajo.core.bvh.gpu.triangle_bvh import GpuTriangleBvh
from bajo.core.bvh.gpu.utils import _upload_rays, _download_full_hit_checksum


comptime DEFAULT_OBJ_PATH = "./assets/bunny/bunny.obj"
comptime PRIMARY_WIDTH = 640
comptime PRIMARY_HEIGHT = 360
comptime PRIMARY_VIEWS = 3
comptime BENCH_REPEATS = 8
comptime MISS_PRIM = UInt32(0xFFFFFFFF)


def _make_single_instance(bounds: AABB) -> List[Instance]:
    return [Instance(Mat44f32.identity(), Mat44f32.identity(), 0, bounds)]


def _make_translated_grid_instances(
    bounds: AABB, count_x: Int, count_y: Int
) -> List[Instance]:
    var out = List[Instance](capacity=count_x * count_y)

    var extent = bounds._max - bounds._min
    var spacing_x = max(extent.x * 2.25, 0.25)
    var spacing_y = max(extent.y * 2.25, 0.25)

    for y in range(count_y):
        for x in range(count_x):
            var tx = (Float32(x) - Float32(count_x - 1) * 0.5) * spacing_x
            var ty = (Float32(y) - Float32(count_y - 1) * 0.5) * spacing_y
            var tz = Float32(0.0)

            out.append(
                Instance(
                    _translation(tx, ty, tz),
                    _inv_translation(tx, ty, tz),
                    0,
                    bounds,
                )
            )

    return out^


def _instances_bounds(instances: List[Instance]) -> AABB:
    var out = AABB.invalid()
    for i in range(len(instances)):
        out.grow(instances[i].bounds)
    return out


def _make_cpu_instances(instances: List[Instance]) -> List[Instance]:
    var out = List[Instance](capacity=len(instances))
    for i in range(len(instances)):
        var inst = Instance()
        inst.transform = Mat44f32.identity()
        inst.inv_transform = instances[i].inv_transform.copy()
        inst.bounds = instances[i].bounds
        inst.blas_idx = instances[i].blas_idx
        out.append(inst^)
    return out^


def _cpu_tlas_triangle_reference[
    tlas_width: Int, blas_width: Int
](
    cpu_blas: TriangleBvh[blas_width],
    instances: List[Instance],
    rays: List[Ray],
) -> Tuple[Float64, UInt32, UInt64]:
    var cpu_instances = _make_cpu_instances(instances)
    var tlas = Tlas[tlas_width](cpu_instances)

    var blases = List[TriangleBvh[blas_width]](capacity=1)
    blases.append(cpu_blas.copy())

    var checksum = Float64(0.0)
    var hits = UInt32(0)
    var inst_checksum = UInt64(0)

    for i in range(len(rays)):
        var ray = rays[i].copy()
        var hit = tlas.traverse_triangles[blas_width](
            ray,
            blases.unsafe_ptr(),
        )
        checksum += hit_t_for_checksum(hit.t)
        if hit.t < Float32(1.0e20):
            hits += 1
            inst_checksum += UInt64(hit.inst)

    return (checksum, hits, inst_checksum)


def _cpu_tlas_triangle_shadow_reference[
    tlas_width: Int,
    blas_width: Int,
](
    cpu_blas: TriangleBvh[blas_width],
    instances: List[Instance],
    rays: List[Ray],
) -> UInt32:
    var cpu_instances = _make_cpu_instances(instances)
    var tlas = Tlas[tlas_width](cpu_instances)

    var blases = List[TriangleBvh[blas_width]](capacity=1)
    blases.append(cpu_blas.copy())

    var occluded = UInt32(0)

    for i in range(len(rays)):
        var ray = rays[i].copy()
        if tlas.is_occluded_triangles[blas_width](ray, blases.unsafe_ptr()):
            occluded += 1

    return occluded


def _download_tlas_hit_checksum(
    hits_f32: DeviceBuffer[DType.float32],
    hits_u32: DeviceBuffer[DType.uint32],
    ray_count: Int,
) raises -> Tuple[Float64, UInt32, UInt64]:
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)
    var inst_checksum = UInt64(0)

    with hits_f32.map_to_host() as hf:
        for i in range(ray_count):
            var t = hf[i * 3 + 0]
            checksum += hit_t_for_checksum(t)
            if t < Float32(1.0e20):
                hit_count += 1

    with hits_u32.map_to_host() as hu:
        for i in range(ray_count):
            var inst = UInt32(hu[i * 2 + 1])
            if inst != MISS_PRIM:
                inst_checksum += UInt64(inst)

    return (checksum, hit_count, inst_checksum)


def _download_shadow_count(
    flags: DeviceBuffer[DType.uint32], ray_count: Int
) raises -> UInt32:
    var out = UInt32(0)
    with flags.map_to_host() as f:
        for i in range(ray_count):
            if UInt32(f[i]) != 0:
                out += 1
    return out


def _bench_direct_triangle_primary[
    width: Int,
](
    ctx: DeviceContext,
    blas: GpuTriangleBvh[width],
    rays_flat: List[Float32],
    ray_count: Int,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32]:
    var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
    var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
    var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count)

    _upload_rays(ctx, d_rays, rays_flat)
    blas.launch_uploaded_primary(ctx, d_rays, d_hits_f32, d_hits_u32, ray_count)
    ctx.synchronize()

    var best_ns = Int.MAX
    var checksum = Float64(0.0)
    var hits = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        blas.launch_uploaded_primary(
            ctx, d_rays, d_hits_f32, d_hits_u32, ray_count
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_ns = min(best_ns, Int(t1 - t0))

        var downloaded = _download_full_hit_checksum(ctx, d_hits_f32, ray_count)
        checksum = downloaded[0]
        hits = downloaded[1]

    return (best_ns, checksum, hits)


def _bench_direct_triangle_shadow[
    width: Int,
](
    ctx: DeviceContext,
    blas: GpuTriangleBvh[width],
    rays_flat: List[Float32],
    ray_count: Int,
    repeats: Int,
) raises -> Tuple[Int, UInt32]:
    var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
    var d_flags = ctx.enqueue_create_buffer[DType.uint32](ray_count)

    _upload_rays(ctx, d_rays, rays_flat)
    blas.launch_uploaded_shadow(ctx, d_rays, d_flags, ray_count)
    ctx.synchronize()

    var best_ns = Int.MAX
    var occluded = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        blas.launch_uploaded_shadow(ctx, d_rays, d_flags, ray_count)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_ns = min(best_ns, Int(t1 - t0))
        occluded = _download_shadow_count(d_flags, ray_count)

    return (best_ns, occluded)


def _bench_tlas_triangles_primary[
    tlas_width: Int,
    blas_width: Int,
](
    ctx: DeviceContext,
    tlas: GpuTlas[tlas_width],
    blas: GpuTriangleBvh[blas_width],
    rays_flat: List[Float32],
    ray_count: Int,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32, UInt64]:
    var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
    var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
    var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count * 2)

    _upload_rays(ctx, d_rays, rays_flat)
    tlas.launch_uploaded_triangle_primary[blas_width](
        ctx,
        blas,
        d_rays,
        d_hits_f32,
        d_hits_u32,
        ray_count,
    )
    ctx.synchronize()

    var best_ns = Int.MAX
    var checksum = Float64(0.0)
    var hits = UInt32(0)
    var inst_checksum = UInt64(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        tlas.launch_uploaded_triangle_primary[blas_width](
            ctx,
            blas,
            d_rays,
            d_hits_f32,
            d_hits_u32,
            ray_count,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_ns = min(best_ns, Int(t1 - t0))

        var downloaded = _download_tlas_hit_checksum(
            d_hits_f32, d_hits_u32, ray_count
        )
        checksum = downloaded[0]
        hits = downloaded[1]
        inst_checksum = downloaded[2]

    return (best_ns, checksum, hits, inst_checksum)


def _bench_tlas_triangles_shadow[
    tlas_width: Int,
    blas_width: Int,
](
    ctx: DeviceContext,
    tlas: GpuTlas[tlas_width],
    blas: GpuTriangleBvh[blas_width],
    rays_flat: List[Float32],
    ray_count: Int,
    repeats: Int,
) raises -> Tuple[Int, UInt32]:
    var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
    var d_flags = ctx.enqueue_create_buffer[DType.uint32](ray_count)

    _upload_rays(ctx, d_rays, rays_flat)
    tlas.launch_uploaded_triangle_shadow[blas_width](
        ctx, blas, d_rays, d_flags, ray_count
    )
    ctx.synchronize()

    var best_ns = Int.MAX
    var occluded = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        tlas.launch_uploaded_triangle_shadow[blas_width](
            ctx, blas, d_rays, d_flags, ray_count
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_ns = min(best_ns, Int(t1 - t0))
        occluded = _download_shadow_count(d_flags, ray_count)

    return (best_ns, occluded)


def _bench_direct_sphere_primary[
    width: Int,
](
    ctx: DeviceContext,
    blas: GpuSphereBvh[width],
    rays_flat: List[Float32],
    ray_count: Int,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32]:
    var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
    var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
    var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count)

    _upload_rays(ctx, d_rays, rays_flat)
    blas.launch_uploaded_primary(ctx, d_rays, d_hits_f32, d_hits_u32, ray_count)
    ctx.synchronize()

    var best_ns = Int.MAX
    var checksum = Float64(0.0)
    var hits = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        blas.launch_uploaded_primary(
            ctx, d_rays, d_hits_f32, d_hits_u32, ray_count
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_ns = min(best_ns, Int(t1 - t0))

        var downloaded = _download_full_hit_checksum(ctx, d_hits_f32, ray_count)
        checksum = downloaded[0]
        hits = downloaded[1]

    return (best_ns, checksum, hits)


def _bench_direct_sphere_shadow[
    width: Int,
](
    ctx: DeviceContext,
    blas: GpuSphereBvh[width],
    rays_flat: List[Float32],
    ray_count: Int,
    repeats: Int,
) raises -> Tuple[Int, UInt32]:
    var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
    var d_flags = ctx.enqueue_create_buffer[DType.uint32](ray_count)

    _upload_rays(ctx, d_rays, rays_flat)
    blas.launch_uploaded_shadow(ctx, d_rays, d_flags, ray_count)
    ctx.synchronize()

    var best_ns = Int.MAX
    var occluded = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        blas.launch_uploaded_shadow(ctx, d_rays, d_flags, ray_count)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_ns = min(best_ns, Int(t1 - t0))
        occluded = _download_shadow_count(d_flags, ray_count)

    return (best_ns, occluded)


def _bench_tlas_spheres_primary[
    tlas_width: Int,
    blas_width: Int,
](
    ctx: DeviceContext,
    tlas: GpuTlas[tlas_width],
    blas: GpuSphereBvh[blas_width],
    rays_flat: List[Float32],
    ray_count: Int,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32, UInt64]:
    var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
    var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
    var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count * 2)

    _upload_rays(ctx, d_rays, rays_flat)
    tlas.launch_uploaded_sphere_primary[blas_width](
        ctx,
        blas,
        d_rays,
        d_hits_f32,
        d_hits_u32,
        ray_count,
    )
    ctx.synchronize()

    var best_ns = Int.MAX
    var checksum = Float64(0.0)
    var hits = UInt32(0)
    var inst_checksum = UInt64(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        tlas.launch_uploaded_sphere_primary[blas_width](
            ctx, blas, d_rays, d_hits_f32, d_hits_u32, ray_count
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_ns = min(best_ns, Int(t1 - t0))

        var downloaded = _download_tlas_hit_checksum(
            d_hits_f32, d_hits_u32, ray_count
        )
        checksum = downloaded[0]
        hits = downloaded[1]
        inst_checksum = downloaded[2]

    return (best_ns, checksum, hits, inst_checksum)


def _bench_tlas_spheres_shadow[
    tlas_width: Int,
    blas_width: Int,
](
    ctx: DeviceContext,
    tlas: GpuTlas[tlas_width],
    blas: GpuSphereBvh[blas_width],
    rays_flat: List[Float32],
    ray_count: Int,
    repeats: Int,
) raises -> Tuple[Int, UInt32]:
    var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
    var d_flags = ctx.enqueue_create_buffer[DType.uint32](ray_count)

    _upload_rays(ctx, d_rays, rays_flat)
    tlas.launch_uploaded_sphere_shadow[blas_width](
        ctx, blas, d_rays, d_flags, ray_count
    )
    ctx.synchronize()

    var best_ns = Int.MAX
    var occluded = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        tlas.launch_uploaded_sphere_shadow[blas_width](
            ctx, blas, d_rays, d_flags, ray_count
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_ns = min(best_ns, Int(t1 - t0))
        occluded = _download_shadow_count(d_flags, ray_count)

    return (best_ns, occluded)


def _print_blas_build_table_header():
    print("| case | build ms | collapse ms | pack ms |")


def _print_blas_build_row(
    label: String,
    build_ns: Int,
    collapse_ns: Int,
    pack_ns: Int,
):
    print(
        t"| {label} | {round(ns_to_ms(build_ns), 3)} | "
        t"{round(ns_to_ms(collapse_ns), 3)} | "
        t"{round(ns_to_ms(pack_ns), 3)} |"
    )


def _print_direct_table_header():
    print(
        "| case | primary ms | P MRays/s | hits | checksum | shadow ms | S"
        " MRays/s | occluded |"
    )


def _print_direct_row(
    label: String,
    primary_ns: Int,
    primary_checksum: Float64,
    primary_hits: UInt32,
    shadow_ns: Int,
    shadow_occluded: UInt32,
    ray_count: Int,
):
    print(
        t"| {label} | {round(ns_to_ms(primary_ns), 3)} | "
        t"{round(ns_to_mrays_per_s(primary_ns, ray_count), 3)} | "
        t"{primary_hits} | {round(primary_checksum, 3)} | "
        t"{round(ns_to_ms(shadow_ns), 3)} | "
        t"{round(ns_to_mrays_per_s(shadow_ns, ray_count), 3)} | "
        t"{shadow_occluded} |"
    )


def _print_cpu_reference_table_header():
    print("| case | inst | hits | checksum | shadow hits | inst checksum |")


def _print_cpu_reference_row(
    label: String,
    inst_count: Int,
    checksum: Float64,
    hits: UInt32,
    shadow_hits: UInt32,
    inst_checksum: UInt64,
):
    print(
        t"| {label} | {inst_count} | {hits} | {round(checksum, 3)} | "
        t"{shadow_hits} | {inst_checksum} |"
    )


def _print_tlas_table_header(has_reference: Bool):
    if has_reference:
        print(
            "| case | inst | build ms | collapse ms | primary ms | P MRays/s |"
            " hits | checksum | Δsum | Δhits | inst sum | Δinst | shadow ms | S"
            " MRays/s | occluded | Δocc | status |"
        )
    else:
        print(
            "| case | inst | build ms | collapse ms | primary ms | P MRays/s |"
            " hits | checksum | inst sum | shadow ms | S MRays/s | occluded |"
        )


def _print_tlas_row(
    label: String,
    inst_count: Int,
    build_ns: Int,
    collapse_ns: Int,
    primary_ns: Int,
    shadow_ns: Int,
    ray_count: Int,
    checksum: Float64,
    hits: UInt32,
    inst_checksum: UInt64,
    occluded: UInt32,
    has_reference: Bool,
    reference_checksum: Float64,
    reference_hits: UInt32,
    reference_inst_checksum: UInt64,
    reference_occluded: UInt32,
):
    var build_ms = round(ns_to_ms(build_ns), 3)
    var collapse_ms = round(ns_to_ms(collapse_ns), 3)
    var primary_ms = round(ns_to_ms(primary_ns), 3)
    var shadow_ms = round(ns_to_ms(shadow_ns), 3)
    var primary_mrays = round(ns_to_mrays_per_s(primary_ns, ray_count), 3)
    var shadow_mrays = round(ns_to_mrays_per_s(shadow_ns, ray_count), 3)
    var checksum_r = round(checksum, 3)

    if has_reference:
        var diff = round(abs(checksum - reference_checksum), 6)
        var hit_diff = Int(hits) - Int(reference_hits)
        var inst_diff = Int(inst_checksum) - Int(reference_inst_checksum)
        var occ_diff = Int(occluded) - Int(reference_occluded)

        if (
            diff <= 0.000001
            and hit_diff == 0
            and inst_diff == 0
            and occ_diff == 0
        ):
            print(
                t"| {label} | {inst_count} | {build_ms} | {collapse_ms} | "
                t"{primary_ms} | {primary_mrays} | {hits} | {checksum_r} | "
                t"{diff} | {hit_diff} | {inst_checksum} | {inst_diff} | "
                t"{shadow_ms} | {shadow_mrays} | {occluded} | {occ_diff} | OK |"
            )
        else:
            print(
                t"| {label} | {inst_count} | {build_ms} | {collapse_ms} |"
                t" {primary_ms} | {primary_mrays} | {hits} | {checksum_r} |"
                t" {diff} | {hit_diff} | {inst_checksum} | {inst_diff} |"
                t" {shadow_ms} | {shadow_mrays} | {occluded} | {occ_diff} |"
                t" CHECK |"
            )
    else:
        print(
            t"| {label} | {inst_count} | {build_ms} | {collapse_ms} | "
            t"{primary_ms} | {primary_mrays} | {hits} | {checksum_r} | "
            t"{inst_checksum} | {shadow_ms} | {shadow_mrays} | {occluded} |"
        )


def _run_triangle_tlas_width[
    tlas_width: Int,
    blas_width: Int,
](
    mut ctx: DeviceContext,
    blas: GpuTriangleBvh[blas_width],
    instances: List[Instance],
    rays_flat: List[Float32],
    ray_count: Int,
    repeats: Int,
    label: String,
    has_reference: Bool,
    reference_checksum: Float64,
    reference_hits: UInt32,
    reference_inst_checksum: UInt64,
    reference_shadow: UInt32,
) raises:
    _ = GpuTlas[tlas_width](ctx, instances)
    ctx.synchronize()

    var b0 = perf_counter_ns()
    var tlas = GpuTlas[tlas_width](ctx, instances)
    ctx.synchronize()
    var b1 = perf_counter_ns()

    var primary = _bench_tlas_triangles_primary[tlas_width, blas_width](
        ctx,
        tlas,
        blas,
        rays_flat,
        ray_count,
        repeats,
    )
    var shadow = _bench_tlas_triangles_shadow[tlas_width, blas_width](
        ctx,
        tlas,
        blas,
        rays_flat,
        ray_count,
        repeats,
    )

    _print_tlas_row(
        label,
        len(instances),
        Int(b1 - b0),
        tlas.tree.collapse_ns,
        primary[0],
        shadow[0],
        ray_count,
        primary[1],
        primary[2],
        primary[3],
        shadow[1],
        has_reference,
        reference_checksum,
        reference_hits,
        reference_inst_checksum,
        reference_shadow,
    )

    keep(tlas.tree.leaf_block_count)
    keep(primary[2])
    keep(shadow[1])


def _run_sphere_tlas_width[
    tlas_width: Int,
    blas_width: Int,
](
    mut ctx: DeviceContext,
    blas: GpuSphereBvh[blas_width],
    instances: List[Instance],
    rays_flat: List[Float32],
    ray_count: Int,
    repeats: Int,
    label: String,
    has_reference: Bool,
    reference_checksum: Float64,
    reference_hits: UInt32,
    reference_shadow: UInt32,
) raises:
    _ = GpuTlas[tlas_width](ctx, instances)
    ctx.synchronize()

    var b0 = perf_counter_ns()
    var tlas = GpuTlas[tlas_width](ctx, instances)
    ctx.synchronize()
    var b1 = perf_counter_ns()

    var primary = _bench_tlas_spheres_primary[tlas_width, blas_width](
        ctx,
        tlas,
        blas,
        rays_flat,
        ray_count,
        repeats,
    )
    var shadow = _bench_tlas_spheres_shadow[tlas_width, blas_width](
        ctx,
        tlas,
        blas,
        rays_flat,
        ray_count,
        repeats,
    )

    _print_tlas_row(
        label,
        len(instances),
        Int(b1 - b0),
        tlas.tree.collapse_ns,
        primary[0],
        shadow[0],
        ray_count,
        primary[1],
        primary[2],
        primary[3],
        shadow[1],
        has_reference,
        reference_checksum,
        reference_hits,
        UInt64(0),
        reference_shadow,
    )

    keep(tlas.tree.leaf_block_count)
    keep(primary[2])
    keep(shadow[1])


def _make_sphere_scene(count_x: Int, count_y: Int) -> Tuple[List[Sphere], AABB]:
    var spheres = List[Sphere](capacity=count_x * count_y)
    var bounds = AABB.invalid()

    for y in range(count_y):
        for x in range(count_x):
            var cx = (Float32(x) - Float32(count_x - 1) * 0.5) * 3.0
            var cy = (Float32(y) - Float32(count_y - 1) * 0.5) * 3.0
            var cz = 8.0 + Float32((x + y) % 5) * 0.5
            var r = Float32(0.8)
            spheres.append(Sphere(Vec3f32(cx, cy, cz), r))
            bounds.grow(Vec3f32(cx - r, cy - r, cz - r))
            bounds.grow(Vec3f32(cx + r, cy + r, cz + r))

    return (spheres^, bounds)


def _bench_triangle_instance_set[
    side: Int
](
    mut ctx: DeviceContext,
    blas: GpuTriangleBvh[4],
    cpu_blas: TriangleBvh[8],
    blas_bounds: AABB,
) raises:
    var instances = _make_translated_grid_instances(blas_bounds, side, side)
    var bounds = _instances_bounds(instances)
    var rays = generate_primary_rays(
        bounds, PRIMARY_WIDTH, PRIMARY_HEIGHT, PRIMARY_VIEWS
    )
    var rays_flat = flatten_rays(rays)
    var has_ref = side == 4

    var ref_primary = (Float64(0.0), UInt32(0), UInt64(0))
    var ref_shadow = UInt32(0)
    if has_ref:
        ref_primary = _cpu_tlas_triangle_reference[2, 8](
            cpu_blas, instances, rays
        )
        ref_shadow = _cpu_tlas_triangle_shadow_reference[2, 8](
            cpu_blas, instances, rays
        )
        print("\nCPU reference")
        _print_cpu_reference_table_header()
        _print_cpu_reference_row(
            "CPU Tlas[2]/TriangleBvh[8]",
            len(instances),
            ref_primary[0],
            ref_primary[1],
            ref_shadow,
            ref_primary[2],
        )

    print(t"\nTriangle TLAS translated grid {side}x{side}")
    print("--------------------------------")
    print(t"Instances: {len(instances)}")
    print(t"Rays: {len(rays)}")
    print_vec3_rounded("TLAS bounds min:", bounds._min)
    print_vec3_rounded("TLAS bounds max:", bounds._max)
    print("\nTLAS results")
    _print_tlas_table_header(has_ref)

    _run_triangle_tlas_width[2, 4](
        ctx,
        blas,
        instances,
        rays_flat,
        len(rays),
        BENCH_REPEATS,
        String(t"tlas2/blas4 grid {side}x{side}"),
        has_ref,
        ref_primary[0],
        ref_primary[1],
        ref_primary[2],
        ref_shadow,
    )
    _run_triangle_tlas_width[4, 4](
        ctx,
        blas,
        instances,
        rays_flat,
        len(rays),
        BENCH_REPEATS,
        String(t"tlas4/blas4 grid {side}x{side}"),
        has_ref,
        ref_primary[0],
        ref_primary[1],
        ref_primary[2],
        ref_shadow,
    )
    _run_triangle_tlas_width[8, 4](
        ctx,
        blas,
        instances,
        rays_flat,
        len(rays),
        BENCH_REPEATS,
        String(t"tlas8/blas4 grid {side}x{side}"),
        has_ref,
        ref_primary[0],
        ref_primary[1],
        ref_primary[2],
        ref_shadow,
    )


def main() raises:
    print("GPU TLAS benchmark")
    print(t"Path: {DEFAULT_OBJ_PATH}")
    print(t"Image rays: {PRIMARY_WIDTH} x {PRIMARY_HEIGHT} x {PRIMARY_VIEWS}")
    print(t"Repeats: {BENCH_REPEATS}")

    print("\nLoading + packing OBJ...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles(DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var blas_bounds = compute_bounds(tri_vertices)
    print(t"Triangles: {len(tri_vertices) / 3}")
    print(t"Load+pack ms: {round(ns_to_ms(Int(load_t1 - load_t0)), 3)}")
    print_vec3_rounded("BLAS bounds min:", blas_bounds._min)
    print_vec3_rounded("BLAS bounds max:", blas_bounds._max)

    comptime if has_accelerator():
        with DeviceContext() as ctx:
            print("\nBuilding GpuTriangleBvh[4]...")
            _ = GpuTriangleBvh[4](ctx, tri_vertices)
            ctx.synchronize()

            var blas_b0 = perf_counter_ns()
            var blas = GpuTriangleBvh[4](ctx, tri_vertices)
            ctx.synchronize()
            var blas_b1 = perf_counter_ns()

            _print_blas_build_table_header()
            _print_blas_build_row(
                "GpuTriangleBvh[4]",
                Int(blas_b1 - blas_b0),
                blas.tree.collapse_ns,
                blas.leaf_pack_ns,
            )

            print("Building CPU TriangleBvh[8] LBVH reference for 4x4 grid...")
            var cpu_blas = TriangleBvh[8].__init__["lbvh"](
                tri_vertices.unsafe_ptr(),
                UInt32(len(tri_vertices) / 3),
            )

            print("\nSingle instance triangle TLAS")
            print("-----------------------------")
            var single_instances = _make_single_instance(blas_bounds)
            var single_bounds = _instances_bounds(single_instances)
            var single_rays = generate_primary_rays(
                single_bounds,
                PRIMARY_WIDTH,
                PRIMARY_HEIGHT,
                PRIMARY_VIEWS,
            )
            var single_rays_flat = flatten_rays(single_rays)
            print(t"Rays: {len(single_rays)}")

            var direct = _bench_direct_triangle_primary[4](
                ctx,
                blas,
                single_rays_flat,
                len(single_rays),
                BENCH_REPEATS,
            )
            var direct_shadow = _bench_direct_triangle_shadow[4](
                ctx,
                blas,
                single_rays_flat,
                len(single_rays),
                BENCH_REPEATS,
            )
            print("\nDirect BLAS")
            _print_direct_table_header()
            _print_direct_row(
                "direct triangle BLAS",
                direct[0],
                direct[1],
                direct[2],
                direct_shadow[0],
                direct_shadow[1],
                len(single_rays),
            )

            print("\nTLAS results")
            _print_tlas_table_header(True)

            _run_triangle_tlas_width[2, 4](
                ctx,
                blas,
                single_instances,
                single_rays_flat,
                len(single_rays),
                BENCH_REPEATS,
                "tlas2/blas4 single",
                True,
                direct[1],
                direct[2],
                UInt64(0),
                direct_shadow[1],
            )
            _run_triangle_tlas_width[4, 4](
                ctx,
                blas,
                single_instances,
                single_rays_flat,
                len(single_rays),
                BENCH_REPEATS,
                "tlas4/blas4 single",
                True,
                direct[1],
                direct[2],
                UInt64(0),
                direct_shadow[1],
            )
            _run_triangle_tlas_width[8, 4](
                ctx,
                blas,
                single_instances,
                single_rays_flat,
                len(single_rays),
                BENCH_REPEATS,
                "tlas8/blas4 single",
                True,
                direct[1],
                direct[2],
                UInt64(0),
                direct_shadow[1],
            )

            _bench_triangle_instance_set[4](ctx, blas, cpu_blas, blas_bounds)
            _bench_triangle_instance_set[8](ctx, blas, cpu_blas, blas_bounds)
            _bench_triangle_instance_set[16](ctx, blas, cpu_blas, blas_bounds)

            print("\nSphere TLAS")
            print("-----------")
            var sphere_scene = _make_sphere_scene(32, 32)
            var spheres = sphere_scene[0].copy()
            var sphere_bounds = sphere_scene[1].copy()
            print(t"Spheres: {len(spheres)}")
            print_vec3_rounded("Sphere BLAS bounds min:", sphere_bounds._min)
            print_vec3_rounded("Sphere BLAS bounds max:", sphere_bounds._max)

            print("Building GpuSphereBvh[4]...")
            _ = GpuSphereBvh[4](ctx, spheres)
            ctx.synchronize()

            var sph_b0 = perf_counter_ns()
            var sphere_blas = GpuSphereBvh[4](ctx, spheres)
            ctx.synchronize()
            var sph_b1 = perf_counter_ns()
            _print_blas_build_table_header()
            _print_blas_build_row(
                "GpuSphereBvh[4]",
                Int(sph_b1 - sph_b0),
                sphere_blas.tree.collapse_ns,
                sphere_blas.leaf_pack_ns,
            )

            var sphere_single = _make_single_instance(sphere_bounds)
            var sphere_rays = generate_primary_rays(
                sphere_bounds, PRIMARY_WIDTH, PRIMARY_HEIGHT, PRIMARY_VIEWS
            )
            var sphere_rays_flat = flatten_rays(sphere_rays)
            var direct_sphere = _bench_direct_sphere_primary[4](
                ctx,
                sphere_blas,
                sphere_rays_flat,
                len(sphere_rays),
                BENCH_REPEATS,
            )
            var direct_sphere_shadow = _bench_direct_sphere_shadow[4](
                ctx,
                sphere_blas,
                sphere_rays_flat,
                len(sphere_rays),
                BENCH_REPEATS,
            )
            print("\nDirect sphere BLAS")
            _print_direct_table_header()
            _print_direct_row(
                "direct sphere BLAS",
                direct_sphere[0],
                direct_sphere[1],
                direct_sphere[2],
                direct_sphere_shadow[0],
                direct_sphere_shadow[1],
                len(sphere_rays),
            )

            print("\nSphere single-instance TLAS")
            _print_tlas_table_header(True)

            _run_sphere_tlas_width[2, 4](
                ctx,
                sphere_blas,
                sphere_single,
                sphere_rays_flat,
                len(sphere_rays),
                BENCH_REPEATS,
                "sph tlas2/blas4 single",
                True,
                direct_sphere[1],
                direct_sphere[2],
                direct_sphere_shadow[1],
            )
            _run_sphere_tlas_width[4, 4](
                ctx,
                sphere_blas,
                sphere_single,
                sphere_rays_flat,
                len(sphere_rays),
                BENCH_REPEATS,
                "sph tlas4/blas4 single",
                True,
                direct_sphere[1],
                direct_sphere[2],
                direct_sphere_shadow[1],
            )
            _run_sphere_tlas_width[8, 4](
                ctx,
                sphere_blas,
                sphere_single,
                sphere_rays_flat,
                len(sphere_rays),
                BENCH_REPEATS,
                "sph tlas8/blas4 single",
                True,
                direct_sphere[1],
                direct_sphere[2],
                direct_sphere_shadow[1],
            )

            var sphere_grid = _make_translated_grid_instances(
                sphere_bounds, 4, 4
            )
            var sphere_grid_bounds = _instances_bounds(sphere_grid)
            var sphere_grid_rays = generate_primary_rays(
                sphere_grid_bounds,
                PRIMARY_WIDTH,
                PRIMARY_HEIGHT,
                PRIMARY_VIEWS,
            )
            var sphere_grid_rays_flat = flatten_rays(sphere_grid_rays)
            print("\nSphere TLAS translated grid 4x4")
            print(t"Instances: {len(sphere_grid)}")
            print(t"Rays: {len(sphere_grid_rays)}")
            print_vec3_rounded("TLAS bounds min:", sphere_grid_bounds._min)
            print_vec3_rounded("TLAS bounds max:", sphere_grid_bounds._max)
            _print_tlas_table_header(False)
            _run_sphere_tlas_width[2, 4](
                ctx,
                sphere_blas,
                sphere_grid,
                sphere_grid_rays_flat,
                len(sphere_grid_rays),
                BENCH_REPEATS,
                "sph tlas2/blas4 grid 4x4",
                False,
                0.0,
                UInt32(0),
                UInt32(0),
            )
            _run_sphere_tlas_width[4, 4](
                ctx,
                sphere_blas,
                sphere_grid,
                sphere_grid_rays_flat,
                len(sphere_grid_rays),
                BENCH_REPEATS,
                "sph tlas4/blas4 grid 4x4",
                False,
                0.0,
                UInt32(0),
                UInt32(0),
            )
            _run_sphere_tlas_width[8, 4](
                ctx,
                sphere_blas,
                sphere_grid,
                sphere_grid_rays_flat,
                len(sphere_grid_rays),
                BENCH_REPEATS,
                "sph tlas8/blas4 grid 4x4",
                False,
                0.0,
                UInt32(0),
                UInt32(0),
            )
    else:
        print("No compatible GPU found; skipped Mojo GPU TLAS benchmark.")
