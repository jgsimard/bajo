from std.benchmark import keep
from std.math import abs, round, min, max
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu import DeviceContext, DeviceBuffer

from bajo.core.utils import (
    ns_to_ms,
    ns_to_mrays_per_s,
)
from bajo.core import AABB, Vec3f32, Affine3f32
from bajo.bvh.camera import Camera
from bajo.bvh.types import Ray, Sphere, Instance, BlasSet
from bajo.bvh.host_utils import (
    compute_bounds,
    hit_t_for_checksum,
)
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.tlas import Tlas
from bajo.bvh.gpu.tlas import GpuTriangleTlas, GpuSphereTlas
from bajo.bvh.gpu.sphere_bvh import GpuSphereBvh, build_sphere_blas_set
from bajo.bvh.gpu.triangle_bvh import GpuTriangleBvh, build_triangle_blas_set
from bajo.bvh.constants import TRACE, Primitive, MISS_PRIM
from bajo.obj.pack import pack_obj_triangles
from bajo.bvh.gpu.utils import GpuBuildTimings, upload_list, upload_vertices

from bench.bvh.bench_printing import (
    print_gpu_build_timing_rows,
    print_transposed_header,
    print_transposed_row,
    print_transposed_ms_row,
)


comptime DEFAULT_OBJ_PATH = "./assets/bunny/bunny.obj"
comptime MULTI_OBJ_PATH_1 = "./assets/buddha/buddha.obj"
comptime MULTI_OBJ_PATH_2 = "./assets/dragon/dragon.obj"
comptime PRIMARY_WIDTH = 640
comptime PRIMARY_HEIGHT = 360
comptime PRIMARY_VIEWS = 3
comptime BENCH_REPEATS = 8


def _make_single_instance(bounds: AABB, primitive: Primitive) -> List[Instance]:
    return [
        Instance(
            Affine3f32.identity(),
            Affine3f32.identity(),
            0,
            bounds,
            primitive,
        )
    ]


def _make_translated_grid_instances(
    bounds: AABB,
    count_x: Int,
    count_y: Int,
    primitive: Primitive,
) -> List[Instance]:
    var out = List[Instance](capacity=count_x * count_y)

    var extent = bounds._max - bounds._min
    var spacing_x = max(extent.x * 2.25, 0.25)
    var spacing_y = max(extent.y * 2.25, 0.25)

    for y in range(count_y):
        for x in range(count_x):
            var t = Vec3f32(
                (Float32(x) - Float32(count_x - 1) * 0.5) * spacing_x,
                (Float32(y) - Float32(count_y - 1) * 0.5) * spacing_y,
                Float32(0.0),
            )

            out.append(
                Instance(
                    Affine3f32.from_translation(t),
                    Affine3f32.from_translation(-t),
                    0,
                    bounds,
                    primitive,
                )
            )

    return out^


def _make_multi_blas_grid_instances(
    bounds_list: List[AABB],
    count_x: Int,
    count_y: Int,
) -> List[Instance]:
    var out = List[Instance](capacity=count_x * count_y)

    var max_extent = Float32(0.25)
    for bounds in bounds_list:
        var extent = bounds.extent()
        var e = max(max(extent.x, extent.y), extent.z)
        if e > max_extent:
            max_extent = e

    var spacing = max(max_extent * 2.75, 0.25)

    for y in range(count_y):
        for x in range(count_x):
            var idx = y * count_x + x
            var blas_idx = UInt32(idx % len(bounds_list))
            var t = Vec3f32(
                (Float32(x) - Float32(count_x - 1) * 0.5) * spacing,
                (Float32(y) - Float32(count_y - 1) * 0.5) * spacing,
                Float32((idx * 17) % 13) * max_extent * 0.025,
            )

            ref bounds = bounds_list[Int(blas_idx)]
            out.append(
                Instance(
                    Affine3f32.from_translation(t),
                    Affine3f32.from_translation(-t),
                    blas_idx,
                    bounds,
                    Primitive.TRIANGLE,
                )
            )

    return out^


def _instances_bounds(instances: List[Instance]) -> AABB:
    var out = AABB.invalid()
    for instance in instances:
        out.grow(instance.bounds)
    return out


def _make_camera_rays_and_params(
    bounds: AABB,
    width: Int,
    height: Int,
    views: Int,
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
        var target = center
        var camera = Camera(
            eye,
            target,
            Vec3f32(0.0, 1.0, 0.0),
            Float32(0.75),
        )
        params.extend(camera.flatten())

        for py in range(height):
            for px in range(width):
                rays.append(camera.make_ray(px, py, width, height))

    return (rays^, params^)


def _cpu_tlas_triangle_reference[
    tlas_width: Int,
    blas_width: Int,
](
    cpu_blas: TriangleBvh[blas_width],
    instances: List[Instance],
    rays: List[Ray],
) -> Tuple[Float64, UInt32, UInt64]:
    var tlas = Tlas[tlas_width](instances)

    var blases = [cpu_blas.copy()]

    var checksum = Float64(0.0)
    var hits = UInt32(0)
    var inst_checksum = UInt64(0)

    for ray in rays:
        var hit = tlas.trace[TriangleBvh[blas_width], TRACE.CLOSEST_HIT](
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
    var tlas = Tlas[tlas_width](instances)

    var blases = [cpu_blas.copy()]

    var occluded = UInt32(0)

    for ray in rays:
        var hit = tlas.trace[TriangleBvh[blas_width], TRACE.ANY_HIT](
            ray,
            blases.unsafe_ptr(),
        )
        if hit.is_occluded():
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


def _download_direct_hit_checksum(
    hits_f32: DeviceBuffer[DType.float32],
    ray_count: Int,
) raises -> Tuple[Float64, UInt32]:
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    with hits_f32.map_to_host() as hf:
        for i in range(ray_count):
            var t = hf[i * 3 + 0]
            checksum += hit_t_for_checksum(t)
            if t < Float32(1.0e20):
                hit_count += 1

    return (checksum, hit_count)


def _bench_direct_triangle_camera[
    width: Int,
](
    mut ctx: DeviceContext,
    blas: GpuTriangleBvh[width],
    camera_params: List[Float32],
    ray_count: Int,
    width_px: Int,
    height_px: Int,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32]:
    var d_camera = upload_list(ctx, camera_params)
    var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
    var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count)

    blas.launch_camera(
        ctx,
        d_camera,
        d_hits_f32,
        d_hits_u32,
        ray_count,
        width_px,
        height_px,
    )
    ctx.synchronize()

    var best_ns = Int.MAX
    var checksum = Float64(0.0)
    var hits = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        blas.launch_camera(
            ctx,
            d_camera,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            width_px,
            height_px,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_ns = min(best_ns, Int(t1 - t0))

        var downloaded = _download_direct_hit_checksum(d_hits_f32, ray_count)
        checksum = downloaded[0]
        hits = downloaded[1]

    return (best_ns, checksum, hits)


def _bench_tlas_triangles_camera[
    tlas_width: Int,
    blas_width: Int,
](
    mut ctx: DeviceContext,
    tlas: GpuTriangleTlas[tlas_width, blas_width],
    blases: BlasSet[blas_width],
    camera_params: List[Float32],
    ray_count: Int,
    width: Int,
    height: Int,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32, UInt64]:
    var d_camera = upload_list(ctx, camera_params)
    var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
    var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count * 2)

    tlas.launch_camera(
        ctx,
        blases,
        d_camera,
        d_hits_f32,
        d_hits_u32,
        ray_count,
        width,
        height,
    )
    ctx.synchronize()

    var best_ns = Int.MAX
    var checksum = Float64(0.0)
    var hits = UInt32(0)
    var inst_checksum = UInt64(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        tlas.launch_camera(
            ctx,
            blases,
            d_camera,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            width,
            height,
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


def _bench_direct_sphere_camera[
    width: Int,
](
    mut ctx: DeviceContext,
    blas: GpuSphereBvh[width],
    camera_params: List[Float32],
    ray_count: Int,
    width_px: Int,
    height_px: Int,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32]:
    var d_camera = upload_list(ctx, camera_params)
    var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
    var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count)

    blas.launch_camera(
        ctx,
        d_camera,
        d_hits_f32,
        d_hits_u32,
        ray_count,
        width_px,
        height_px,
    )
    ctx.synchronize()

    var best_ns = Int.MAX
    var checksum = Float64(0.0)
    var hits = UInt32(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        blas.launch_camera(
            ctx,
            d_camera,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            width_px,
            height_px,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        best_ns = min(best_ns, Int(t1 - t0))

        var downloaded = _download_direct_hit_checksum(d_hits_f32, ray_count)
        checksum = downloaded[0]
        hits = downloaded[1]

    return (best_ns, checksum, hits)


def _bench_tlas_spheres_camera[
    tlas_width: Int,
    blas_width: Int,
](
    mut ctx: DeviceContext,
    tlas: GpuSphereTlas[tlas_width, blas_width],
    blases: BlasSet[blas_width],
    camera_params: List[Float32],
    ray_count: Int,
    width: Int,
    height: Int,
    repeats: Int,
) raises -> Tuple[Int, Float64, UInt32, UInt64]:
    var d_camera = upload_list(ctx, camera_params)
    var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
    var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count * 2)

    tlas.launch_camera(
        ctx,
        blases,
        d_camera,
        d_hits_f32,
        d_hits_u32,
        ray_count,
        width,
        height,
    )
    ctx.synchronize()

    var best_ns = Int.MAX
    var checksum = Float64(0.0)
    var hits = UInt32(0)
    var inst_checksum = UInt64(0)

    for _ in range(repeats):
        var t0 = perf_counter_ns()
        tlas.launch_camera(
            ctx,
            blases,
            d_camera,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            width,
            height,
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


def _print_blas_build_table_header():
    var c0 = String("case").ascii_ljust(18)
    var c1 = String("build").ascii_rjust(8)
    var c2 = String("collapse").ascii_rjust(10)
    var c3 = String("pack").ascii_rjust(8)

    print(t"{c0} {c1} {c2} {c3}")
    print("------------------ -------- ---------- --------")


def _print_blas_build_row(
    label: String,
    build_ns: Int,
    timings: GpuBuildTimings,
):
    var build_ms = round(ns_to_ms(build_ns), 3)
    var collapse_ms = round(ns_to_ms(timings.collapse_ns), 3)
    var pack_ms = round(ns_to_ms(timings.leaf_pack_ns), 3)

    var c0 = label.ascii_ljust(18)
    var c1 = String(t"{build_ms}").ascii_rjust(8)
    var c2 = String(t"{collapse_ms}").ascii_rjust(10)
    var c3 = String(t"{pack_ms}").ascii_rjust(8)

    print(t"{c0} {c1} {c2} {c3}")


def _print_direct_table_header():
    var c0 = String("case").ascii_ljust(22)
    var c1 = String("primary").ascii_rjust(9)
    var c2 = String("P MRay/s").ascii_rjust(9)
    var c3 = String("hits").ascii_rjust(8)
    var c4 = String("checksum").ascii_rjust(12)

    print(t"{c0} {c1} {c2} {c3} {c4}")
    print("---------------------- --------- --------- -------- ------------")


def _print_direct_row(
    label: String,
    primary_ns: Int,
    primary_checksum: Float64,
    primary_hits: UInt32,
    ray_count: Int,
):
    var primary_ms = round(ns_to_ms(primary_ns), 3)
    var primary_mrays = round(ns_to_mrays_per_s(primary_ns, ray_count), 3)
    var checksum = round(primary_checksum, 3)

    var c0 = label.ascii_ljust(22)
    var c1 = String(t"{primary_ms}").ascii_rjust(9)
    var c2 = String(t"{primary_mrays}").ascii_rjust(9)
    var c3 = String(t"{primary_hits}").ascii_rjust(8)
    var c4 = String(t"{checksum}").ascii_rjust(12)
    print(t"{c0} {c1} {c2} {c3} {c4}")


def _print_cpu_reference_table_header():
    var c0 = String("case").ascii_ljust(30)
    var c1 = String("inst").ascii_rjust(5)
    var c2 = String("hits").ascii_rjust(8)
    var c3 = String("checksum").ascii_rjust(12)
    var c4 = String("shadow").ascii_rjust(8)
    var c5 = String("instsum").ascii_rjust(12)

    print(t"{c0} {c1} {c2} {c3} {c4} {c5}")
    print(
        "------------------------------ ----- -------- ------------ --------"
        " ------------"
    )


def _print_cpu_reference_row(
    label: String,
    inst_count: Int,
    checksum: Float64,
    hits: UInt32,
    shadow_hits: UInt32,
    inst_checksum: UInt64,
):
    var checksum_r = round(checksum, 3)

    var c0 = label.ascii_ljust(30)
    var c1 = String(t"{inst_count}").ascii_rjust(5)
    var c2 = String(t"{hits}").ascii_rjust(8)
    var c3 = String(t"{checksum_r}").ascii_rjust(12)
    var c4 = String(t"{shadow_hits}").ascii_rjust(8)
    var c5 = String(t"{inst_checksum}").ascii_rjust(12)

    print(t"{c0} {c1} {c2} {c3} {c4} {c5}")


@fieldwise_init
struct TlasBenchResult(Writable):
    var label: String
    var inst_count: Int
    var build_ns: Int
    var timings: GpuBuildTimings
    var primary_ns: Int
    var ray_count: Int
    var checksum: Float64
    var hits: UInt32
    var inst_checksum: UInt64
    var has_reference: Bool
    var reference_checksum: Float64
    var reference_hits: UInt32
    var reference_inst_checksum: UInt64


def _tlas_result_sum_diff(row: TlasBenchResult) -> Float64:
    return abs(row.checksum - row.reference_checksum)


def _tlas_result_hit_diff(row: TlasBenchResult) -> Int:
    return Int(row.hits) - Int(row.reference_hits)


def _tlas_result_inst_diff(row: TlasBenchResult) -> Int:
    return Int(row.inst_checksum) - Int(row.reference_inst_checksum)


def _tlas_result_status(row: TlasBenchResult) -> String:
    var diff = _tlas_result_sum_diff(row)
    var hit_diff = _tlas_result_hit_diff(row)
    var inst_diff = _tlas_result_inst_diff(row)

    if diff <= Float64(0.001) and hit_diff == 0 and inst_diff == 0:
        return String("OK")

    return String("CHECK")


def _print_tlas_results_transposed(
    row0: TlasBenchResult,
    row1: TlasBenchResult,
    row2: TlasBenchResult,
):
    var value_width = 15

    print_transposed_header(
        value_width,
        row0.label,
        row1.label,
        row2.label,
    )

    print_transposed_row(
        String("instances"),
        value_width,
        row0.inst_count,
        row1.inst_count,
        row2.inst_count,
    )

    print_gpu_build_timing_rows(
        row0.build_ns,
        row0.timings,
        row1.build_ns,
        row1.timings,
        row2.build_ns,
        row2.timings,
        False,
        value_width,
    )

    print_transposed_ms_row(
        String("camera"),
        row0.primary_ns,
        row1.primary_ns,
        row2.primary_ns,
        value_width,
    )
    print_transposed_row(
        String("MRay/s"),
        value_width,
        round(ns_to_mrays_per_s(row0.primary_ns, row0.ray_count), 3),
        round(ns_to_mrays_per_s(row1.primary_ns, row1.ray_count), 3),
        round(ns_to_mrays_per_s(row2.primary_ns, row2.ray_count), 3),
    )
    print_transposed_row(
        String("hits"),
        value_width,
        row0.hits,
        row1.hits,
        row2.hits,
    )
    print_transposed_row(
        String("checksum"),
        value_width,
        round(row0.checksum, 3),
        round(row1.checksum, 3),
        round(row2.checksum, 3),
    )
    print_transposed_row(
        String("instsum"),
        value_width,
        row0.inst_checksum,
        row1.inst_checksum,
        row2.inst_checksum,
    )

    if row0.has_reference:
        print_transposed_row(
            String("dsum"),
            value_width,
            round(_tlas_result_sum_diff(row0), 6),
            round(_tlas_result_sum_diff(row1), 6),
            round(_tlas_result_sum_diff(row2), 6),
        )
        print_transposed_row(
            String("dhit"),
            value_width,
            _tlas_result_hit_diff(row0),
            _tlas_result_hit_diff(row1),
            _tlas_result_hit_diff(row2),
        )
        print_transposed_row(
            String("dinst"),
            value_width,
            _tlas_result_inst_diff(row0),
            _tlas_result_inst_diff(row1),
            _tlas_result_inst_diff(row2),
        )
        print_transposed_row(
            String("status"),
            value_width,
            _tlas_result_status(row0),
            _tlas_result_status(row1),
            _tlas_result_status(row2),
        )


def _run_triangle_tlas_width[
    tlas_width: Int,
    blas_width: Int,
](
    mut ctx: DeviceContext,
    blases: BlasSet[blas_width],
    instances: List[Instance],
    camera_params: List[Float32],
    ray_count: Int,
    repeats: Int,
    label: String,
    has_reference: Bool,
    reference_checksum: Float64,
    reference_hits: UInt32,
    reference_inst_checksum: UInt64,
) raises -> TlasBenchResult:
    _ = GpuTriangleTlas[tlas_width, blas_width](ctx, instances)
    ctx.synchronize()

    var b0 = perf_counter_ns()
    var tlas = GpuTriangleTlas[tlas_width, blas_width](ctx, instances)
    ctx.synchronize()
    var b1 = perf_counter_ns()

    var primary = _bench_tlas_triangles_camera[tlas_width, blas_width](
        ctx,
        tlas,
        blases,
        camera_params,
        ray_count,
        PRIMARY_WIDTH,
        PRIMARY_HEIGHT,
        repeats,
    )

    return TlasBenchResult(
        label,
        len(instances),
        Int(b1 - b0),
        tlas.core.timings,
        primary[0],
        ray_count,
        primary[1],
        primary[2],
        primary[3],
        has_reference,
        reference_checksum,
        reference_hits,
        reference_inst_checksum,
    )


def _run_sphere_tlas_width[
    tlas_width: Int,
    blas_width: Int,
](
    mut ctx: DeviceContext,
    blases: BlasSet[blas_width],
    instances: List[Instance],
    camera_params: List[Float32],
    ray_count: Int,
    repeats: Int,
    label: String,
    has_reference: Bool,
    reference_checksum: Float64,
    reference_hits: UInt32,
) raises -> TlasBenchResult:
    _ = GpuSphereTlas[tlas_width, blas_width](ctx, instances)
    ctx.synchronize()

    var b0 = perf_counter_ns()
    var tlas = GpuSphereTlas[tlas_width, blas_width](ctx, instances)
    ctx.synchronize()
    var b1 = perf_counter_ns()

    var primary = _bench_tlas_spheres_camera[tlas_width, blas_width](
        ctx,
        tlas,
        blases,
        camera_params,
        ray_count,
        PRIMARY_WIDTH,
        PRIMARY_HEIGHT,
        repeats,
    )

    return TlasBenchResult(
        label,
        len(instances),
        Int(b1 - b0),
        tlas.core.timings,
        primary[0],
        ray_count,
        primary[1],
        primary[2],
        primary[3],
        has_reference,
        reference_checksum,
        reference_hits,
        UInt64(0),
    )


def _bench_triangle_multi_blas_instance_set(
    mut ctx: DeviceContext,
    blases: BlasSet[4],
    blas_bounds: List[AABB],
) raises:
    var instances = _make_multi_blas_grid_instances(blas_bounds, 32, 16)
    var bounds = _instances_bounds(instances)
    var camera_data = _make_camera_rays_and_params(
        bounds,
        PRIMARY_WIDTH,
        PRIMARY_HEIGHT,
        PRIMARY_VIEWS,
    )
    var camera_params = camera_data[1].copy()
    var ray_count = len(camera_data[0])

    print("\nTriangle TLAS multi-BLAS grid 32x16")
    print("-----------------------------------")
    print(t"BLASes: {len(blas_bounds)}")
    print(t"Instances: {len(instances)}")
    print(t"Rays: {ray_count}")
    print("TLAS bounds min:", round(bounds._min, 3))
    print("TLAS bounds max:", round(bounds._max, 3))

    var tlas2 = _run_triangle_tlas_width[2, 4](
        ctx,
        blases,
        instances,
        camera_params,
        ray_count,
        BENCH_REPEATS,
        "t2/b4 32x16",
        False,
        0.0,
        UInt32(0),
        UInt64(0),
    )
    var tlas4 = _run_triangle_tlas_width[4, 4](
        ctx,
        blases,
        instances,
        camera_params,
        ray_count,
        BENCH_REPEATS,
        "t4/b4 32x16",
        False,
        0.0,
        UInt32(0),
        UInt64(0),
    )
    var tlas8 = _run_triangle_tlas_width[8, 4](
        ctx,
        blases,
        instances,
        camera_params,
        ray_count,
        BENCH_REPEATS,
        "t8/b4 32x16",
        False,
        0.0,
        UInt32(0),
        UInt64(0),
    )
    _print_tlas_results_transposed(tlas2, tlas4, tlas8)


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
    blases: BlasSet[4],
    cpu_blas: TriangleBvh[8],
    blas_bounds: AABB,
) raises:
    var instances = _make_translated_grid_instances(
        blas_bounds,
        side,
        side,
        Primitive.TRIANGLE,
    )
    var bounds = _instances_bounds(instances)
    var camera_data = _make_camera_rays_and_params(
        bounds,
        PRIMARY_WIDTH,
        PRIMARY_HEIGHT,
        PRIMARY_VIEWS,
    )
    var rays = camera_data[0].copy()
    var camera_params = camera_data[1].copy()
    var ray_count = len(rays)
    var has_ref = side == 4

    var ref_primary = (Float64(0.0), UInt32(0), UInt64(0))

    if has_ref:
        ref_primary = _cpu_tlas_triangle_reference[2, 8](
            cpu_blas,
            instances,
            rays,
        )
        var ref_shadow = _cpu_tlas_triangle_shadow_reference[2, 8](
            cpu_blas,
            instances,
            rays,
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
    print(t"Rays: {ray_count}")
    print("TLAS bounds min:", bounds._min)
    print("TLAS bounds max:", bounds._max)

    print("\nTLAS camera results")
    var tlas2 = _run_triangle_tlas_width[2, 4](
        ctx,
        blases,
        instances,
        camera_params,
        ray_count,
        BENCH_REPEATS,
        String(t"t2/b4 {side}x{side}"),
        has_ref,
        ref_primary[0],
        ref_primary[1],
        ref_primary[2],
    )
    var tlas4 = _run_triangle_tlas_width[4, 4](
        ctx,
        blases,
        instances,
        camera_params,
        ray_count,
        BENCH_REPEATS,
        String(t"t4/b4 {side}x{side}"),
        has_ref,
        ref_primary[0],
        ref_primary[1],
        ref_primary[2],
    )
    var tlas8 = _run_triangle_tlas_width[8, 4](
        ctx,
        blases,
        instances,
        camera_params,
        ray_count,
        BENCH_REPEATS,
        String(t"t8/b4 {side}x{side}"),
        has_ref,
        ref_primary[0],
        ref_primary[1],
        ref_primary[2],
    )
    _print_tlas_results_transposed(tlas2, tlas4, tlas8)


def main() raises:
    print("GPU TLAS benchmark")
    print(t"Path: {DEFAULT_OBJ_PATH}")
    print(t"Image rays: {PRIMARY_WIDTH} x {PRIMARY_HEIGHT} x {PRIMARY_VIEWS}")
    print(t"Repeats: {BENCH_REPEATS}")

    print("\nLoading + packing OBJ...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles(DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var tri_count = len(tri_vertices) / 3
    var blas_bounds = compute_bounds(tri_vertices)

    print(t"Triangles: {tri_count}")
    print(t"Load+pack ms: {round(ns_to_ms(Int(load_t1 - load_t0)), 3)}")
    print("BLAS bounds min:", round(blas_bounds._min, 3))
    print("BLAS bounds max:", round(blas_bounds._max, 3))

    print("\nLoading multi-BLAS OBJ set...")
    var multi_vertices = List[List[Vec3f32]](capacity=3)
    var multi_bounds = List[AABB](capacity=3)
    multi_vertices.append(tri_vertices.copy())
    multi_bounds.append(blas_bounds)

    var multi_load_t0 = perf_counter_ns()
    var multi_tri1 = pack_obj_triangles(MULTI_OBJ_PATH_1)
    var multi_tri2 = pack_obj_triangles(MULTI_OBJ_PATH_2)
    var multi_load_t1 = perf_counter_ns()
    multi_vertices.append(multi_tri1.copy())
    multi_vertices.append(multi_tri2.copy())
    multi_bounds.append(compute_bounds(multi_tri1))
    multi_bounds.append(compute_bounds(multi_tri2))
    print(t"Multi-BLAS assets: {len(multi_vertices)}")
    print(
        t"Extra load+pack ms:"
        t" {round(ns_to_ms(Int(multi_load_t1 - multi_load_t0)), 3)}"
    )

    comptime if not has_accelerator():
        raise "No compatible GPU found; skipped Mojo GPU TLAS benchmark."

    with DeviceContext() as ctx:
        print("\nUploading triangle vertices...")
        var d_vertices = upload_vertices(ctx, tri_vertices)
        ctx.synchronize()

        print("\nBuilding GpuTriangleBvh[4]...")
        _ = GpuTriangleBvh[4](ctx, d_vertices)
        ctx.synchronize()

        var blas_b0 = perf_counter_ns()
        var blas = GpuTriangleBvh[4](ctx, d_vertices)
        ctx.synchronize()
        var blas_b1 = perf_counter_ns()

        _print_blas_build_table_header()
        _print_blas_build_row(
            "GpuTriangleBvh[4]",
            Int(blas_b1 - blas_b0),
            blas.timings,
        )

        print("Building GpuTriangleBlasSet[4]...")
        var triangle_blas_set = build_triangle_blas_set[4](
            ctx, [tri_vertices.copy()]
        )
        ctx.synchronize()

        print("Building multi-object GpuTriangleBlasSet[4]...")
        var multi_triangle_blas_set = build_triangle_blas_set[4](
            ctx, multi_vertices
        )
        ctx.synchronize()

        print("Building CPU TriangleBvh[8] LBVH reference for 4x4 grid...")
        var cpu_blas = TriangleBvh[8].__init__["lbvh"](tri_vertices^)

        print("\nSingle instance triangle TLAS")
        print("-----------------------------")
        var single_instances = _make_single_instance(
            blas_bounds,
            Primitive.TRIANGLE,
        )
        var single_bounds = _instances_bounds(single_instances)
        var single_camera_data = _make_camera_rays_and_params(
            single_bounds,
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            PRIMARY_VIEWS,
        )
        var single_rays = single_camera_data[0].copy()
        var single_camera_params = single_camera_data[1].copy()
        var single_ray_count = len(single_rays)

        print(t"Rays: {single_ray_count}")

        var direct = _bench_direct_triangle_camera[4](
            ctx,
            blas,
            single_camera_params,
            single_ray_count,
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            BENCH_REPEATS,
        )

        print("\nDirect BLAS")
        _print_direct_table_header()
        _print_direct_row(
            "direct triangle BLAS",
            direct[0],
            direct[1],
            direct[2],
            single_ray_count,
        )

        print("\nTLAS camera results")
        var tlas2 = _run_triangle_tlas_width[2, 4](
            ctx,
            triangle_blas_set,
            single_instances,
            single_camera_params,
            single_ray_count,
            BENCH_REPEATS,
            "t2/b4 1x1",
            True,
            direct[1],
            direct[2],
            UInt64(0),
        )
        var tlas4 = _run_triangle_tlas_width[4, 4](
            ctx,
            triangle_blas_set,
            single_instances,
            single_camera_params,
            single_ray_count,
            BENCH_REPEATS,
            "t4/b4 1x1",
            True,
            direct[1],
            direct[2],
            UInt64(0),
        )
        var tlas8 = _run_triangle_tlas_width[8, 4](
            ctx,
            triangle_blas_set,
            single_instances,
            single_camera_params,
            single_ray_count,
            BENCH_REPEATS,
            "t8/b4 1x1",
            True,
            direct[1],
            direct[2],
            UInt64(0),
        )
        _print_tlas_results_transposed(tlas2, tlas4, tlas8)

        _bench_triangle_instance_set[4](
            ctx, triangle_blas_set, cpu_blas, blas_bounds
        )
        _bench_triangle_instance_set[8](
            ctx, triangle_blas_set, cpu_blas, blas_bounds
        )
        _bench_triangle_instance_set[16](
            ctx, triangle_blas_set, cpu_blas, blas_bounds
        )
        _bench_triangle_multi_blas_instance_set(
            ctx,
            multi_triangle_blas_set,
            multi_bounds,
        )

        print("\nSphere TLAS")
        print("-----------")
        var sphere_scene = _make_sphere_scene(32, 32)
        var spheres = sphere_scene[0].copy()
        var sphere_bounds = sphere_scene[1].copy()
        print(t"Spheres: {len(spheres)}")
        print("Sphere BLAS bounds min:", round(sphere_bounds._min, 3))
        print("Sphere BLAS bounds max:", round(sphere_bounds._max, 3))

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
            sphere_blas.timings,
        )

        print("Building GpuSphere BlasSet[4]...")
        var sphere_blas_set = build_sphere_blas_set[4](ctx, [spheres^])
        ctx.synchronize()

        var sphere_single = _make_single_instance(
            sphere_bounds,
            Primitive.SPHERE,
        )
        var sphere_camera_data = _make_camera_rays_and_params(
            sphere_bounds,
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            PRIMARY_VIEWS,
        )
        var sphere_rays = sphere_camera_data[0].copy()
        var sphere_camera_params = sphere_camera_data[1].copy()
        var sphere_ray_count = len(sphere_rays)

        var direct_sphere = _bench_direct_sphere_camera[4](
            ctx,
            sphere_blas,
            sphere_camera_params,
            sphere_ray_count,
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            BENCH_REPEATS,
        )

        print("\nDirect sphere BLAS")
        _print_direct_table_header()
        _print_direct_row(
            "direct sphere BLAS",
            direct_sphere[0],
            direct_sphere[1],
            direct_sphere[2],
            sphere_ray_count,
        )

        print("\nSphere single-instance TLAS")
        var sph_tlas2 = _run_sphere_tlas_width[2, 4](
            ctx,
            sphere_blas_set,
            sphere_single,
            sphere_camera_params,
            sphere_ray_count,
            BENCH_REPEATS,
            "sph t2/b4 1x1",
            True,
            direct_sphere[1],
            direct_sphere[2],
        )
        var sph_tlas4 = _run_sphere_tlas_width[4, 4](
            ctx,
            sphere_blas_set,
            sphere_single,
            sphere_camera_params,
            sphere_ray_count,
            BENCH_REPEATS,
            "sph t4/b4 1x1",
            True,
            direct_sphere[1],
            direct_sphere[2],
        )
        var sph_tlas8 = _run_sphere_tlas_width[8, 4](
            ctx,
            sphere_blas_set,
            sphere_single,
            sphere_camera_params,
            sphere_ray_count,
            BENCH_REPEATS,
            "sph t8/b4 1x1",
            True,
            direct_sphere[1],
            direct_sphere[2],
        )
        _print_tlas_results_transposed(sph_tlas2, sph_tlas4, sph_tlas8)

        var sphere_grid = _make_translated_grid_instances(
            sphere_bounds,
            4,
            4,
            Primitive.SPHERE,
        )
        var sphere_grid_bounds = _instances_bounds(sphere_grid)
        var sphere_grid_camera_data = _make_camera_rays_and_params(
            sphere_grid_bounds,
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            PRIMARY_VIEWS,
        )
        var sphere_grid_camera_params = sphere_grid_camera_data[1].copy()
        var sphere_grid_ray_count = len(sphere_grid_camera_data[0])

        print("\nSphere TLAS translated 4x4")
        print(t"Instances: {len(sphere_grid)}")
        print(t"Rays: {sphere_grid_ray_count}")
        print("TLAS bounds min:", round(sphere_grid_bounds._min, 3))
        print("TLAS bounds max:", round(sphere_grid_bounds._max, 3))

        var sph_grid_tlas2 = _run_sphere_tlas_width[2, 4](
            ctx,
            sphere_blas_set,
            sphere_grid,
            sphere_grid_camera_params,
            sphere_grid_ray_count,
            BENCH_REPEATS,
            "sph t2/b4 4x4",
            False,
            0.0,
            UInt32(0),
        )
        var sph_grid_tlas4 = _run_sphere_tlas_width[4, 4](
            ctx,
            sphere_blas_set,
            sphere_grid,
            sphere_grid_camera_params,
            sphere_grid_ray_count,
            BENCH_REPEATS,
            "sph t4/b4 4x4",
            False,
            0.0,
            UInt32(0),
        )
        var sph_grid_tlas8 = _run_sphere_tlas_width[8, 4](
            ctx,
            sphere_blas_set,
            sphere_grid,
            sphere_grid_camera_params,
            sphere_grid_ray_count,
            BENCH_REPEATS,
            "sph t8/b4 4x4",
            False,
            0.0,
            UInt32(0),
        )
        _print_tlas_results_transposed(
            sph_grid_tlas2,
            sph_grid_tlas4,
            sph_grid_tlas8,
        )
