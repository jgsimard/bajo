"""Instanced closest-hit controls for cross-library CPU BVH reports."""

from std.math import round
from std.time import perf_counter_ns

from bajo.benchmark.bvh_fixtures import make_camera_rays_and_params
from bajo.benchmark.bvh_reporting import TablePrinter
from bajo.bvh.constants import PrimitiveKind, TraceMode, f32_max
from bajo.bvh.cpu import CpuBlasSet, CpuBvhBuildMethod
from bajo.bvh.cpu.blas_set import build_cpu_triangle_blas_set
from bajo.bvh.cpu.blas_set import (
    trace_blas_set,
    trace_blas_set_packet,
    trace_blas_set_packet_any_hit,
)
from bajo.bvh.cpu.tlas import CpuTlas
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Instance
from bajo.core import (
    AABB,
    Affine3f32,
    Point3,
    Point3f32,
    Ray,
    Rayf32,
    Vec3,
    Vec3f32,
)
from bajo.core.utils import ns_to_mrays_per_s, ns_to_ms
from bajo.parser.obj.pack import pack_obj_triangles


comptime OBJ_PATH = "./assets/dragon/dragon.obj"
comptime INSTANCE_X = 12
comptime INSTANCE_Y = 9
comptime INSTANCE_COUNT = INSTANCE_X * INSTANCE_Y
comptime RAY_WIDTH = 512
comptime RAY_HEIGHT = 288
comptime FOV_SCALE = 0.2
comptime TRAVERSAL_REPEATS = 8
comptime TRAVERSAL_BATCHES = 8
comptime BUILD_REPEATS = 5


@fieldwise_init
struct TraceResult(Copyable):
    var ns: Int
    var checksum: Float64
    var hits: Int


def _median_ns(values: List[Int]) -> Int:
    var sorted_values = values.copy()
    sort(sorted_values)
    return sorted_values[len(sorted_values) / 2]


def make_instances(
    bounds: AABB[.LOCAL],
) -> Tuple[List[Instance], AABB[.WORLD]]:
    var instances = List[Instance](capacity=INSTANCE_COUNT)
    var world_bounds = AABB[.WORLD].invalid()
    var extent = bounds.extent()
    var spacing_x = max(extent.x, Float32(1.0e-6)) * 1.25
    var spacing_y = max(extent.y, Float32(1.0e-6)) * 1.25
    for y in range(INSTANCE_Y):
        for x in range(INSTANCE_X):
            var tx = (Float32(x) - Float32(INSTANCE_X - 1) * 0.5) * spacing_x
            var ty = (Float32(y) - Float32(INSTANCE_Y - 1) * 0.5) * spacing_y
            var transform = Affine3f32[.LOCAL, .WORLD].from_translation(
                Vec3f32[.WORLD](tx, ty, 0.0)
            )
            var instance = Instance(
                transform, UInt32(0), bounds, PrimitiveKind.TRIANGLE
            )
            world_bounds.grow(instance.bounds)
            instances.append(instance^)
    return (instances^, world_bounds)


def trace_scalar[
    mode: TraceMode,
](
    tlas: CpuTlas[4, 1],
    blases: CpuBlasSet[.TRIANGLE, 16, 16],
    rays: List[Rayf32[.WORLD]],
) -> Tuple[Float64, Int]:
    var checksum = Float64(0.0)
    var hits = 0
    for ray in rays:
        var hit = tlas.trace_blases[16, 16, mode](ray, blases)
        comptime if mode == .ANY_HIT:
            if hit.is_occluded():
                hits += 1
        else:
            if hit.is_hit():
                checksum += (
                    Float64(hit.t)
                    + Float64(hit.u)
                    + Float64(hit.v)
                    + Float64(hit.normal.x)
                    + Float64(hit.normal.y)
                    + Float64(hit.normal.z)
                    + Float64(hit.prim)
                )
                hits += 1
    comptime if mode == .ANY_HIT:
        checksum = Float64(hits)
    return (checksum, hits)


def trace_packet_closest[
    length: SIMDLength,
](
    tlas: CpuTlas[4, 1],
    blases: CpuBlasSet[.TRIANGLE, 16, 16],
    rays: List[Rayf32[.WORLD]],
) -> Tuple[Float64, Int]:
    var checksum = Float64(0.0)
    var hits = 0
    for base in range(0, len(rays), length):
        var ox = SIMD[.float32, length](0.0)
        var oy = SIMD[.float32, length](0.0)
        var oz = SIMD[.float32, length](0.0)
        var dx = SIMD[.float32, length](0.0)
        var dy = SIMD[.float32, length](0.0)
        var dz = SIMD[.float32, length](1.0)
        var t_min = SIMD[.float32, length](0.0)
        var t_max = SIMD[.float32, length](f32_max)
        var valid = SIMD[.bool, length](fill=False)
        var lane_count = min(length, len(rays) - base)
        comptime for lane in range(length):
            if lane < lane_count:
                ref ray = rays.unsafe_get(base + lane)
                ox[lane] = ray.o.x
                oy[lane] = ray.o.y
                oz[lane] = ray.o.z
                dx[lane] = ray.d.x
                dy[lane] = ray.d.y
                dz[lane] = ray.d.z
                t_min[lane] = ray.t_min
                t_max[lane] = ray.t_max
                valid[lane] = True
        var packet = Ray[.float32, .WORLD, length](
            Point3[.float32, .WORLD, length](ox, oy, oz),
            Vec3[.float32, .WORLD, length](dx, dy, dz),
            t_min,
            t_max,
        )
        var hit = tlas.trace_blases_packet[16, 16, length](
            packet, blases, valid
        )
        comptime for lane in range(length):
            if lane < lane_count and hit.is_hit()[lane]:
                checksum += (
                    Float64(hit.t[lane])
                    + Float64(hit.u[lane])
                    + Float64(hit.v[lane])
                    + Float64(hit.normal.x[lane])
                    + Float64(hit.normal.y[lane])
                    + Float64(hit.normal.z[lane])
                    + Float64(hit.prim[lane])
                )
                hits += 1
    return (checksum, hits)


def trace_packet_any[
    length: SIMDLength,
](
    tlas: CpuTlas[4, 1],
    blases: CpuBlasSet[.TRIANGLE, 16, 16],
    rays: List[Rayf32[.WORLD]],
) -> Tuple[Float64, Int]:
    var hits = 0
    for base in range(0, len(rays), length):
        var ox = SIMD[.float32, length](0.0)
        var oy = SIMD[.float32, length](0.0)
        var oz = SIMD[.float32, length](0.0)
        var dx = SIMD[.float32, length](0.0)
        var dy = SIMD[.float32, length](0.0)
        var dz = SIMD[.float32, length](1.0)
        var t_min = SIMD[.float32, length](0.0)
        var t_max = SIMD[.float32, length](f32_max)
        var valid = SIMD[.bool, length](fill=False)
        var lane_count = min(length, len(rays) - base)
        comptime for lane in range(length):
            if lane < lane_count:
                ref ray = rays.unsafe_get(base + lane)
                ox[lane] = ray.o.x
                oy[lane] = ray.o.y
                oz[lane] = ray.o.z
                dx[lane] = ray.d.x
                dy[lane] = ray.d.y
                dz[lane] = ray.d.z
                t_min[lane] = ray.t_min
                t_max[lane] = ray.t_max
                valid[lane] = True
        var packet = Ray[.float32, .WORLD, length](
            Point3[.float32, .WORLD, length](ox, oy, oz),
            Vec3[.float32, .WORLD, length](dx, dy, dz),
            t_min,
            t_max,
        )
        var occluded = tlas.trace_blases_packet_any_hit[16, 16, length](
            packet, blases, valid
        )
        hits += Int(occluded.cast[.uint32]().reduce_add())
    return (Float64(hits), hits)


def benchmark_scalar[
    mode: TraceMode,
](
    tlas: CpuTlas[4, 1],
    blases: CpuBlasSet[.TRIANGLE, 16, 16],
    rays: List[Rayf32[.WORLD]],
) -> TraceResult:
    var summary = trace_scalar[mode](tlas, blases, rays)
    var samples = List[Int](capacity=TRAVERSAL_REPEATS)
    for _ in range(TRAVERSAL_REPEATS):
        var start = perf_counter_ns()
        for _ in range(TRAVERSAL_BATCHES):
            summary = trace_scalar[mode](tlas, blases, rays)
        samples.append(Int(perf_counter_ns() - start) / TRAVERSAL_BATCHES)
    return TraceResult(_median_ns(samples), summary[0], summary[1])


def benchmark_packet_closest[
    length: SIMDLength,
](
    tlas: CpuTlas[4, 1],
    blases: CpuBlasSet[.TRIANGLE, 16, 16],
    rays: List[Rayf32[.WORLD]],
) -> TraceResult:
    var summary = trace_packet_closest[length](tlas, blases, rays)
    var samples = List[Int](capacity=TRAVERSAL_REPEATS)
    for _ in range(TRAVERSAL_REPEATS):
        var start = perf_counter_ns()
        for _ in range(TRAVERSAL_BATCHES):
            summary = trace_packet_closest[length](tlas, blases, rays)
        samples.append(Int(perf_counter_ns() - start) / TRAVERSAL_BATCHES)
    return TraceResult(_median_ns(samples), summary[0], summary[1])


def benchmark_packet_any[
    length: SIMDLength,
](
    tlas: CpuTlas[4, 1],
    blases: CpuBlasSet[.TRIANGLE, 16, 16],
    rays: List[Rayf32[.WORLD]],
) -> TraceResult:
    var summary = trace_packet_any[length](tlas, blases, rays)
    var samples = List[Int](capacity=TRAVERSAL_REPEATS)
    for _ in range(TRAVERSAL_REPEATS):
        var start = perf_counter_ns()
        for _ in range(TRAVERSAL_BATCHES):
            summary = trace_packet_any[length](tlas, blases, rays)
        samples.append(Int(perf_counter_ns() - start) / TRAVERSAL_BATCHES)
    return TraceResult(_median_ns(samples), summary[0], summary[1])


def trace_flat_packet[
    length: SIMDLength,
    mode: TraceMode,
    common_octant: Bool,
](
    blas: CpuBlasSet[.TRIANGLE, 16, 16],
    rays: List[Rayf32[.WORLD]],
) -> Tuple[
    Float64, Int
]:
    var checksum = Float64(0.0)
    var hits = 0
    for base in range(0, len(rays), length):
        var ox = SIMD[.float32, length](0.0)
        var oy = SIMD[.float32, length](0.0)
        var oz = SIMD[.float32, length](0.0)
        var dx = SIMD[.float32, length](0.0)
        var dy = SIMD[.float32, length](0.0)
        var dz = SIMD[.float32, length](1.0)
        var valid = SIMD[.bool, length](fill=False)
        var lane_count = min(length, len(rays) - base)
        comptime for lane in range(length):
            if lane < lane_count:
                ref ray = rays.unsafe_get(base + lane)
                ox[lane] = ray.o.x
                oy[lane] = ray.o.y
                oz[lane] = ray.o.z
                dx[lane] = ray.d.x
                dy[lane] = ray.d.y
                dz[lane] = ray.d.z
                valid[lane] = True
        var packet = Ray[.float32, .WORLD, length](
            Point3[.float32, .WORLD, length](ox, oy, oz),
            Vec3[.float32, .WORLD, length](dx, dy, dz),
            SIMD[.float32, length](0.0),
            SIMD[.float32, length](f32_max),
        )
        comptime if mode == .ANY_HIT:
            var occluded = trace_blas_set_packet_any_hit[
                16, 16, length, common_octant, .WORLD
            ](blas, UInt32(0), packet, valid)
            hits += Int(occluded.cast[.uint32]().reduce_add())
        else:
            var hit = trace_blas_set_packet[
                16, 16, length, common_octant, .WORLD
            ](blas, UInt32(0), packet, valid)
            comptime for lane in range(length):
                if lane < lane_count and hit.is_hit()[lane]:
                    checksum += (
                        Float64(hit.t[lane])
                        + Float64(hit.u[lane])
                        + Float64(hit.v[lane])
                        + Float64(hit.normal.x[lane])
                        + Float64(hit.normal.y[lane])
                        + Float64(hit.normal.z[lane])
                        + Float64(hit.prim[lane])
                    )
                    hits += 1
    comptime if mode == .ANY_HIT:
        checksum = Float64(hits)
    return (checksum, hits)


def trace_flat_scalar[
    mode: TraceMode,
](
    blas: CpuBlasSet[.TRIANGLE, 16, 16],
    rays: List[Rayf32[.WORLD]],
) -> Tuple[
    Float64, Int
]:
    var checksum = Float64(0.0)
    var hits = 0
    for ray in rays:
        var hit = trace_blas_set[16, 16, mode, .WORLD](blas, UInt32(0), ray)
        comptime if mode == .ANY_HIT:
            if hit.is_occluded():
                hits += 1
        else:
            if hit.is_hit():
                checksum += (
                    Float64(hit.t)
                    + Float64(hit.u)
                    + Float64(hit.v)
                    + Float64(hit.normal.x)
                    + Float64(hit.normal.y)
                    + Float64(hit.normal.z)
                    + Float64(hit.prim)
                )
                hits += 1
    comptime if mode == .ANY_HIT:
        checksum = Float64(hits)
    return (checksum, hits)


def benchmark_flat_scalar[
    mode: TraceMode,
](
    blas: CpuBlasSet[.TRIANGLE, 16, 16],
    rays: List[Rayf32[.WORLD]],
) -> TraceResult:
    var summary = trace_flat_scalar[mode](blas, rays)
    var samples = List[Int](capacity=TRAVERSAL_REPEATS)
    for _ in range(TRAVERSAL_REPEATS):
        var start = perf_counter_ns()
        for _ in range(TRAVERSAL_BATCHES):
            summary = trace_flat_scalar[mode](blas, rays)
        samples.append(Int(perf_counter_ns() - start) / TRAVERSAL_BATCHES)
    return TraceResult(_median_ns(samples), summary[0], summary[1])


def benchmark_flat_packet[
    length: SIMDLength,
    mode: TraceMode,
    common_octant: Bool,
](
    blas: CpuBlasSet[.TRIANGLE, 16, 16],
    rays: List[Rayf32[.WORLD]],
) -> TraceResult:
    var summary = trace_flat_packet[length, mode, common_octant](blas, rays)
    var samples = List[Int](capacity=TRAVERSAL_REPEATS)
    for _ in range(TRAVERSAL_REPEATS):
        var start = perf_counter_ns()
        for _ in range(TRAVERSAL_BATCHES):
            summary = trace_flat_packet[length, mode, common_octant](blas, rays)
        samples.append(Int(perf_counter_ns() - start) / TRAVERSAL_BATCHES)
    return TraceResult(_median_ns(samples), summary[0], summary[1])


def print_result(
    table: TablePrinter,
    method: String,
    traversal: String,
    build_ns: Int,
    result: TraceResult,
    ray_count: Int,
) raises:
    table.result_line(
        split_method=method,
        bounds_width="4",
        leaf_width="1",
        traversal=traversal,
        build_ms=String(round(ns_to_ms(build_ns), 3)),
        trace_ms=String(round(ns_to_ms(result.ns), 3)),
        MRay_s=String(round(ns_to_mrays_per_s(result.ns, ray_count), 3)),
        hits=String(result.hits),
        checksum=String(round(result.checksum, 3)),
    )


def make_table() -> TablePrinter:
    return TablePrinter(
        split_method=12,
        bounds_width=12,
        leaf_width=10,
        traversal=17,
        build_ms=10,
        trace_ms=10,
        MRay_s=8,
        hits=8,
        checksum=15,
    )


def benchmark_method[
    method: CpuBvhBuildMethod,
](
    case_name: String,
    vertices: List[Point3f32[.LOCAL]],
    instances: List[Instance],
    rays: List[Rayf32[.WORLD]],
) raises:
    var blases = build_cpu_triangle_blas_set[16, 16, method, .LOCAL](
        [vertices.copy()]
    )
    var tlas = CpuTlas[4, 1].__init__[method](instances)
    var build_times = List[Int](capacity=BUILD_REPEATS)
    for _ in range(BUILD_REPEATS):
        var start = perf_counter_ns()
        var sample_blases = build_cpu_triangle_blas_set[16, 16, method, .LOCAL](
            [vertices.copy()]
        )
        var sample_tlas = CpuTlas[4, 1].__init__[method](instances)
        build_times.append(Int(perf_counter_ns() - start))
        blases = sample_blases^
        tlas = sample_tlas^
    var build_ns = _median_ns(build_times)

    var closest_scalar = benchmark_scalar[.CLOSEST_HIT](tlas, blases, rays)
    var closest_packet4 = benchmark_packet_closest[4](tlas, blases, rays)
    var closest_packet8 = benchmark_packet_closest[8](tlas, blases, rays)
    var closest_packet16 = benchmark_packet_closest[16](tlas, blases, rays)
    var any_scalar = benchmark_scalar[.ANY_HIT](tlas, blases, rays)
    var any_packet4 = benchmark_packet_any[4](tlas, blases, rays)
    var any_packet8 = benchmark_packet_any[8](tlas, blases, rays)
    var any_packet16 = benchmark_packet_any[16](tlas, blases, rays)

    print(t"\n{case_name} closest-hit benchmark")
    print(t"Triangles: {len(vertices) / 3}")
    print(t"Instances: {len(instances)}")
    print(t"Rays: {len(rays)}")
    var table = make_table()
    table.header()
    print_result(
        table, method.name(), "scalar1", build_ns, closest_scalar, len(rays)
    )
    print_result(
        table,
        method.name(),
        "packet4",
        build_ns,
        closest_packet4,
        len(rays),
    )
    print_result(
        table,
        method.name(),
        "packet8",
        build_ns,
        closest_packet8,
        len(rays),
    )
    print_result(
        table,
        method.name(),
        "packet16",
        build_ns,
        closest_packet16,
        len(rays),
    )
    print(t"\n{case_name} any-hit benchmark")
    print(t"Triangles: {len(vertices) / 3}")
    print(t"Instances: {len(instances)}")
    print(t"Rays: {len(rays)}")
    table = make_table()
    table.header()
    print_result(
        table, method.name(), "scalar1", build_ns, any_scalar, len(rays)
    )
    print_result(
        table,
        method.name(),
        "packet4",
        build_ns,
        any_packet4,
        len(rays),
    )
    print_result(
        table,
        method.name(),
        "packet8",
        build_ns,
        any_packet8,
        len(rays),
    )
    print_result(
        table,
        method.name(),
        "packet16",
        build_ns,
        any_packet16,
        len(rays),
    )


def benchmark_scene(
    case_name: String,
    vertices: List[Point3f32[.LOCAL]],
) raises:
    var bounds = compute_bounds(vertices)
    var scene = make_instances(bounds)
    var instances = scene[0].copy()
    var camera = make_camera_rays_and_params(
        scene[1], RAY_WIDTH, RAY_HEIGHT, 1, FOV_SCALE
    )
    var rays = camera[0].copy()

    comptime for method in [
        CpuBvhBuildMethod.SAH,
        CpuBvhBuildMethod.LBVH,
        CpuBvhBuildMethod.HPLOC,
    ]:
        benchmark_method[method](case_name, vertices, instances, rays)


def make_single_triangle() -> List[Point3f32[.LOCAL]]:
    """Minimal BLAS: leaves only TLAS, transform, and hit-payload overhead."""
    return [
        Point3f32[.LOCAL](-0.75, -0.75, 0.0),
        Point3f32[.LOCAL](0.75, -0.75, 0.0),
        Point3f32[.LOCAL](0.0, 0.75, 0.0),
    ]


def make_flattened_triangle_grid() -> List[Point3f32[.WORLD]]:
    """Same world geometry as the instances, stored in one packet BLAS."""
    var vertices = List[Point3f32[.WORLD]](capacity=3 * INSTANCE_COUNT)
    comptime spacing_x = Float32(1.5 * 1.25)
    comptime spacing_y = Float32(1.5 * 1.25)
    for y in range(INSTANCE_Y):
        for x in range(INSTANCE_X):
            var tx = (Float32(x) - Float32(INSTANCE_X - 1) * 0.5) * spacing_x
            var ty = (Float32(y) - Float32(INSTANCE_Y - 1) * 0.5) * spacing_y
            vertices.append(Point3f32[.WORLD](tx - 0.75, ty - 0.75, 0.0))
            vertices.append(Point3f32[.WORLD](tx + 0.75, ty - 0.75, 0.0))
            vertices.append(Point3f32[.WORLD](tx, ty + 0.75, 0.0))
    return vertices^


def benchmark_flattened_mode[
    mode: TraceMode,
](
    blas: CpuBlasSet[.TRIANGLE, 16, 16],
    vertices: List[Point3f32[.WORLD]],
    rays: List[Rayf32[.WORLD]],
    build_ns: Int,
) raises:
    comptime suffix = "closest-hit" if mode == .CLOSEST_HIT else "any-hit"
    print(t"\nFlattened triangle grid {suffix} benchmark")
    print(t"Triangles: {len(vertices) / 3}")
    print("Instances: 1")
    print(t"Rays: {len(rays)}")
    var table = make_table()
    table.header()
    var scalar = benchmark_flat_scalar[mode](blas, rays)
    table.result_line(
        split_method="sah",
        bounds_width="16",
        leaf_width="16",
        traversal="scalar1",
        build_ms=String(round(ns_to_ms(build_ns), 3)),
        trace_ms=String(round(ns_to_ms(scalar.ns), 3)),
        MRay_s=String(round(ns_to_mrays_per_s(scalar.ns, len(rays)), 3)),
        hits=String(scalar.hits),
        checksum=String(round(scalar.checksum, 3)),
    )
    comptime for length in [4, 8, 16]:
        comptime for common_octant in [False, True]:
            var result = benchmark_flat_packet[length, mode, common_octant](
                blas, rays
            )
            comptime prefix = "coh-" if common_octant else ""
            table.result_line(
                split_method="sah",
                bounds_width="16",
                leaf_width="16",
                traversal=String(t"{prefix}packet{length}"),
                build_ms=String(round(ns_to_ms(build_ns), 3)),
                trace_ms=String(round(ns_to_ms(result.ns), 3)),
                MRay_s=String(
                    round(ns_to_mrays_per_s(result.ns, len(rays)), 3)
                ),
                hits=String(result.hits),
                checksum=String(round(result.checksum, 3)),
            )


def benchmark_flattened_scene() raises:
    var vertices = make_flattened_triangle_grid()
    var camera = make_camera_rays_and_params(
        compute_bounds(vertices), RAY_WIDTH, RAY_HEIGHT, 1, FOV_SCALE
    )
    var rays = camera[0].copy()
    var blas = build_cpu_triangle_blas_set[16, 16, .SAH, .WORLD](
        [vertices.copy()]
    )
    var build_times = List[Int](capacity=BUILD_REPEATS)
    for _ in range(BUILD_REPEATS):
        var start = perf_counter_ns()
        var sample = build_cpu_triangle_blas_set[16, 16, .SAH, .WORLD](
            [vertices.copy()]
        )
        build_times.append(Int(perf_counter_ns() - start))
        blas = sample^
    var build_ns = _median_ns(build_times)
    benchmark_flattened_mode[.CLOSEST_HIT](blas, vertices, rays, build_ns)
    benchmark_flattened_mode[.ANY_HIT](blas, vertices, rays, build_ns)


def run_benchmark() raises:
    print("CPU instanced closest-hit diagnostic benchmark")
    print(t"Traversal repeats: {TRAVERSAL_REPEATS}")
    print(t"Timing batches: {TRAVERSAL_BATCHES}")
    var dragon = pack_obj_triangles[.LOCAL](OBJ_PATH)
    benchmark_scene("Instanced Dragon", dragon)
    benchmark_scene("Instanced triangle", make_single_triangle())
    benchmark_flattened_scene()


def main() raises:
    run_benchmark()
