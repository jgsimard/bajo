"""Controlled scalar versus shared-stack packet triangle BVH benchmark."""

from std.math import round
from std.time import perf_counter_ns

from bajo.bvh.constants import TRACE, f32_max
from bajo.core import Ray
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Hit
from bajo.core import Frame, Point3, Point3f32, Vec3, Rayf32
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.parser.obj.pack import pack_obj_triangles
from bench.bvh.fixtures import (
    make_camera_rays_and_params,
    make_grid_triangles,
    make_hit_and_miss_rays,
)


comptime OBJ_PATH = "./assets/dragon/dragon.obj"
comptime DRAGON_WIDTH = 1024
comptime DRAGON_HEIGHT = 576
comptime FOV_SCALE = 0.2
comptime REPEATS = 8


@fieldwise_init
struct PacketTiming(Copyable):
    var ns: Int
    var checksum: Float64
    var hits: Int


def trace_scalar[
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
    unmasked: Bool = False,
](
    bvh: TriangleBvh[Frame.WORLD, bounds_width, leaf_width],
    rays: List[Rayf32[Frame.WORLD]],
) -> Tuple[Float64, Int]:
    var checksum = Float64(0.0)
    var hits = 0
    for ray in rays:
        var hit: Hit[Frame.WORLD]
        comptime if unmasked:
            hit = bvh.trace_scalar_unmasked[TRACE.CLOSEST_HIT](ray)
        else:
            hit = bvh.trace[TRACE.CLOSEST_HIT](ray)
        if hit.t < f32_max:
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
    return (checksum, hits)


def trace_packet[
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
    length: SIMDLength,
    common_octant: Bool,
](
    bvh: TriangleBvh[Frame.WORLD, bounds_width, leaf_width],
    rays: List[Rayf32[Frame.WORLD]],
) -> Tuple[Float64, Int]:
    var checksum = Float64(0.0)
    var hits = 0
    for base in range(0, len(rays), length):
        var ox = SIMD[DType.float32, length](0.0)
        var oy = SIMD[DType.float32, length](0.0)
        var oz = SIMD[DType.float32, length](0.0)
        var dx = SIMD[DType.float32, length](0.0)
        var dy = SIMD[DType.float32, length](0.0)
        var dz = SIMD[DType.float32, length](1.0)
        var t_min = SIMD[DType.float32, length](0.0)
        var t_max = SIMD[DType.float32, length](f32_max)
        var valid = SIMD[DType.bool, length](fill=False)
        var lane_count = min(length, len(rays) - base)
        for lane in range(lane_count):
            ref ray = rays[base + lane]
            ox[lane] = ray.o.x
            oy[lane] = ray.o.y
            oz[lane] = ray.o.z
            dx[lane] = ray.d.x
            dy[lane] = ray.d.y
            dz[lane] = ray.d.z
            t_min[lane] = ray.t_min
            t_max[lane] = ray.t_max
            valid[lane] = True

        var packet = Ray[DType.float32, Frame.WORLD, length](
            Point3[DType.float32, Frame.WORLD, length](ox, oy, oz),
            Vec3[DType.float32, Frame.WORLD, length](dx, dy, dz),
            t_min,
            t_max,
        )
        var packet_hit: Hit[Frame.WORLD, length]
        comptime if common_octant:
            packet_hit = bvh.trace_packet_common_octant(packet, valid)
        else:
            packet_hit = bvh.trace[TRACE.CLOSEST_HIT](packet, valid)
        for lane in range(lane_count):
            if packet_hit.prim[lane] != UInt32(0xFFFFFFFF):
                checksum += (
                    Float64(packet_hit.t[lane])
                    + Float64(packet_hit.u[lane])
                    + Float64(packet_hit.v[lane])
                    + Float64(packet_hit.normal.x[lane])
                    + Float64(packet_hit.normal.y[lane])
                    + Float64(packet_hit.normal.z[lane])
                    + Float64(packet_hit.prim[lane])
                )
                hits += 1
    return (checksum, hits)


def benchmark[
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
    length: SIMDLength,
    common_octant: Bool = False,
    unmasked_scalar: Bool = False,
](
    bvh: TriangleBvh[Frame.WORLD, bounds_width, leaf_width],
    rays: List[Rayf32[Frame.WORLD]],
) -> PacketTiming:
    var summary: Tuple[Float64, Int]
    comptime if length == 1:
        summary = trace_scalar[bounds_width, leaf_width, unmasked_scalar](
            bvh, rays
        )
    else:
        summary = trace_packet[bounds_width, leaf_width, length, common_octant](
            bvh, rays
        )
    var best = Int.MAX
    for _ in range(REPEATS):
        var t0 = perf_counter_ns()
        comptime if length == 1:
            summary = trace_scalar[bounds_width, leaf_width, unmasked_scalar](
                bvh, rays
            )
        else:
            summary = trace_packet[
                bounds_width, leaf_width, length, common_octant
            ](bvh, rays)
        var elapsed = Int(perf_counter_ns() - t0)
        best = min(best, elapsed)
    return PacketTiming(best, summary[0], summary[1])


def print_timing(label: String, result: PacketTiming, ray_count: Int):
    print(
        t"  {label}: {round(ns_to_ms(result.ns), 3)} ms, "
        t"{round(ns_to_mrays_per_s(result.ns, ray_count), 3)} MRay/s, "
        t"hits={result.hits}, checksum={round(result.checksum, 3)}"
    )


def benchmark_scene[
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
    split_method: String,
](
    label: String,
    vertices: List[Point3f32[Frame.WORLD]],
    rays: List[Rayf32[Frame.WORLD]],
):
    print(
        t"\n{label} / {split_method} / BVH{Int(bounds_width)} "
        t"leaf{Int(leaf_width)}"
    )
    var bvh = TriangleBvh[Frame.WORLD, bounds_width, leaf_width].__init__[
        split_method
    ](vertices)
    print_timing(
        "scalar",
        benchmark[bounds_width, leaf_width, 1](bvh, rays),
        len(rays),
    )
    print_timing(
        "unmasked-scalar",
        benchmark[bounds_width, leaf_width, 1, False, True](bvh, rays),
        len(rays),
    )
    print_timing(
        "packet4",
        benchmark[bounds_width, leaf_width, 4](bvh, rays),
        len(rays),
    )
    print_timing(
        "packet8",
        benchmark[bounds_width, leaf_width, 8](bvh, rays),
        len(rays),
    )
    print_timing(
        "packet16",
        benchmark[bounds_width, leaf_width, 16](bvh, rays),
        len(rays),
    )
    print_timing(
        "coh-packet4",
        benchmark[bounds_width, leaf_width, 4, True](bvh, rays),
        len(rays),
    )
    print_timing(
        "coh-packet8",
        benchmark[bounds_width, leaf_width, 8, True](bvh, rays),
        len(rays),
    )
    print_timing(
        "coh-packet16",
        benchmark[bounds_width, leaf_width, 16, True](bvh, rays),
        len(rays),
    )


def main() raises:
    print("CPU shared-stack packet BVH benchmark")
    var grid_vertices = make_grid_triangles()
    var grid_rays = make_hit_and_miss_rays()
    benchmark_scene[16, 16, "median"]("Regular grid", grid_vertices, grid_rays)
    benchmark_scene[8, 8, "median"]("Regular grid", grid_vertices, grid_rays)
    benchmark_scene[16, 16, "lbvh"]("Regular grid", grid_vertices, grid_rays)
    benchmark_scene[8, 8, "lbvh"]("Regular grid", grid_vertices, grid_rays)
    benchmark_scene[16, 16, "hploc"]("Regular grid", grid_vertices, grid_rays)

    var dragon_vertices = pack_obj_triangles[Frame.WORLD](OBJ_PATH)
    var bounds = compute_bounds(dragon_vertices)
    var camera = make_camera_rays_and_params(
        bounds, DRAGON_WIDTH, DRAGON_HEIGHT, 1, FOV_SCALE
    )
    var dragon_rays = camera[0].copy()
    benchmark_scene[16, 16, "sah"](
        "Dragon camera rays", dragon_vertices, dragon_rays
    )
    benchmark_scene[16, 16, "hploc"](
        "Dragon camera rays", dragon_vertices, dragon_rays
    )
