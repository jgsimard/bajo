"""Packet visibility-query benchmark for the production CpuScene path."""

from std.math import round
from std.time import perf_counter_ns

from bajo.benchmark.rt_fixtures import (
    make_bounded_grid_rays,
    make_grid_triangle_world,
)
from bajo.benchmark.bvh_fixtures import make_grid_triangles
from bajo.bvh.cpu import (
    CpuBlasSet,
    CpuBvhBuildMethod,
    build_cpu_triangle_blas_set,
    trace_blas_set_packet,
    trace_blas_set_packet_any_hit,
)
from bajo.core import Point3, Ray, Rayf32, Vec3
from bajo.core.utils import ns_to_mrays_per_s
from bajo.rt import CpuScene


comptime PACKET_WIDTH = 16
comptime REPEATS = 8


@fieldwise_init
struct Timing(Copyable):
    var ns: Int
    var hits: Int


def trace_scalar(world: CpuScene[16, 16], rays: List[Rayf32[.WORLD]]) -> Int:
    var hits = 0
    for ray in rays:
        if world.occluded(ray):
            hits += 1
    return hits


def trace_packet(world: CpuScene[16, 16], rays: List[Rayf32[.WORLD]]) -> Int:
    var hits = 0
    for base in range(0, len(rays), PACKET_WIDTH):
        var ox = SIMD[.float32, PACKET_WIDTH](0.0)
        var oy = SIMD[.float32, PACKET_WIDTH](0.0)
        var oz = SIMD[.float32, PACKET_WIDTH](0.0)
        var dx = SIMD[.float32, PACKET_WIDTH](0.0)
        var dy = SIMD[.float32, PACKET_WIDTH](0.0)
        var dz = SIMD[.float32, PACKET_WIDTH](1.0)
        var t_min = SIMD[.float32, PACKET_WIDTH](0.0)
        var t_max = SIMD[.float32, PACKET_WIDTH](0.0)
        var valid = SIMD[.bool, PACKET_WIDTH](fill=False)
        var lane_count = min(PACKET_WIDTH, len(rays) - base)
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

        var packet = Ray[.float32, .WORLD, PACKET_WIDTH](
            Point3[.float32, .WORLD, PACKET_WIDTH](ox, oy, oz),
            Vec3[.float32, .WORLD, PACKET_WIDTH](dx, dy, dz),
            t_min,
            t_max,
        )
        var occluded = world.occluded[PACKET_WIDTH](packet, valid)
        for lane in range(lane_count):
            if occluded[lane]:
                hits += 1
    return hits


def trace_low_level_packet[
    any_hit: Bool
](
    bvh: CpuBlasSet[.TRIANGLE, PACKET_WIDTH, PACKET_WIDTH],
    rays: List[Rayf32[.WORLD]],
) -> Int:
    """Same-binary reference closest-hit versus specialized any-hit A/B."""
    var hits = 0
    for base in range(0, len(rays), PACKET_WIDTH):
        var ox = SIMD[.float32, PACKET_WIDTH](0.0)
        var oy = SIMD[.float32, PACKET_WIDTH](0.0)
        var oz = SIMD[.float32, PACKET_WIDTH](0.0)
        var dx = SIMD[.float32, PACKET_WIDTH](0.0)
        var dy = SIMD[.float32, PACKET_WIDTH](0.0)
        var dz = SIMD[.float32, PACKET_WIDTH](1.0)
        var t_min = SIMD[.float32, PACKET_WIDTH](0.0)
        var t_max = SIMD[.float32, PACKET_WIDTH](0.0)
        var valid = SIMD[.bool, PACKET_WIDTH](fill=False)
        var lane_count = min(PACKET_WIDTH, len(rays) - base)
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

        var packet = Ray[.float32, .WORLD, PACKET_WIDTH](
            Point3[.float32, .WORLD, PACKET_WIDTH](ox, oy, oz),
            Vec3[.float32, .WORLD, PACKET_WIDTH](dx, dy, dz),
            t_min,
            t_max,
        )
        var occluded: SIMD[.bool, PACKET_WIDTH]
        comptime if any_hit:
            occluded = trace_blas_set_packet_any_hit[
                PACKET_WIDTH,
                PACKET_WIDTH,
                PACKET_WIDTH,
                False,
                .WORLD,
            ](bvh, UInt32(0), packet, valid)
        else:
            occluded = trace_blas_set_packet[
                PACKET_WIDTH,
                PACKET_WIDTH,
                PACKET_WIDTH,
                False,
                .WORLD,
            ](bvh, UInt32(0), packet, valid).is_hit()
        for lane in range(lane_count):
            if occluded[lane]:
                hits += 1
    return hits


def benchmark_scalar(
    world: CpuScene[16, 16], rays: List[Rayf32[.WORLD]]
) -> Timing:
    var hits = trace_scalar(world, rays)
    var best = Int.MAX
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        hits = trace_scalar(world, rays)
        best = min(best, Int(perf_counter_ns() - start))
    return Timing(best, hits)


def benchmark_packet(
    world: CpuScene[16, 16], rays: List[Rayf32[.WORLD]]
) -> Timing:
    var hits = trace_packet(world, rays)
    var best = Int.MAX
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        hits = trace_packet(world, rays)
        best = min(best, Int(perf_counter_ns() - start))
    return Timing(best, hits)


def benchmark_low_level_packet[
    any_hit: Bool
](
    bvh: CpuBlasSet[.TRIANGLE, PACKET_WIDTH, PACKET_WIDTH],
    rays: List[Rayf32[.WORLD]],
) -> Timing:
    var hits = trace_low_level_packet[any_hit](bvh, rays)
    var best = Int.MAX
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        hits = trace_low_level_packet[any_hit](bvh, rays)
        best = min(best, Int(perf_counter_ns() - start))
    return Timing(best, hits)


def main() raises:
    var world = make_grid_triangle_world()
    var rays = make_bounded_grid_rays()
    var vertices = make_grid_triangles()
    var bvh = build_cpu_triangle_blas_set[
        PACKET_WIDTH,
        PACKET_WIDTH,
        CpuBvhBuildMethod.SAH,
        .WORLD,
    ]([vertices^])
    var scalar = benchmark_scalar(world, rays)
    var packet = benchmark_packet(world, rays)
    var reference = benchmark_low_level_packet[False](bvh, rays)
    var specialized = benchmark_low_level_packet[True](bvh, rays)

    print("CPU packet any-hit benchmark; regular triangle scene; best of 8")
    print(
        t"rays={len(rays)} scalar_hits={scalar.hits} packet_hits={packet.hits}"
    )
    print(
        t"scalar_ns={scalar.ns} scalar_mray_s="
        t"{round(ns_to_mrays_per_s(scalar.ns, len(rays)), 3)}"
    )
    print(
        t"packet16_ns={packet.ns} packet16_mray_s="
        t"{round(ns_to_mrays_per_s(packet.ns, len(rays)), 3)}"
    )
    print(
        t"reference_closest_ns={reference.ns} reference_hits={reference.hits}"
    )
    print(
        t"specialized_any_ns={specialized.ns} specialized_hits={specialized.hits}"
    )
