"""Octant queue and packet-width tail-compaction traversal benchmark."""

from std.benchmark import keep
from std.math import round
from std.time import perf_counter_ns

from bajo.bvh.constants import f32_max
from bajo.bvh.cpu import CpuBlasSet, CpuBvhBuildMethod
from bajo.bvh.cpu.blas_set import (
    _trace_blas_set_packet_policy,
    build_cpu_triangle_blas_set,
    trace_blas_set,
)
from bajo.bvh.cpu.triangle_bvh import TrianglePacketConfig
from bajo.bvh.types import Hit
from bajo.core import Point3, Point3f32, Ray, Rayf32, Vec3, Vec3f32
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.parser.obj.pack import pack_obj_triangles


comptime OBJ_PATH = "./assets/dragon/dragon.obj"
comptime BOUNDS_WIDTH = 16
comptime RAY_COUNT = 262144
comptime REPEATS = 7


@fieldwise_init
struct Timing(Copyable):
    var ns: Int
    var checksum: Float64
    var hits: Int


@always_inline
def _accumulate_hit[
    length: SIMDLength
](
    packet_hit: Hit[.WORLD, length],
    lane_count: Int,
    mut checksum: Float64,
    mut hits: Int,
):
    comptime for lane in range(length):
        if lane < lane_count and packet_hit.prim[lane] != UInt32(0xFFFFFFFF):
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


@always_inline
def _trace_chunk[
    length: SIMDLength,
    common_octant: Bool,
    config: TrianglePacketConfig = .PURE,
](
    bvh: CpuBlasSet[.TRIANGLE, BOUNDS_WIDTH, BOUNDS_WIDTH],
    rays: List[Rayf32[.WORLD]],
    base: Int,
    lane_count: Int,
) -> Hit[.WORLD, length]:
    var ox = SIMD[.float32, length](0.0)
    var oy = SIMD[.float32, length](0.0)
    var oz = SIMD[.float32, length](0.0)
    var dx = SIMD[.float32, length](0.0)
    var dy = SIMD[.float32, length](0.0)
    var dz = SIMD[.float32, length](1.0)
    var t_min = SIMD[.float32, length](0.0)
    var t_max = SIMD[.float32, length](f32_max)
    var valid = SIMD[.bool, length](fill=False)
    comptime for lane in range(length):
        if lane < lane_count:
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
    var packet = Ray[.float32, .WORLD, length](
        Point3[.float32, .WORLD, length](ox, oy, oz),
        Vec3[.float32, .WORLD, length](dx, dy, dz),
        t_min,
        t_max,
    )
    return _trace_blas_set_packet_policy[
        BOUNDS_WIDTH,
        BOUNDS_WIDTH,
        length,
        common_octant,
        .WORLD,
        config,
    ](bvh, UInt32(0), packet, valid)


@always_inline
def _trace_one_scalar(
    bvh: CpuBlasSet[.TRIANGLE, BOUNDS_WIDTH, BOUNDS_WIDTH],
    ray: Rayf32[.WORLD],
    mut checksum: Float64,
    mut hits: Int,
):
    var hit = trace_blas_set[
        BOUNDS_WIDTH,
        BOUNDS_WIDTH,
        .CLOSEST_HIT,
        .WORLD,
    ](bvh, UInt32(0), ray)
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


def _trace_scalar(
    bvh: CpuBlasSet[.TRIANGLE, BOUNDS_WIDTH, BOUNDS_WIDTH],
    rays: List[Rayf32[.WORLD]],
) -> Tuple[Float64, Int]:
    var checksum = Float64(0.0)
    var hits = 0
    for ray in rays:
        _trace_one_scalar(bvh, ray, checksum, hits)
    return (checksum, hits)


def _trace_packet16_noncoherent[
    config: TrianglePacketConfig = .PURE,
](
    bvh: CpuBlasSet[.TRIANGLE, BOUNDS_WIDTH, BOUNDS_WIDTH],
    rays: List[Rayf32[.WORLD]],
) -> Tuple[Float64, Int]:
    var checksum = Float64(0.0)
    var hits = 0
    for base in range(0, len(rays), 16):
        var lane_count = min(16, len(rays) - base)
        var packet_hit = _trace_chunk[16, False, config](
            bvh, rays, base, lane_count
        )
        _accumulate_hit(packet_hit, lane_count, checksum, hits)
    return (checksum, hits)


@always_inline
def _octant(ray: Rayf32[.WORLD]) -> Int:
    return (
        Int(ray.d.x >= 0.0)
        | (Int(ray.d.y >= 0.0) << 1)
        | (Int(ray.d.z >= 0.0) << 2)
    )


def _trace_octant_queues[
    window_size: Int,
    compact_tails: Bool,
](
    bvh: CpuBlasSet[.TRIANGLE, BOUNDS_WIDTH, BOUNDS_WIDTH],
    rays: List[Rayf32[.WORLD]],
) -> Tuple[Float64, Int]:
    var buckets = [List[Rayf32[.WORLD]]() for _ in range(8)]
    var checksum = Float64(0.0)
    var hits = 0
    for window_first in range(0, len(rays), window_size):
        for bucket_idx in range(8):
            buckets[bucket_idx].clear()
        var window_end = min(window_first + window_size, len(rays))
        for ray_idx in range(window_first, window_end):
            ref ray = rays[ray_idx]
            buckets[_octant(ray)].append(ray)

        for bucket_idx in range(8):
            ref bucket = buckets[bucket_idx]
            var base = 0
            while base + 16 <= len(bucket):
                var packet_hit = _trace_chunk[16, True](bvh, bucket, base, 16)
                _accumulate_hit(packet_hit, 16, checksum, hits)
                base += 16
            comptime if compact_tails:
                if base + 8 <= len(bucket):
                    var packet_hit = _trace_chunk[8, True](bvh, bucket, base, 8)
                    _accumulate_hit(packet_hit, 8, checksum, hits)
                    base += 8
                if base + 4 <= len(bucket):
                    var packet_hit = _trace_chunk[4, True](bvh, bucket, base, 4)
                    _accumulate_hit(packet_hit, 4, checksum, hits)
                    base += 4
                while base < len(bucket):
                    _trace_one_scalar(bvh, bucket[base], checksum, hits)
                    base += 1
            else:
                if base < len(bucket):
                    var lane_count = len(bucket) - base
                    var packet_hit = _trace_chunk[16, True](
                        bvh, bucket, base, lane_count
                    )
                    _accumulate_hit(packet_hit, lane_count, checksum, hits)
    return (checksum, hits)


def _make_eight_octant_rays(
    vertices: List[Point3f32[.WORLD]], count: Int
) -> List[Rayf32[.WORLD]]:
    var triangle_count = len(vertices) // 3
    var rays = List[Rayf32[.WORLD]](capacity=count)
    comptime direction_scale = Float32(0.5773502691896258)
    for i in range(count):
        var prim_idx = (i * 7919) % triangle_count
        var center = (
            vertices[prim_idx * 3]
            .unsafe_add(vertices[prim_idx * 3 + 1])
            .unsafe_add(vertices[prim_idx * 3 + 2])
            / 3.0
        )
        var octant = i & 7
        var sx = -direction_scale
        var sy = -direction_scale
        var sz = -direction_scale
        if (octant & 1) != 0:
            sx = direction_scale
        if (octant & 2) != 0:
            sy = direction_scale
        if (octant & 4) != 0:
            sz = direction_scale
        var direction = Vec3f32[.WORLD](sx, sy, sz)
        rays.append(Rayf32[.WORLD](center - direction * 100.0, direction))
    return rays^


def _benchmark[
    method: Int,
](
    bvh: CpuBlasSet[.TRIANGLE, BOUNDS_WIDTH, BOUNDS_WIDTH],
    rays: List[Rayf32[.WORLD]],
) -> Timing:
    var summary: Tuple[Float64, Int]
    comptime if method == 0:
        summary = _trace_scalar(bvh, rays)
    elif method == 1:
        summary = _trace_packet16_noncoherent(bvh, rays)
    elif method == 2:
        summary = _trace_octant_queues[64, False](bvh, rays)
    elif method == 3:
        summary = _trace_octant_queues[64, True](bvh, rays)
    elif method == 4:
        summary = _trace_octant_queues[256, False](bvh, rays)
    elif method == 5:
        summary = _trace_octant_queues[256, True](bvh, rays)
    elif method == 6:
        summary = _trace_octant_queues[1024, False](bvh, rays)
    elif method == 7:
        summary = _trace_octant_queues[1024, True](bvh, rays)
    elif method == 8:
        summary = _trace_octant_queues[8192, False](bvh, rays)
    elif method == 9:
        summary = _trace_octant_queues[8192, True](bvh, rays)
    elif method == 10:
        summary = _trace_octant_queues[RAY_COUNT, False](bvh, rays)
    elif method == 11:
        summary = _trace_octant_queues[RAY_COUNT, True](bvh, rays)
    elif method == 12:
        summary = _trace_packet16_noncoherent[TrianglePacketConfig.PRODUCTION](
            bvh, rays
        )
    elif method == 13:
        summary = _trace_packet16_noncoherent[
            TrianglePacketConfig.scalar_continuation[4]()
        ](bvh, rays)
    elif method == 14:
        summary = _trace_packet16_noncoherent[
            TrianglePacketConfig.scalar_continuation[8]()
        ](bvh, rays)
    elif method == 15:
        summary = _trace_packet16_noncoherent[
            TrianglePacketConfig.scalar_continuation[12]()
        ](bvh, rays)
    elif method == 16:
        summary = _trace_packet16_noncoherent[
            TrianglePacketConfig.scalar_continuation[15]()
        ](bvh, rays)
    elif method == 17:
        summary = _trace_packet16_noncoherent[
            TrianglePacketConfig.scalar_both[8]()
        ](bvh, rays)
    else:
        comptime assert False
    var best = Int.MAX
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        comptime if method == 0:
            summary = _trace_scalar(bvh, rays)
        elif method == 1:
            summary = _trace_packet16_noncoherent(bvh, rays)
        elif method == 2:
            summary = _trace_octant_queues[64, False](bvh, rays)
        elif method == 3:
            summary = _trace_octant_queues[64, True](bvh, rays)
        elif method == 4:
            summary = _trace_octant_queues[256, False](bvh, rays)
        elif method == 5:
            summary = _trace_octant_queues[256, True](bvh, rays)
        elif method == 6:
            summary = _trace_octant_queues[1024, False](bvh, rays)
        elif method == 7:
            summary = _trace_octant_queues[1024, True](bvh, rays)
        elif method == 8:
            summary = _trace_octant_queues[8192, False](bvh, rays)
        elif method == 9:
            summary = _trace_octant_queues[8192, True](bvh, rays)
        elif method == 10:
            summary = _trace_octant_queues[RAY_COUNT, False](bvh, rays)
        elif method == 11:
            summary = _trace_octant_queues[RAY_COUNT, True](bvh, rays)
        elif method == 12:
            summary = _trace_packet16_noncoherent[
                TrianglePacketConfig.PRODUCTION
            ](bvh, rays)
        elif method == 13:
            summary = _trace_packet16_noncoherent[
                TrianglePacketConfig.scalar_continuation[4]()
            ](bvh, rays)
        elif method == 14:
            summary = _trace_packet16_noncoherent[
                TrianglePacketConfig.scalar_continuation[8]()
            ](bvh, rays)
        elif method == 15:
            summary = _trace_packet16_noncoherent[
                TrianglePacketConfig.scalar_continuation[12]()
            ](bvh, rays)
        elif method == 16:
            summary = _trace_packet16_noncoherent[
                TrianglePacketConfig.scalar_continuation[15]()
            ](bvh, rays)
        elif method == 17:
            summary = _trace_packet16_noncoherent[
                TrianglePacketConfig.scalar_both[8]()
            ](bvh, rays)
        best = min(best, Int(perf_counter_ns() - start))
    keep(summary[0])
    keep(summary[1])
    return Timing(best, summary[0], summary[1])


def _print(label: String, timing: Timing):
    print(
        t"{label}\t{round(ns_to_ms(timing.ns), 3)}\t"
        t"{round(ns_to_mrays_per_s(timing.ns, RAY_COUNT), 3)}\t"
        t"{timing.hits}\t{round(timing.checksum, 3)}"
    )


def main() raises:
    var vertices = pack_obj_triangles[.WORLD](OBJ_PATH)
    var rays = _make_eight_octant_rays(vertices, RAY_COUNT)
    var bvh = build_cpu_triangle_blas_set[
        BOUNDS_WIDTH,
        BOUNDS_WIDTH,
        CpuBvhBuildMethod.SAH,
        .WORLD,
    ]([vertices^])
    print("Dragon eight-octant ray queue benchmark; best of 7")
    print("Policy\tTrace ms\tMRay/s\tHits\tChecksum")
    _print("scalar", _benchmark[0](bvh, rays))
    _print("packet16-noncoherent", _benchmark[1](bvh, rays))
    _print("packet16-production", _benchmark[12](bvh, rays))
    _print("scalar-continuation-4", _benchmark[13](bvh, rays))
    _print("scalar-continuation-8", _benchmark[14](bvh, rays))
    _print("scalar-continuation-12", _benchmark[15](bvh, rays))
    _print("scalar-continuation-15", _benchmark[16](bvh, rays))
    _print("scalar-both-8", _benchmark[17](bvh, rays))
    _print("octant64-packet16", _benchmark[2](bvh, rays))
    _print("octant64-compact-tail", _benchmark[3](bvh, rays))
    _print("octant256-packet16", _benchmark[4](bvh, rays))
    _print("octant256-compact-tail", _benchmark[5](bvh, rays))
    _print("octant1024-packet16", _benchmark[6](bvh, rays))
    _print("octant1024-compact-tail", _benchmark[7](bvh, rays))
    _print("octant8192-packet16", _benchmark[8](bvh, rays))
    _print("octant8192-compact-tail", _benchmark[9](bvh, rays))
    _print("octant-global-packet16", _benchmark[10](bvh, rays))
    _print("octant-global-compact-tail", _benchmark[11](bvh, rays))
