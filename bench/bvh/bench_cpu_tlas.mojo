"""CPU TLAS leaf-width comparison on the viewer's LBVH scene."""

from std.benchmark import keep
from std.math import round
from std.sys import simd_width_of
from std.time import perf_counter_ns

from bajo.bvh.constants import TraceMode
from bajo.bvh.cpu.tlas import CpuTlas
from bajo.bvh.cpu import CpuBlasSet
from bajo.bvh.types import Instance
from bajo.core import Point3, Ray, Rayf32, Vec3
from bajo.core.utils import ns_to_ms, ns_to_mrays_per_s
from bajo.benchmark.bvh_reporting import TablePrinter
from bajo.benchmark.timing import TimingSummary, summarize_timings
from examples.lbvh_scene import make_lbvh_camera, make_lbvh_world


comptime IMAGE_WIDTH = 320
comptime IMAGE_HEIGHT = 214
comptime WARMUPS = 2
comptime PAIRED_REPEATS = 12


@fieldwise_init
struct TraceChecksum(Copyable):
    var distance: Float64
    var hits: UInt64
    var instances: UInt64


def _trace[
    width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TraceMode,
](
    tlas: CpuTlas[width, leaf_width],
    blases: CpuBlasSet[.TRIANGLE, width],
    rays: List[Rayf32[.WORLD]],
) -> TraceChecksum:
    var distance = Float64(0.0)
    var hits = UInt64(0)
    var instances = UInt64(0)
    for ray in rays:
        var hit = tlas.trace_blases[width, width, mode](ray, blases)
        comptime if mode == .ANY_HIT:
            if hit.is_occluded():
                hits += 1
        else:
            if hit.is_hit():
                distance += Float64(hit.t)
                hits += 1
                instances += UInt64(hit.inst)

    keep(distance)
    keep(hits)
    keep(instances)
    return TraceChecksum(distance, hits, instances)


def _timed_trace[
    width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TraceMode,
](
    tlas: CpuTlas[width, leaf_width],
    blases: CpuBlasSet[.TRIANGLE, width],
    rays: List[Rayf32[.WORLD]],
) -> Int:
    var start = perf_counter_ns()
    _ = _trace[width, leaf_width, mode](tlas, blases, rays)
    return Int(perf_counter_ns() - start)


def _trace_packet[
    width: SIMDLength,
    length: SIMDLength,
    simd_normal_transforms: Bool = False,
](
    tlas: CpuTlas[width, 1],
    blases: CpuBlasSet[.TRIANGLE, width],
    rays: List[Rayf32[.WORLD]],
) -> TraceChecksum:
    var distance = Float64(0.0)
    var hits = UInt64(0)
    var instances = UInt64(0)
    for base in range(0, len(rays), length):
        var ox = SIMD[.float32, length](0.0)
        var oy = SIMD[.float32, length](0.0)
        var oz = SIMD[.float32, length](0.0)
        var dx = SIMD[.float32, length](0.0)
        var dy = SIMD[.float32, length](0.0)
        var dz = SIMD[.float32, length](1.0)
        var t_min = SIMD[.float32, length](0.0)
        var t_max = SIMD[.float32, length](0.0)
        var valid = SIMD[.bool, length](fill=False)
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
        var packet = Ray[.float32, .WORLD, length](
            Point3[.float32, .WORLD, length](ox, oy, oz),
            Vec3[.float32, .WORLD, length](dx, dy, dz),
            t_min,
            t_max,
        )
        var packet_hit = tlas.trace_blases_packet[
            width,
            width,
            length,
            simd_normal_transforms,
        ](packet, blases, valid)
        for lane in range(lane_count):
            if packet_hit.is_hit()[lane]:
                distance += Float64(packet_hit.t[lane])
                hits += 1
                instances += UInt64(packet_hit.inst[lane])
    keep(distance)
    keep(hits)
    keep(instances)
    return TraceChecksum(distance, hits, instances)


def _benchmark_packet[
    width: SIMDLength,
](
    tlas: CpuTlas[width, 1],
    blases: CpuBlasSet[.TRIANGLE, width],
    rays: List[Rayf32[.WORLD]],
):
    comptime length = 16
    _ = _trace_packet[width, length, False](tlas, blases, rays)
    _ = _trace_packet[width, length, True](tlas, blases, rays)
    var scalar_times = List[Int](capacity=PAIRED_REPEATS)
    var simd_times = List[Int](capacity=PAIRED_REPEATS)
    for pair in range(PAIRED_REPEATS):
        var start = perf_counter_ns()
        if pair % 2 == 0:
            _ = _trace_packet[width, length, False](tlas, blases, rays)
            scalar_times.append(Int(perf_counter_ns() - start))
            start = perf_counter_ns()
            _ = _trace_packet[width, length, True](tlas, blases, rays)
            simd_times.append(Int(perf_counter_ns() - start))
        else:
            _ = _trace_packet[width, length, True](tlas, blases, rays)
            simd_times.append(Int(perf_counter_ns() - start))
            start = perf_counter_ns()
            _ = _trace_packet[width, length, False](tlas, blases, rays)
            scalar_times.append(Int(perf_counter_ns() - start))
    var scalar_summary = summarize_timings(scalar_times)
    var simd_summary = summarize_timings(simd_times)
    var scalar_checksum = _trace_packet[width, length, False](
        tlas, blases, rays
    )
    var simd_checksum = _trace_packet[width, length, True](tlas, blases, rays)
    print("\nPacket TLAS; scalar vs SIMD normal finalization")
    print(
        t"scalar median={round(ns_to_ms(scalar_summary.median_ns), 3)} ms, "
        t"simd median={round(ns_to_ms(simd_summary.median_ns), 3)} ms, "
        t"delta={round(Float64(simd_summary.median_ns - scalar_summary.median_ns) * 100.0 / Float64(scalar_summary.median_ns), 3)}%"
    )
    print(
        t"scalar hits/checksum/inst={scalar_checksum.hits}/"
        t"{round(scalar_checksum.distance, 3)}/{scalar_checksum.instances}; "
        t"simd={simd_checksum.hits}/"
        t"{round(simd_checksum.distance, 3)}/{simd_checksum.instances}"
    )


def _trace_packet_any_hit_reference[
    width: SIMDLength,
    length: SIMDLength,
](
    tlas: CpuTlas[width, 1],
    blases: CpuBlasSet[.TRIANGLE, width],
    rays: List[Rayf32[.WORLD]],
) -> UInt64:
    var hits = UInt64(0)
    for base in range(0, len(rays), length):
        var lane_count = min(length, len(rays) - base)
        for lane in range(lane_count):
            if tlas.trace_blases[width, width, .ANY_HIT](
                rays[base + lane], blases
            ).is_occluded():
                hits += 1
    keep(hits)
    return hits


def _trace_packet_any_hit[
    width: SIMDLength,
    length: SIMDLength,
](
    tlas: CpuTlas[width, 1],
    blases: CpuBlasSet[.TRIANGLE, width],
    rays: List[Rayf32[.WORLD]],
) -> UInt64:
    var hits = UInt64(0)
    for base in range(0, len(rays), length):
        var ox = SIMD[.float32, length](0.0)
        var oy = SIMD[.float32, length](0.0)
        var oz = SIMD[.float32, length](0.0)
        var dx = SIMD[.float32, length](0.0)
        var dy = SIMD[.float32, length](0.0)
        var dz = SIMD[.float32, length](1.0)
        var t_min = SIMD[.float32, length](0.0)
        var t_max = SIMD[.float32, length](0.0)
        var valid = SIMD[.bool, length](fill=False)
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
        var packet = Ray[.float32, .WORLD, length](
            Point3[.float32, .WORLD, length](ox, oy, oz),
            Vec3[.float32, .WORLD, length](dx, dy, dz),
            t_min,
            t_max,
        )
        var occluded = tlas.trace_blases_packet_any_hit[width, width, length](
            packet, blases, valid
        )
        hits += UInt64(occluded.cast[.uint64]().reduce_add())
    keep(hits)
    return hits


def _timed_packet_any_hit_reference[
    width: SIMDLength,
    length: SIMDLength,
](
    tlas: CpuTlas[width, 1],
    blases: CpuBlasSet[.TRIANGLE, width],
    rays: List[Rayf32[.WORLD]],
) -> Int:
    var start = perf_counter_ns()
    _ = _trace_packet_any_hit_reference[width, length](tlas, blases, rays)
    return Int(perf_counter_ns() - start)


def _timed_packet_any_hit[
    width: SIMDLength,
    length: SIMDLength,
](
    tlas: CpuTlas[width, 1],
    blases: CpuBlasSet[.TRIANGLE, width],
    rays: List[Rayf32[.WORLD]],
) -> Int:
    var start = perf_counter_ns()
    _ = _trace_packet_any_hit[width, length](tlas, blases, rays)
    return Int(perf_counter_ns() - start)


def _benchmark_packet_any_hit[
    width: SIMDLength,
](
    tlas: CpuTlas[width, 1],
    blases: CpuBlasSet[.TRIANGLE, width],
    rays: List[Rayf32[.WORLD]],
):
    comptime length = 16
    _ = _trace_packet_any_hit_reference[width, length](tlas, blases, rays)
    _ = _trace_packet_any_hit[width, length](tlas, blases, rays)
    var reference_times = List[Int](capacity=PAIRED_REPEATS)
    var packet_times = List[Int](capacity=PAIRED_REPEATS)
    for pair in range(PAIRED_REPEATS):
        if pair % 2 == 0:
            reference_times.append(
                _timed_packet_any_hit_reference[width, length](
                    tlas, blases, rays
                )
            )
            packet_times.append(
                _timed_packet_any_hit[width, length](tlas, blases, rays)
            )
        else:
            packet_times.append(
                _timed_packet_any_hit[width, length](tlas, blases, rays)
            )
            reference_times.append(
                _timed_packet_any_hit_reference[width, length](
                    tlas, blases, rays
                )
            )
    var reference = summarize_timings(reference_times)
    var packet = summarize_timings(packet_times)
    var reference_hits = _trace_packet_any_hit_reference[width, length](
        tlas, blases, rays
    )
    var packet_hits = _trace_packet_any_hit[width, length](tlas, blases, rays)
    print("\nPacket TLAS any-hit; scalar-per-lane reference vs shared stack")
    print(
        t"reference median={round(ns_to_ms(reference.median_ns), 3)} ms,"
        t" packet16 median={round(ns_to_ms(packet.median_ns), 3)} ms,"
        t" delta={round(Float64(packet.median_ns - reference.median_ns) * 100.0 / Float64(reference.median_ns), 3)}%,"
        t" hits={reference_hits}/{packet_hits}"
    )


def _benchmark_refit[
    width: SIMDLength,
](instances: List[Instance]) raises:
    var tlas = CpuTlas[width, 1](instances)
    tlas.refit(instances)
    var warm_rebuild = CpuTlas[width, 1](instances)
    keep(warm_rebuild.bounds().surface_area())

    var rebuild_times = List[Int](capacity=PAIRED_REPEATS)
    var refit_times = List[Int](capacity=PAIRED_REPEATS)
    for pair in range(PAIRED_REPEATS):
        if pair % 2 == 0:
            var start = perf_counter_ns()
            var rebuilt = CpuTlas[width, 1](instances)
            rebuild_times.append(Int(perf_counter_ns() - start))
            keep(rebuilt.bounds().surface_area())
            start = perf_counter_ns()
            tlas.refit(instances)
            refit_times.append(Int(perf_counter_ns() - start))
        else:
            var start = perf_counter_ns()
            tlas.refit(instances)
            refit_times.append(Int(perf_counter_ns() - start))
            start = perf_counter_ns()
            var rebuilt = CpuTlas[width, 1](instances)
            rebuild_times.append(Int(perf_counter_ns() - start))
            keep(rebuilt.bounds().surface_area())

    var rebuild = summarize_timings(rebuild_times)
    var refit = summarize_timings(refit_times)
    print(
        "\nDynamic TLAS update; full LBVH rebuild vs topology-preserving refit"
    )
    print(
        t"rebuild median={round(ns_to_ms(rebuild.median_ns), 6)} ms, "
        t"refit median={round(ns_to_ms(refit.median_ns), 6)} ms, "
        t"delta={round(Float64(refit.median_ns - rebuild.median_ns) * 100.0 / Float64(rebuild.median_ns), 3)}%"
    )


def _print_row(
    table: TablePrinter,
    label: String,
    summary: TimingSummary,
    ray_count: Int,
    delta: Float64,
    checksum: TraceChecksum,
) raises:
    table.result_line(
        layout=label,
        median_ms=String(round(ns_to_ms(summary.median_ns), 3)),
        min_ms=String(round(ns_to_ms(summary.min_ns), 3)),
        max_ms=String(round(ns_to_ms(summary.max_ns), 3)),
        MRay_s=String(
            round(ns_to_mrays_per_s(summary.median_ns, ray_count), 3)
        ),
        delta_pct=String(round(delta, 3)),
        hits=String(checksum.hits),
        checksum=String(round(checksum.distance, 3)),
        inst_sum=String(checksum.instances),
    )


def _benchmark_mode[
    width: SIMDLength,
    mode: TraceMode,
](
    label: String,
    tlas_leaf1: CpuTlas[width, 1],
    tlas_native: CpuTlas[width, width],
    blases: CpuBlasSet[.TRIANGLE, width],
    rays: List[Rayf32[.WORLD]],
) raises:
    for _ in range(WARMUPS):
        _ = _trace[width, 1, mode](tlas_leaf1, blases, rays)
        _ = _trace[width, width, mode](tlas_native, blases, rays)

    var leaf1_times = List[Int](capacity=PAIRED_REPEATS)
    var native_times = List[Int](capacity=PAIRED_REPEATS)
    for pair in range(PAIRED_REPEATS):
        if pair % 2 == 0:
            leaf1_times.append(
                _timed_trace[width, 1, mode](tlas_leaf1, blases, rays)
            )
            native_times.append(
                _timed_trace[width, width, mode](tlas_native, blases, rays)
            )
        else:
            native_times.append(
                _timed_trace[width, width, mode](tlas_native, blases, rays)
            )
            leaf1_times.append(
                _timed_trace[width, 1, mode](tlas_leaf1, blases, rays)
            )

    var leaf1_summary = summarize_timings(leaf1_times)
    var native_summary = summarize_timings(native_times)
    var leaf1_checksum = _trace[width, 1, mode](tlas_leaf1, blases, rays)
    var native_checksum = _trace[width, width, mode](tlas_native, blases, rays)
    var delta = (
        Float64(leaf1_summary.median_ns - native_summary.median_ns)
        * 100.0
        / Float64(native_summary.median_ns)
    )

    print(t"\n{label}")
    var table = TablePrinter(
        layout=18,
        median_ms=10,
        min_ms=10,
        max_ms=10,
        MRay_s=10,
        delta_pct=10,
        hits=8,
        checksum=16,
        inst_sum=12,
    )
    table.header()
    _print_row(
        table,
        String(t"TLAS{Int(width)}/leaf1"),
        leaf1_summary,
        len(rays),
        delta,
        leaf1_checksum,
    )
    _print_row(
        table,
        String(t"TLAS{Int(width)}/leaf{Int(width)}"),
        native_summary,
        len(rays),
        0.0,
        native_checksum,
    )
    print(
        t"checksum delta:"
        t" {round(leaf1_checksum.distance - native_checksum.distance, 6)}, hit"
        t" delta: {Int(leaf1_checksum.hits) - Int(native_checksum.hits)},"
        t" instance delta:"
        t" {Int(leaf1_checksum.instances) - Int(native_checksum.instances)}"
    )


def run_benchmark() raises:
    comptime width = simd_width_of[DType.float32]()
    print("CPU TLAS leaf-width benchmark")
    print(
        t"LBVH viewer scene, {IMAGE_WIDTH}x{IMAGE_HEIGHT}, native float32"
        t" SIMD={Int(width)}"
    )
    print(t"2 warmups + {PAIRED_REPEATS} alternating pairs")

    var world = make_lbvh_world[width, width]()
    var camera = make_lbvh_camera()
    var rays = List[Rayf32[.WORLD]](capacity=IMAGE_WIDTH * IMAGE_HEIGHT)
    for py in range(IMAGE_HEIGHT):
        for px in range(IMAGE_WIDTH):
            rays.append(camera.make_ray(px, py, IMAGE_WIDTH, IMAGE_HEIGHT))

    var build_start = perf_counter_ns()
    var tlas_leaf1 = CpuTlas[width, 1](world.scene_data().triangle_instances())
    var leaf1_build_ns = Int(perf_counter_ns() - build_start)
    build_start = perf_counter_ns()
    var tlas_native = CpuTlas[width, width](
        world.scene_data().triangle_instances()
    )
    var native_build_ns = Int(perf_counter_ns() - build_start)

    print(
        t"instances={len(world.scene_data().triangle_instances())},"
        t" rays={len(rays)}"
    )
    print(
        t"build leaf1/native: {round(ns_to_ms(leaf1_build_ns), 3)} / "
        t"{round(ns_to_ms(native_build_ns), 3)} ms"
    )
    _benchmark_refit[width](world.scene_data().triangle_instances())
    _benchmark_mode[width, .CLOSEST_HIT](
        "Primary closest-hit",
        tlas_leaf1,
        tlas_native,
        world.triangle_mesh_blases.value(),
        rays,
    )
    _benchmark_mode[width, .ANY_HIT](
        "Primary any-hit",
        tlas_leaf1,
        tlas_native,
        world.triangle_mesh_blases.value(),
        rays,
    )
    _benchmark_packet[width](
        tlas_leaf1, world.triangle_mesh_blases.value(), rays
    )
    _benchmark_packet_any_hit[width](
        tlas_leaf1, world.triangle_mesh_blases.value(), rays
    )


def main() raises:
    run_benchmark()
