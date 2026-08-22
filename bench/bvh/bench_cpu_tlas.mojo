"""CPU TLAS leaf-width comparison on the viewer's LBVH scene."""

from std.benchmark import keep
from std.math import round
from std.sys import simd_width_of
from std.time import perf_counter_ns

from bajo.bvh.constants import TRACE
from bajo.bvh.cpu.tlas import Tlas
from bajo.bvh.cpu import CpuBlasSet
from bajo.core import Frame, Rayf32
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
    mode: TRACE,
](
    tlas: Tlas[width, leaf_width],
    blases: CpuBlasSet[width],
    rays: List[Rayf32[Frame.WORLD]],
) -> TraceChecksum:
    var distance = Float64(0.0)
    var hits = UInt64(0)
    var instances = UInt64(0)
    for ray in rays:
        var hit = tlas.trace_triangle_blases[width, width, mode](ray, blases)
        comptime if mode == TRACE.ANY_HIT:
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
    mode: TRACE,
](
    tlas: Tlas[width, leaf_width],
    blases: CpuBlasSet[width],
    rays: List[Rayf32[Frame.WORLD]],
) -> Int:
    var start = perf_counter_ns()
    _ = _trace[width, leaf_width, mode](tlas, blases, rays)
    return Int(perf_counter_ns() - start)


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
    mode: TRACE,
](
    label: String,
    tlas_leaf1: Tlas[width, 1],
    tlas_native: Tlas[width, width],
    blases: CpuBlasSet[width],
    rays: List[Rayf32[Frame.WORLD]],
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
    var rays = List[Rayf32[Frame.WORLD]](capacity=IMAGE_WIDTH * IMAGE_HEIGHT)
    for py in range(IMAGE_HEIGHT):
        for px in range(IMAGE_WIDTH):
            rays.append(camera.make_ray(px, py, IMAGE_WIDTH, IMAGE_HEIGHT))

    var build_start = perf_counter_ns()
    var tlas_leaf1 = Tlas[width, 1](world.scene_data().triangle_instances)
    var leaf1_build_ns = Int(perf_counter_ns() - build_start)
    build_start = perf_counter_ns()
    var tlas_native = Tlas[width, width](world.scene_data().triangle_instances)
    var native_build_ns = Int(perf_counter_ns() - build_start)

    print(
        t"instances={len(world.scene_data().triangle_instances)},"
        t" rays={len(rays)}"
    )
    print(
        t"build leaf1/native: {round(ns_to_ms(leaf1_build_ns), 3)} / "
        t"{round(ns_to_ms(native_build_ns), 3)} ms"
    )
    _benchmark_mode[width, TRACE.CLOSEST_HIT](
        "Primary closest-hit",
        tlas_leaf1,
        tlas_native,
        world.triangle_mesh_blases.value(),
        rays,
    )
    _benchmark_mode[width, TRACE.ANY_HIT](
        "Primary any-hit",
        tlas_leaf1,
        tlas_native,
        world.triangle_mesh_blases.value(),
        rays,
    )


def main() raises:
    run_benchmark()
