"""Depth-first CPU render timings and deterministic output checksums."""

from std.math import round
from std.time import perf_counter_ns

from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Integrator,
    Camera,
    RenderSettings,
    CpuScene,
    render_depth_first,
)
from examples.rtiaw import make_weekend_world
from bajo.benchmark.cpu_harness import pixel_checksum
from bajo.benchmark.rt_fixtures import (
    make_mixed_triangle_world,
    mixed_triangle_camera,
    weekend_camera,
)


comptime TIMING_WIDTH = 960
comptime TIMING_HEIGHT = 540
comptime TIMING_SPP = 8
comptime TIMING_REPEATS = 7
comptime MAX_DEPTH = 8
comptime RNG_SEED = UInt64(2026)


@fieldwise_init
struct TimingResult(Copyable):
    var median_total_ns: Int
    var median_init_ns: Int
    var median_render_ns: Int
    var min_render_ns: Int
    var max_render_ns: Int
    var checksum: Float64


def time_render[
    integrator: Integrator
](
    settings: RenderSettings, camera: Camera, world: CpuScene[16, 16]
) raises -> TimingResult:
    # Warm all compiled code and renderer-owned allocations before measuring.
    var warmup = render_depth_first[integrator](settings, camera, world)
    var checksum = pixel_checksum(warmup.pixels)
    var total_times = List[Int](capacity=TIMING_REPEATS)
    var init_times = List[Int](capacity=TIMING_REPEATS)
    var render_times = List[Int](capacity=TIMING_REPEATS)

    for _ in range(TIMING_REPEATS):
        var result = render_depth_first[integrator](settings, camera, world)
        var current_checksum = pixel_checksum(result.pixels)
        debug_assert["safe", _use_compiler_assume=True](
            current_checksum == checksum, "render checksum changed between runs"
        )
        total_times.append(result.timings.total_ns)
        init_times.append(result.timings.init_ns)
        render_times.append(result.timings.render_ns)

    sort(total_times)
    sort(init_times)
    sort(render_times)
    var middle = (TIMING_REPEATS - 1) >> 1
    return TimingResult(
        total_times[middle],
        init_times[middle],
        render_times[middle],
        render_times[0],
        render_times[TIMING_REPEATS - 1],
        checksum,
    )


def print_timing(label: String, result: TimingResult, sample_count: Int):
    var primary_msamples_s = (
        Float64(sample_count) / Float64(result.median_render_ns) * 1.0e3
    )
    var spread_percent = 100.0 * (
        Float64(result.max_render_ns - result.min_render_ns)
        / Float64(result.median_render_ns)
    )
    print(
        t"  {label}: {round(ns_to_ms(result.median_total_ns), 3)} ms total, "
        t"{round(ns_to_ms(result.median_render_ns), 3)} ms kernel, "
        t"{round(primary_msamples_s, 3)} Mprimary/s, "
        t"{round(ns_to_ms(result.median_init_ns), 3)} ms init"
    )
    print(
        t"    kernel min..max: {round(ns_to_ms(result.min_render_ns), 3)}.."
        t"{round(ns_to_ms(result.max_render_ns), 3)} ms "
        t"({round(spread_percent, 2)}% span), "
        t"checksum={round(result.checksum, 3)}"
    )


def run_benchmark() raises:
    print("CPU ray tracer end-to-end benchmark")
    print(
        t"timing: {TIMING_WIDTH}x{TIMING_HEIGHT} x {TIMING_SPP} spp, "
        t"depth {MAX_DEPTH}, median of {TIMING_REPEATS} after warmup"
    )

    var sphere_build_t0 = perf_counter_ns()
    var sphere_world = make_weekend_world()
    var sphere_build_ns = Int(perf_counter_ns() - sphere_build_t0)
    var sphere_camera = weekend_camera()

    var triangle_build_t0 = perf_counter_ns()
    var triangle_world = make_mixed_triangle_world()
    var triangle_build_ns = Int(perf_counter_ns() - triangle_build_t0)
    var triangle_cam = mixed_triangle_camera(triangle_world)

    var timing_settings = RenderSettings(
        TIMING_WIDTH, TIMING_HEIGHT, TIMING_SPP, RNG_SEED, MAX_DEPTH
    )
    var sample_count = TIMING_WIDTH * TIMING_HEIGHT * TIMING_SPP
    print("\nEnd-to-end production renderer timings")
    print(
        t"  scene build: spheres={round(ns_to_ms(sphere_build_ns), 3)} ms, "
        t"mixed triangles={round(ns_to_ms(triangle_build_ns), 3)} ms"
    )
    print_timing(
        "sphere / path",
        time_render[.PATH](timing_settings, sphere_camera, sphere_world),
        sample_count,
    )
    print_timing(
        "sphere / AO",
        time_render[.AO](timing_settings, sphere_camera, sphere_world),
        sample_count,
    )
    print_timing(
        "sphere / normals",
        time_render[.NORMALS](timing_settings, sphere_camera, sphere_world),
        sample_count,
    )
    print_timing(
        "triangles / path",
        time_render[.PATH](timing_settings, triangle_cam, triangle_world),
        sample_count,
    )
    print_timing(
        "triangles / AO",
        time_render[.AO](timing_settings, triangle_cam, triangle_world),
        sample_count,
    )
    print_timing(
        "triangles / normals",
        time_render[.NORMALS](timing_settings, triangle_cam, triangle_world),
        sample_count,
    )


def main() raises:
    run_benchmark()
