"""Diagnostic CPU renderer scheduling sweep."""

from std.math import round

from bajo.core.utils import ns_to_ms
from bajo.rt import Camera, RenderSettings, CpuScene
from bajo.rt.cpu import CpuSchedulerMode, render_depth_first
from bajo.benchmark.cpu_harness import pixel_checksum
from bajo.benchmark.rt_fixtures import (
    make_mixed_triangle_world,
    mixed_triangle_camera,
    weekend_camera,
)
from examples.rtiaw import make_weekend_world


comptime WIDTH = 960
comptime HEIGHT = 540
comptime SPP = 8
comptime MAX_DEPTH = 8
comptime REPEATS = 7
comptime RNG_SEED = UInt64(2026)


@fieldwise_init
struct ScheduleResult(Copyable):
    var median_ns: Int
    var min_ns: Int
    var max_ns: Int
    var checksum: Float64


def run_schedule[
    TILE_WIDTH: Int, TILE_HEIGHT: Int, scheduler_mode: CpuSchedulerMode
](
    settings: RenderSettings, camera: Camera, world: CpuScene[16, 16]
) raises -> Tuple[Int, Float64] where (
    TILE_WIDTH > 0,
    "tile width must be positive",
) where (
    TILE_HEIGHT > 0,
    "tile height must be positive",
):
    var result = render_depth_first[
        .PATH,
        TILE_WIDTH,
        TILE_HEIGHT,
        scheduler_mode,
    ](settings, camera, world)
    return (result.timings.render_ns, pixel_checksum(result.pixels))


def record_schedule[
    TILE_WIDTH: Int, TILE_HEIGHT: Int, scheduler_mode: CpuSchedulerMode
](
    mut times: List[Int],
    checksum: Float64,
    settings: RenderSettings,
    camera: Camera,
    world: CpuScene[16, 16],
) raises where (TILE_WIDTH > 0, "tile width must be positive") where (
    TILE_HEIGHT > 0,
    "tile height must be positive",
):
    var sample = run_schedule[TILE_WIDTH, TILE_HEIGHT, scheduler_mode](
        settings, camera, world
    )
    debug_assert["safe", _use_compiler_assume=True](
        sample[1] == checksum, "schedule changed the rendered pixels"
    )
    times.append(sample[0])


def summarize(mut times: List[Int], checksum: Float64) -> ScheduleResult:
    sort(times)
    return ScheduleResult(times[3], times[0], times[REPEATS - 1], checksum)


def print_result(
    label: String, result: ScheduleResult, baseline: ScheduleResult
):
    print(
        t"  {label}: {round(ns_to_ms(result.median_ns), 3)} ms median, "
        t"{round(ns_to_ms(result.min_ns), 3)}.."
        t"{round(ns_to_ms(result.max_ns), 3)} ms, "
        t"{round(Float64(baseline.median_ns) / Float64(result.median_ns), 3)}x"
    )


def benchmark_world(
    label: String,
    settings: RenderSettings,
    camera: Camera,
    world: CpuScene[16, 16],
) raises:
    print(t"\n{label}")
    # A tile wider than the image and one pixel high reproduces scanlines.
    var legacy_warmup = run_schedule[4096, 1, CpuSchedulerMode.TASK_PARTITIONS](
        settings, camera, world
    )
    var checksum = legacy_warmup[1]
    _ = run_schedule[4096, 1, CpuSchedulerMode.LOGICAL_CORES](
        settings, camera, world
    )
    _ = run_schedule[16, 16, CpuSchedulerMode.TASK_PARTITIONS](
        settings, camera, world
    )
    _ = run_schedule[32, 8, CpuSchedulerMode.TASK_PARTITIONS](
        settings, camera, world
    )
    _ = run_schedule[64, 8, CpuSchedulerMode.TASK_PARTITIONS](
        settings, camera, world
    )

    var legacy_times = List[Int](capacity=REPEATS)
    var core_times = List[Int](capacity=REPEATS)
    var tile16_times = List[Int](capacity=REPEATS)
    var tile32_times = List[Int](capacity=REPEATS)
    var tile64_times = List[Int](capacity=REPEATS)
    for iteration in range(REPEATS):
        if iteration % 2 == 0:
            record_schedule[4096, 1, CpuSchedulerMode.TASK_PARTITIONS](
                legacy_times, checksum, settings, camera, world
            )
            record_schedule[4096, 1, CpuSchedulerMode.LOGICAL_CORES](
                core_times, checksum, settings, camera, world
            )
            record_schedule[16, 16, CpuSchedulerMode.TASK_PARTITIONS](
                tile16_times, checksum, settings, camera, world
            )
            record_schedule[32, 8, CpuSchedulerMode.TASK_PARTITIONS](
                tile32_times, checksum, settings, camera, world
            )
            record_schedule[64, 8, CpuSchedulerMode.TASK_PARTITIONS](
                tile64_times, checksum, settings, camera, world
            )
        else:
            record_schedule[64, 8, CpuSchedulerMode.TASK_PARTITIONS](
                tile64_times, checksum, settings, camera, world
            )
            record_schedule[32, 8, CpuSchedulerMode.TASK_PARTITIONS](
                tile32_times, checksum, settings, camera, world
            )
            record_schedule[16, 16, CpuSchedulerMode.TASK_PARTITIONS](
                tile16_times, checksum, settings, camera, world
            )
            record_schedule[4096, 1, CpuSchedulerMode.LOGICAL_CORES](
                core_times, checksum, settings, camera, world
            )
            record_schedule[4096, 1, CpuSchedulerMode.TASK_PARTITIONS](
                legacy_times, checksum, settings, camera, world
            )

    var legacy = summarize(legacy_times, checksum)
    print_result("scanline / height workers", legacy, legacy)
    print_result(
        "scanline / 16 partitions",
        summarize(core_times, checksum),
        legacy,
    )
    print_result(
        "16x16 / tile partitions",
        summarize(tile16_times, checksum),
        legacy,
    )
    print_result(
        "32x8 / tile partitions",
        summarize(tile32_times, checksum),
        legacy,
    )
    print_result(
        "64x8 / tile partitions",
        summarize(tile64_times, checksum),
        legacy,
    )

    print(t"checksum: {round(legacy.checksum, 3)}")


def main() raises:
    print("CPU ray tracer scheduling and image-tile benchmark")
    print(t"{WIDTH}x{HEIGHT} x {SPP} spp, path depth {MAX_DEPTH}")
    print(
        "interleaved median of 7 after warmup; relative to original scanlines"
    )

    var settings = RenderSettings(WIDTH, HEIGHT, SPP, RNG_SEED, MAX_DEPTH)
    var sphere_world = make_weekend_world()
    var sphere_camera = weekend_camera()
    benchmark_world("Weekend spheres", settings, sphere_camera, sphere_world)

    var triangle_world = make_mixed_triangle_world()
    var triangle_cam = mixed_triangle_camera(triangle_world)
    benchmark_world(
        "Mixed standalone/instanced triangles",
        settings,
        triangle_cam,
        triangle_world,
    )
