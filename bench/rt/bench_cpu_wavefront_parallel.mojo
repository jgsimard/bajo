"""Interleaved CPU wavefront chunk-parallel scheduling benchmark."""

from std.math import abs, round

from bajo.core import Point3W, Vec3W
from bajo.core.utils import ns_to_ms
from bajo.rt import Camera, RENDER, RenderResult, RenderSettings, World
from bajo.rt.cpu import (
    CPU_WAVEFRONT_SERIAL_CHUNK_PATHS,
    WAVE_PARALLEL_LOGICAL_CORES,
    WAVE_PARALLEL_RUNTIME_DEFAULT,
    WAVE_PARALLEL_TASK_PARTITIONS,
    render_wavefront,
)
from bench.rt.bench_cpu_end_to_end import (
    make_triangle_world,
    pixel_checksum,
    sort_timings,
    triangle_camera,
)
from examples.rtiaw import make_weekend_world


comptime WIDTH = 320
comptime HEIGHT = 180
comptime SPP = 4
comptime MAX_DEPTH = 8
comptime REPEATS = 7
comptime RNG_SEED = UInt64(2026)


@fieldwise_init
struct Timing(Copyable):
    var median_ns: Int
    var min_ns: Int
    var max_ns: Int
    var checksum: Float64


def _render_configuration[
    CHUNK_PATHS: Int,
    SCHEDULER_MODE: Int = WAVE_PARALLEL_TASK_PARTITIONS,
    PARALLEL: Bool = True,
](settings: RenderSettings, camera: Camera, world: World) -> RenderResult:
    return render_wavefront[
        RENDER.PATH,
        MAX_DEPTH,
        1,
        CHUNK_PATHS,
        PARALLEL,
        SCHEDULER_MODE,
    ](settings, camera, world)


def _record[
    CHUNK_PATHS: Int,
    SCHEDULER_MODE: Int = WAVE_PARALLEL_TASK_PARTITIONS,
    PARALLEL: Bool = True,
](
    mut times: List[Int],
    checksum: Float64,
    settings: RenderSettings,
    camera: Camera,
    world: World,
):
    var result = _render_configuration[CHUNK_PATHS, SCHEDULER_MODE, PARALLEL](
        settings, camera, world
    )
    debug_assert["safe", _use_compiler_assume=True](
        abs(pixel_checksum(result.pixels) - checksum) <= 0.1,
        "parallel chunk schedule changed the output checksum",
    )
    times.append(result.timings.render_ns)


def _summarize(mut times: List[Int], checksum: Float64) -> Timing:
    sort_timings(times)
    return Timing(times[3], times[0], times[REPEATS - 1], checksum)


def _print(label: String, timing: Timing, serial: Timing):
    print(
        t"  {label}: {round(ns_to_ms(timing.median_ns), 3)} ms, "
        t"{round(ns_to_ms(timing.min_ns), 3)}.."
        t"{round(ns_to_ms(timing.max_ns), 3)} ms, "
        t"{round(Float64(serial.median_ns) / Float64(timing.median_ns), 3)}x"
    )


def benchmark_world(
    label: String, settings: RenderSettings, camera: Camera, world: World
):
    print(t"\n{label}")
    var warmup = _render_configuration[
        CPU_WAVEFRONT_SERIAL_CHUNK_PATHS,
        WAVE_PARALLEL_TASK_PARTITIONS,
        False,
    ](settings, camera, world)
    var checksum = pixel_checksum(warmup.pixels)
    _ = _render_configuration[512](settings, camera, world)
    _ = _render_configuration[1024](settings, camera, world)
    _ = _render_configuration[2048](settings, camera, world)
    _ = _render_configuration[4096](settings, camera, world)
    _ = _render_configuration[8192](settings, camera, world)
    _ = _render_configuration[8192, WAVE_PARALLEL_RUNTIME_DEFAULT](
        settings, camera, world
    )
    _ = _render_configuration[8192, WAVE_PARALLEL_LOGICAL_CORES](
        settings, camera, world
    )

    var serial_times = List[Int](capacity=REPEATS)
    var parallel512_times = List[Int](capacity=REPEATS)
    var parallel1k_times = List[Int](capacity=REPEATS)
    var parallel2k_times = List[Int](capacity=REPEATS)
    var parallel4k_times = List[Int](capacity=REPEATS)
    var parallel8k_times = List[Int](capacity=REPEATS)
    var default_times = List[Int](capacity=REPEATS)
    var core_times = List[Int](capacity=REPEATS)
    for iteration in range(REPEATS):
        if iteration % 2 == 0:
            _record[
                CPU_WAVEFRONT_SERIAL_CHUNK_PATHS,
                WAVE_PARALLEL_TASK_PARTITIONS,
                False,
            ](serial_times, checksum, settings, camera, world)
            _record[512](parallel512_times, checksum, settings, camera, world)
            _record[1024](parallel1k_times, checksum, settings, camera, world)
            _record[2048](parallel2k_times, checksum, settings, camera, world)
            _record[4096](parallel4k_times, checksum, settings, camera, world)
            _record[8192](parallel8k_times, checksum, settings, camera, world)
            _record[8192, WAVE_PARALLEL_RUNTIME_DEFAULT](
                default_times, checksum, settings, camera, world
            )
            _record[8192, WAVE_PARALLEL_LOGICAL_CORES](
                core_times, checksum, settings, camera, world
            )
        else:
            _record[8192, WAVE_PARALLEL_LOGICAL_CORES](
                core_times, checksum, settings, camera, world
            )
            _record[8192, WAVE_PARALLEL_RUNTIME_DEFAULT](
                default_times, checksum, settings, camera, world
            )
            _record[8192](parallel8k_times, checksum, settings, camera, world)
            _record[4096](parallel4k_times, checksum, settings, camera, world)
            _record[2048](parallel2k_times, checksum, settings, camera, world)
            _record[1024](parallel1k_times, checksum, settings, camera, world)
            _record[512](parallel512_times, checksum, settings, camera, world)
            _record[
                CPU_WAVEFRONT_SERIAL_CHUNK_PATHS,
                WAVE_PARALLEL_TASK_PARTITIONS,
                False,
            ](serial_times, checksum, settings, camera, world)

    var serial = _summarize(serial_times, checksum)
    _print("serial 8K", serial, serial)
    _print(
        "parallel 512 / task partitions",
        _summarize(parallel512_times, checksum),
        serial,
    )
    _print(
        "parallel 1K / task partitions",
        _summarize(parallel1k_times, checksum),
        serial,
    )
    _print(
        "parallel 2K / task partitions",
        _summarize(parallel2k_times, checksum),
        serial,
    )
    _print(
        "parallel 4K / task partitions",
        _summarize(parallel4k_times, checksum),
        serial,
    )
    _print(
        "parallel 8K / task partitions",
        _summarize(parallel8k_times, checksum),
        serial,
    )
    _print(
        "parallel 8K / runtime default",
        _summarize(default_times, checksum),
        serial,
    )
    _print(
        "parallel 8K / logical cores", _summarize(core_times, checksum), serial
    )
    print(t"  checksum={round(checksum, 3)}")


def main():
    print("CPU wavefront parallel chunk benchmark")
    print(
        t"{WIDTH}x{HEIGHT} x {SPP} spp, depth {MAX_DEPTH}; interleaved median"
        t" of {REPEATS}"
    )
    var settings = RenderSettings(WIDTH, HEIGHT, SPP, RNG_SEED)
    var sphere_world = make_weekend_world()
    var sphere_camera = Camera.from_vfov(
        Point3W(13.0, 2.0, 3.0),
        Point3W(0.0, 0.0, 0.0),
        Vec3W(0.0, 1.0, 0.0),
        20.0,
        10.0,
        0.6,
    )
    benchmark_world("Weekend spheres", settings, sphere_camera, sphere_world)
    var triangle_world = make_triangle_world()
    var triangle_cam = triangle_camera(triangle_world)
    benchmark_world(
        "Mixed standalone/instanced triangles",
        settings,
        triangle_cam,
        triangle_world,
    )
