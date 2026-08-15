"""Compare CPU renderer architectures on the full RTIAW example workload."""

from std.math import round

from bajo.core import Point3W, Vec3W
from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Camera,
    RENDER,
    RenderResult,
    RenderSettings,
    World,
    render_depth_first,
)
from bajo.rt.cpu import render_wavefront
from bajo.rt.cpu.wavefront import (
    CPU_WAVEFRONT_SERIAL_CHUNK_PATHS,
    WAVE_PARALLEL_TASK_PARTITIONS,
)
from bench.rt.bench_cpu_end_to_end import pixel_checksum, sort_timings
from examples.rtiaw import make_weekend_world


comptime WIDTH = 480
comptime HEIGHT = 270
comptime SPP = 10
comptime MAX_DEPTH = 64
comptime RNG_SEED = UInt64(1234)
comptime REPEATS = 9

comptime MODE_DEPTH_FIRST = 0
comptime MODE_WAVEFRONT_SERIAL = 1
comptime MODE_WAVEFRONT_PARALLEL = 2


@fieldwise_init
struct Timing(Copyable):
    var median_total_ns: Int
    var median_render_ns: Int
    var min_render_ns: Int
    var max_render_ns: Int
    var checksum: Float64


def _render_mode[
    MODE: Int, CHUNK_PATHS: Int = 0
](settings: RenderSettings, camera: Camera, world: World) -> RenderResult:
    comptime if MODE == MODE_DEPTH_FIRST:
        return render_depth_first[RENDER.PATH, MAX_DEPTH](
            settings, camera, world
        )
    elif MODE == MODE_WAVEFRONT_SERIAL:
        return render_wavefront[
            RENDER.PATH,
            MAX_DEPTH,
            16,
            CPU_WAVEFRONT_SERIAL_CHUNK_PATHS,
            False,
        ](settings, camera, world)
    else:
        comptime assert MODE == MODE_WAVEFRONT_PARALLEL
        comptime assert CHUNK_PATHS > 0
        return render_wavefront[
            RENDER.PATH,
            MAX_DEPTH,
            16,
            CHUNK_PATHS,
            True,
            WAVE_PARALLEL_TASK_PARTITIONS,
        ](settings, camera, world)


def _warmup[
    MODE: Int, CHUNK_PATHS: Int = 0
](settings: RenderSettings, camera: Camera, world: World) -> Float64:
    return pixel_checksum(
        _render_mode[MODE, CHUNK_PATHS](settings, camera, world).pixels
    )


def _record[
    MODE: Int, CHUNK_PATHS: Int = 0
](
    mut total_times: List[Int],
    mut render_times: List[Int],
    checksum: Float64,
    settings: RenderSettings,
    camera: Camera,
    world: World,
):
    var result = _render_mode[MODE, CHUNK_PATHS](settings, camera, world)
    debug_assert["safe", _use_compiler_assume=True](
        pixel_checksum(result.pixels) == checksum,
        "RTIAW output changed between benchmark samples",
    )
    total_times.append(result.timings.total_ns)
    render_times.append(result.timings.render_ns)


def _summarize(
    mut total_times: List[Int],
    mut render_times: List[Int],
    checksum: Float64,
) -> Timing:
    sort_timings(total_times)
    sort_timings(render_times)
    return Timing(
        total_times[REPEATS // 2],
        render_times[REPEATS // 2],
        render_times[0],
        render_times[REPEATS - 1],
        checksum,
    )


def _print(label: String, timing: Timing):
    print(
        t"  {label}: total={round(ns_to_ms(timing.median_total_ns), 3)} ms, "
        t"kernel={round(ns_to_ms(timing.median_render_ns), 3)} ms, "
        t"range={round(ns_to_ms(timing.min_render_ns), 3)}.."
        t"{round(ns_to_ms(timing.max_render_ns), 3)} ms, "
        t"checksum={round(timing.checksum, 3)}"
    )


def main():
    print("Ray Tracing in One Weekend CPU architecture benchmark")
    print(
        t"{WIDTH}x{HEIGHT} x {SPP} spp, depth {MAX_DEPTH}; interleaved median"
        t" of {REPEATS}"
    )
    var settings = RenderSettings(WIDTH, HEIGHT, SPP, RNG_SEED)
    var world = make_weekend_world()
    var camera = Camera.from_vfov(
        Point3W(13.0, 2.0, 3.0),
        Point3W(0.0, 0.0, 0.0),
        Vec3W(0.0, 1.0, 0.0),
        20.0,
        10.0,
        0.6,
    )

    var depth_checksum = _warmup[MODE_DEPTH_FIRST](settings, camera, world)
    var serial_checksum = _warmup[MODE_WAVEFRONT_SERIAL](
        settings, camera, world
    )
    var parallel_checksum = _warmup[MODE_WAVEFRONT_PARALLEL, 512](
        settings, camera, world
    )
    var parallel1k_checksum = _warmup[MODE_WAVEFRONT_PARALLEL, 1024](
        settings, camera, world
    )
    var parallel2k_checksum = _warmup[MODE_WAVEFRONT_PARALLEL, 2048](
        settings, camera, world
    )
    var parallel4k_checksum = _warmup[MODE_WAVEFRONT_PARALLEL, 4096](
        settings, camera, world
    )
    debug_assert["safe", _use_compiler_assume=True](
        parallel_checksum == serial_checksum
        and parallel1k_checksum == serial_checksum
        and parallel2k_checksum == serial_checksum
        and parallel4k_checksum == serial_checksum,
        "parallel wavefront changed serial wavefront pixels",
    )

    var depth_total = List[Int](capacity=REPEATS)
    var depth_render = List[Int](capacity=REPEATS)
    var serial_total = List[Int](capacity=REPEATS)
    var serial_render = List[Int](capacity=REPEATS)
    var parallel512_total = List[Int](capacity=REPEATS)
    var parallel512_render = List[Int](capacity=REPEATS)
    var parallel1k_total = List[Int](capacity=REPEATS)
    var parallel1k_render = List[Int](capacity=REPEATS)
    var parallel2k_total = List[Int](capacity=REPEATS)
    var parallel2k_render = List[Int](capacity=REPEATS)
    var parallel4k_total = List[Int](capacity=REPEATS)
    var parallel4k_render = List[Int](capacity=REPEATS)
    for iteration in range(REPEATS):
        if iteration % 2 == 0:
            _record[MODE_DEPTH_FIRST](
                depth_total,
                depth_render,
                depth_checksum,
                settings,
                camera,
                world,
            )
            _record[MODE_WAVEFRONT_SERIAL](
                serial_total,
                serial_render,
                serial_checksum,
                settings,
                camera,
                world,
            )
            _record[MODE_WAVEFRONT_PARALLEL, 512](
                parallel512_total,
                parallel512_render,
                parallel_checksum,
                settings,
                camera,
                world,
            )
            _record[MODE_WAVEFRONT_PARALLEL, 1024](
                parallel1k_total,
                parallel1k_render,
                parallel_checksum,
                settings,
                camera,
                world,
            )
            _record[MODE_WAVEFRONT_PARALLEL, 2048](
                parallel2k_total,
                parallel2k_render,
                parallel_checksum,
                settings,
                camera,
                world,
            )
            _record[MODE_WAVEFRONT_PARALLEL, 4096](
                parallel4k_total,
                parallel4k_render,
                parallel_checksum,
                settings,
                camera,
                world,
            )
        else:
            _record[MODE_WAVEFRONT_PARALLEL, 4096](
                parallel4k_total,
                parallel4k_render,
                parallel_checksum,
                settings,
                camera,
                world,
            )
            _record[MODE_WAVEFRONT_PARALLEL, 2048](
                parallel2k_total,
                parallel2k_render,
                parallel_checksum,
                settings,
                camera,
                world,
            )
            _record[MODE_WAVEFRONT_PARALLEL, 1024](
                parallel1k_total,
                parallel1k_render,
                parallel_checksum,
                settings,
                camera,
                world,
            )
            _record[MODE_WAVEFRONT_PARALLEL, 512](
                parallel512_total,
                parallel512_render,
                parallel_checksum,
                settings,
                camera,
                world,
            )
            _record[MODE_WAVEFRONT_SERIAL](
                serial_total,
                serial_render,
                serial_checksum,
                settings,
                camera,
                world,
            )
            _record[MODE_DEPTH_FIRST](
                depth_total,
                depth_render,
                depth_checksum,
                settings,
                camera,
                world,
            )

    var depth = _summarize(depth_total, depth_render, depth_checksum)
    var serial = _summarize(serial_total, serial_render, serial_checksum)
    var parallel512 = _summarize(
        parallel512_total, parallel512_render, parallel_checksum
    )
    var parallel1k = _summarize(
        parallel1k_total, parallel1k_render, parallel_checksum
    )
    var parallel2k = _summarize(
        parallel2k_total, parallel2k_render, parallel_checksum
    )
    var parallel4k = _summarize(
        parallel4k_total, parallel4k_render, parallel_checksum
    )
    _print("parallel depth-first 16x16", depth)
    _print("serial packet16 wavefront 8K", serial)
    _print("parallel packet16 wavefront 512", parallel512)
    _print("parallel packet16 wavefront 1K", parallel1k)
    _print("parallel packet16 wavefront 2K", parallel2k)
    _print("parallel packet16 wavefront 4K", parallel4k)
