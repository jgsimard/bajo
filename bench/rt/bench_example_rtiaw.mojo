"""Compare CPU renderer architectures on the full RTIAW example workload."""

from std.math import round

from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Camera,
    RenderResult,
    RenderSettings,
    CpuScene,
    render_depth_first,
)
from bajo.rt.cpu import render_wavefront
from bajo.rt.cpu.wavefront import (
    CPU_WAVEFRONT_SERIAL_CHUNK_PATHS,
)
from bajo.rt.cpu.scheduler_mode import CpuSchedulerMode
from bajo.benchmark.cpu_harness import pixel_checksum
from bajo.benchmark.rt_fixtures import weekend_camera
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


# Mode, chunk size, and output label in forward sampling order.
comptime CASES = (
    (MODE_DEPTH_FIRST, 0, "parallel depth-first 16x16"),
    (MODE_WAVEFRONT_SERIAL, 0, "serial packet16 wavefront 8K"),
    (MODE_WAVEFRONT_PARALLEL, 512, "parallel packet16 wavefront 512"),
    (MODE_WAVEFRONT_PARALLEL, 1024, "parallel packet16 wavefront 1K"),
    (MODE_WAVEFRONT_PARALLEL, 2048, "parallel packet16 wavefront 2K"),
    (MODE_WAVEFRONT_PARALLEL, 4096, "parallel packet16 wavefront 4K"),
)


@fieldwise_init
struct Samples(Copyable):
    var total_times: List[Int]
    var render_times: List[Int]
    var checksum: Float64


@fieldwise_init
struct Timing(Copyable):
    var median_total_ns: Int
    var median_render_ns: Int
    var min_render_ns: Int
    var max_render_ns: Int
    var checksum: Float64


def _render_mode[
    MODE: Int, CHUNK_PATHS: Int = 0
](
    settings: RenderSettings, camera: Camera, world: CpuScene[16, 16]
) raises -> RenderResult:
    comptime __match MODE:
        case 0:
            return render_depth_first[.PATH](settings, camera, world)
        case 1:
            return render_wavefront[
                .PATH,
                16,
                CPU_WAVEFRONT_SERIAL_CHUNK_PATHS,
                False,
            ](settings, camera, world)
        case 2:
            comptime assert CHUNK_PATHS > 0
            return render_wavefront[
                .PATH,
                16,
                CHUNK_PATHS,
                True,
                CpuSchedulerMode.TASK_PARTITIONS,
            ](settings, camera, world)
        case _:
            comptime assert False


def _warmup[
    MODE: Int, CHUNK_PATHS: Int = 0
](
    settings: RenderSettings, camera: Camera, world: CpuScene[16, 16]
) raises -> Float64:
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
    world: CpuScene[16, 16],
) raises:
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
    sort(total_times)
    sort(render_times)
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


def main() raises:
    print("Ray Tracing in One Weekend CPU architecture benchmark")
    print(
        t"{WIDTH}x{HEIGHT} x {SPP} spp, depth {MAX_DEPTH}; interleaved median"
        t" of {REPEATS}"
    )
    var settings = RenderSettings(WIDTH, HEIGHT, SPP, RNG_SEED, MAX_DEPTH)
    var world = make_weekend_world()
    var camera = weekend_camera()

    var samples = List[Samples](capacity=len(CASES))
    comptime for i in range(len(CASES)):
        var checksum = _warmup[CASES[i][0], CASES[i][1]](
            settings, camera, world
        )
        comptime if CASES[i][0] == MODE_WAVEFRONT_PARALLEL:
            debug_assert["safe", _use_compiler_assume=True](
                checksum == samples[1].checksum,
                "parallel wavefront changed serial wavefront pixels",
            )
        samples.append(
            Samples(
                List[Int](capacity=REPEATS),
                List[Int](capacity=REPEATS),
                checksum,
            )
        )

    for iteration in range(REPEATS):
        # Alternate forward and reverse order to balance timing drift.
        comptime for reverse in range(2):
            if iteration % 2 == reverse:
                comptime for position in range(len(CASES)):
                    comptime i = len(
                        CASES
                    ) - 1 - position if reverse else position
                    _record[CASES[i][0], CASES[i][1]](
                        samples[i].total_times,
                        samples[i].render_times,
                        samples[i].checksum,
                        settings,
                        camera,
                        world,
                    )

    comptime for i in range(len(CASES)):
        _print(
            CASES[i][2],
            _summarize(
                samples[i].total_times,
                samples[i].render_times,
                samples[i].checksum,
            ),
        )
