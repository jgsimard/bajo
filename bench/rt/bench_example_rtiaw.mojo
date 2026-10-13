"""Compare CPU renderer architectures on the full RTIAW example workload."""

from std.math import round
from std.sys.arg import argv

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
from bajo.benchmark.cpu_harness import (
    CpuRenderSamples,
    CpuRenderTiming,
    pixel_checksum,
)
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


def _render_mode[
    MODE: Int, CHUNK_PATHS: Int = 0
](
    settings: RenderSettings, camera: Camera, world: CpuScene[16, 16]
) raises -> RenderResult:
    comptime match MODE:
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


def _print(label: String, timing: CpuRenderTiming):
    print(
        t"  {label}: total={round(ns_to_ms(timing.median_total_ns), 3)} ms, "
        t"kernel={round(ns_to_ms(timing.median_render_ns), 3)} ms, "
        t"range={round(ns_to_ms(timing.min_render_ns), 3)}.."
        t"{round(ns_to_ms(timing.max_render_ns), 3)} ms, "
        t"checksum={round(timing.checksum, 3)}"
    )


def _run[full_sweep: Bool]() raises:
    comptime case_count = len(CASES) if full_sweep else 3
    print("Ray Tracing in One Weekend CPU architecture benchmark")
    print(
        t"{WIDTH}x{HEIGHT} x {SPP} spp, depth {MAX_DEPTH}; interleaved median"
        t" of {REPEATS}"
    )
    var settings = RenderSettings(WIDTH, HEIGHT, SPP, RNG_SEED, MAX_DEPTH)
    var world = make_weekend_world()
    var camera = weekend_camera()

    print(
        "Full chunk-size sweep" if full_sweep else "Three default architectures"
    )
    var samples = List[CpuRenderSamples](capacity=case_count)
    comptime for i in range(case_count):
        comptime case_idx = i if full_sweep or i < 2 else 3
        var warmup = _render_mode[CASES[case_idx][0], CASES[case_idx][1]](
            settings, camera, world
        )
        var checksum = pixel_checksum(warmup.pixels)
        comptime if CASES[case_idx][0] == MODE_WAVEFRONT_PARALLEL:
            debug_assert["safe", _use_compiler_assume=True](
                checksum == samples[1].checksum,
                "parallel wavefront changed serial wavefront pixels",
            )
        samples.append(CpuRenderSamples(checksum, REPEATS))

    for iteration in range(REPEATS):
        # Alternate forward and reverse order to balance timing drift.
        comptime for reverse in range(2):
            if iteration % 2 == reverse:
                comptime for position in range(case_count):
                    comptime i = case_count - 1 - position if reverse else position
                    comptime case_idx = i if full_sweep or i < 2 else 3
                    var result = _render_mode[
                        CASES[case_idx][0], CASES[case_idx][1]
                    ](settings, camera, world)
                    samples[i].record(result)

    comptime for i in range(case_count):
        comptime case_idx = i if full_sweep or i < 2 else 3
        _print(CASES[case_idx][2], samples[i].summarize())


def main() raises:
    var args = argv()
    if len(args) == 1:
        _run[False]()
    elif len(args) == 2 and args[1] == "--full":
        _run[True]()
    else:
        raise Error("usage: bench_rtiaw [--full]")
