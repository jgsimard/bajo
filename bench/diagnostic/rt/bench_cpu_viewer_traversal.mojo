"""CPU wavefront traversal benchmark for the viewer's heavy triangle scene."""

from std.math import round
from std.sys import simd_width_of
from std.sys.defines import get_defined_int

from bajo.core.utils import ns_to_ms
from bajo.bvh.cpu import CpuBvhBuildMethod, CpuTraversalMode
from bajo.rt import (
    CpuSceneConfig,
    CpuSchedulerMode,
    RenderSettings,
    render_wavefront_configured,
)
from bajo.benchmark.bvh_reporting import TablePrinter
from bajo.benchmark.cpu_harness import pixel_checksum
from bajo.benchmark.timing import summarize_timings
from examples.lbvh_scene import make_lbvh_camera, make_lbvh_world


comptime IMAGE_WIDTH = 320
comptime IMAGE_HEIGHT = 214
comptime SAMPLES_PER_PIXEL = 4
comptime MAX_DEPTH = 8
comptime RNG_SEED = UInt64(1234)
comptime WARMUPS = 2
comptime REPEATS = 12
comptime VIEWER_BUILD = get_defined_int["VIEWER_BUILD", 0]()
comptime VIEWER_TRAVERSAL = get_defined_int["VIEWER_TRAVERSAL", 0]()
comptime BENCH_SCENE_CONFIG = CpuSceneConfig(
    CpuBvhBuildMethod.SAH if VIEWER_BUILD
    == 0 else CpuBvhBuildMethod.LBVH if VIEWER_BUILD
    == 1 else CpuBvhBuildMethod.HPLOC if VIEWER_BUILD
    == 2 else CpuBvhBuildMethod.MEDIAN,
    CpuTraversalMode.AUTO_COHERENT if VIEWER_TRAVERSAL
    == 0 else CpuTraversalMode.FIXED_PACKET if VIEWER_TRAVERSAL
    == 1 else CpuTraversalMode.ADAPTIVE,
)


def main() raises:
    comptime width = simd_width_of[DType.float32]()
    var settings = RenderSettings(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        SAMPLES_PER_PIXEL,
        RNG_SEED,
        MAX_DEPTH,
    )
    var camera = make_lbvh_camera()
    var world = make_lbvh_world[width, width, BENCH_SCENE_CONFIG]()

    print("CPU viewer traversal benchmark")
    print(
        t"{IMAGE_WIDTH}x{IMAGE_HEIGHT}, spp={SAMPLES_PER_PIXEL}, "
        t"depth={MAX_DEPTH}, seed={RNG_SEED}, native float32 SIMD={Int(width)}"
    )
    print(t"{WARMUPS} warmups + {REPEATS} measured renders")

    var table = TablePrinter(
        workload=14,
        median_ms=10,
        min_ms=10,
        max_ms=10,
        total_ms=10,
        checksum=18,
    )
    table.header()

    def benchmark[
        PARALLEL: Bool,
    ](label: String) raises {imm}:
        for _ in range(WARMUPS):
            _ = render_wavefront_configured[
                BENCH_SCENE_CONFIG,
                .PATH,
                16,
                1024,
                PARALLEL,
                CpuSchedulerMode.TASK_PARTITIONS,
                width,
                width,
                16,
                8,
                4,
            ](settings, camera, world)

        var render_times = List[Int](capacity=REPEATS)
        var total_times = List[Int](capacity=REPEATS)
        var checksum = Float64(0.0)
        for _ in range(REPEATS):
            var result = render_wavefront_configured[
                BENCH_SCENE_CONFIG,
                .PATH,
                16,
                1024,
                PARALLEL,
                CpuSchedulerMode.TASK_PARTITIONS,
                width,
                width,
                16,
                8,
                4,
            ](settings, camera, world)
            render_times.append(result.timings.render_ns)
            total_times.append(result.timings.total_ns)
            checksum = pixel_checksum(result.pixels)

        var render = summarize_timings(render_times)
        var total = summarize_timings(total_times)
        table.result_line(
            workload=label,
            median_ms=String(round(ns_to_ms(render.median_ns), 3)),
            min_ms=String(round(ns_to_ms(render.min_ns), 3)),
            max_ms=String(round(ns_to_ms(render.max_ns), 3)),
            total_ms=String(round(ns_to_ms(total.median_ns), 3)),
            checksum=String(round(checksum, 6)),
        )

    benchmark[True]("parallel")
    benchmark[False]("serial")
