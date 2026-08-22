"""Pure-Mojo CPU PATH benchmark for the viewer's LBVH scene."""

from std.math import round
from std.sys import simd_width_of

from bajo.core.utils import ns_to_ms
from bajo.rt import RENDER, RenderSettings, render_depth_first
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
    var world = make_lbvh_world[width, width]()

    print("CPU LBVH viewer PATH benchmark (pure Mojo)")
    print(
        t"{IMAGE_WIDTH}x{IMAGE_HEIGHT}, spp={SAMPLES_PER_PIXEL}, "
        t"depth={MAX_DEPTH}, seed={RNG_SEED}, native float32 SIMD={Int(width)}"
    )
    print(t"{WARMUPS} warmups + {REPEATS} measured renders")

    for _ in range(WARMUPS):
        _ = render_depth_first[RENDER.PATH](settings, camera, world)

    var render_times = List[Int](capacity=REPEATS)
    var total_times = List[Int](capacity=REPEATS)
    var checksum = Float64(0.0)
    for _ in range(REPEATS):
        var result = render_depth_first[RENDER.PATH](settings, camera, world)
        render_times.append(result.timings.render_ns)
        total_times.append(result.timings.total_ns)
        checksum = pixel_checksum(result.pixels)

    var render = summarize_timings(render_times)
    var total = summarize_timings(total_times)
    var table = TablePrinter(
        workload=14,
        median_ms=10,
        min_ms=10,
        max_ms=10,
        total_ms=10,
        checksum=18,
    )
    table.header()
    table.result_line(
        workload="LBVH PATH",
        median_ms=String(round(ns_to_ms(render.median_ns), 3)),
        min_ms=String(round(ns_to_ms(render.min_ns), 3)),
        max_ms=String(round(ns_to_ms(render.max_ns), 3)),
        total_ms=String(round(ns_to_ms(total.median_ns), 3)),
        checksum=String(round(checksum, 6)),
    )
