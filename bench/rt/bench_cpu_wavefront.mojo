"""Serial CPU wavefront timings across packet widths and integrators."""

from std.math import round

from bajo.core import Point3W, Vec3W
from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Camera,
    Integrator,
    RenderSettings,
    CpuScene,
)
from bajo.rt.cpu import render_wavefront
from bajo.benchmark.cpu_harness import (
    CpuRenderTiming as WaveTiming,
    benchmark_cpu_render,
)
from bajo.benchmark.rt_fixtures import (
    make_mixed_triangle_world,
    mixed_triangle_camera,
    weekend_camera,
)
from examples.cornell_box import make_cornell_world
from examples.rtiaw import make_weekend_world


comptime TIMING_WIDTH = 320
comptime TIMING_HEIGHT = 180
comptime TIMING_SPP = 4
comptime TIMING_REPEATS = 7
comptime MAX_DEPTH = 8
comptime RNG_SEED = UInt64(2026)
comptime CHUNK_PATHS = 8192


def time_wavefront[
    length: SIMDLength, integrator: Integrator = .PATH
](
    settings: RenderSettings, camera: Camera, world: CpuScene[16, 16]
) raises -> WaveTiming:
    return benchmark_cpu_render[
        render_wavefront[integrator, length, CHUNK_PATHS, False]
    ](settings, camera, world, TIMING_REPEATS)


def print_timing(label: String, timing: WaveTiming):
    var primary_count = TIMING_WIDTH * TIMING_HEIGHT * TIMING_SPP
    var primary_mray_s = (
        Float64(primary_count) / Float64(timing.median_render_ns) * 1.0e3
    )
    print(t"  {label}")
    print(
        t"    total={round(ns_to_ms(timing.median_total_ns), 3)} ms, "
        t"init={round(ns_to_ms(timing.median_init_ns), 3)} ms, "
        t"render={round(ns_to_ms(timing.median_render_ns), 3)} ms"
    )
    print(
        t"    render range={round(ns_to_ms(timing.min_render_ns), 3)}.."
        t"{round(ns_to_ms(timing.max_render_ns), 3)} ms, "
        t"primary={round(primary_mray_s, 3)} M/s, "
        t"checksum={round(timing.checksum, 3)}"
    )


def benchmark_scene(
    label: String,
    timing_settings: RenderSettings,
    camera: Camera,
    world: CpuScene[16, 16],
) raises:
    print_timing(
        label + " / packet width 1",
        time_wavefront[1](timing_settings, camera, world),
    )
    print_timing(
        label + " / packet width 4",
        time_wavefront[4](timing_settings, camera, world),
    )
    print_timing(
        label + " / packet width 8",
        time_wavefront[8](timing_settings, camera, world),
    )
    print_timing(
        label + " / packet width 16",
        time_wavefront[16](timing_settings, camera, world),
    )


def benchmark_direct_lighting(
    settings: RenderSettings, camera: Camera, world: CpuScene[16, 16]
) raises:
    print_timing(
        "Cornell NEE / packet width 1",
        time_wavefront[1, .NEE](settings, camera, world),
    )
    print_timing(
        "Cornell NEE / packet width 8",
        time_wavefront[8, .NEE](settings, camera, world),
    )
    print_timing(
        "Cornell NEE / packet width 16",
        time_wavefront[16, .NEE](settings, camera, world),
    )
    print_timing(
        "Cornell MIS / packet width 1",
        time_wavefront[1, .MIS](settings, camera, world),
    )
    print_timing(
        "Cornell MIS / packet width 8",
        time_wavefront[8, .MIS](settings, camera, world),
    )
    print_timing(
        "Cornell MIS / packet width 16",
        time_wavefront[16, .MIS](settings, camera, world),
    )


def main() raises:
    print("CPU serial wavefront benchmark")
    print(
        t"timing: {TIMING_WIDTH}x{TIMING_HEIGHT} x {TIMING_SPP} spp, "
        t"depth {MAX_DEPTH}, median of {TIMING_REPEATS} after warmup"
    )

    var timing_settings = RenderSettings(
        TIMING_WIDTH, TIMING_HEIGHT, TIMING_SPP, RNG_SEED, MAX_DEPTH
    )

    var sphere_world = make_weekend_world()
    var sphere_camera = weekend_camera()
    benchmark_scene(
        "Weekend spheres",
        timing_settings,
        sphere_camera,
        sphere_world,
    )

    var triangle_world = make_mixed_triangle_world()
    var triangle_cam = mixed_triangle_camera(triangle_world)
    benchmark_scene(
        "Mixed triangles",
        timing_settings,
        triangle_cam,
        triangle_world,
    )

    var cornell_world = make_cornell_world()
    var cornell_camera = Camera.from_vfov(
        Point3W(0.0, 1.0, 3.2),
        Point3W(0.0, 1.0, -1.0),
        Vec3W(0.0, 1.0, 0.0),
        28.0,
        4.2,
    )
    benchmark_direct_lighting(timing_settings, cornell_camera, cornell_world)
