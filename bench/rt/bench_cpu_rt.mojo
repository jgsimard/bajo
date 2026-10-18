"""CPU MIS rendering: scalar, SIMD and parallel SIMD on the same scene.

Usage: bench_cpu_rt [cornell|weekend|mixed [width height spp repeats]].
Each case warms up, then reports timed medians and an output checksum.
Scene construction and checksum calculation are outside the timed region.
"""

from std.math import round
from std.sys import simd_width_of
from std.sys.arg import argv

from bajo.benchmark.cpu_harness import benchmark_cpu_render
from bajo.benchmark.rt_fixtures import (
    make_mixed_triangle_world,
    mixed_triangle_camera,
    weekend_camera,
)
from bajo.core import Point3W, Vec3W
from bajo.core.utils import ns_to_ms
from bajo.parser.number import parse_int_f32
from bajo.rt import Camera, CpuScene, RenderSettings, render_wavefront
from examples.cornell_box import make_cornell_scene
from examples.rtiaw import make_weekend_scene


def benchmark[
    length: SIMDLength, parallel: Bool
](
    label: String,
    settings: RenderSettings,
    camera: Camera,
    world: CpuScene[16, 16],
    repeats: Int,
) raises:
    var timing = benchmark_cpu_render[
        render_wavefront[.MIS, length, 1024, parallel]
    ](settings, camera, world, repeats)
    print(
        label,
        "width",
        Int(length),
        "total_ms",
        round(ns_to_ms(timing.median_total_ns), 3),
        "render_ms",
        round(ns_to_ms(timing.median_render_ns), 3),
        "min_ms",
        round(ns_to_ms(timing.min_render_ns), 3),
        "max_ms",
        round(ns_to_ms(timing.max_render_ns), 3),
        "checksum",
        timing.checksum,
    )


def main() raises:
    var args = argv()
    if len(args) != 1 and len(args) != 2 and len(args) != 6:
        raise Error(
            "usage: bench_cpu_rt [cornell|weekend|mixed"
            " [width height spp repeats]]"
        )
    var name = String(args[1]) if len(args) > 1 else String("cornell")
    var settings = RenderSettings(160, 90, 2, UInt64(2026), 8)
    var repeats = 3
    if len(args) == 6:
        settings.image_width = parse_int_f32(StringSpan(args[2]), "width")
        settings.image_height = parse_int_f32(StringSpan(args[3]), "height")
        settings.samples_per_pixel = parse_int_f32(StringSpan(args[4]), "spp")
        repeats = parse_int_f32(StringSpan(args[5]), "repeats")
    if repeats < 1:
        raise Error("repeats must be positive")
    var camera = weekend_camera()
    var world: CpuScene[16, 16]
    match name:
        case "cornell":
            world = CpuScene[16, 16](make_cornell_scene())
            camera = Camera.from_vfov(
                Point3W(0.0, 1.0, 3.2),
                Point3W(0.0, 1.0, -1.0),
                Vec3W(0.0, 1.0, 0.0),
                28.0,
                4.2,
            )
        case "weekend":
            world = CpuScene[16, 16](make_weekend_scene())
        case "mixed":
            world = make_mixed_triangle_world()
            camera = mixed_triangle_camera(world)
        case _:
            raise Error("unknown benchmark scene: " + name)
    print(
        name,
        settings.image_width,
        settings.image_height,
        "spp",
        settings.samples_per_pixel,
        "repeats",
        repeats,
    )
    benchmark[1, False]("scalar", settings, camera, world, repeats)
    benchmark[simd_width_of[Float32](), False](
        "SIMD", settings, camera, world, repeats
    )
    benchmark[simd_width_of[Float32](), True](
        "parallel SIMD", settings, camera, world, repeats
    )
