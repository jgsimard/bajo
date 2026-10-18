"""Render a Cornell box with plain path tracing and next-event estimation."""

from std.math import round

from bajo.bvh.cpu import CpuBvhBuildMethod
from bajo.core import Point3W, Vec3W
from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Camera,
    Color,
    RenderSettings,
    SceneBuilder,
    CpuScene,
    render_wavefront,
    write_ppm_from_colors,
)


comptime IMAGE_WIDTH = 256
comptime IMAGE_HEIGHT = 256
comptime SAMPLES_PER_PIXEL = 32
comptime MAX_DEPTH = 12
comptime RNG_SEED = UInt64(2026)
comptime PATH_OUTPUT = "cornell_path.ppm"
comptime NEE_OUTPUT = "cornell_nee.ppm"
comptime MIS_OUTPUT = "cornell_mis.ppm"


def make_cornell_world[
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
    build_method: CpuBvhBuildMethod = .SAH,
]() raises -> CpuScene[world_bvh_width, instance_bvh_width]:
    var builder = SceneBuilder()
    var white = builder.add_lambertian(Color(0.73, 0.73, 0.73))
    var red = builder.add_lambertian(Color(0.65, 0.05, 0.05))
    var green = builder.add_lambertian(Color(0.12, 0.45, 0.15))
    var light = builder.add_emissive(Color(15.0, 15.0, 15.0))

    # Floor, ceiling, back, red left wall, and green right wall.
    builder.add_quad(
        Point3W(-1.0, 0.0, 0.0),
        Point3W(1.0, 0.0, 0.0),
        Point3W(1.0, 0.0, -2.0),
        Point3W(-1.0, 0.0, -2.0),
        white,
    )
    builder.add_quad(
        Point3W(-1.0, 2.0, -2.0),
        Point3W(1.0, 2.0, -2.0),
        Point3W(1.0, 2.0, 0.0),
        Point3W(-1.0, 2.0, 0.0),
        white,
    )
    builder.add_quad(
        Point3W(-1.0, 0.0, -2.0),
        Point3W(1.0, 0.0, -2.0),
        Point3W(1.0, 2.0, -2.0),
        Point3W(-1.0, 2.0, -2.0),
        white,
    )
    builder.add_quad(
        Point3W(-1.0, 0.0, 0.0),
        Point3W(-1.0, 0.0, -2.0),
        Point3W(-1.0, 2.0, -2.0),
        Point3W(-1.0, 2.0, 0.0),
        red,
    )
    builder.add_quad(
        Point3W(1.0, 0.0, -2.0),
        Point3W(1.0, 0.0, 0.0),
        Point3W(1.0, 2.0, 0.0),
        Point3W(1.0, 2.0, -2.0),
        green,
    )

    # Downward-facing rectangular area light.
    builder.add_quad(
        Point3W(-0.28, 1.99, -1.28),
        Point3W(0.28, 1.99, -1.28),
        Point3W(0.28, 1.99, -0.72),
        Point3W(-0.28, 1.99, -0.72),
        light,
    )

    builder.add_box(
        Point3W(-0.72, 0.0, -1.55),
        Point3W(-0.12, 1.15, -0.85),
        white,
    )
    builder.add_box(
        Point3W(0.18, 0.0, -1.15),
        Point3W(0.72, 0.62, -0.48),
        white,
    )

    var scene = builder^.finish()
    return CpuScene[world_bvh_width, instance_bvh_width].__init__[build_method](
        scene^
    )


def main() raises:
    print("Cornell box: path tracing, NEE, and MIS")
    var settings = RenderSettings(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        SAMPLES_PER_PIXEL,
        RNG_SEED,
        MAX_DEPTH,
    )
    var camera = Camera.from_vfov(
        Point3W(0.0, 1.0, 3.2),
        Point3W(0.0, 1.0, -1.0),
        Vec3W(0.0, 1.0, 0.0),
        28.0,
        4.2,
    )
    var world = make_cornell_world()

    var path_result = render_wavefront[.PATH](settings, camera, world)
    write_ppm_from_colors(
        PATH_OUTPUT, IMAGE_WIDTH, IMAGE_HEIGHT, path_result.pixels
    )

    var nee_result = render_wavefront[.NEE](settings, camera, world)
    write_ppm_from_colors(
        NEE_OUTPUT, IMAGE_WIDTH, IMAGE_HEIGHT, nee_result.pixels
    )

    var mis_result = render_wavefront[.MIS](settings, camera, world)
    write_ppm_from_colors(
        MIS_OUTPUT, IMAGE_WIDTH, IMAGE_HEIGHT, mis_result.pixels
    )

    print(
        t"path: {round(ns_to_ms(path_result.timings.total_ns), 3)} ms ->"
        t" {PATH_OUTPUT}"
    )
    print(
        t"NEE : {round(ns_to_ms(nee_result.timings.total_ns), 3)} ms ->"
        t" {NEE_OUTPUT}"
    )
    print(
        t"MIS : {round(ns_to_ms(mis_result.timings.total_ns), 3)} ms ->"
        t" {MIS_OUTPUT}"
    )
