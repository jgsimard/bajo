"""Veach-style MIS comparison with glossy bars and spherical emitters."""

from std.math import round
from std.time import perf_counter_ns

from bajo.core import Point3W, Vec3W
from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Camera,
    Color,
    RENDER,
    RenderSettings,
    SceneBuilder,
    SurfaceId,
    CpuScene,
    render_wavefront,
    write_ppm_from_colors,
)


comptime IMAGE_WIDTH = 480
comptime IMAGE_HEIGHT = 270
comptime SAMPLES_PER_PIXEL = 32
comptime MAX_DEPTH = 10
comptime RNG_SEED = UInt64(2026)
comptime PATH_OUTPUT = "mis_showcase_path.ppm"
comptime NEE_OUTPUT = "mis_showcase_nee.ppm"
comptime MIS_OUTPUT = "mis_showcase_mis.ppm"
comptime COMPARISON_OUTPUT = "mis_showcase_comparison.ppm"


def _add_box(
    mut builder: SceneBuilder,
    minimum: Point3W,
    maximum: Point3W,
    surface: SurfaceId[1],
):
    var x0 = minimum.x
    var y0 = minimum.y
    var z0 = minimum.z
    var x1 = maximum.x
    var y1 = maximum.y
    var z1 = maximum.z
    builder.add_quad(
        Point3W(x0, y0, z0),
        Point3W(x1, y0, z0),
        Point3W(x1, y0, z1),
        Point3W(x0, y0, z1),
        surface,
    )
    builder.add_quad(
        Point3W(x0, y1, z1),
        Point3W(x1, y1, z1),
        Point3W(x1, y1, z0),
        Point3W(x0, y1, z0),
        surface,
    )
    builder.add_quad(
        Point3W(x0, y0, z1),
        Point3W(x1, y0, z1),
        Point3W(x1, y1, z1),
        Point3W(x0, y1, z1),
        surface,
    )
    builder.add_quad(
        Point3W(x1, y0, z0),
        Point3W(x0, y0, z0),
        Point3W(x0, y1, z0),
        Point3W(x1, y1, z0),
        surface,
    )
    builder.add_quad(
        Point3W(x0, y0, z0),
        Point3W(x0, y0, z1),
        Point3W(x0, y1, z1),
        Point3W(x0, y1, z0),
        surface,
    )
    builder.add_quad(
        Point3W(x1, y0, z1),
        Point3W(x1, y0, z0),
        Point3W(x1, y1, z0),
        Point3W(x1, y1, z1),
        surface,
    )


def make_mis_showcase_world[
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
]() raises -> CpuScene[world_bvh_width, instance_bvh_width]:
    var builder = SceneBuilder()
    var room = builder.add_lambertian(Color(0.48, 0.48, 0.48))
    var rough = builder.add_metal(Color(0.64), 0.50)
    var medium = builder.add_metal(Color(0.64), 0.20)
    var glossy = builder.add_metal(Color(0.64), 0.10)
    var polished = builder.add_metal(Color(0.64), 0.01)

    # Radiance is inversely proportional to sphere area, so each light has
    # roughly equal power while its apparent solid angle changes radically.
    var point_like = builder.add_emissive(Color(100.0))
    var small = builder.add_emissive(Color(20.0))
    var medium_light = builder.add_emissive(Color(10.0))
    var large = builder.add_emissive(Color(1.0))

    builder.add_sphere(
        Point3W(-1.65, 2.65, -3.25),
        0.05,
        point_like,
    )
    builder.add_sphere(
        Point3W(-0.55, 2.65, -3.25),
        0.12,
        small,
    )
    builder.add_sphere(
        Point3W(0.60, 2.65, -3.25),
        0.20,
        medium_light,
    )
    builder.add_sphere(
        Point3W(1.75, 2.65, -3.25),
        0.45,
        large,
    )

    builder.add_quad(
        Point3W(-4.0, 0.0, 1.0),
        Point3W(4.0, 0.0, 1.0),
        Point3W(4.0, 0.0, -4.2),
        Point3W(-4.0, 0.0, -4.2),
        room,
    )
    builder.add_quad(
        Point3W(-4.0, 0.0, 1.0),
        Point3W(-4.0, 0.0, -4.1),
        Point3W(-4.0, 4.0, -4.1),
        Point3W(-4.0, 4.0, 1.0),
        room,
    )
    builder.add_quad(
        Point3W(4.0, 0.0, -4.1),
        Point3W(4.0, 0.0, 1.0),
        Point3W(4.0, 4.0, 1.0),
        Point3W(4.0, 4.0, -4.1),
        room,
    )
    builder.add_quad(
        Point3W(-4.0, 0.0, -4.1),
        Point3W(4.0, 0.0, -4.1),
        Point3W(4.0, 4.0, -4.1),
        Point3W(-4.0, 4.0, -4.1),
        room,
    )

    # Rough at the back, polished at the front. The same four lights reflect
    # in every bar, covering both broad and sharply peaked BSDF distributions.
    builder.add_quad(
        Point3W(-2.25, 0.82, -2.72),
        Point3W(2.25, 0.82, -2.72),
        Point3W(2.25, 1.22, -3.30),
        Point3W(-2.25, 1.22, -3.30),
        rough,
    )
    builder.add_quad(
        Point3W(-2.45, 0.58, -1.80),
        Point3W(2.45, 0.58, -1.80),
        Point3W(2.45, 0.83, -2.42),
        Point3W(-2.45, 0.83, -2.42),
        medium,
    )
    builder.add_quad(
        Point3W(-2.65, 0.34, -0.86),
        Point3W(2.65, 0.34, -0.86),
        Point3W(2.65, 0.50, -1.52),
        Point3W(-2.65, 0.50, -1.52),
        glossy,
    )
    builder.add_quad(
        Point3W(-2.85, 0.12, 0.12),
        Point3W(2.85, 0.12, 0.12),
        Point3W(2.85, 0.21, -0.58),
        Point3W(-2.85, 0.21, -0.58),
        polished,
    )

    var scene = builder^.finish()
    return CpuScene[world_bvh_width, instance_bvh_width](scene^)


def _write_comparison(
    path_pixels: List[Color],
    nee_pixels: List[Color],
    mis_pixels: List[Color],
) raises:
    var width = 3 * IMAGE_WIDTH
    var comparison = List[Color](length=width * IMAGE_HEIGHT, fill=Color(0.0))
    for y in range(IMAGE_HEIGHT):
        for x in range(IMAGE_WIDTH):
            var source = y * IMAGE_WIDTH + x
            var target = y * width + x
            comparison[target] = path_pixels[source]
            comparison[target + IMAGE_WIDTH] = nee_pixels[source]
            comparison[target + 2 * IMAGE_WIDTH] = mis_pixels[source]
    write_ppm_from_colors(COMPARISON_OUTPUT, width, IMAGE_HEIGHT, comparison)


def main() raises:
    print("Veach-style MIS showcase at equal spp")
    var settings = RenderSettings(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        SAMPLES_PER_PIXEL,
        RNG_SEED,
        MAX_DEPTH,
    )
    var camera = Camera.from_vfov(
        Point3W(0.0, 3.0, 6.2),
        Point3W(0.0, 0.85, -1.65),
        Vec3W(0.0, 1.0, 0.0),
        31.0,
        8.10,
    )
    var world = make_mis_showcase_world()

    var path_t0 = perf_counter_ns()
    var path_result = render_wavefront[RENDER.PATH](settings, camera, world)
    var path_t1 = perf_counter_ns()
    write_ppm_from_colors(
        PATH_OUTPUT, IMAGE_WIDTH, IMAGE_HEIGHT, path_result.pixels
    )

    var nee_t0 = perf_counter_ns()
    var nee_result = render_wavefront[RENDER.NEE](settings, camera, world)
    var nee_t1 = perf_counter_ns()
    write_ppm_from_colors(
        NEE_OUTPUT, IMAGE_WIDTH, IMAGE_HEIGHT, nee_result.pixels
    )

    var mis_t0 = perf_counter_ns()
    var mis_result = render_wavefront[RENDER.MIS](settings, camera, world)
    var mis_t1 = perf_counter_ns()
    write_ppm_from_colors(
        MIS_OUTPUT, IMAGE_WIDTH, IMAGE_HEIGHT, mis_result.pixels
    )
    _write_comparison(path_result.pixels, nee_result.pixels, mis_result.pixels)

    print(
        t"path: {round(ns_to_ms(Int(path_t1 - path_t0)), 3)} ms ->"
        t" {PATH_OUTPUT}"
    )
    print(
        t"NEE : {round(ns_to_ms(Int(nee_t1 - nee_t0)), 3)} ms -> {NEE_OUTPUT}"
    )
    print(
        t"MIS : {round(ns_to_ms(Int(mis_t1 - mis_t0)), 3)} ms -> {MIS_OUTPUT}"
    )
    print(t"comparison (path | NEE | MIS) -> {COMPARISON_OUTPUT}")
