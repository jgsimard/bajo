"""Render a Cornell box with plain path tracing and next-event estimation."""

from std.math import round
from std.time import perf_counter_ns

from bajo.core import Frame, Point3W, Point3f32, Vec3W
from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Camera,
    Color,
    Instance,
    RENDER,
    RenderSettings,
    Sphere,
    SurfaceId,
    SurfaceStore,
    CpuScene,
    add_triangle,
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


def _add_quad(
    mut vertices: List[Point3W],
    mut surfaces: List[SurfaceId[1]],
    a: Point3W,
    b: Point3W,
    c: Point3W,
    d: Point3W,
    surface: SurfaceId[1],
):
    add_triangle(vertices, surfaces, a, b, c, surface)
    add_triangle(vertices, surfaces, a, c, d, surface)


def _add_box(
    mut vertices: List[Point3W],
    mut surfaces: List[SurfaceId[1]],
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

    _add_quad(
        vertices,
        surfaces,
        Point3W(x0, y0, z0),
        Point3W(x1, y0, z0),
        Point3W(x1, y0, z1),
        Point3W(x0, y0, z1),
        surface,
    )
    _add_quad(
        vertices,
        surfaces,
        Point3W(x0, y1, z1),
        Point3W(x1, y1, z1),
        Point3W(x1, y1, z0),
        Point3W(x0, y1, z0),
        surface,
    )
    _add_quad(
        vertices,
        surfaces,
        Point3W(x0, y0, z1),
        Point3W(x1, y0, z1),
        Point3W(x1, y1, z1),
        Point3W(x0, y1, z1),
        surface,
    )
    _add_quad(
        vertices,
        surfaces,
        Point3W(x1, y0, z0),
        Point3W(x0, y0, z0),
        Point3W(x0, y1, z0),
        Point3W(x1, y1, z0),
        surface,
    )
    _add_quad(
        vertices,
        surfaces,
        Point3W(x0, y0, z0),
        Point3W(x0, y0, z1),
        Point3W(x0, y1, z1),
        Point3W(x0, y1, z0),
        surface,
    )
    _add_quad(
        vertices,
        surfaces,
        Point3W(x1, y0, z1),
        Point3W(x1, y0, z0),
        Point3W(x1, y1, z0),
        Point3W(x1, y1, z1),
        surface,
    )


def make_cornell_world[
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
]() -> CpuScene[world_bvh_width, instance_bvh_width]:
    var store = SurfaceStore()
    var white = store.add_lambertian(Color(0.73, 0.73, 0.73))
    var red = store.add_lambertian(Color(0.65, 0.05, 0.05))
    var green = store.add_lambertian(Color(0.12, 0.45, 0.15))
    var light = store.add_emissive(Color(15.0, 15.0, 15.0))

    var vertices = List[Point3W]()
    var surfaces = List[SurfaceId[1]]()

    # Floor, ceiling, back, red left wall, and green right wall.
    _add_quad(
        vertices,
        surfaces,
        Point3W(-1.0, 0.0, 0.0),
        Point3W(1.0, 0.0, 0.0),
        Point3W(1.0, 0.0, -2.0),
        Point3W(-1.0, 0.0, -2.0),
        white,
    )
    _add_quad(
        vertices,
        surfaces,
        Point3W(-1.0, 2.0, -2.0),
        Point3W(1.0, 2.0, -2.0),
        Point3W(1.0, 2.0, 0.0),
        Point3W(-1.0, 2.0, 0.0),
        white,
    )
    _add_quad(
        vertices,
        surfaces,
        Point3W(-1.0, 0.0, -2.0),
        Point3W(1.0, 0.0, -2.0),
        Point3W(1.0, 2.0, -2.0),
        Point3W(-1.0, 2.0, -2.0),
        white,
    )
    _add_quad(
        vertices,
        surfaces,
        Point3W(-1.0, 0.0, 0.0),
        Point3W(-1.0, 0.0, -2.0),
        Point3W(-1.0, 2.0, -2.0),
        Point3W(-1.0, 2.0, 0.0),
        red,
    )
    _add_quad(
        vertices,
        surfaces,
        Point3W(1.0, 0.0, -2.0),
        Point3W(1.0, 0.0, 0.0),
        Point3W(1.0, 2.0, 0.0),
        Point3W(1.0, 2.0, -2.0),
        green,
    )

    # Downward-facing rectangular area light.
    _add_quad(
        vertices,
        surfaces,
        Point3W(-0.28, 1.99, -1.28),
        Point3W(0.28, 1.99, -1.28),
        Point3W(0.28, 1.99, -0.72),
        Point3W(-0.28, 1.99, -0.72),
        light,
    )

    _add_box(
        vertices,
        surfaces,
        Point3W(-0.72, 0.0, -1.55),
        Point3W(-0.12, 1.15, -0.85),
        white,
    )
    _add_box(
        vertices,
        surfaces,
        Point3W(0.18, 0.0, -1.15),
        Point3W(0.72, 0.62, -0.48),
        white,
    )

    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var instances = List[Instance]()
    var instance_surfaces = List[SurfaceId[1]]()
    return CpuScene[world_bvh_width, instance_bvh_width](
        spheres^,
        sphere_surfaces^,
        vertices^,
        surfaces^,
        meshes^,
        instances^,
        instance_surfaces^,
        store^,
    )


def main() raises:
    print("Cornell box: path tracing, NEE, and MIS")
    var settings = RenderSettings(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        SAMPLES_PER_PIXEL,
        RNG_SEED,
    )
    var camera = Camera.from_vfov(
        Point3W(0.0, 1.0, 3.2),
        Point3W(0.0, 1.0, -1.0),
        Vec3W(0.0, 1.0, 0.0),
        28.0,
        4.2,
    )
    var world = make_cornell_world()

    var path_t0 = perf_counter_ns()
    var path_result = render_wavefront[RENDER.PATH, MAX_DEPTH](
        settings, camera, world
    )
    var path_t1 = perf_counter_ns()
    write_ppm_from_colors(
        PATH_OUTPUT, IMAGE_WIDTH, IMAGE_HEIGHT, path_result.pixels
    )

    var nee_t0 = perf_counter_ns()
    var nee_result = render_wavefront[RENDER.NEE, MAX_DEPTH](
        settings, camera, world
    )
    var nee_t1 = perf_counter_ns()
    write_ppm_from_colors(
        NEE_OUTPUT, IMAGE_WIDTH, IMAGE_HEIGHT, nee_result.pixels
    )

    var mis_t0 = perf_counter_ns()
    var mis_result = render_wavefront[RENDER.MIS, MAX_DEPTH](
        settings, camera, world
    )
    var mis_t1 = perf_counter_ns()
    write_ppm_from_colors(
        MIS_OUTPUT, IMAGE_WIDTH, IMAGE_HEIGHT, mis_result.pixels
    )

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
