from std.time import perf_counter_ns

from bajo.core import Vec3f32, length
from bajo.core.random import Rng
from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Camera,
    Color,
    Instance,
    Point3,
    RENDER_PATH,
    RenderSettings,
    Sphere,
    SurfaceId,
    SurfaceStore,
    World,
    add_sphere,
    render,
    write_ppm_from_colors,
)


comptime OUTPUT_PATH = "rtiaw.ppm"
comptime IMAGE_WIDTH = 480
comptime IMAGE_HEIGHT = 270
comptime SAMPLES_PER_PIXEL = 10
comptime MAX_DEPTH = 12
comptime RNG_SEED = UInt64(1234)
comptime RENDER_ALGORITHM = RENDER_PATH


def make_weekend_world() -> World:
    var rng = Rng(seed=42, id=7)
    var surfaces = SurfaceStore()
    var spheres = List[Sphere]()
    var sphere_surfaces = List[SurfaceId]()
    var triangle_vertices = List[Vec3f32]()
    var triangle_surfaces = List[SurfaceId]()
    var triangle_meshes = List[List[Vec3f32]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId]()

    var ground_surface = surfaces.add_lambertian(Color(0.5, 0.5, 0.5))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3(0.0, -1000.0, 0.0),
        1000.0,
        ground_surface,
    )

    for a in range(-11, 11):
        for b in range(-11, 11):
            var choose_mat = rng.f32()
            var center = Point3(
                Float32(a) + 0.9 * rng.f32(),
                0.2,
                Float32(b) + 0.9 * rng.f32(),
            )

            if length(center - Point3(4.0, 0.2, 0.0)) > 0.9:
                if choose_mat < 0.8:
                    var albedo = rng.vec3f32() * rng.vec3f32()
                    var surface = surfaces.add_lambertian(albedo)
                    add_sphere(spheres, sphere_surfaces, center, 0.2, surface)
                elif choose_mat < 0.95:
                    var albedo = rng.vec3f32(0.5, 1.0)
                    var fuzz = rng.f32(0.0, 0.5)
                    var surface = surfaces.add_metal(albedo, fuzz)
                    add_sphere(spheres, sphere_surfaces, center, 0.2, surface)
                else:
                    var surface = surfaces.add_dielectric(1.5)
                    add_sphere(spheres, sphere_surfaces, center, 0.2, surface)

    var glass = surfaces.add_dielectric(1.5)
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3(0.0, 1.0, 0.0),
        1.0,
        glass,
    )

    var diffuse = surfaces.add_lambertian(Color(0.4, 0.2, 0.1))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3(-4.0, 1.0, 0.0),
        1.0,
        diffuse,
    )

    var metal = surfaces.add_metal(Color(0.7, 0.6, 0.5), 0.0)
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3(4.0, 1.0, 0.0),
        1.0,
        metal,
    )

    return World(
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )


def main() raises:
    print("Ray Tracing in One Weekend, bajo CPU")
    var settings = RenderSettings(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        SAMPLES_PER_PIXEL,
        RNG_SEED,
    )
    print(
        t"image: {settings.image_width}x{settings.image_height} | "
        t"samples: {SAMPLES_PER_PIXEL} | depth: {MAX_DEPTH}"
    )

    var world = make_weekend_world()
    var camera = Camera.from_vfov(
        Point3(13.0, 2.0, 3.0),
        Point3(0.0, 0.0, 0.0),
        Vec3f32(0.0, 1.0, 0.0),
        20.0,
        10.0,
        0.6,
    )

    print(t"spheres: {len(world.spheres)}")
    print(
        t"surfaces:"
        t" {len(world.surfaces.lambertians) + len(world.surfaces.metals) + len(world.surfaces.dielectrics)}"
    )

    var t0 = perf_counter_ns()
    var result = render[RENDER_ALGORITHM, MAX_DEPTH](settings, camera, world)
    var t1 = perf_counter_ns()

    write_ppm_from_colors(
        OUTPUT_PATH,
        settings.image_width,
        settings.image_height,
        result.pixels,
    )
    print(t"render ms: {round(ns_to_ms(Int(t1 - t0)), 3)}")
    print(t"  total  : {round(ns_to_ms(result.timings.total_ns), 3)} ms")
    print(t"  init   : {round(ns_to_ms(result.timings.init_ns), 3)} ms")
    print(t"  kernel : {round(ns_to_ms(result.timings.render_ns), 3)} ms")
    print(t"  pixels : {result.timings.pixel_count}")
    print(t"  samples: {result.timings.sample_count}")
    print(t"  depth  : {result.timings.max_depth}")
    print(t"wrote {OUTPUT_PATH}")
