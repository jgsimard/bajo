from std.time import perf_counter_ns

from bajo.bvh.cpu import CpuBvhBuildMethod
from bajo.core import Vec3f32, length, Vec3W, Point3W, Point3
from bajo.core.random import Rng
from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Camera,
    Color,
    Integrator,
    RenderSettings,
    SceneBuilder,
    CpuScene,
    render_wavefront,
    write_ppm_from_colors,
)


comptime OUTPUT_PATH = "rtiaw.ppm"
comptime IMAGE_WIDTH = 600
comptime IMAGE_HEIGHT = 400
comptime SAMPLES_PER_PIXEL = 10
comptime MAX_DEPTH = 32
comptime RNG_SEED = UInt64(1234)
comptime INTEGRATOR = Integrator.PATH


def make_weekend_world[
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
    build_method: CpuBvhBuildMethod = .SAH,
]() raises -> CpuScene[world_bvh_width, instance_bvh_width]:
    var rng = Rng(seed=42, id=7)
    var builder = SceneBuilder()

    var ground_surface = builder.add_lambertian(Color(0.5, 0.5, 0.5))
    builder.add_sphere(
        Point3W(0.0, -1000.0, 0.0),
        1000.0,
        ground_surface,
    )

    for a in range(-11, 11):
        for b in range(-11, 11):
            var choose_mat = rng.f32()
            var center = Point3W(
                Float32(a) + 0.9 * rng.f32(),
                0.2,
                Float32(b) + 0.9 * rng.f32(),
            )

            if length(center - Point3W(4.0, 0.2, 0.0)) > 0.9:
                if choose_mat < 0.8:
                    var albedo = rng.vec3f32[.WORLD]() * rng.vec3f32[.WORLD]()
                    var surface = builder.add_lambertian(albedo)
                    builder.add_sphere(center, 0.2, surface)
                elif choose_mat < 0.95:
                    var albedo = rng.vec3f32[.WORLD](0.5, 1.0)
                    var fuzz = rng.f32(0.0, 0.5)
                    var surface = builder.add_metal(albedo, fuzz)
                    builder.add_sphere(center, 0.2, surface)
                else:
                    var surface = builder.add_dielectric(1.5)
                    builder.add_sphere(center, 0.2, surface)

    var glass = builder.add_dielectric(1.5)
    builder.add_sphere(
        Point3W(0.0, 1.0, 0.0),
        1.0,
        glass,
    )

    var diffuse = builder.add_lambertian(Color(0.4, 0.2, 0.1))
    builder.add_sphere(
        Point3W(-4.0, 1.0, 0.0),
        1.0,
        diffuse,
    )

    var metal = builder.add_metal(Color(0.7, 0.6, 0.5), 0.0)
    builder.add_sphere(
        Point3W(4.0, 1.0, 0.0),
        1.0,
        metal,
    )

    var scene = builder^.finish()
    return CpuScene[world_bvh_width, instance_bvh_width].__init__[build_method](
        scene^
    )


def main() raises:
    print("Ray Tracing in One Weekend, bajo CPU")
    var settings = RenderSettings(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        SAMPLES_PER_PIXEL,
        RNG_SEED,
        MAX_DEPTH,
    )
    print(
        t"image: {settings.image_width}x{settings.image_height} | "
        t"samples: {SAMPLES_PER_PIXEL} | depth: {MAX_DEPTH}"
    )

    var world = make_weekend_world()
    var camera = Camera.from_vfov(
        Point3W(13.0, 2.0, 3.0),
        Point3W(0.0, 0.0, 0.0),
        Vec3W(0.0, 1.0, 0.0),
        20.0,
        10.0,
        0.6,
    )

    print(t"spheres: {len(world.scene_data().spheres())}")
    print(
        t"surfaces:"
        t" {len(world.scene_data().surfaces().lambertians) + len(world.scene_data().surfaces().metals) + len(world.scene_data().surfaces().dielectrics)}"
    )

    var t0 = perf_counter_ns()
    var result = render_wavefront[INTEGRATOR](settings, camera, world)
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
