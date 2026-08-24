"""Shade-heavy GPU RT benchmark for single deferred material kinds."""

from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.core import Point3f32, Vec3f32
from bajo.rt import (
    Camera,
    Color,
    Integrator,
    RenderSettings,
    SceneBuilder,
    SceneData,
    SurfaceId,
)
from bajo.rt.gpu.config import GPU_RT_BVH_WIDE4
from bajo.rt.gpu.render import enqueue_render_gpu
from bajo.rt.gpu.resources import GpuRtRenderTarget, download_gpu_pixels
from bajo.rt.gpu.scene import GpuRtScene, prepare_gpu_scene
from bajo.benchmark.gpu_harness import (
    BENCH_REPEATS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_DEPTH,
    RNG_SEED,
    SAMPLES_PER_PIXEL,
    GpuRtBenchResult,
    finalize_gpu_rt_timings,
    gpu_rt_checksum,
    print_gpu_rt_result,
)


def _camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.55, 3.2),
        Point3f32[.WORLD](0.0, 0.35, -1.5),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        35.0,
        4.2,
    )


def _add_object_spheres(mut builder: SceneBuilder, surface: SurfaceId[1]):
    builder.add_sphere(Point3f32[.WORLD](-1.05, 0.00, -1.25), 0.48, surface)
    builder.add_sphere(Point3f32[.WORLD](0.00, 0.00, -1.10), 0.55, surface)
    builder.add_sphere(Point3f32[.WORLD](1.05, 0.00, -1.35), 0.48, surface)
    builder.add_sphere(Point3f32[.WORLD](-0.58, 0.88, -1.65), 0.43, surface)
    builder.add_sphere(Point3f32[.WORLD](0.58, 0.88, -1.65), 0.43, surface)
    builder.add_sphere(Point3f32[.WORLD](0.00, 1.58, -2.05), 0.38, surface)


def _make_metal_world() raises -> SceneData:
    var builder = SceneBuilder()
    var ground = builder.add_lambertian(Color(0.42, 0.44, 0.48))
    var metal = builder.add_metal(Color(0.84, 0.74, 0.55), 0.12)
    var light = builder.add_emissive(Color(7.0, 6.5, 5.5))
    builder.add_sphere(Point3f32[.WORLD](0.0, -100.55, -1.5), 100.0, ground)
    _add_object_spheres(builder, metal)
    builder.add_sphere(Point3f32[.WORLD](0.0, 2.75, -1.75), 0.48, light)
    return builder^.finish()


def _make_dielectric_world() raises -> SceneData:
    var builder = SceneBuilder()
    var ground = builder.add_lambertian(Color(0.42, 0.44, 0.48))
    var glass = builder.add_dielectric(1.5)
    var light = builder.add_emissive(Color(7.0, 6.5, 5.5))
    builder.add_sphere(Point3f32[.WORLD](0.0, -100.55, -1.5), 100.0, ground)
    _add_object_spheres(builder, glass)
    builder.add_sphere(Point3f32[.WORLD](0.0, 2.75, -1.75), 0.48, light)
    return builder^.finish()


def _bench_integrator[
    integrator: Integrator,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtScene[.SPHERES, GPU_RT_BVH_WIDE4],
    settings: RenderSettings,
) raises -> GpuRtBenchResult:
    enqueue_render_gpu[integrator](ctx, target, world, settings)
    ctx.synchronize()

    var submit_times = List[Int](capacity=BENCH_REPEATS)
    var render_times = List[Int](capacity=BENCH_REPEATS)
    for _ in range(BENCH_REPEATS):
        var render_t0 = perf_counter_ns()
        enqueue_render_gpu[integrator](ctx, target, world, settings)
        var submit_t1 = perf_counter_ns()
        ctx.synchronize()
        var render_t1 = perf_counter_ns()
        submit_times.append(Int(submit_t1 - render_t0))
        render_times.append(Int(render_t1 - render_t0))

    var pixels = download_gpu_pixels(ctx, target)
    return finalize_gpu_rt_timings(
        submit_times, render_times, gpu_rt_checksum(pixels)
    )


def main() raises:
    comptime if not has_accelerator():
        print("GPU RT material benchmark skipped: no accelerator")
        return

    var settings = RenderSettings(
        IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES_PER_PIXEL, RNG_SEED, MAX_DEPTH
    )
    var sample_count = IMAGE_WIDTH * IMAGE_HEIGHT * SAMPLES_PER_PIXEL
    print("GPU RT deferred-material benchmark")
    print(
        t"Spheres, {IMAGE_WIDTH}x{IMAGE_HEIGHT}, spp={SAMPLES_PER_PIXEL}, "
        t"max_depth={MAX_DEPTH}, median of {BENCH_REPEATS}"
    )

    with DeviceContext() as ctx:
        var metal_data = _make_metal_world()
        var dielectric_data = _make_dielectric_world()
        var metal_world = prepare_gpu_scene[.SPHERES](ctx, metal_data)
        var dielectric_world = prepare_gpu_scene[.SPHERES](ctx, dielectric_data)
        var target = GpuRtRenderTarget(ctx, settings, _camera())
        ctx.synchronize()
        print(
            "hot timings exclude scene setup, target allocation, and download"
        )
        print_gpu_rt_result(
            "METAL-PATH",
            _bench_integrator[.PATH](ctx, target, metal_world, settings),
            sample_count,
        )
        print_gpu_rt_result(
            "DIELECTRIC-PATH",
            _bench_integrator[.PATH](ctx, target, dielectric_world, settings),
            sample_count,
        )
        print_gpu_rt_result(
            "METAL-NEE",
            _bench_integrator[.NEE](ctx, target, metal_world, settings),
            sample_count,
        )
        print_gpu_rt_result(
            "DIELECTRIC-NEE",
            _bench_integrator[.NEE](ctx, target, dielectric_world, settings),
            sample_count,
        )
        print_gpu_rt_result(
            "METAL-MIS",
            _bench_integrator[.MIS](ctx, target, metal_world, settings),
            sample_count,
        )
        print_gpu_rt_result(
            "DIELECTRIC-MIS",
            _bench_integrator[.MIS](ctx, target, dielectric_world, settings),
            sample_count,
        )
