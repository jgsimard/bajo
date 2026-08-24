"""GPU RT direct-light benchmark with 64 emissive spheres."""

from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.core import Point3f32
from bajo.rt import Color, Integrator, RenderSettings, SceneBuilder, SceneData
from bajo.rt.gpu.render import enqueue_render_gpu
from bajo.rt.gpu.resources import GpuRtRenderTarget, download_gpu_pixels
from bajo.rt.gpu.scene import GpuRtScene, prepare_gpu_scene
from bajo.benchmark.gpu_harness import (
    BENCH_REPEATS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_DEPTH,
    RNG_SEED,
    GpuRtBenchResult,
    finalize_gpu_rt_timings,
    gpu_rt_camera,
    gpu_rt_checksum,
    print_gpu_rt_result,
)


comptime SPHERE_LIGHT_SPP = 64


def _make_world() raises -> SceneData:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.7))
    var light = builder.add_emissive(Color(8.0))
    builder.add_quad(
        Point3f32[.WORLD](-1.2, 0.0, -2.0),
        Point3f32[.WORLD](1.2, 0.0, -2.0),
        Point3f32[.WORLD](1.2, 2.0, -2.0),
        Point3f32[.WORLD](-1.2, 2.0, -2.0),
        matte,
    )
    for light_y in range(8):
        for light_x in range(8):
            builder.add_sphere(
                Point3f32[.WORLD](
                    -0.84 + Float32(light_x) * 0.24,
                    0.20 + Float32(light_y) * 0.22,
                    -1.45,
                ),
                0.035,
                light,
            )
    return builder^.finish()


def _bench_integrator[
    integrator: Integrator,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtScene[.SPHERES_TRIANGLES],
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
        print("GPU RT sphere-light benchmark skipped: no accelerator")
        return

    var settings = RenderSettings(
        IMAGE_WIDTH, IMAGE_HEIGHT, SPHERE_LIGHT_SPP, RNG_SEED, MAX_DEPTH
    )
    var sample_count = IMAGE_WIDTH * IMAGE_HEIGHT * SPHERE_LIGHT_SPP
    print("GPU RT 64-sphere-light benchmark")
    print(
        t"Diffuse receiver, {IMAGE_WIDTH}x{IMAGE_HEIGHT}, "
        t"spp={SPHERE_LIGHT_SPP}, max_depth={MAX_DEPTH}, "
        t"median of {BENCH_REPEATS}"
    )

    with DeviceContext() as ctx:
        var data = _make_world()
        var world = prepare_gpu_scene[.SPHERES_TRIANGLES](ctx, data)
        var target = GpuRtRenderTarget(ctx, settings, gpu_rt_camera())
        ctx.synchronize()
        print(
            "hot timings exclude scene setup, target allocation, and download"
        )
        print_gpu_rt_result(
            "NEE-SPHERE-64",
            _bench_integrator[.NEE](ctx, target, world, settings),
            sample_count,
        )
        print_gpu_rt_result(
            "MIS-SPHERE-64",
            _bench_integrator[.MIS](ctx, target, world, settings),
            sample_count,
        )
