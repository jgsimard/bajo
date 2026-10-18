"""GPU RT workloads: triangles (default) or mixed spheres/triangles."""

from std.math import round
from std.sys import has_accelerator
from std.sys.arg import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.core import Point3f32, Vec3f32
from bajo.core.utils import ns_to_ms
from bajo.rt import Camera, SceneData, Integrator, RenderSettings
from bajo.rt.gpu.config import GpuRtBvhFormat, GpuRtSceneKind, GPU_RT_BVH_WIDE4
from bajo.rt.gpu.scene import prepare_gpu_scene
from bajo.benchmark.gpu_harness import (
    BENCH_REPEATS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_DEPTH,
    RNG_SEED,
    SAMPLES_PER_PIXEL,
    bench_gpu_integrator,
    gpu_rt_camera,
    make_many_light_scene,
    print_gpu_rt_result,
)
from examples.cornell_box import make_cornell_scene
from examples.mis_showcase import make_mis_showcase_scene


def _benchmark[
    kind: GpuRtSceneKind
](
    mut ctx: DeviceContext,
    world: SceneData,
    camera: Camera,
    settings: RenderSettings,
) raises:
    comptime sphere_format = (
        GpuRtBvhFormat(8, 4, .WIDE) if kind
        == .SPHERES_TRIANGLES else GPU_RT_BVH_WIDE4
    )
    var sample_count = IMAGE_WIDTH * IMAGE_HEIGHT * SAMPLES_PER_PIXEL
    var scene_t0 = perf_counter_ns()
    var gpu_world = prepare_gpu_scene[kind, sphere_format](ctx, world)
    ctx.synchronize()
    var scene_ns = Int(perf_counter_ns() - scene_t0)
    var target_t0 = perf_counter_ns()
    var target = gpu_world.make_render_target(ctx, settings, camera)
    ctx.synchronize()
    var target_ns = Int(perf_counter_ns() - target_t0)
    print(t"scene upload + BVHs: {round(ns_to_ms(scene_ns), 3)} ms")
    print(t"reusable target allocation: {round(ns_to_ms(target_ns), 3)} ms")
    print("hot timings exclude scene setup, target allocation, and download")
    comptime cases = (
        (Integrator.PATH, "PATH"),
        (Integrator.AO, "AO"),
        (Integrator.NEE, "NEE"),
        (Integrator.MIS, "MIS"),
    )
    comptime for i in range(len(cases)):
        var result = bench_gpu_integrator[
            cases[i][0], kind=kind, sphere_format=sphere_format
        ](ctx, target, gpu_world, settings)
        print_gpu_rt_result(cases[i][1], result, sample_count)

    comptime if kind == .TRIANGLES:
        var many_light_world = make_many_light_scene()
        var many_scene_t0 = perf_counter_ns()
        var many_gpu_world = prepare_gpu_scene[.TRIANGLES](
            ctx, many_light_world
        )
        ctx.synchronize()
        var many_scene_ns = Int(perf_counter_ns() - many_scene_t0)
        var result = bench_gpu_integrator[.NEE](
            ctx, target, many_gpu_world, settings
        )
        print(
            t"64-light scene upload + BVH:"
            t" {round(ns_to_ms(many_scene_ns), 3)} ms"
        )
        print_gpu_rt_result("NEE-64", result, sample_count)


def main() raises:
    var args = argv()
    var workload = String(args[1]) if len(args) == 2 else String("triangles")
    if len(args) > 2 or (workload != "triangles" and workload != "mixed"):
        raise Error("usage: bench_gpu_rt [triangles|mixed]")
    comptime if not has_accelerator():
        print("GPU RT benchmark skipped: no accelerator")
        return
    var settings = RenderSettings(
        IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES_PER_PIXEL, RNG_SEED, MAX_DEPTH
    )
    print("GPU RT persistent wavefront benchmark:", workload)
    print(
        t"{IMAGE_WIDTH}x{IMAGE_HEIGHT}, spp={SAMPLES_PER_PIXEL}, "
        t"max_depth={MAX_DEPTH}, median of {BENCH_REPEATS}"
    )
    with DeviceContext() as ctx:
        if workload == "mixed":
            var camera = Camera.from_vfov(
                Point3f32[.WORLD](0.0, 3.0, 6.2),
                Point3f32[.WORLD](0.0, 0.85, -1.65),
                Vec3f32[.WORLD](0.0, 1.0, 0.0),
                31.0,
                8.10,
            )
            var world = make_mis_showcase_scene()
            _benchmark[.SPHERES_TRIANGLES](ctx, world, camera, settings)
        else:
            var world = make_cornell_scene()
            _benchmark[.TRIANGLES](ctx, world, gpu_rt_camera(), settings)
