"""Long-workload benchmark for the persistent GPU wavefront renderer."""

from std.math import round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.core.utils import ns_to_ms
from bajo.rt import RenderSettings
from bajo.rt.gpu.resources import GpuRtRenderTarget
from bajo.rt.gpu.scene import prepare_gpu_scene
from bajo.benchmark.gpu_harness import (
    BENCH_REPEATS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEAF_WIDTH,
    MAX_DEPTH,
    NODE_WIDTH,
    RNG_SEED,
    SAMPLES_PER_PIXEL,
    bench_gpu_triangle_integrator,
    gpu_rt_camera,
    make_many_light_world,
    print_gpu_rt_result,
)
from examples.cornell_box import make_cornell_world


def main() raises:
    comptime if not has_accelerator():
        print("GPU RT benchmark skipped: no accelerator")
        return

    var settings = RenderSettings(
        IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES_PER_PIXEL, RNG_SEED, MAX_DEPTH
    )
    var camera = gpu_rt_camera()
    var world = make_cornell_world()
    var sample_count = IMAGE_WIDTH * IMAGE_HEIGHT * SAMPLES_PER_PIXEL

    print("GPU RT persistent wavefront benchmark")
    print(
        t"Cornell triangles, {IMAGE_WIDTH}x{IMAGE_HEIGHT}, "
        t"spp={SAMPLES_PER_PIXEL}, max_depth={MAX_DEPTH}, "
        t"median of {BENCH_REPEATS}"
    )

    with DeviceContext() as ctx:
        var scene_t0 = perf_counter_ns()
        var gpu_world = prepare_gpu_scene[.TRIANGLES](ctx, world.scene_data())
        ctx.synchronize()
        var scene_ns = Int(perf_counter_ns() - scene_t0)

        var target_t0 = perf_counter_ns()
        var target = GpuRtRenderTarget(ctx, settings, camera)
        ctx.synchronize()
        var target_ns = Int(perf_counter_ns() - target_t0)

        print(t"scene upload + BVH: {round(ns_to_ms(scene_ns), 3)} ms")
        print(t"reusable target allocation: {round(ns_to_ms(target_ns), 3)} ms")
        print(
            "hot timings exclude scene setup, target allocation, and download"
        )

        var path = bench_gpu_triangle_integrator[.PATH](
            ctx, target, gpu_world, settings
        )
        var ao = bench_gpu_triangle_integrator[.AO](
            ctx, target, gpu_world, settings
        )
        var nee = bench_gpu_triangle_integrator[.NEE](
            ctx, target, gpu_world, settings
        )
        var mis = bench_gpu_triangle_integrator[.MIS](
            ctx, target, gpu_world, settings
        )

        print_gpu_rt_result("PATH", path, sample_count)
        print_gpu_rt_result("AO", ao, sample_count)
        print_gpu_rt_result("NEE", nee, sample_count)
        print_gpu_rt_result("MIS", mis, sample_count)

        var many_light_world = make_many_light_world()
        var many_scene_t0 = perf_counter_ns()
        var many_gpu_world = prepare_gpu_scene[.TRIANGLES](
            ctx, many_light_world.scene_data()
        )
        ctx.synchronize()
        var many_scene_ns = Int(perf_counter_ns() - many_scene_t0)
        var many_light_nee = bench_gpu_triangle_integrator[.NEE](
            ctx, target, many_gpu_world, settings
        )
        print(
            t"64-light scene upload + BVH: "
            t"{round(ns_to_ms(many_scene_ns), 3)} ms"
        )
        print_gpu_rt_result("NEE-64", many_light_nee, sample_count)
