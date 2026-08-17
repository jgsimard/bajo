"""Long-workload benchmark for mixed sphere/triangle GPU RT."""

from std.math import round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.core import Frame, Point3f32, Vec3f32
from bajo.core.utils import ns_to_ms
from bajo.rt import Camera, RENDER, RenderSettings
from bajo.rt.gpu.mixed_path import GpuRtMixedScene, enqueue_render_gpu_mixed
from bajo.rt.gpu.resources import GpuRtRenderTarget, download_gpu_pixels
from bench.rt.gpu_harness import (
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
from examples.mis_showcase import make_mis_showcase_world


comptime NODE_WIDTH = 8
comptime LEAF_WIDTH = 4


def _camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[Frame.WORLD](0.0, 3.0, 6.2),
        Point3f32[Frame.WORLD](0.0, 0.85, -1.65),
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
        31.0,
        8.10,
    )


def _bench_algorithm[
    ALGORITHM: RENDER,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtMixedScene[NODE_WIDTH, LEAF_WIDTH],
    settings: RenderSettings,
) raises -> GpuRtBenchResult:
    enqueue_render_gpu_mixed[ALGORITHM, NODE_WIDTH, LEAF_WIDTH](
        ctx, target, world, settings
    )
    ctx.synchronize()

    var submit_times = List[Int](capacity=BENCH_REPEATS)
    var render_times = List[Int](capacity=BENCH_REPEATS)
    for _ in range(BENCH_REPEATS):
        var render_t0 = perf_counter_ns()
        enqueue_render_gpu_mixed[ALGORITHM, NODE_WIDTH, LEAF_WIDTH](
            ctx, target, world, settings
        )
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
        print("Mixed GPU RT benchmark skipped: no accelerator")
        return

    var settings = RenderSettings(
        IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES_PER_PIXEL, RNG_SEED, MAX_DEPTH
    )
    var world = make_mis_showcase_world()
    var sample_count = IMAGE_WIDTH * IMAGE_HEIGHT * SAMPLES_PER_PIXEL
    print("GPU RT persistent mixed-geometry benchmark")
    print(
        t"Veach spheres + triangles, {IMAGE_WIDTH}x{IMAGE_HEIGHT}, "
        t"spp={SAMPLES_PER_PIXEL}, max_depth={MAX_DEPTH}, "
        t"median of {BENCH_REPEATS}"
    )

    with DeviceContext() as ctx:
        var scene_t0 = perf_counter_ns()
        var gpu_world = GpuRtMixedScene[NODE_WIDTH, LEAF_WIDTH](
            ctx, world.scene
        )
        ctx.synchronize()
        var scene_ns = Int(perf_counter_ns() - scene_t0)
        var target = GpuRtRenderTarget(ctx, settings, _camera())
        ctx.synchronize()

        print(t"scene upload + BVHs: {round(ns_to_ms(scene_ns), 3)} ms")
        print(
            "hot timings exclude scene setup, target allocation, and download"
        )
        print_gpu_rt_result(
            "PATH",
            _bench_algorithm[RENDER.PATH](ctx, target, gpu_world, settings),
            sample_count,
        )
        print_gpu_rt_result(
            "AO",
            _bench_algorithm[RENDER.AO](ctx, target, gpu_world, settings),
            sample_count,
        )
        print_gpu_rt_result(
            "NEE",
            _bench_algorithm[RENDER.NEE](ctx, target, gpu_world, settings),
            sample_count,
        )
        print_gpu_rt_result(
            "MIS",
            _bench_algorithm[RENDER.MIS](ctx, target, gpu_world, settings),
            sample_count,
        )
