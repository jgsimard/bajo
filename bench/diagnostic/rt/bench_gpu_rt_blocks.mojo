"""Diagnostic long-workload GPU RT persistent-block sweep."""

from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from bajo.rt import RENDER, RenderSettings
from bajo.rt.gpu.resources import GpuRtRenderTarget
from bajo.rt.gpu.triangle_path import GpuRtTriangleWorld
from bench.rt.gpu_harness import (
    BENCH_REPEATS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEAF_WIDTH,
    MAX_DEPTH,
    NODE_WIDTH,
    RNG_SEED,
    SAMPLES_PER_PIXEL,
    bench_gpu_triangle_algorithm,
    gpu_rt_camera,
    print_gpu_rt_result,
)
from examples.cornell_box import make_cornell_world


def _run_cap[
    MAX_BLOCKS: Int,
    SHADOW_MAX_BLOCKS: Int = MAX_BLOCKS,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtTriangleWorld[NODE_WIDTH, LEAF_WIDTH],
    settings: RenderSettings,
    sample_count: Int,
    label: String,
) raises:
    print(t"\n{label}")
    var path = bench_gpu_triangle_algorithm[
        RENDER.PATH,
        NODE_WIDTH,
        LEAF_WIDTH,
        MAX_BLOCKS,
        SHADOW_MAX_BLOCKS,
    ](ctx, target, world, settings)
    var nee = bench_gpu_triangle_algorithm[
        RENDER.NEE,
        NODE_WIDTH,
        LEAF_WIDTH,
        MAX_BLOCKS,
        SHADOW_MAX_BLOCKS,
    ](ctx, target, world, settings)
    print_gpu_rt_result("PATH", path, sample_count)
    print_gpu_rt_result("NEE", nee, sample_count)


def main() raises:
    comptime if not has_accelerator():
        print("GPU RT block-cap benchmark skipped: no accelerator")
        return

    var settings = RenderSettings(
        IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES_PER_PIXEL, RNG_SEED, MAX_DEPTH
    )
    var camera = gpu_rt_camera()
    var world = make_cornell_world()
    var sample_count = IMAGE_WIDTH * IMAGE_HEIGHT * SAMPLES_PER_PIXEL
    print("GPU RT persistent grid block-cap sweep")
    print(
        t"Cornell node8/leaf4, "
        t"{IMAGE_WIDTH}x{IMAGE_HEIGHT}, spp={SAMPLES_PER_PIXEL}, "
        t"depth={MAX_DEPTH}, median of {BENCH_REPEATS}"
    )
    with DeviceContext() as ctx:
        var gpu_world = GpuRtTriangleWorld[NODE_WIDTH, LEAF_WIDTH](ctx, world)
        var target = GpuRtRenderTarget(ctx, settings, camera)
        ctx.synchronize()
        _run_cap[64](
            ctx, target, gpu_world, settings, sample_count, "64 blocks"
        )
        _run_cap[128](
            ctx, target, gpu_world, settings, sample_count, "128 blocks"
        )
        _run_cap[256](
            ctx, target, gpu_world, settings, sample_count, "256 blocks"
        )
        _run_cap[512](
            ctx, target, gpu_world, settings, sample_count, "512 blocks"
        )
        _run_cap[1024](
            ctx, target, gpu_world, settings, sample_count, "1024 blocks"
        )
        _run_cap[2048](
            ctx, target, gpu_world, settings, sample_count, "2048 blocks"
        )
        _run_cap[4096](
            ctx, target, gpu_world, settings, sample_count, "4096 blocks"
        )
        _run_cap[8192](
            ctx, target, gpu_world, settings, sample_count, "8192 blocks"
        )
        _run_cap[16384](
            ctx, target, gpu_world, settings, sample_count, "16384 blocks"
        )
        _run_cap[32768](
            ctx, target, gpu_world, settings, sample_count, "32768 blocks"
        )
        _run_cap[65536](
            ctx, target, gpu_world, settings, sample_count, "65536 blocks"
        )
        _run_cap[1024, 64](
            ctx,
            target,
            gpu_world,
            settings,
            sample_count,
            "1024 core / 64 shadow blocks",
        )
        _run_cap[1024, 128](
            ctx,
            target,
            gpu_world,
            settings,
            sample_count,
            "1024 core / 128 shadow blocks",
        )
        _run_cap[1024, 256](
            ctx,
            target,
            gpu_world,
            settings,
            sample_count,
            "1024 core / 256 shadow blocks",
        )
        _run_cap[1024, 512](
            ctx,
            target,
            gpu_world,
            settings,
            sample_count,
            "1024 core / 512 shadow blocks",
        )
