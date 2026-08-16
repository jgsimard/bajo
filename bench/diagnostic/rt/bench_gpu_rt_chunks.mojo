"""Diagnostic long-workload GPU RT path-capacity sweep."""

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


def _run_capacity[
    PATH_CAPACITY: Int,
](
    mut ctx: DeviceContext,
    world: GpuRtTriangleWorld[NODE_WIDTH, LEAF_WIDTH],
    settings: RenderSettings,
    sample_count: Int,
    label: String,
) raises:
    var target = GpuRtRenderTarget(
        ctx, settings, gpu_rt_camera(), PATH_CAPACITY
    )
    ctx.synchronize()
    print(t"\n{label}")
    var path = bench_gpu_triangle_algorithm[RENDER.PATH](
        ctx, target, world, settings
    )
    var nee = bench_gpu_triangle_algorithm[RENDER.NEE](
        ctx, target, world, settings
    )
    print_gpu_rt_result("PATH", path, sample_count)
    print_gpu_rt_result("NEE", nee, sample_count)


def main() raises:
    comptime if not has_accelerator():
        print("GPU RT chunk-capacity benchmark skipped: no accelerator")
        return

    var settings = RenderSettings(
        IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES_PER_PIXEL, RNG_SEED, MAX_DEPTH
    )
    var world = make_cornell_world()
    var sample_count = IMAGE_WIDTH * IMAGE_HEIGHT * SAMPLES_PER_PIXEL
    print("GPU RT path-capacity sweep")
    print(
        t"Cornell node8/leaf4, {IMAGE_WIDTH}x{IMAGE_HEIGHT}, "
        t"spp={SAMPLES_PER_PIXEL}, depth={MAX_DEPTH}, "
        t"median of {BENCH_REPEATS}; total samples={sample_count}"
    )
    with DeviceContext() as ctx:
        var gpu_world = GpuRtTriangleWorld[NODE_WIDTH, LEAF_WIDTH](ctx, world)
        ctx.synchronize()
        _run_capacity[131072](
            ctx, gpu_world, settings, sample_count, "128K paths (64 chunks)"
        )
        _run_capacity[262144](
            ctx, gpu_world, settings, sample_count, "256K paths (32 chunks)"
        )
        _run_capacity[524288](
            ctx, gpu_world, settings, sample_count, "512K paths (16 chunks)"
        )
        _run_capacity[1048576](
            ctx, gpu_world, settings, sample_count, "1M paths (8 chunks)"
        )
        _run_capacity[2097152](
            ctx, gpu_world, settings, sample_count, "2M paths (4 chunks)"
        )
        _run_capacity[4194304](
            ctx, gpu_world, settings, sample_count, "4M paths (2 chunks)"
        )
        _run_capacity[8388608](
            ctx, gpu_world, settings, sample_count, "8M paths (1 chunk)"
        )
