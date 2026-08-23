"""Diagnostic long-workload GPU RT node/leaf-width sweep."""

from std.math import round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.core.utils import ns_to_ms
from bajo.rt import RENDER, RenderSettings, CpuScene
from bajo.rt.gpu.resources import GpuRtRenderTarget
from bajo.rt.gpu.triangle_path import GpuRtTriangleScene
from bajo.benchmark.gpu_harness import (
    BENCH_REPEATS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_DEPTH,
    RNG_SEED,
    SAMPLES_PER_PIXEL,
    bench_gpu_triangle_algorithm,
    gpu_rt_camera,
    print_gpu_rt_result,
)
from examples.cornell_box import make_cornell_world


def _run_layout[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    mut ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: CpuScene[],
    settings: RenderSettings,
    sample_count: Int,
    label: String,
) raises:
    var build_t0 = perf_counter_ns()
    var gpu_world = GpuRtTriangleScene[node_width, leaf_width](
        ctx, world.scene_data()
    )
    ctx.synchronize()
    var build_ns = Int(perf_counter_ns() - build_t0)
    print(t"\n{label}: scene upload + BVH={round(ns_to_ms(build_ns), 3)} ms")
    var path = bench_gpu_triangle_algorithm[
        .PATH, node_width, leaf_width
    ](ctx, target, gpu_world, settings)
    var nee = bench_gpu_triangle_algorithm[.NEE, node_width, leaf_width](
        ctx, target, gpu_world, settings
    )
    print_gpu_rt_result("PATH", path, sample_count)
    print_gpu_rt_result("NEE", nee, sample_count)


def main() raises:
    comptime if not has_accelerator():
        print("GPU RT BVH-width benchmark skipped: no accelerator")
        return

    var settings = RenderSettings(
        IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES_PER_PIXEL, RNG_SEED, MAX_DEPTH
    )
    var camera = gpu_rt_camera()
    var world = make_cornell_world()
    var sample_count = IMAGE_WIDTH * IMAGE_HEIGHT * SAMPLES_PER_PIXEL
    print("GPU RT triangle BVH width sweep")
    print(
        t"Cornell, {IMAGE_WIDTH}x{IMAGE_HEIGHT}, spp={SAMPLES_PER_PIXEL}, "
        t"depth={MAX_DEPTH}, median of {BENCH_REPEATS}"
    )

    with DeviceContext() as ctx:
        var target = GpuRtRenderTarget(ctx, settings, camera)
        ctx.synchronize()
        _run_layout[2, 2](ctx, target, world, settings, sample_count, "n2/l2")
        _run_layout[2, 4](ctx, target, world, settings, sample_count, "n2/l4")
        _run_layout[4, 4](ctx, target, world, settings, sample_count, "n4/l4")
        _run_layout[8, 4](ctx, target, world, settings, sample_count, "n8/l4")
        _run_layout[8, 8](ctx, target, world, settings, sample_count, "n8/l8")
