"""Diagnostic long-workload instanced-BLAS layout sweep."""

from std.math import max, round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.bvh.gpu import GpuBvhLayout
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.host_utils import compute_bounds
from bajo.core import Affine3f32, Point3f32, Vec3f32
from bajo.core.utils import ns_to_ms
from bajo.parser.obj.pack import pack_obj_triangles
from bajo.rt import (
    Camera,
    Color,
    Integrator,
    RenderSettings,
    SceneBuilder,
    CpuScene,
)
from bajo.rt.gpu.policy import (
    GpuRtBvhFormat,
    GPU_RT_BVH_CWBVH8,
    GPU_RT_BVH_WIDE4,
)
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


comptime DRAGON_PATH = "./assets/dragon/dragon.obj"


def _dragon_instance_world() raises -> CpuScene[]:
    var mesh = pack_obj_triangles[.LOCAL](DRAGON_PATH)
    var bounds = compute_bounds(mesh)
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.65, 0.65, 0.65))
    _ = builder.add_triangle_mesh_instance(
        mesh,
        Affine3f32[.LOCAL, .WORLD].identity(),
        bounds,
        matte,
    )
    var scene = builder^.finish()
    return CpuScene[](scene^)


def _dragon_camera(world: CpuScene[]) -> Camera:
    var bounds = compute_bounds(world.scene_data().triangle_meshes()[0])
    var center_local = bounds.centroid()
    var center = Point3f32[.WORLD](
        center_local.x, center_local.y, center_local.z
    )
    var extent = bounds.extent()
    var scene_width = max(max(extent.x, extent.y), extent.z)
    return Camera(
        center + Vec3f32[.WORLD](0.0, extent.y * 0.15, -2.5 * scene_width),
        center,
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        0.20,
    )


def _bench_integrator[
    integrator: Integrator,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    layout: GpuBvhLayout,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtScene[
        .INSTANCES,
        GPU_RT_BVH_WIDE4,
        GPU_RT_BVH_CWBVH8,
        GpuRtBvhFormat(2, 2, .WIDE),
        GpuRtBvhFormat(
            8 if layout.compressed else blas_node_width,
            blas_leaf_width,
            layout,
        ),
    ],
    settings: RenderSettings,
) raises -> GpuRtBenchResult:
    comptime effective_width = 8 if layout.compressed else blas_node_width
    enqueue_render_gpu[integrator](ctx, target, world, settings)
    ctx.synchronize()
    var submit = List[Int](capacity=BENCH_REPEATS)
    var render = List[Int](capacity=BENCH_REPEATS)
    for _ in range(BENCH_REPEATS):
        var t0 = perf_counter_ns()
        enqueue_render_gpu[integrator](ctx, target, world, settings)
        var t1 = perf_counter_ns()
        ctx.synchronize()
        var t2 = perf_counter_ns()
        submit.append(Int(t1 - t0))
        render.append(Int(t2 - t0))
    var pixels = download_gpu_pixels(ctx, target)
    return finalize_gpu_rt_timings(submit, render, gpu_rt_checksum(pixels))


def _run_layout[
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    method: GpuBvhBuildMethod,
    layout: GpuBvhLayout = .WIDE,
](
    mut ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: CpuScene[],
    settings: RenderSettings,
    sample_count: Int,
    label: String,
) raises:
    comptime effective_width = 8 if layout.compressed else blas_node_width
    var t0 = perf_counter_ns()
    var gpu_world = prepare_gpu_scene[
        .INSTANCES,
        tlas_format=GpuRtBvhFormat(2, 2, .WIDE),
        blas_format=GpuRtBvhFormat(
            effective_width, blas_leaf_width, layout
        ),
        blas_build_method=method,
    ](ctx, world.scene_data())
    ctx.synchronize()
    print(
        t"\n{label}: build={round(ns_to_ms(Int(perf_counter_ns() - t0)), 3)} ms"
    )
    print_gpu_rt_result(
        "PATH",
        _bench_integrator[
            .PATH, blas_node_width, blas_leaf_width, layout
        ](ctx, target, gpu_world, settings),
        sample_count,
    )
    print_gpu_rt_result(
        "AO",
        _bench_integrator[
            .AO, blas_node_width, blas_leaf_width, layout
        ](ctx, target, gpu_world, settings),
        sample_count,
    )
    print_gpu_rt_result(
        "NEE",
        _bench_integrator[
            .NEE, blas_node_width, blas_leaf_width, layout
        ](ctx, target, gpu_world, settings),
        sample_count,
    )
    print_gpu_rt_result(
        "MIS",
        _bench_integrator[
            .MIS, blas_node_width, blas_leaf_width, layout
        ](ctx, target, gpu_world, settings),
        sample_count,
    )


def main() raises:
    comptime if not has_accelerator():
        print("GPU RT instanced-BLAS benchmark skipped: no accelerator")
        return
    var settings = RenderSettings(
        IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES_PER_PIXEL, RNG_SEED, MAX_DEPTH
    )
    var world = _dragon_instance_world()
    var sample_count = IMAGE_WIDTH * IMAGE_HEIGHT * SAMPLES_PER_PIXEL
    print("GPU RT geometry-heavy instanced-BLAS long benchmark")
    print(
        t"Dragon instance, TLAS2/leaf2, {IMAGE_WIDTH}x{IMAGE_HEIGHT},"
        t" spp={SAMPLES_PER_PIXEL}, depth={MAX_DEPTH},"
        t" median of {BENCH_REPEATS}"
    )
    with DeviceContext() as ctx:
        var target = GpuRtRenderTarget(ctx, settings, _dragon_camera(world))
        ctx.synchronize()
        _run_layout[4, 4, .LBVH](
            ctx, target, world, settings, sample_count, "LBVH wide4/leaf4"
        )
        _run_layout[8, 4, .LBVH](
            ctx, target, world, settings, sample_count, "LBVH wide8/leaf4"
        )
        _run_layout[8, 4, .LBVH, .CWBVH8](
            ctx, target, world, settings, sample_count, "LBVH CWBVH8"
        )
        _run_layout[8, 4, .HPLOC, .CWBVH8](
            ctx, target, world, settings, sample_count, "H-PLOC CWBVH8"
        )
