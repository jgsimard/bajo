"""Diagnostic long-workload combined-scene width sweep."""

from std.math import round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.gpu import GpuBvhLayout
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.core import Affine3f32, Point3f32, Vec3f32
from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Camera,
    Color,
    Integrator,
    RenderSettings,
    SceneBuilder,
    CpuScene,
)
from bajo.rt.gpu.config import GpuRtBvhFormat
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


def _combined_grid_world() raises -> CpuScene[16, 16]:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.62, 0.58, 0.50))
    var wall = builder.add_lambertian(Color(0.22, 0.28, 0.36))
    var light = builder.add_emissive(Color(5.0, 4.5, 3.8))

    var mesh = List[Point3f32[.LOCAL]]()
    mesh.append(Point3f32[.LOCAL](-0.45, -0.45, 0.0))
    mesh.append(Point3f32[.LOCAL](0.45, -0.45, 0.0))
    mesh.append(Point3f32[.LOCAL](0.45, 0.45, 0.0))
    mesh.append(Point3f32[.LOCAL](-0.45, -0.45, 0.0))
    mesh.append(Point3f32[.LOCAL](0.45, 0.45, 0.0))
    mesh.append(Point3f32[.LOCAL](-0.45, 0.45, 0.0))
    var mesh_bounds = compute_bounds(mesh)
    var mesh_idx = builder.add_triangle_mesh_instance(
        mesh,
        Affine3f32[.LOCAL, .WORLD].from_translation(
            Vec3f32[.WORLD](-7.5, -7.5, -5.0)
        ),
        mesh_bounds,
        matte,
    )
    for y in range(16):
        for x in range(16):
            if x == 0 and y == 0:
                continue
            builder.add_triangle_instance(
                mesh_idx,
                Affine3f32[.LOCAL, .WORLD].from_translation(
                    Vec3f32[.WORLD](
                        Float32(x) - 7.5,
                        Float32(y) - 7.5,
                        -5.0 - 0.08 * Float32((x + 3 * y) % 5),
                    )
                ),
                mesh_bounds,
                matte,
            )

    for x in range(7):
        builder.add_sphere(
            Point3f32[.WORLD](Float32(x) * 2.0 - 6.0, -8.4, -3.7),
            0.55,
            matte,
        )
    builder.add_sphere(
        Point3f32[.WORLD](0.0, 8.0, -3.5),
        0.8,
        light,
    )

    builder.add_quad(
        Point3f32[.WORLD](-10.0, -10.0, -6.0),
        Point3f32[.WORLD](10.0, -10.0, -6.0),
        Point3f32[.WORLD](10.0, 10.0, -6.0),
        Point3f32[.WORLD](-10.0, 10.0, -6.0),
        wall,
    )
    var scene = builder^.finish()
    return CpuScene[16, 16](scene^)


def _camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.0, 18.0),
        Point3f32[.WORLD](0.0, 0.0, -5.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        42.0,
    )


def _bench_integrator[
    integrator: Integrator,
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength = 4,
    blas_leaf_width: SIMDLength = 4,
    blas_layout: GpuBvhLayout = .WIDE,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtScene[
        .ALL,
        GpuRtBvhFormat(4, 4, .WIDE),
        GpuRtBvhFormat(4, 4, .WIDE),
        GpuRtBvhFormat(tlas_node_width, tlas_leaf_width, .WIDE),
        GpuRtBvhFormat(blas_node_width, blas_leaf_width, blas_layout),
    ],
    settings: RenderSettings,
) raises -> GpuRtBenchResult:
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
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength = 4,
    blas_leaf_width: SIMDLength = 4,
    blas_build_method: GpuBvhBuildMethod = .LBVH,
    blas_layout: GpuBvhLayout = .WIDE,
](
    mut ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: CpuScene[16, 16],
    settings: RenderSettings,
    sample_count: Int,
    label: String,
) raises:
    var t0 = perf_counter_ns()
    var gpu_world = prepare_gpu_scene[
        .ALL,
        sphere_format=GpuRtBvhFormat(4, 4, .WIDE),
        triangle_format=GpuRtBvhFormat(4, 4, .WIDE),
        tlas_format=GpuRtBvhFormat(tlas_node_width, tlas_leaf_width, .WIDE),
        blas_format=GpuRtBvhFormat(
            blas_node_width,
            blas_leaf_width,
            blas_layout,
        ),
        triangle_build_method=.LBVH,
        blas_build_method=blas_build_method,
    ](ctx, world.scene_data())
    ctx.synchronize()
    print(
        t"\n{label}: build={round(ns_to_ms(Int(perf_counter_ns() - t0)), 3)} ms"
    )
    print_gpu_rt_result(
        "PATH",
        _bench_integrator[
            .PATH,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            blas_layout,
        ](ctx, target, gpu_world, settings),
        sample_count,
    )
    print_gpu_rt_result(
        "AO",
        _bench_integrator[
            .AO,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            blas_layout,
        ](ctx, target, gpu_world, settings),
        sample_count,
    )
    print_gpu_rt_result(
        "NEE",
        _bench_integrator[
            .NEE,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            blas_layout,
        ](ctx, target, gpu_world, settings),
        sample_count,
    )
    print_gpu_rt_result(
        "MIS",
        _bench_integrator[
            .MIS,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            blas_layout,
        ](ctx, target, gpu_world, settings),
        sample_count,
    )


def main() raises:
    comptime if not has_accelerator():
        print("GPU RT combined-width benchmark skipped: no accelerator")
        return
    var settings = RenderSettings(
        IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES_PER_PIXEL, RNG_SEED, MAX_DEPTH
    )
    var world = _combined_grid_world()
    var sample_count = IMAGE_WIDTH * IMAGE_HEIGHT * SAMPLES_PER_PIXEL
    print("GPU RT combined-scene independent-width long benchmark")
    print(
        t"static4/leaf4, 256 instances, wide4 and CWBVH8 BLAS policies,"
        t" {IMAGE_WIDTH}x{IMAGE_HEIGHT}, spp={SAMPLES_PER_PIXEL},"
        t" depth={MAX_DEPTH}, median of {BENCH_REPEATS}"
    )
    with DeviceContext() as ctx:
        var target = GpuRtRenderTarget(ctx, settings, _camera())
        ctx.synchronize()
        _run_layout[4, 4](
            ctx, target, world, settings, sample_count, "Former shared TLAS4/4"
        )
        _run_layout[2, 2](
            ctx, target, world, settings, sample_count, "Independent TLAS2/2"
        )
        _run_layout[2, 1](
            ctx, target, world, settings, sample_count, "Default TLAS2/1"
        )
        _run_layout[2, 1, 8, 4, .HPLOC, .CWBVH8](
            ctx,
            target,
            world,
            settings,
            sample_count,
            "Default TLAS2/1 + H-PLOC CWBVH8 BLAS",
        )
        _run_layout[8, 4](
            ctx, target, world, settings, sample_count, "Independent TLAS8/4"
        )
