"""Pure-Mojo paired PATH benchmark for the viewer's heavy LBVH scene."""

from std.math import round
from std.sys import has_accelerator, simd_width_of
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.bvh.gpu.tlas_diagnostics import (
    GpuTlasTraversalStats,
    launch_triangle_tlas_camera_diagnostics,
    summarize_tlas_diagnostics,
)
from bajo.bvh.types import Hit
from bajo.core.utils import ns_to_ms
from bajo.rt import RenderSettings, CpuScene
from bajo.rt.gpu.config import GpuRtBvhFormat
from bajo.rt.gpu.render import enqueue_render_gpu
from bajo.rt.gpu.resources import GpuRtRenderTarget, download_gpu_pixels
from bajo.rt.gpu.scene import GpuRtScene, prepare_gpu_scene
from bajo.benchmark.bvh_reporting import TablePrinter
from bajo.benchmark.gpu_harness import gpu_rt_checksum
from bajo.benchmark.timing import TimingSummary, summarize_timings
from examples.lbvh_scene import make_lbvh_camera, make_lbvh_world


comptime IMAGE_WIDTH = 320
comptime IMAGE_HEIGHT = 214
comptime SAMPLES_PER_PIXEL = 4
comptime MAX_DEPTH = 8
comptime RNG_SEED = UInt64(1234)
comptime WARMUPS = 2
comptime PAIRED_REPEATS = 12


def _warm_world_build[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    mut ctx: DeviceContext,
    world: CpuScene[world_bvh_width, instance_bvh_width],
) raises:
    """Absorb one-time driver, allocator, and builder initialization."""
    _ = prepare_gpu_scene[
        .ALL,
        sphere_format=GpuRtBvhFormat(4, 4, .WIDE),
        triangle_format=GpuRtBvhFormat(4, 4, .WIDE),
        tlas_format=GpuRtBvhFormat(2, 1, .WIDE),
        blas_format=GpuRtBvhFormat(8, 4, .CWBVH8),
        triangle_build_method=.LBVH,
    ](ctx, world.scene_data())
    ctx.synchronize()


def _enqueue[
    tlas_leaf_width: SIMDLength,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtScene[
        .ALL,
        GpuRtBvhFormat(4, 4, .WIDE),
        GpuRtBvhFormat(4, 4, .WIDE),
        GpuRtBvhFormat(2, tlas_leaf_width, .WIDE),
        GpuRtBvhFormat(8, 4, .CWBVH8),
    ],
    settings: RenderSettings,
) raises:
    enqueue_render_gpu[.PATH](ctx, target, world, settings)


def _timed_enqueue[
    tlas_leaf_width: SIMDLength,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtScene[
        .ALL,
        GpuRtBvhFormat(4, 4, .WIDE),
        GpuRtBvhFormat(4, 4, .WIDE),
        GpuRtBvhFormat(2, tlas_leaf_width, .WIDE),
        GpuRtBvhFormat(8, 4, .CWBVH8),
    ],
    settings: RenderSettings,
) raises -> Int:
    var t0 = perf_counter_ns()
    _enqueue[tlas_leaf_width](ctx, target, world, settings)
    ctx.synchronize()
    return Int(perf_counter_ns() - t0)


def _ms(ns: Int) -> String:
    return String(t"{round(ns_to_ms(ns), 3)}")


def _per_ray(total: UInt64, rays: UInt64) -> Float64:
    if rays == 0:
        return 0.0
    return Float64(total) / Float64(rays)


def _print_row(
    table: TablePrinter,
    label: String,
    summary: TimingSummary,
    checksum: Float64,
    delta_percent: Float64,
) raises:
    table.result_line(
        layout=label,
        median_ms=_ms(summary.median_ns),
        min_ms=_ms(summary.min_ns),
        max_ms=_ms(summary.max_ns),
        delta_pct=String(t"{round(delta_percent, 3)}"),
        checksum=String(t"{round(checksum, 3)}"),
    )


def main() raises:
    comptime if not has_accelerator():
        print("GPU LBVH viewer benchmark skipped: no accelerator")
        return

    comptime host_width = simd_width_of[DType.float32]()
    var settings = RenderSettings(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        SAMPLES_PER_PIXEL,
        RNG_SEED,
        MAX_DEPTH,
    )
    var camera = make_lbvh_camera()
    var world = make_lbvh_world[host_width, host_width]()

    print("GPU LBVH viewer PATH benchmark (pure Mojo)")
    print(
        t"{IMAGE_WIDTH}x{IMAGE_HEIGHT}, spp={SAMPLES_PER_PIXEL},"
        t" depth={MAX_DEPTH}, two warmups + {PAIRED_REPEATS} alternating pairs"
    )
    print(
        t"instances={len(world.scene_data().triangle_instances())},"
        t" meshes={len(world.scene_data().triangle_meshes())},"
        t" static triangles={len(world.scene_data().triangle_vertices()) / 3},"
        t" spheres={len(world.scene_data().spheres())}"
    )

    with DeviceContext() as ctx:
        _warm_world_build(ctx, world)

        var build21_t0 = perf_counter_ns()
        var gpu21 = prepare_gpu_scene[
            .ALL,
            sphere_format=GpuRtBvhFormat(4, 4, .WIDE),
            triangle_format=GpuRtBvhFormat(4, 4, .WIDE),
            tlas_format=GpuRtBvhFormat(2, 1, .WIDE),
            blas_format=GpuRtBvhFormat(8, 4, .CWBVH8),
            triangle_build_method=.LBVH,
        ](ctx, world.scene_data())
        ctx.synchronize()
        var build21_ns = Int(perf_counter_ns() - build21_t0)

        var build22_t0 = perf_counter_ns()
        var gpu22 = prepare_gpu_scene[
            .ALL,
            sphere_format=GpuRtBvhFormat(4, 4, .WIDE),
            triangle_format=GpuRtBvhFormat(4, 4, .WIDE),
            tlas_format=GpuRtBvhFormat(2, 2, .WIDE),
            blas_format=GpuRtBvhFormat(8, 4, .CWBVH8),
            triangle_build_method=.LBVH,
        ](ctx, world.scene_data())
        ctx.synchronize()
        var build22_ns = Int(perf_counter_ns() - build22_t0)

        var target = GpuRtRenderTarget(ctx, settings, camera)
        ctx.synchronize()

        for _ in range(WARMUPS):
            _enqueue[1](ctx, target, gpu21, settings)
            _enqueue[2](ctx, target, gpu22, settings)
            ctx.synchronize()

        var times21 = List[Int](capacity=PAIRED_REPEATS)
        var times22 = List[Int](capacity=PAIRED_REPEATS)
        for pair in range(PAIRED_REPEATS):
            if pair % 2 == 0:
                times21.append(_timed_enqueue[1](ctx, target, gpu21, settings))
                times22.append(_timed_enqueue[2](ctx, target, gpu22, settings))
            else:
                times22.append(_timed_enqueue[2](ctx, target, gpu22, settings))
                times21.append(_timed_enqueue[1](ctx, target, gpu21, settings))

        _enqueue[1](ctx, target, gpu21, settings)
        ctx.synchronize()
        var checksum21 = gpu_rt_checksum(download_gpu_pixels(ctx, target))
        _enqueue[2](ctx, target, gpu22, settings)
        ctx.synchronize()
        var checksum22 = gpu_rt_checksum(download_gpu_pixels(ctx, target))

        var summary21 = summarize_timings(times21)
        var summary22 = summarize_timings(times22)
        var delta21 = Float64(0.0)
        if summary22.median_ns > 0:
            delta21 = (
                Float64(summary21.median_ns - summary22.median_ns)
                * 100.0
                / Float64(summary22.median_ns)
            )

        var table = TablePrinter(
            layout=20,
            median_ms=10,
            min_ms=10,
            max_ms=10,
            delta_pct=10,
            checksum=16,
        )
        print("\nPATH tracing")
        table.header()
        _print_row(table, "TLAS2/leaf1", summary21, checksum21, delta21)
        _print_row(table, "TLAS2/leaf2", summary22, checksum22, 0.0)

        print("\nBuild guardrail")
        print(t"TLAS2/leaf1 world build: {_ms(build21_ns)} ms")
        print(t"TLAS2/leaf2 world build: {_ms(build22_ns)} ms")
        print(t"checksum delta: {round(checksum21 - checksum22, 6)}")

        var primary_ray_count = IMAGE_WIDTH * IMAGE_HEIGHT
        var diagnostic_hits = ctx.enqueue_create_buffer[.float32](
            primary_ray_count * Hit[.WORLD].STRIDE
        )
        for _ in range(WARMUPS):
            gpu21._tlas.value().launch_camera(
                ctx,
                gpu21._instance_blases.value(),
                target.camera,
                diagnostic_hits,
                primary_ray_count,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
            )
            gpu22._tlas.value().launch_camera(
                ctx,
                gpu22._instance_blases.value(),
                target.camera,
                diagnostic_hits,
                primary_ray_count,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
            )
            ctx.synchronize()
        var primary21 = List[Int](capacity=PAIRED_REPEATS)
        var primary22 = List[Int](capacity=PAIRED_REPEATS)
        for pair in range(PAIRED_REPEATS):
            if pair % 2 == 0:
                var t21 = perf_counter_ns()
                gpu21._tlas.value().launch_camera(
                    ctx,
                    gpu21._instance_blases.value(),
                    target.camera,
                    diagnostic_hits,
                    primary_ray_count,
                    IMAGE_WIDTH,
                    IMAGE_HEIGHT,
                )
                ctx.synchronize()
                primary21.append(Int(perf_counter_ns() - t21))
                var t22 = perf_counter_ns()
                gpu22._tlas.value().launch_camera(
                    ctx,
                    gpu22._instance_blases.value(),
                    target.camera,
                    diagnostic_hits,
                    primary_ray_count,
                    IMAGE_WIDTH,
                    IMAGE_HEIGHT,
                )
                ctx.synchronize()
                primary22.append(Int(perf_counter_ns() - t22))
            else:
                var t22 = perf_counter_ns()
                gpu22._tlas.value().launch_camera(
                    ctx,
                    gpu22._instance_blases.value(),
                    target.camera,
                    diagnostic_hits,
                    primary_ray_count,
                    IMAGE_WIDTH,
                    IMAGE_HEIGHT,
                )
                ctx.synchronize()
                primary22.append(Int(perf_counter_ns() - t22))
                var t21 = perf_counter_ns()
                gpu21._tlas.value().launch_camera(
                    ctx,
                    gpu21._instance_blases.value(),
                    target.camera,
                    diagnostic_hits,
                    primary_ray_count,
                    IMAGE_WIDTH,
                    IMAGE_HEIGHT,
                )
                ctx.synchronize()
                primary21.append(Int(perf_counter_ns() - t21))
        var primary_summary21 = summarize_timings(primary21)
        var primary_summary22 = summarize_timings(primary22)
        var primary_table = TablePrinter(
            layout=20,
            median_ms=10,
            min_ms=10,
            max_ms=10,
            MRay_s=10,
        )
        print("\nPrimary closest-hit TLAS to BLAS tracing")
        primary_table.header()
        primary_table.result_line(
            layout="TLAS2/leaf1",
            median_ms=_ms(primary_summary21.median_ns),
            min_ms=_ms(primary_summary21.min_ns),
            max_ms=_ms(primary_summary21.max_ns),
            MRay_s=String(
                t"{round(Float64(primary_ray_count) * 1.0e3 / Float64(primary_summary21.median_ns), 3)}"
            ),
        )
        primary_table.result_line(
            layout="TLAS2/leaf2",
            median_ms=_ms(primary_summary22.median_ns),
            min_ms=_ms(primary_summary22.min_ns),
            max_ms=_ms(primary_summary22.max_ns),
            MRay_s=String(
                t"{round(Float64(primary_ray_count) * 1.0e3 / Float64(primary_summary22.median_ns), 3)}"
            ),
        )

        var diagnostic_stats = ctx.enqueue_create_buffer[.uint32](
            primary_ray_count * GpuTlasTraversalStats.STRIDE
        )
        launch_triangle_tlas_camera_diagnostics(
            ctx,
            gpu21._tlas.value(),
            gpu21._instance_blases.value(),
            target.camera,
            diagnostic_hits,
            diagnostic_stats,
            primary_ray_count,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
        )
        ctx.synchronize()
        var nested = summarize_tlas_diagnostics(
            diagnostic_stats, primary_ray_count
        )
        var shared_percent = Float64(0.0)
        if nested.winner_lanes > 0:
            shared_percent = (
                100.0
                * Float64(nested.winner_lanes_sharing_blas)
                / Float64(nested.winner_lanes)
            )
        print("\nPrimary-ray nested traversal diagnostics")
        print(
            t"TLAS"
            t" nodes/ray={round(_per_ray(nested.tlas_node_visits, nested.rays), 3)}"
            t" leaves/ray={round(_per_ray(nested.tlas_leaves, nested.rays), 3)}"
            t" max_stack={nested.tlas_max_stack}"
        )
        print(
            t"BLAS"
            t" dispatches/ray={round(_per_ray(nested.blas_dispatches, nested.rays), 3)}"
            t" hits={nested.blas_hits} misses={nested.blas_misses}"
            t" replacements/ray={round(_per_ray(nested.hit_replacements, nested.rays), 3)}"
        )
        print(
            t"BLAS"
            t" nodes/dispatch={round(_per_ray(nested.blas_node_visits, nested.blas_dispatches), 3)}"
            t" leaves/dispatch={round(_per_ray(nested.blas_leaves, nested.blas_dispatches), 3)}"
            t" triangles/dispatch={round(_per_ray(nested.blas_primitives, nested.blas_dispatches), 3)}"
            t" max_stack={nested.blas_max_stack}"
        )
        print(
            t"winner lanes sharing their BLAS within a warp:"
            t" {round(shared_percent, 2)}%"
        )
