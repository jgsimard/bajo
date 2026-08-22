"""Diagnostic long-workload GPU RT TLAS-layout sweep."""

from std.math import round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.host_utils import compute_bounds
from bajo.core import Affine3f32, Frame, Point3f32, Vec3f32
from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Camera,
    Color,
    Instance,
    RENDER,
    RenderSettings,
    Sphere,
    SurfaceId,
    SurfaceStore,
    CpuScene,
    add_triangle_instance,
    add_triangle_mesh_instance,
)
from bajo.rt.gpu.instance_path import (
    GpuRtTriangleInstanceScene,
    enqueue_render_gpu_triangle_instances,
)
from bajo.rt.gpu.resources import GpuRtRenderTarget, download_gpu_pixels
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
)
from bajo.benchmark.bvh_reporting import TablePrinter


@fieldwise_init
struct TlasRtLayoutResult(Copyable):
    var label: String
    var build_ns: Int
    var path: GpuRtBenchResult
    var ao: GpuRtBenchResult
    var nee: GpuRtBenchResult
    var mis: GpuRtBenchResult


def _warm_world_build(mut ctx: DeviceContext, world: CpuScene[]) raises:
    _ = GpuRtTriangleInstanceScene[2, 4, 1, 4](ctx, world.scene_data())
    ctx.synchronize()


def _instance_grid_world() -> CpuScene[]:
    var store = SurfaceStore()
    var matte = store.add_lambertian(Color(0.62, 0.58, 0.50))
    var mesh = List[Point3f32[Frame.LOCAL]]()
    mesh.append(Point3f32[Frame.LOCAL](-0.45, -0.45, 0.0))
    mesh.append(Point3f32[Frame.LOCAL](0.45, -0.45, 0.0))
    mesh.append(Point3f32[Frame.LOCAL](0.45, 0.45, 0.0))
    mesh.append(Point3f32[Frame.LOCAL](-0.45, -0.45, 0.0))
    mesh.append(Point3f32[Frame.LOCAL](0.45, 0.45, 0.0))
    mesh.append(Point3f32[Frame.LOCAL](-0.45, 0.45, 0.0))
    var mesh_bounds = compute_bounds(mesh)
    var meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var instances = List[Instance]()
    var instance_surfaces = List[SurfaceId[1]]()
    var mesh_idx = add_triangle_mesh_instance(
        meshes,
        instances,
        instance_surfaces,
        mesh,
        Affine3f32[Frame.LOCAL, Frame.WORLD].from_translation(
            Vec3f32[Frame.WORLD](-7.5, -7.5, -5.0)
        ),
        mesh_bounds,
        matte,
    )
    for y in range(16):
        for x in range(16):
            if x == 0 and y == 0:
                continue
            add_triangle_instance(
                instances,
                instance_surfaces,
                mesh_idx,
                Affine3f32[Frame.LOCAL, Frame.WORLD].from_translation(
                    Vec3f32[Frame.WORLD](
                        Float32(x) - 7.5,
                        Float32(y) - 7.5,
                        -5.0 - 0.08 * Float32((x + 3 * y) % 5),
                    )
                ),
                mesh_bounds,
                matte,
            )
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    return CpuScene[](
        spheres^,
        sphere_surfaces^,
        vertices^,
        triangle_surfaces^,
        meshes^,
        instances^,
        instance_surfaces^,
        store^,
    )


def _camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[Frame.WORLD](0.0, 0.0, 18.0),
        Point3f32[Frame.WORLD](0.0, 0.0, -5.0),
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
        42.0,
    )


def _bench_algorithm[
    ALGORITHM: RENDER,
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    tlas_build_method: GpuBvhBuildMethod,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtTriangleInstanceScene[
        tlas_node_width,
        blas_node_width,
        tlas_leaf_width,
        blas_leaf_width,
        GpuBvhBuildMethod.HPLOC,
        False,
        tlas_build_method,
    ],
    settings: RenderSettings,
) raises -> GpuRtBenchResult:
    enqueue_render_gpu_triangle_instances[
        ALGORITHM,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        GpuBvhBuildMethod.HPLOC,
        False,
        tlas_build_method,
    ](ctx, target, world, settings)
    ctx.synchronize()
    var submit = List[Int](capacity=BENCH_REPEATS)
    var render = List[Int](capacity=BENCH_REPEATS)
    for _ in range(BENCH_REPEATS):
        var t0 = perf_counter_ns()
        enqueue_render_gpu_triangle_instances[
            ALGORITHM,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            GpuBvhBuildMethod.HPLOC,
            False,
            tlas_build_method,
        ](ctx, target, world, settings)
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
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    tlas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
](
    mut ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: CpuScene[],
    settings: RenderSettings,
    label: String,
) raises -> TlasRtLayoutResult:
    var t0 = perf_counter_ns()
    var gpu_world = GpuRtTriangleInstanceScene[
        tlas_node_width,
        blas_node_width,
        tlas_leaf_width,
        blas_leaf_width,
        GpuBvhBuildMethod.HPLOC,
        False,
        tlas_build_method,
    ](ctx, world.scene_data())
    ctx.synchronize()
    var build_ns = Int(perf_counter_ns() - t0)
    var path = _bench_algorithm[
        RENDER.PATH,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        tlas_build_method,
    ](ctx, target, gpu_world, settings)
    var ao = _bench_algorithm[
        RENDER.AO,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        tlas_build_method,
    ](ctx, target, gpu_world, settings)
    var nee = _bench_algorithm[
        RENDER.NEE,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        tlas_build_method,
    ](ctx, target, gpu_world, settings)
    var mis = _bench_algorithm[
        RENDER.MIS,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        tlas_build_method,
    ](ctx, target, gpu_world, settings)
    return TlasRtLayoutResult(label, build_ns, path^, ao^, nee^, mis^)


def _print_algorithm_row(
    table: TablePrinter,
    layout: String,
    algorithm: String,
    build_ns: Int,
    result: GpuRtBenchResult,
    sample_count: Int,
) raises:
    var throughput = (
        Float64(sample_count) / Float64(result.median_render_ns) * 1.0e3
    )
    table.result_line(
        layout=layout,
        algorithm=algorithm,
        build_ms=String(t"{round(ns_to_ms(build_ns), 3)}"),
        submit_ms=String(t"{round(ns_to_ms(result.median_submit_ns), 3)}"),
        render_ms=String(t"{round(ns_to_ms(result.median_render_ns), 3)}"),
        min_ms=String(t"{round(ns_to_ms(result.min_render_ns), 3)}"),
        max_ms=String(t"{round(ns_to_ms(result.max_render_ns), 3)}"),
        Msample_s=String(t"{round(throughput, 3)}"),
        checksum=String(t"{round(result.checksum, 3)}"),
    )


def _print_layout(
    table: TablePrinter,
    result: TlasRtLayoutResult,
    sample_count: Int,
) raises:
    _print_algorithm_row(
        table, result.label, "PATH", result.build_ns, result.path, sample_count
    )
    _print_algorithm_row(
        table, result.label, "AO", result.build_ns, result.ao, sample_count
    )
    _print_algorithm_row(
        table, result.label, "NEE", result.build_ns, result.nee, sample_count
    )
    _print_algorithm_row(
        table, result.label, "MIS", result.build_ns, result.mis, sample_count
    )


def main() raises:
    comptime if not has_accelerator():
        print("GPU RT TLAS benchmark skipped: no accelerator")
        return
    var settings = RenderSettings(
        IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES_PER_PIXEL, RNG_SEED, MAX_DEPTH
    )
    var world = _instance_grid_world()
    var sample_count = IMAGE_WIDTH * IMAGE_HEIGHT * SAMPLES_PER_PIXEL
    print("GPU RT TLAS long-workload layout benchmark")
    print(
        t"256 instances, {IMAGE_WIDTH}x{IMAGE_HEIGHT}, spp={SAMPLES_PER_PIXEL},"
        t" depth={MAX_DEPTH}, median of {BENCH_REPEATS}"
    )
    with DeviceContext() as ctx:
        var target = GpuRtRenderTarget(ctx, settings, _camera())
        ctx.synchronize()
        _warm_world_build(ctx, world)
        var rows = List[TlasRtLayoutResult]()
        rows.append(
            _run_layout[2, 1, 4, 4](
                ctx,
                target,
                world,
                settings,
                "TLAS2/leaf1 BLAS4/leaf4",
            )
        )
        rows.append(
            _run_layout[2, 2, 4, 4](
                ctx,
                target,
                world,
                settings,
                "TLAS2/leaf2 BLAS4/leaf4",
            )
        )
        rows.append(
            _run_layout[2, 1, 4, 4, GpuBvhBuildMethod.HPLOC](
                ctx,
                target,
                world,
                settings,
                "H-PLOC TLAS2/leaf1 BLAS4/leaf4",
            )
        )
        rows.append(
            _run_layout[4, 4, 4, 4](
                ctx,
                target,
                world,
                settings,
                "TLAS4/leaf4 BLAS4/leaf4",
            )
        )
        rows.append(
            _run_layout[8, 4, 4, 4](
                ctx,
                target,
                world,
                settings,
                "TLAS8/leaf4 BLAS4/leaf4",
            )
        )
        rows.append(
            _run_layout[8, 8, 4, 4](
                ctx,
                target,
                world,
                settings,
                "TLAS8/leaf8 BLAS4/leaf4",
            )
        )

        var table = TablePrinter(
            layout=32,
            algorithm=9,
            build_ms=9,
            submit_ms=10,
            render_ms=10,
            min_ms=9,
            max_ms=9,
            Msample_s=11,
            checksum=16,
        )
        print("\nResults")
        table.header()
        for row in rows:
            _print_layout(table, row, sample_count)
