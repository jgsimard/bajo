"""Diagnostic long-workload combined-scene width sweep."""

from std.math import round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
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
    add_sphere,
    add_triangle,
    add_triangle_instance,
    add_triangle_mesh_instance,
)
from bajo.rt.gpu.combined_instance_path import (
    GpuRtCombinedInstanceScene,
    enqueue_render_gpu_combined_instances,
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
    print_gpu_rt_result,
)


def _combined_grid_world() -> CpuScene[]:
    var store = SurfaceStore()
    var matte = store.add_lambertian(Color(0.62, 0.58, 0.50))
    var wall = store.add_lambertian(Color(0.22, 0.28, 0.36))
    var light = store.add_emissive(Color(5.0, 4.5, 3.8))

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
    for x in range(7):
        add_sphere(
            spheres,
            sphere_surfaces,
            Point3f32[Frame.WORLD](Float32(x) * 2.0 - 6.0, -8.4, -3.7),
            0.55,
            matte,
        )
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 8.0, -3.5),
        0.8,
        light,
    )

    var vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    add_triangle(
        vertices,
        triangle_surfaces,
        Point3f32[Frame.WORLD](-10.0, -10.0, -6.0),
        Point3f32[Frame.WORLD](10.0, -10.0, -6.0),
        Point3f32[Frame.WORLD](10.0, 10.0, -6.0),
        wall,
    )
    add_triangle(
        vertices,
        triangle_surfaces,
        Point3f32[Frame.WORLD](-10.0, -10.0, -6.0),
        Point3f32[Frame.WORLD](10.0, 10.0, -6.0),
        Point3f32[Frame.WORLD](-10.0, 10.0, -6.0),
        wall,
    )
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
    blas_node_width: SIMDLength = 4,
    blas_leaf_width: SIMDLength = 4,
    blas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
    blas_compressed: Bool = False,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtCombinedInstanceScene[
        True,
        True,
        4,
        4,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        blas_build_method,
        blas_compressed,
        4,
        4,
        GpuBvhBuildMethod.LBVH,
        False,
    ],
    settings: RenderSettings,
) raises -> GpuRtBenchResult:
    enqueue_render_gpu_combined_instances[
        ALGORITHM,
        True,
        True,
        4,
        4,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        blas_build_method,
        blas_compressed,
        4,
        4,
        GpuBvhBuildMethod.LBVH,
        False,
    ](ctx, target, world, settings)
    ctx.synchronize()
    var submit = List[Int](capacity=BENCH_REPEATS)
    var render = List[Int](capacity=BENCH_REPEATS)
    for _ in range(BENCH_REPEATS):
        var t0 = perf_counter_ns()
        enqueue_render_gpu_combined_instances[
            ALGORITHM,
            True,
            True,
            4,
            4,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            blas_build_method,
            blas_compressed,
            4,
            4,
            GpuBvhBuildMethod.LBVH,
            False,
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
    blas_node_width: SIMDLength = 4,
    blas_leaf_width: SIMDLength = 4,
    blas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
    blas_compressed: Bool = False,
](
    mut ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: CpuScene[],
    settings: RenderSettings,
    sample_count: Int,
    label: String,
) raises:
    var t0 = perf_counter_ns()
    var gpu_world = GpuRtCombinedInstanceScene[
        True,
        True,
        4,
        4,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        blas_build_method,
        blas_compressed,
        4,
        4,
        GpuBvhBuildMethod.LBVH,
        False,
    ](ctx, world.scene_data())
    ctx.synchronize()
    print(
        t"\n{label}: build={round(ns_to_ms(Int(perf_counter_ns() - t0)), 3)} ms"
    )
    print_gpu_rt_result(
        "PATH",
        _bench_algorithm[
            RENDER.PATH,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            blas_build_method,
            blas_compressed,
        ](ctx, target, gpu_world, settings),
        sample_count,
    )
    print_gpu_rt_result(
        "AO",
        _bench_algorithm[
            RENDER.AO,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            blas_build_method,
            blas_compressed,
        ](ctx, target, gpu_world, settings),
        sample_count,
    )
    print_gpu_rt_result(
        "NEE",
        _bench_algorithm[
            RENDER.NEE,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            blas_build_method,
            blas_compressed,
        ](ctx, target, gpu_world, settings),
        sample_count,
    )
    print_gpu_rt_result(
        "MIS",
        _bench_algorithm[
            RENDER.MIS,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            blas_build_method,
            blas_compressed,
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
        _run_layout[2, 1, 8, 4, GpuBvhBuildMethod.HPLOC, True](
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
