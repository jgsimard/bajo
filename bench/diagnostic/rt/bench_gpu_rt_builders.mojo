"""Diagnostic long-workload comparison of GPU RT BVH builders."""

from std.math import max, round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.host_utils import compute_bounds
from bajo.core import Frame, Point3f32, Vec3f32
from bajo.core.utils import ns_to_ms
from bajo.parser.obj.pack import pack_obj_triangles
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
)
from bajo.rt.gpu.common_kernels import GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.resources import GpuRtRenderTarget
from bajo.rt.gpu.triangle_path import GpuRtTriangleScene
from bajo.benchmark.gpu_harness import (
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


comptime DRAGON_PATH = "./assets/dragon/dragon.obj"


def _dragon_world() raises -> CpuScene[]:
    var store = SurfaceStore()
    var matte = store.add_lambertian(Color(0.65, 0.65, 0.65))
    var vertices = pack_obj_triangles[Frame.WORLD](DRAGON_PATH)
    var triangle_surfaces = List[SurfaceId[1]](
        length=len(vertices) / 3, fill=matte
    )
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var instances = List[Instance]()
    var instance_surfaces = List[SurfaceId[1]]()
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


def _dragon_camera(world: CpuScene[]) -> Camera:
    var bounds = compute_bounds(world.scene_data().triangle_vertices)
    var center = bounds.centroid()
    var extent = bounds.extent()
    var scene_width = max(max(extent.x, extent.y), extent.z)
    return Camera(
        center + Vec3f32[Frame.WORLD](0.0, extent.y * 0.15, -2.5 * scene_width),
        center,
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
        0.20,
    )


def _run_builder[
    method: GpuBvhBuildMethod,
    compressed: Bool = False,
](
    mut ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: CpuScene[],
    settings: RenderSettings,
    sample_count: Int,
    label: String,
) raises:
    var build_t0 = perf_counter_ns()
    var gpu_world = GpuRtTriangleScene[
        NODE_WIDTH, LEAF_WIDTH, method, compressed
    ](ctx, world.scene_data())
    ctx.synchronize()
    var build_ns = Int(perf_counter_ns() - build_t0)
    print(t"\n{label}: scene upload + BVH={round(ns_to_ms(build_ns), 3)} ms")
    print_gpu_rt_result(
        "PATH",
        bench_gpu_triangle_algorithm[
            RENDER.PATH,
            NODE_WIDTH,
            LEAF_WIDTH,
            GPU_RT_MAX_BLOCKS,
            GPU_RT_MAX_BLOCKS,
            method,
            compressed,
        ](ctx, target, gpu_world, settings),
        sample_count,
    )
    print_gpu_rt_result(
        "AO",
        bench_gpu_triangle_algorithm[
            RENDER.AO,
            NODE_WIDTH,
            LEAF_WIDTH,
            GPU_RT_MAX_BLOCKS,
            GPU_RT_MAX_BLOCKS,
            method,
            compressed,
        ](ctx, target, gpu_world, settings),
        sample_count,
    )
    print_gpu_rt_result(
        "NEE",
        bench_gpu_triangle_algorithm[
            RENDER.NEE,
            NODE_WIDTH,
            LEAF_WIDTH,
            GPU_RT_MAX_BLOCKS,
            GPU_RT_MAX_BLOCKS,
            method,
            compressed,
        ](ctx, target, gpu_world, settings),
        sample_count,
    )
    print_gpu_rt_result(
        "MIS",
        bench_gpu_triangle_algorithm[
            RENDER.MIS,
            NODE_WIDTH,
            LEAF_WIDTH,
            GPU_RT_MAX_BLOCKS,
            GPU_RT_MAX_BLOCKS,
            method,
            compressed,
        ](ctx, target, gpu_world, settings),
        sample_count,
    )


def main() raises:
    comptime if not has_accelerator():
        print("GPU RT builder benchmark skipped: no accelerator")
        return
    var settings = RenderSettings(
        IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES_PER_PIXEL, RNG_SEED, MAX_DEPTH
    )
    var world = make_cornell_world()
    var sample_count = IMAGE_WIDTH * IMAGE_HEIGHT * SAMPLES_PER_PIXEL
    print("GPU RT metaprogrammed BVH-builder comparison")
    print(
        t"Cornell node{NODE_WIDTH}/leaf{LEAF_WIDTH},"
        t" {IMAGE_WIDTH}x{IMAGE_HEIGHT}, spp={SAMPLES_PER_PIXEL},"
        t" depth={MAX_DEPTH}, median of {BENCH_REPEATS}"
    )
    with DeviceContext() as ctx:
        var target = GpuRtRenderTarget(ctx, settings, gpu_rt_camera())
        ctx.synchronize()
        _run_builder[GpuBvhBuildMethod.LBVH](
            ctx, target, world, settings, sample_count, "LBVH"
        )
        _run_builder[GpuBvhBuildMethod.HPLOC](
            ctx, target, world, settings, sample_count, "H-PLOC"
        )
        _run_builder[GpuBvhBuildMethod.LBVH, True](
            ctx, target, world, settings, sample_count, "LBVH CWBVH8"
        )
        _run_builder[GpuBvhBuildMethod.HPLOC, True](
            ctx, target, world, settings, sample_count, "H-PLOC CWBVH8"
        )

    print("\nGeometry-heavy incoherent RT scene")
    var dragon = _dragon_world()
    with DeviceContext() as ctx:
        var target = GpuRtRenderTarget(ctx, settings, _dragon_camera(dragon))
        ctx.synchronize()
        _run_builder[GpuBvhBuildMethod.LBVH](
            ctx, target, dragon, settings, sample_count, "Dragon LBVH"
        )
        _run_builder[GpuBvhBuildMethod.HPLOC](
            ctx, target, dragon, settings, sample_count, "Dragon H-PLOC"
        )
        _run_builder[GpuBvhBuildMethod.LBVH, True](
            ctx, target, dragon, settings, sample_count, "Dragon LBVH CWBVH8"
        )
        _run_builder[GpuBvhBuildMethod.HPLOC, True](
            ctx,
            target,
            dragon,
            settings,
            sample_count,
            "Dragon H-PLOC CWBVH8",
        )
