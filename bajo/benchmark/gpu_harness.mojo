"""Shared fixtures, timing, and reporting for long GPU RT benchmarks."""

from std.math import round
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.bvh.gpu import GpuBvhBuildMethod, GpuBvhLayout
from bajo.core import Frame, Point3f32, Vec3f32
from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Camera,
    Color,
    Integrator,
    RenderSettings,
    SceneBuilder,
    CpuScene,
)
from bajo.rt.gpu.common_kernels import GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.resources import GpuRtRenderTarget, download_gpu_pixels
from bajo.rt.gpu.triangle_path import (
    GpuRtTriangleScene,
)
from bajo.rt.gpu.render import enqueue_render_gpu
from .timing import summarize_timings


comptime IMAGE_WIDTH = 1024
comptime IMAGE_HEIGHT = 1024
comptime SAMPLES_PER_PIXEL = 8
comptime MAX_DEPTH = 8
comptime BENCH_REPEATS = 9
comptime RNG_SEED = UInt64(2026)
comptime NODE_WIDTH = 8
comptime LEAF_WIDTH = 4


@fieldwise_init
struct GpuRtBenchResult(Copyable):
    var median_submit_ns: Int
    var min_submit_ns: Int
    var max_submit_ns: Int
    var median_render_ns: Int
    var min_render_ns: Int
    var max_render_ns: Int
    var checksum: Float64


def gpu_rt_camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[.WORLD](0.0, 1.0, 3.2),
        Point3f32[.WORLD](0.0, 1.0, -1.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        28.0,
        4.2,
    )


def make_many_light_world() raises -> CpuScene[]:
    """Diffuse receiver plus 64 emissive triangles for selection scaling."""
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.7, 0.7, 0.7))
    var light = builder.add_emissive(Color(8.0, 8.0, 8.0))

    builder.add_quad(
        Point3f32[.WORLD](-1.2, 0.0, -2.0),
        Point3f32[.WORLD](1.2, 0.0, -2.0),
        Point3f32[.WORLD](1.2, 2.0, -2.0),
        Point3f32[.WORLD](-1.2, 2.0, -2.0),
        matte,
    )

    for light_y in range(8):
        for light_x in range(8):
            var x = -0.84 + Float32(light_x) * 0.24
            var y = 0.20 + Float32(light_y) * 0.22
            var half_size = Float32(0.035)
            builder.add_triangle(
                Point3f32[.WORLD](x - half_size, y - half_size, -1.45),
                Point3f32[.WORLD](x, y + half_size, -1.45),
                Point3f32[.WORLD](x + half_size, y - half_size, -1.45),
                light,
            )

    var scene = builder^.finish()
    return CpuScene[](scene^)


def gpu_rt_checksum(pixels: List[Color]) -> Float64:
    var result = Float64(0.0)
    for i, pixel in enumerate(pixels):
        var weight = Float64((i % 251) + 1)
        result += weight * (
            Float64(pixel.x)
            + Float64(3.0) * Float64(pixel.y)
            + Float64(7.0) * Float64(pixel.z)
        )
    return result


def finalize_gpu_rt_timings(
    mut submit_times: List[Int],
    mut render_times: List[Int],
    checksum: Float64,
) -> GpuRtBenchResult:
    var submit = summarize_timings(submit_times)
    var render = summarize_timings(render_times)
    return GpuRtBenchResult(
        submit.median_ns,
        submit.min_ns,
        submit.max_ns,
        render.median_ns,
        render.min_ns,
        render.max_ns,
        checksum,
    )


def bench_gpu_triangle_algorithm[
    ALGORITHM: Integrator,
    node_width: SIMDLength = NODE_WIDTH,
    leaf_width: SIMDLength = LEAF_WIDTH,
    MAX_BLOCKS: Int = GPU_RT_MAX_BLOCKS,
    SHADOW_MAX_BLOCKS: Int = MAX_BLOCKS,
    build_method: GpuBvhBuildMethod = .HPLOC,
    layout: GpuBvhLayout = GpuBvhLayout(
        node_width == 8 and leaf_width == 4
    ),
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtTriangleScene[node_width, leaf_width, build_method, layout],
    settings: RenderSettings,
) raises -> GpuRtBenchResult:
    enqueue_render_gpu[
        ALGORITHM,
        node_width,
        leaf_width,
        MAX_BLOCKS,
        SHADOW_MAX_BLOCKS,
        build_method,
        layout,
    ](ctx, target, world, settings)
    ctx.synchronize()

    var submit_times = List[Int](capacity=BENCH_REPEATS)
    var render_times = List[Int](capacity=BENCH_REPEATS)
    for _ in range(BENCH_REPEATS):
        var render_t0 = perf_counter_ns()
        enqueue_render_gpu[
            ALGORITHM,
            node_width,
            leaf_width,
            MAX_BLOCKS,
            SHADOW_MAX_BLOCKS,
            build_method,
            layout,
        ](ctx, target, world, settings)
        var submit_t1 = perf_counter_ns()
        ctx.synchronize()
        var render_t1 = perf_counter_ns()
        submit_times.append(Int(submit_t1 - render_t0))
        render_times.append(Int(render_t1 - render_t0))

    var pixels = download_gpu_pixels(ctx, target)
    return finalize_gpu_rt_timings(
        submit_times, render_times, gpu_rt_checksum(pixels)
    )


def print_gpu_rt_result(
    label: String, result: GpuRtBenchResult, sample_count: Int
):
    var render_ms = ns_to_ms(result.median_render_ns)
    var msamples_per_second = (
        Float64(sample_count) / Float64(result.median_render_ns) * 1.0e3
    )
    print(
        t"{label}  "
        t"submit={round(ns_to_ms(result.median_submit_ns), 3)} ms median "
        t"[{round(ns_to_ms(result.min_submit_ns), 3)}.."
        t"{round(ns_to_ms(result.max_submit_ns), 3)}]  "
        t"render={round(render_ms, 3)} ms median "
        t"[{round(ns_to_ms(result.min_render_ns), 3)}.."
        t"{round(ns_to_ms(result.max_render_ns), 3)}]  "
        t"throughput={round(msamples_per_second, 3)} Msample/s  "
        t"checksum={round(result.checksum, 3)}"
    )
