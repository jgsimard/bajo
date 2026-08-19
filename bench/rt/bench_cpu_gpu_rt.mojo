"""Direct CPU/GPU RT comparison on identical long workloads."""

from std.math import abs, max, round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.core.utils import ns_to_ms
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.rt import Camera, Color, RENDER, RenderResult, RenderSettings, World
from bajo.rt.cpu import render_depth_first, render_wavefront
from bajo.rt.gpu.resources import GpuRtRenderTarget, download_gpu_pixels
from bajo.rt.gpu.common_kernels import GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.triangle_path import (
    GpuRtTriangleScene,
    enqueue_render_gpu_triangles,
)
from bench.rt.gpu_harness import (
    BENCH_REPEATS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEAF_WIDTH,
    MAX_DEPTH,
    NODE_WIDTH,
    RNG_SEED,
    SAMPLES_PER_PIXEL,
    gpu_rt_camera,
    gpu_rt_checksum,
    make_many_light_world,
)
from examples.cornell_box import make_cornell_world


comptime CPU_PACKET_WIDTH = 16
comptime CPU_CHUNK_PATHS = 1024


@fieldwise_init
struct CpuTiming:
    var median_total_ns: Int
    var min_total_ns: Int
    var max_total_ns: Int
    var median_render_ns: Int
    var min_render_ns: Int
    var max_render_ns: Int
    var checksum: Float64


@fieldwise_init
struct GpuTiming:
    var median_submit_ns: Int
    var min_submit_ns: Int
    var max_submit_ns: Int
    var median_device_ns: Int
    var min_device_ns: Int
    var max_device_ns: Int
    var median_host_ns: Int
    var min_host_ns: Int
    var max_host_ns: Int
    var checksum: Float64


def _render_cpu[
    ALGORITHM: RENDER
](settings: RenderSettings, camera: Camera, world: World[]) -> RenderResult:
    comptime if ALGORITHM == RENDER.AO:
        return render_depth_first[ALGORITHM](settings, camera, world)
    else:
        return render_wavefront[
            ALGORITHM,
            CPU_PACKET_WIDTH,
            CPU_CHUNK_PATHS,
            True,
        ](settings, camera, world)


def _bench_cpu[
    ALGORITHM: RENDER
](settings: RenderSettings, camera: Camera, world: World[]) -> CpuTiming:
    var warmup = _render_cpu[ALGORITHM](settings, camera, world)
    var checksum = gpu_rt_checksum(warmup.pixels)
    var total_times = List[Int](capacity=BENCH_REPEATS)
    var render_times = List[Int](capacity=BENCH_REPEATS)
    for _ in range(BENCH_REPEATS):
        var result = _render_cpu[ALGORITHM](settings, camera, world)
        var current_checksum = gpu_rt_checksum(result.pixels)
        debug_assert["safe", _use_compiler_assume=True](
            current_checksum == checksum,
            "CPU comparison checksum changed between runs",
        )
        total_times.append(result.timings.total_ns)
        render_times.append(result.timings.render_ns)
    sort(total_times)
    sort(render_times)
    var middle = (BENCH_REPEATS - 1) >> 1
    return CpuTiming(
        total_times[middle],
        total_times[0],
        total_times[BENCH_REPEATS - 1],
        render_times[middle],
        render_times[0],
        render_times[BENCH_REPEATS - 1],
        checksum,
    )


def _bench_gpu[
    ALGORITHM: RENDER,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
    compressed: Bool = False,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtTriangleScene[NODE_WIDTH, LEAF_WIDTH, build_method, compressed],
    settings: RenderSettings,
) raises -> GpuTiming:
    enqueue_render_gpu_triangles[
        ALGORITHM,
        NODE_WIDTH,
        LEAF_WIDTH,
        GPU_RT_MAX_BLOCKS,
        GPU_RT_MAX_BLOCKS,
        build_method,
        compressed,
    ](ctx, target, world, settings)
    var warmup_pixels = download_gpu_pixels(ctx, target)
    var checksum = gpu_rt_checksum(warmup_pixels)
    var submit_times = List[Int](capacity=BENCH_REPEATS)
    var device_times = List[Int](capacity=BENCH_REPEATS)
    var host_times = List[Int](capacity=BENCH_REPEATS)
    for _ in range(BENCH_REPEATS):
        var t0 = perf_counter_ns()
        enqueue_render_gpu_triangles[
            ALGORITHM,
            NODE_WIDTH,
            LEAF_WIDTH,
            GPU_RT_MAX_BLOCKS,
            GPU_RT_MAX_BLOCKS,
            build_method,
            compressed,
        ](ctx, target, world, settings)
        var submit_t1 = perf_counter_ns()
        ctx.synchronize()
        var device_t1 = perf_counter_ns()
        var pixels = download_gpu_pixels(ctx, target)
        var host_t1 = perf_counter_ns()
        var current_checksum = gpu_rt_checksum(pixels)
        debug_assert["safe", _use_compiler_assume=True](
            current_checksum == checksum,
            "GPU comparison checksum changed between runs",
        )
        submit_times.append(Int(submit_t1 - t0))
        device_times.append(Int(device_t1 - t0))
        host_times.append(Int(host_t1 - t0))
    sort(submit_times)
    sort(device_times)
    sort(host_times)
    var middle = (BENCH_REPEATS - 1) >> 1
    return GpuTiming(
        submit_times[middle],
        submit_times[0],
        submit_times[BENCH_REPEATS - 1],
        device_times[middle],
        device_times[0],
        device_times[BENCH_REPEATS - 1],
        host_times[middle],
        host_times[0],
        host_times[BENCH_REPEATS - 1],
        checksum,
    )


def _print_comparison(
    label: String,
    cpu: CpuTiming,
    gpu: GpuTiming,
    sample_count: Int,
):
    var cpu_total_ms = ns_to_ms(cpu.median_total_ns)
    var cpu_render_ms = ns_to_ms(cpu.median_render_ns)
    var gpu_device_ms = ns_to_ms(gpu.median_device_ns)
    var gpu_host_ms = ns_to_ms(gpu.median_host_ns)
    var checksum_delta = abs(cpu.checksum - gpu.checksum)
    var checksum_delta_ppm = (
        checksum_delta / max(abs(cpu.checksum), 1.0) * 1.0e6
    )
    print(t"\n{label}")
    print(
        t"CPU total={round(cpu_total_ms, 3)} ms median"
        t" [{round(ns_to_ms(cpu.min_total_ns), 3)}..{round(ns_to_ms(cpu.max_total_ns), 3)}],"
        t" render={round(cpu_render_ms, 3)} ms median"
        t" [{round(ns_to_ms(cpu.min_render_ns), 3)}..{round(ns_to_ms(cpu.max_render_ns), 3)}],"
        t" throughput={round(Float64(sample_count) / Float64(cpu.median_total_ns) * 1.0e3, 3)} Msample/s,"
        t" checksum={round(cpu.checksum, 3)}"
    )
    print(
        t"GPU submit={round(ns_to_ms(gpu.median_submit_ns), 3)} ms median"
        t" [{round(ns_to_ms(gpu.min_submit_ns), 3)}..{round(ns_to_ms(gpu.max_submit_ns), 3)}],"
        t" device={round(gpu_device_ms, 3)} ms median"
        t" [{round(ns_to_ms(gpu.min_device_ns), 3)}..{round(ns_to_ms(gpu.max_device_ns), 3)}],"
        t" host-output={round(gpu_host_ms, 3)} ms median"
        t" [{round(ns_to_ms(gpu.min_host_ns), 3)}..{round(ns_to_ms(gpu.max_host_ns), 3)}],"
        t" throughput={round(Float64(sample_count) / Float64(gpu.median_device_ns) * 1.0e3, 3)} Msample/s,"
        t" checksum={round(gpu.checksum, 3)}"
    )
    print(
        t"GPU speedup:"
        t" device/render={round(Float64(cpu.median_render_ns) / Float64(gpu.median_device_ns), 3)}x,"
        t" host-output/total={round(Float64(cpu.median_total_ns) / Float64(gpu.median_host_ns), 3)}x,"
        t" checksum delta={round(checksum_delta, 6)}"
        t" ({round(checksum_delta_ppm, 3)} ppm)"
    )


def main() raises:
    comptime if not has_accelerator():
        print("CPU/GPU RT comparison skipped: no accelerator")
        return
    var settings = RenderSettings(
        IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES_PER_PIXEL, RNG_SEED, MAX_DEPTH
    )
    var camera = gpu_rt_camera()
    var world = make_cornell_world()
    var sample_count = IMAGE_WIDTH * IMAGE_HEIGHT * SAMPLES_PER_PIXEL
    print("CPU/GPU RT long-workload comparison")
    print(
        t"identical Cornell triangles, {IMAGE_WIDTH}x{IMAGE_HEIGHT}, "
        t"spp={SAMPLES_PER_PIXEL}, depth={MAX_DEPTH}, median of {BENCH_REPEATS}"
    )
    print(
        t"CPU: packet{CPU_PACKET_WIDTH}, {CPU_CHUNK_PATHS}-path parallel"
        t" chunks; AO uses tiled depth-first"
    )
    print(
        "GPU device time ends with device-resident pixels; host-output also "
        "includes synchronization and pixel download"
    )

    var cpu_path = _bench_cpu[RENDER.PATH](settings, camera, world)
    var cpu_ao = _bench_cpu[RENDER.AO](settings, camera, world)
    var cpu_nee = _bench_cpu[RENDER.NEE](settings, camera, world)
    var cpu_mis = _bench_cpu[RENDER.MIS](settings, camera, world)
    var many_light_world = make_many_light_world()
    var cpu_nee_64 = _bench_cpu[RENDER.NEE](settings, camera, many_light_world)

    with DeviceContext() as ctx:
        var gpu_world = GpuRtTriangleScene[
            NODE_WIDTH, LEAF_WIDTH, GpuBvhBuildMethod.LBVH, False
        ](ctx, world.scene)
        var target = GpuRtRenderTarget(ctx, settings, camera)
        var many_gpu_world = GpuRtTriangleScene[
            NODE_WIDTH, LEAF_WIDTH, GpuBvhBuildMethod.LBVH, False
        ](ctx, many_light_world.scene)
        var cwbvh_world = GpuRtTriangleScene[
            NODE_WIDTH, LEAF_WIDTH, GpuBvhBuildMethod.HPLOC, True
        ](ctx, world.scene)
        var many_cwbvh_world = GpuRtTriangleScene[
            NODE_WIDTH, LEAF_WIDTH, GpuBvhBuildMethod.HPLOC, True
        ](ctx, many_light_world.scene)
        ctx.synchronize()
        var gpu_path = _bench_gpu[RENDER.PATH](ctx, target, gpu_world, settings)
        var gpu_ao = _bench_gpu[RENDER.AO](ctx, target, gpu_world, settings)
        var gpu_nee = _bench_gpu[RENDER.NEE](ctx, target, gpu_world, settings)
        var gpu_mis = _bench_gpu[RENDER.MIS](ctx, target, gpu_world, settings)
        var gpu_nee_64 = _bench_gpu[RENDER.NEE](
            ctx, target, many_gpu_world, settings
        )
        var cwbvh_path = _bench_gpu[RENDER.PATH, GpuBvhBuildMethod.HPLOC, True](
            ctx, target, cwbvh_world, settings
        )
        var cwbvh_ao = _bench_gpu[RENDER.AO, GpuBvhBuildMethod.HPLOC, True](
            ctx, target, cwbvh_world, settings
        )
        var cwbvh_nee = _bench_gpu[RENDER.NEE, GpuBvhBuildMethod.HPLOC, True](
            ctx, target, cwbvh_world, settings
        )
        var cwbvh_mis = _bench_gpu[RENDER.MIS, GpuBvhBuildMethod.HPLOC, True](
            ctx, target, cwbvh_world, settings
        )
        var cwbvh_nee_64 = _bench_gpu[
            RENDER.NEE, GpuBvhBuildMethod.HPLOC, True
        ](ctx, target, many_cwbvh_world, settings)

        _print_comparison("PATH", cpu_path, gpu_path, sample_count)
        _print_comparison("AO", cpu_ao, gpu_ao, sample_count)
        _print_comparison("NEE", cpu_nee, gpu_nee, sample_count)
        _print_comparison("MIS", cpu_mis, gpu_mis, sample_count)
        _print_comparison("NEE-64", cpu_nee_64, gpu_nee_64, sample_count)
        print("\nBest retained GPU policy: H-PLOC + CWBVH8")
        _print_comparison("PATH CWBVH8", cpu_path, cwbvh_path, sample_count)
        _print_comparison("AO CWBVH8", cpu_ao, cwbvh_ao, sample_count)
        _print_comparison("NEE CWBVH8", cpu_nee, cwbvh_nee, sample_count)
        _print_comparison("MIS CWBVH8", cpu_mis, cwbvh_mis, sample_count)
        _print_comparison(
            "NEE-64 CWBVH8", cpu_nee_64, cwbvh_nee_64, sample_count
        )
