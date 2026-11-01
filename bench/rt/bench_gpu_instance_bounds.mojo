"""Benchmark rendering with tightened instance bounds.

Usage: scene (lbvh or PBRT path), spp, width, height, repeats, output.f32.
Define BAJO_BENCH_PBRT for instance-only PBRT, BAJO_BENCH_PBRT_ALL for mixed
PBRT, BAJO_BENCH_MIS for MIS.
Hot timings exclude preparation and pixel download. Save linear pixels for
correctness comparisons and run candidates serially with baseline brackets.
"""

from std.memory import bitcast
from std.sys import is_defined
from std.sys.arg import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from bajo.parser.number import parse_int_f32
from bajo.parser.pbrt import read_pbrt
from bajo.rt import Camera, Color, SceneData, RenderSettings, Integrator
from bajo.rt.gpu import (
    prepare_gpu_scene,
    enqueue_render_gpu,
    download_gpu_pixels,
)
from bajo.rt.gpu.config import GpuRtSceneKind
from bajo.benchmark.timing import summarize_timings
from examples.lbvh_scene import make_lbvh_scene, make_lbvh_camera


def _write_linear(path: String, pixels: ImmSpan[Color, _]) raises:
    var bytes = List[UInt8](length=len(pixels) * 12, fill=0)
    for i, pixel in enumerate(pixels):
        if not pixel.is_finite()[0]:
            raise Error("non-finite benchmark pixel")
        comptime for channel in range(3):
            var word = bitcast[.uint32](pixel[channel])
            comptime for byte in range(4):
                bytes[i * 12 + channel * 4 + byte] = UInt8(
                    word >> UInt32(byte * 8)
                )
    with open(path, "w") as file:
        file.write_bytes(bytes)


def main() raises:
    var args = argv()
    if len(args) != 7:
        raise Error("expected scene spp width height repeats output.f32")
    var spp = parse_int_f32(StringSpan(args[2]), "spp")
    var width = parse_int_f32(StringSpan(args[3]), "width")
    var height = parse_int_f32(StringSpan(args[4]), "height")
    var repeats = parse_int_f32(StringSpan(args[5]), "repeats")
    if repeats < 1:
        raise Error("repeats must be positive")
    var data: SceneData
    var camera: Camera
    comptime pbrt = is_defined["BAJO_BENCH_PBRT"]() or is_defined[
        "BAJO_BENCH_PBRT_ALL"
    ]()
    comptime if pbrt:
        var parsed = read_pbrt(String(args[1]))
        camera = parsed.camera
        data = parsed^.take_data()
    else:
        if String(args[1]) != "lbvh":
            raise Error("compile with BAJO_BENCH_PBRT for PBRT input")
        data = make_lbvh_scene()
        camera = make_lbvh_camera()
    var settings = RenderSettings(width, height, spp, 2026, 8)
    comptime kind = GpuRtSceneKind.INSTANCES if is_defined[
        "BAJO_BENCH_PBRT"
    ]() else GpuRtSceneKind.ALL
    comptime integrator = Integrator.MIS if is_defined[
        "BAJO_BENCH_MIS"
    ]() else Integrator.PATH
    print(
        "CONFIG",
        args[1],
        width,
        height,
        spp,
        "depth",
        settings.max_depth,
        "refine_bounds",
        True,
    )
    with DeviceContext() as ctx:
        var start = perf_counter_ns()
        var scene = prepare_gpu_scene[kind](ctx, data)
        var target = scene.make_render_target(ctx, settings, camera)
        ctx.synchronize()
        print("SETUP_MS", Float64(perf_counter_ns() - start) / 1e6)
        for _ in range(5):
            enqueue_render_gpu[integrator](ctx, target, scene, settings)
            ctx.synchronize()
        var times = List[Int](capacity=repeats)
        for _ in range(repeats):
            start = perf_counter_ns()
            enqueue_render_gpu[integrator](ctx, target, scene, settings)
            ctx.synchronize()
            times.append(Int(perf_counter_ns() - start))
        var summary = summarize_timings(times)
        print(
            "RENDER_MS",
            Float64(summary.median_ns) / 1e6,
            "MIN_MS",
            Float64(summary.min_ns) / 1e6,
            "MAX_MS",
            Float64(summary.max_ns) / 1e6,
        )
        _write_linear(String(args[6]), download_gpu_pixels(ctx, target))
