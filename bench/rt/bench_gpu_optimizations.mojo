"""Hot GPU rendering benchmark with linear-float output for A/B validation.

Arguments: scene.pbrt spp width height repeats output.f32 [sampler].
Zero width/height retain the scene resolution. Setup and readback are excluded
from render timings; setup is reported separately. Run GPU benchmarks serially.
"""

from std.io.file_descriptor import FileDescriptor
from std.memory import bitcast
from std.sys.arg import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from bajo.parser.number import parse_int_f32
from bajo.parser.pbrt import read_pbrt
from bajo.rt.gpu import (
    prepare_gpu_scene,
    enqueue_render_gpu,
    download_gpu_pixels,
)
from bajo.rt.render_types import Color, RenderSettings
from bajo.rt.scene_data import SceneData
from bajo.bvh import Camera
from bajo.core.random import Sampler
from bajo.rt.gpu.config import GpuRtSceneKind


def _write_linear(path: String, pixels: ImmSpan[Color, _]) raises:
    var bytes = List[UInt8](length=len(pixels) * 12, fill=0)
    var index = 0
    for pixel in pixels:
        var channels = SIMD[.float32, 4](pixel.x, pixel.y, pixel.z, 0.0)
        comptime for channel in range(3):
            var word = bitcast[.uint32](channels[channel])
            comptime for byte in range(4):
                bytes[index + byte] = UInt8(word >> UInt32(byte * 8))
            index += 4
    with open(path, "w") as f:
        var fd = FileDescriptor(f)
        fd.write_bytes(bytes)


def main() raises:
    var args = argv()
    if len(args) < 7:
        raise Error("expected scene spp width height repeats output.f32")
    var parsed = read_pbrt(String(args[1]))
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    settings.samples_per_pixel = parse_int_f32(StringSpan(args[2]), "spp")
    settings.sample_sequence_length = settings.samples_per_pixel
    var width = parse_int_f32(StringSpan(args[3]), "width")
    var height = parse_int_f32(StringSpan(args[4]), "height")
    var repeats = parse_int_f32(StringSpan(args[5]), "repeats")
    if width > 0:
        settings.image_width = width
    if height > 0:
        settings.image_height = height
    if len(args) > 7:
        settings.sampler = Sampler(
            UInt32(parse_int_f32(StringSpan(args[7]), "sampler"))
        )
    print(
        "CONFIG",
        settings.image_width,
        settings.image_height,
        settings.samples_per_pixel,
        settings.max_depth,
        settings.sampler.value,
    )
    var data = parsed^.take_data()
    if len(data.spheres()) > 0:
        _bench[.ALL](data, settings, camera, repeats, String(args[6]))
    else:
        _bench[.INSTANCES](data, settings, camera, repeats, String(args[6]))


def _bench[
    kind: GpuRtSceneKind
](
    data: SceneData,
    settings: RenderSettings,
    camera: Camera,
    repeats: Int,
    output: String,
) raises:
    with DeviceContext() as ctx:
        var init_start = perf_counter_ns()
        var scene = prepare_gpu_scene[kind=kind](ctx, data)
        print(
            "SCENE_INFO",
            "LIGHTS",
            scene.lights.count,
            "TEXTURES",
            len(data.surfaces().image_textures),
            "COATED",
            len(data.surfaces().coated_diffuses),
        )
        var target = scene.make_render_target(ctx, settings, camera)
        ctx.synchronize()
        print(
            "SETUP_MS",
            Float64(perf_counter_ns() - init_start) / 1e6,
            "CAPACITY",
            target.arena.capacity,
        )
        enqueue_render_gpu[.MIS](ctx, target, scene, settings)
        ctx.synchronize()
        for trial in range(repeats):
            var start = perf_counter_ns()
            enqueue_render_gpu[.MIS](ctx, target, scene, settings)
            var submitted = perf_counter_ns()
            ctx.synchronize()
            print(
                "TRIAL",
                trial,
                "RENDER_MS",
                Float64(perf_counter_ns() - start) / 1e6,
                "SUBMIT_MS",
                Float64(submitted - start) / 1e6,
            )
        var pixels = download_gpu_pixels(ctx, target)
        for pixel in pixels:
            if not pixel.is_finite()[0]:
                raise Error("non-finite benchmark pixel")
        _write_linear(output, pixels)
