"""Produce raw/denoised/reference PPMs for Cornell or a local PBRT scene.

Build with --target-accelerator sm_120 to use the GPU. Arguments:
  check_denoising [cornell|scene.pbrt] [output-prefix] [width] [spp] [reference-spp]
"""
from std.sys.arg import argv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from bajo.parser.number import parse_int_f32
from bajo.parser.pbrt import read_pbrt
from bajo.rt import (
    Camera,
    CpuScene,
    RenderSettings,
    render_wavefront,
    write_ppm_from_colors,
)
from bajo.rt.scene_description import SceneDescription
from bajo.rt.scene_data import SceneData
from bajo.rt.gpu.prepared_renderer import GpuRtPreparedRenderer
from bajo.rt.gpu.config import GpuRtSceneKind
from bajo.core import Point3W, Vec3W
from bajo.core.color import Color
from bajo.postprocess import (
    DenoiseStatistics,
    reconstruct_frame,
    AtrousWorkspace,
    atrous_denoise,
)
from bajo.rt.cpu.denoise_guides import make_cpu_denoise_guides
from examples.cornell_box import make_cornell_scene


def report(
    prefix: String,
    settings: RenderSettings,
    raw: ImmSpan[Color, _],
    filtered: ImmSpan[Color, _],
    reference: ImmSpan[Color, _],
) raises:
    var before = Float64(0)
    var after = Float64(0)
    for i in range(len(raw)):
        comptime for c in range(3):
            var d = Float64(raw[i][c] - reference[i][c])
            before += d * d
            d = Float64(filtered[i][c] - reference[i][c])
            after += d * d
    print(
        "linear RGB MSE: raw",
        before / Float64(len(raw) * 3),
        "denoised",
        after / Float64(len(raw) * 3),
        "ratio",
        after / max(before, Float64(1e-30)),
    )
    write_ppm_from_colors(
        prefix + "-raw.ppm", settings.image_width, settings.image_height, raw
    )
    write_ppm_from_colors(
        prefix + "-denoised.ppm",
        settings.image_width,
        settings.image_height,
        filtered,
    )
    write_ppm_from_colors(
        prefix + "-reference.ppm",
        settings.image_width,
        settings.image_height,
        reference,
    )


def cpu_check(
    var data: SceneData,
    camera: Camera,
    low: RenderSettings,
    high: RenderSettings,
    prefix: String,
) raises:
    var world = CpuScene[16, 16](data^)
    var settings = low.copy()
    settings.collect_denoise = True
    var noisy = render_wavefront[.MIS](settings, camera, world)
    var guides = make_cpu_denoise_guides(
        world, camera, low.image_width, low.image_height
    )
    var stats = DenoiseStatistics(low.image_width, low.image_height)
    var t0 = perf_counter_ns()
    stats.add(noisy.denoise_samples.value())
    var frame = reconstruct_frame(stats, guides)
    var workspace = AtrousWorkspace()
    atrous_denoise(frame, guides, workspace)
    print(
        "CPU postprocess ms",
        Float64(perf_counter_ns() - t0) / 1e6,
        "cascade MiB",
        Float64(len(stats.bins) * 8) / 1048576,
    )
    var reference = render_wavefront[.MIS](high, camera, world)
    report(
        prefix,
        low,
        ImmSpan(noisy.pixels),
        ImmSpan(workspace.ping),
        ImmSpan(reference.pixels),
    )


def gpu_check[
    kind: GpuRtSceneKind
](
    data: SceneData,
    camera: Camera,
    low: RenderSettings,
    high: RenderSettings,
    prefix: String,
) raises:
    var renderer = GpuRtPreparedRenderer[kind](low, camera, data)
    var preview = renderer.preview[.MIS](low, camera, 1, True)
    if preview.postprocess_error:
        raise Error(preview.postprocess_error)
    var raw = List[Color]()
    var filtered = List[Color]()
    with renderer.target.pixels.map_to_host() as pixels:
        for i in range(renderer.target.pixel_count):
            raw.append(
                Color(pixels[3 * i], pixels[3 * i + 1], pixels[3 * i + 2])
            )
    with renderer._preview.value().workspace.value().ping.map_to_host() as pixels:
        for i in range(renderer.target.pixel_count):
            filtered.append(
                Color(pixels[3 * i], pixels[3 * i + 1], pixels[3 * i + 2])
            )
    print(
        "GPU batch ms",
        Float64(preview.batch_ns) / 1e6,
        "postprocess ms",
        Float64(preview.postprocess_ns) / 1e6,
        "cascade MiB",
        Float64(len(renderer._preview.value().statistics.value().bins) * 8)
        / 1048576,
    )
    var reference = renderer.render[.MIS](high, camera)
    report(
        prefix, low, ImmSpan(raw), ImmSpan(filtered), ImmSpan(reference.pixels)
    )


def main() raises:
    var args = argv()
    var path = String(args[1]) if len(args) > 1 else String("cornell")
    var prefix = String(args[2]) if len(args) > 2 else String(
        "/tmp/bajo-denoising"
    )
    var width = (
        parse_int_f32(StringSpan(args[3]), "width") if len(args) > 3 else 160
    )
    var spp = (
        parse_int_f32(StringSpan(args[4]), "samples") if len(args) > 4 else 16
    )
    var reference_spp = (
        parse_int_f32(StringSpan(args[5]), "reference samples") if len(args)
        > 5 else 512
    )
    var scene: SceneDescription
    if path == "cornell":
        scene = SceneDescription(
            make_cornell_scene(),
            Camera.from_vfov(
                Point3W(0, 1, 3.2), Point3W(0, 1, -1), Vec3W(0, 1, 0), 40
            ),
            RenderSettings(width, width, spp, 2026, 6),
            .MIS,
        )
    else:
        scene = read_pbrt(path)
        scene.settings.image_height = max(
            1, width * scene.settings.image_height // scene.settings.image_width
        )
        scene.settings.image_width = width
    var low = scene.settings.copy()
    low.samples_per_pixel = spp
    low.sample_sequence_length = reference_spp
    low.sample_offset = 0
    low.rng_seed = 2026
    low.max_depth = 6
    var high = low.copy()
    high.samples_per_pixel = reference_spp
    var camera = scene.camera
    var data = scene^.take_data()
    comptime if has_accelerator():
        if len(data.spheres()) != 0:
            raise Error("quality check supports triangle/instance scenes")
        if len(data.triangle_instances()) == 0:
            gpu_check[.TRIANGLES](data, camera, low, high, prefix)
        elif len(data.triangle_vertices()) == 0:
            gpu_check[.INSTANCES](data, camera, low, high, prefix)
        else:
            gpu_check[.TRIANGLES_INSTANCES](data, camera, low, high, prefix)
    else:
        cpu_check(data^, camera, low, high, prefix)
