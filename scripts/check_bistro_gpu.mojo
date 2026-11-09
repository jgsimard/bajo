"""Render a full-asset Bistro GPU diagnostic; fidelity work is still incomplete.

Usage from the repository root after downloading assets:
    pixi run mojo -I . scripts/check_bistro_gpu.mojo [scene] [output.ppm]
        [width=96] [samples=4] [depth=4] [--color-mode rgb|spectral]

Resolution, sample count, and depth overrides are for diagnostics. Geometry and
decoded texture samples retain their original resolution and precision.
"""

from std.sys.arg import argv
from max.gpu.host import DeviceContext
from bajo.parser.pbrt import read_pbrt
from bajo.parser.number import parse_int_f32
from bajo.rt import write_ppm_from_colors
from bajo.rt.image_output import write_linear_rgb
from bajo.rt.material_storage import TextureDescLayout
from bajo.rt.render_types import RenderColorMode
from bajo.rt.gpu import (
    prepare_gpu_scene,
    enqueue_render_gpu,
    download_gpu_pixels,
)


def main() raises:
    var raw = argv()
    var args = List[String]()
    var color_mode = RenderColorMode.RGB
    var index = 0
    while index < len(raw):
        var value = String(raw[index])
        if value == "--color-mode":
            index += 1
            if index >= len(raw):
                raise Error("--color-mode requires rgb or spectral")
            value = String(raw[index])
            if value != "rgb" and value != "spectral":
                raise Error("--color-mode requires rgb or spectral")
            color_mode = .SPECTRAL if value == "spectral" else .RGB
        else:
            args.append(value^)
        index += 1
    if len(args) > 6:
        raise Error(
            "usage: check_bistro_gpu [scene] [output.ppm] [width] [samples]"
            " [depth]"
        )
    var path = String(args[1]) if len(args) > 1 else String(
        "assets/bistro/bistro_cafe.pbrt"
    )
    var output = String(args[2]) if len(args) > 2 else String(
        "/tmp/bajo_bistro_gpu.ppm"
    )
    var width = (
        parse_int_f32(StringSpan(args[3]), "width") if len(args) > 3 else 96
    )
    var samples = (
        parse_int_f32(StringSpan(args[4]), "samples") if len(args) > 4 else 4
    )
    var depth = (
        parse_int_f32(StringSpan(args[5]), "depth") if len(args) > 5 else 4
    )
    if width <= 0 or samples <= 0 or depth <= 0:
        raise Error(
            "diagnostic width, sample count, and depth must be positive"
        )
    var parsed = read_pbrt(path)
    var settings = parsed.settings.copy()
    settings.color_mode = color_mode
    settings.image_height = max(
        1, width * settings.image_height // settings.image_width
    )
    settings.image_width = width
    settings.samples_per_pixel = samples
    settings.sample_sequence_length = samples
    settings.max_depth = depth
    var camera = parsed.camera
    var data = parsed^.take_data()
    var raw_bytes = 0
    for image in range(len(data.surfaces().image_textures)):
        raw_bytes += 4 * len(data.surfaces().image_textures[image].pixels)
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.INSTANCES](ctx, data)
        if Int(scene.materials.spectra.count) != len(
            data.surfaces().optical_spectra
        ):
            raise Error("GPU preparation discarded retained optical curves")
        with scene.materials.spectra.coated_eta_k.map_to_host() as indices:
            for index in range(len(data.surfaces().coated_diffuses)):
                ref material = data.surfaces().coated_diffuses[index]
                if (
                    indices[2 * index] != material.conductor_eta_spectrum
                    or indices[2 * index + 1] != material.conductor_k_spectrum
                ):
                    raise Error(
                        "GPU preparation changed conductor spectrum IDs"
                    )
        with scene.materials.spectra.dielectric_eta.map_to_host() as indices:
            for index in range(len(data.surfaces().dielectrics)):
                if (
                    indices[index]
                    != data.surfaces().dielectrics[index].eta_spectrum
                ):
                    raise Error(
                        "GPU preparation changed dielectric spectrum IDs"
                    )
        print("Retained GPU optical curves:", scene.materials.spectra.count)
        var paletted = 0
        with scene.materials.texture_descs.map_to_host() as descriptors:
            for image in range(len(data.surfaces().image_textures)):
                paletted += Int(
                    descriptors[
                        image * TextureDescLayout.STRIDE
                        + TextureDescLayout.FORMAT
                    ]
                    == TextureDescLayout.PALETTE_RGB8
                )
        print(
            "Texture RGB bytes:",
            raw_bytes,
            "->",
            4 * len(scene.materials.texture_pixels),
            "paletted images:",
            paletted,
        )
        var target = scene.make_render_target(ctx, settings, camera)
        if color_mode == .SPECTRAL:
            scene.enable_spectral(ctx)
        enqueue_render_gpu[.MIS](ctx, target, scene, settings)
        var pixels = download_gpu_pixels(ctx, target)
        for pixel in pixels:
            if not pixel.is_finite()[0]:
                raise Error("Bistro diagnostic produced a non-finite pixel")
        write_ppm_from_colors(
            output, settings.image_width, settings.image_height, pixels
        )
        write_linear_rgb(output + ".rgb", pixels)
        print(
            "Diagnostic:",
            settings.image_width,
            "x",
            settings.image_height,
            "spp",
            samples,
            "depth",
            depth,
        )
        print("Wrote", output, "and", output + ".rgb")
