"""Load and render a PBRT text scene with Bajo's MIS wavefront integrator."""

from bajo.parser.pbrt import read_pbrt
from bajo.parser.number import parse_int_f32
from bajo.rt import CpuScene, render_wavefront, write_ppm_from_colors
from std.sys.arg import argv


comptime SCENE_PATH = "examples/scenes/pbrt_showcase.pbrt"
comptime OUTPUT_PATH = "pbrt_showcase_mis.ppm"


def main() raises:
    var args = argv()
    if len(args) > 6:
        raise Error(
            "usage: pbrt_scene [scene.pbrt] [output.ppm] [width] [samples]"
            " [depth]"
        )
    var path = String(args[1]) if len(args) > 1 else String(SCENE_PATH)
    var output = String(args[2]) if len(args) > 2 else String(OUTPUT_PATH)
    var scene = read_pbrt(path)
    if len(args) > 3:
        var width = parse_int_f32(StringSpan(args[3]), "image width")
        if width <= 0:
            raise Error("image width must be positive")
        scene.settings.image_height = max(
            1, width * scene.settings.image_height // scene.settings.image_width
        )
        scene.settings.image_width = width
    if len(args) > 4:
        var samples = parse_int_f32(StringSpan(args[4]), "samples")
        if samples <= 0:
            raise Error("samples must be positive")
        scene.settings.samples_per_pixel = samples
        scene.settings.sample_sequence_length = samples
    if len(args) > 5:
        var depth = parse_int_f32(StringSpan(args[5]), "max depth")
        if depth <= 0:
            raise Error("max depth must be positive")
        scene.settings.max_depth = depth
    var settings = scene.settings.copy()
    var camera = scene.camera
    var world = CpuScene[16, 16](scene^.take_data())
    var result = render_wavefront[.MIS](settings, camera, world)
    for pixel in result.pixels:
        if not pixel.is_finite()[0]:
            raise Error("render produced a non-finite pixel")
    write_ppm_from_colors(
        output,
        settings.image_width,
        settings.image_height,
        result.pixels,
    )
    var triangles = len(world.scene_data().triangle_vertices()) // 3
    for instance in world.scene_data().triangle_instances():
        triangles += (
            len(world.scene_data().triangle_meshes()[Int(instance.blas_idx)])
            // 3
        )
    print(
        t"PBRT: {settings.image_width}x{settings.image_height}, "
        t"{settings.samples_per_pixel} spp, "
        t"{len(world.scene_data().spheres())} spheres, "
        t"{triangles} triangles, "
        t"{len(world.scene_data().triangle_instances())} instances"
    )
    print("wrote " + output)
