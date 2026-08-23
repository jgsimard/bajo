"""Load and render a PBRT text scene with Bajo's MIS wavefront integrator."""

from bajo.parser.pbrt import read_pbrt
from bajo.rt import CpuScene, RENDER, render_wavefront, write_ppm_from_colors


comptime SCENE_PATH = "examples/scenes/pbrt_showcase.pbrt"
comptime OUTPUT_PATH = "pbrt_showcase_mis.ppm"
comptime MAX_DEPTH = 10


def main() raises:
    var scene = read_pbrt(SCENE_PATH)
    scene.settings.max_depth = MAX_DEPTH
    var settings = scene.settings.copy()
    var camera = scene.camera
    var world = CpuScene[](scene^.take_data())
    var result = render_wavefront[.MIS](settings, camera, world)
    write_ppm_from_colors(
        OUTPUT_PATH,
        settings.image_width,
        settings.image_height,
        result.pixels,
    )
    print(
        t"PBRT: {settings.image_width}x{settings.image_height}, "
        t"{settings.samples_per_pixel} spp, "
        t"{len(world.scene_data().spheres())} spheres, "
        t"{len(world.scene_data().triangle_vertices()) / 3} triangles"
    )
    print("wrote " + OUTPUT_PATH)
