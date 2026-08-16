"""Load and render a PBRT text scene with Bajo's MIS wavefront integrator."""

from bajo.pbrt import read_pbrt
from bajo.rt import RENDER, render_wavefront, write_ppm_from_colors


comptime SCENE_PATH = "examples/scenes/pbrt_showcase.pbrt"
comptime OUTPUT_PATH = "pbrt_showcase_mis.ppm"
comptime MAX_DEPTH = 10


def main() raises:
    var scene = read_pbrt(SCENE_PATH)
    scene.settings.max_depth = MAX_DEPTH
    var result = render_wavefront[RENDER.MIS](
        scene.settings, scene.camera, scene.world
    )
    write_ppm_from_colors(
        OUTPUT_PATH,
        scene.settings.image_width,
        scene.settings.image_height,
        result.pixels,
    )
    print(
        t"PBRT: {scene.settings.image_width}x{scene.settings.image_height}, "
        t"{scene.settings.samples_per_pixel} spp, "
        t"{len(scene.world.spheres)} spheres, "
        t"{len(scene.world.triangle_vertices) / 3} triangles"
    )
    print("wrote " + OUTPUT_PATH)
