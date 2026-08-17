"""Render the official PBRT-v4 Killeroo gallery scene with Bajo.

Scene source: https://github.com/mmp/pbrt-v4-scenes/tree/master/killeroos
Killeroo model courtesy of headus. The example uses the original scene and
geometry files downloaded by `pixi run download_assets`; only the expensive
film and sampling settings are overridden for an interactive CPU render.
"""

from bajo.pbrt import read_pbrt
from bajo.rt import (
    RENDER,
    RenderSettings,
    render_wavefront,
    write_ppm_from_colors,
)


comptime SCENE_PATH = "assets/pbrt/killeroos/killeroo-simple.pbrt"
comptime OUTPUT_PATH = "pbrt_killeroo_mis.ppm"
comptime IMAGE_WIDTH = 256
comptime IMAGE_HEIGHT = 256
comptime SAMPLES_PER_PIXEL = 16
comptime MAX_DEPTH = 10
comptime RNG_SEED = UInt64(2026)


def main() raises:
    var scene = read_pbrt(SCENE_PATH)
    scene.settings = RenderSettings(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        SAMPLES_PER_PIXEL,
        RNG_SEED,
        MAX_DEPTH,
    )
    var result = render_wavefront[RENDER.MIS](
        scene.settings, scene.camera, scene.world
    )
    write_ppm_from_colors(
        OUTPUT_PATH,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        result.pixels,
    )
    print(
        t"Killeroo: {IMAGE_WIDTH}x{IMAGE_HEIGHT}, {SAMPLES_PER_PIXEL} spp, "
        t"{len(scene.world.scene.spheres)} sphere, "
        t"{len(scene.world.scene.triangle_vertices) / 3} triangles"
    )
    print("wrote " + OUTPUT_PATH)
