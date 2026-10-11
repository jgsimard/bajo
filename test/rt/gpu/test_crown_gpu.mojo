"""Absorption, film response and textured Crown materials on the GPU."""

from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_true, assert_almost_equal
from std.math import exp
from bajo.parser.pbrt import parse_pbrt
from bajo.parser import AssetLoader
from bajo.parser.pbrt import read_pbrt
from bajo.parser.ply import PlyMesh, parse_ply
from bajo.rt import CpuScene, render_wavefront
from bajo.rt.gpu import (
    prepare_gpu_scene,
    GpuRtRenderTarget,
    enqueue_render_gpu,
    download_gpu_pixels,
)
from bajo.rt.material_types import ImageTexture
from test.parser.pbrt.test_pbrt import _triangle_ply


struct _CrownLoader(AssetLoader):
    var source: String

    def __init__(out self, source: String):
        self.source = source

    def read_text(self, path: String) raises -> String:
        return self.source

    def read_ply_mesh(self, path: String) raises -> PlyMesh:
        if path == "scene/cube.ply":
            var mesh = PlyMesh()
            # Outward-facing closed shell; two world units through its center.
            mesh.positions = [
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                1,
                1,
            ]
            mesh.indices = [
                0,
                2,
                1,
                0,
                3,
                2,
                4,
                5,
                6,
                4,
                6,
                7,
                1,
                2,
                6,
                1,
                6,
                5,
                0,
                4,
                7,
                0,
                7,
                3,
                3,
                7,
                6,
                3,
                6,
                2,
                0,
                1,
                5,
                0,
                5,
                4,
            ]
            return mesh^
        var bytes = _triangle_ply()
        return parse_ply(bytes)

    def read_image_texture(self, path: String) raises -> ImageTexture:
        return ImageTexture(
            2,
            2,
            [
                0.1,
                0.1,
                0.1,
                0.9,
                0.9,
                0.9,
                0.4,
                0.4,
                0.4,
                0.7,
                0.7,
                0.7,
            ],
        )


def test_textured_gold_and_dielectric_instances_match_cpu() raises:
    var loader = _CrownLoader(
        """Film "rgb" "integer xresolution" 8 "integer yresolution" 6
Sampler "halton" "integer pixelsamples" 32
Integrator "volpath" "integer maxdepth" 12
LookAt .3 .3 2 .3 .3 0 0 1 0
Camera "perspective" "float fov" 35
WorldBegin
LightSource "infinite" "rgb L" [1 1 1]
Texture "mask" "float" "imagemap" "string filename" "mask.png"
Texture "rough" "float" "scale" "texture tex" "mask" "float scale" .15
Texture "bump" "float" "scale" "texture tex" "mask" "float scale" .01
MakeNamedMaterial "gold" "string type" "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k" "texture roughness" "rough" "texture displacement" "bump"
MakeNamedMaterial "glass" "string type" "dielectric" "float eta" 1.5 "texture displacement" "bump"
Material "mix" "string materials" ["gold" "glass"] "texture amount" "mask"
Shape "plymesh" "string filename" "mesh.ply"
"""
    )
    var parsed = read_pbrt("scene/main.pbrt", loader)
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    var world = CpuScene[4, 4](parsed^.take_data())
    var cpu = render_wavefront[.MIS, 4](settings, camera, world)
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.INSTANCES](ctx, world.scene_data())
        var target = GpuRtRenderTarget(ctx, settings, camera, 3 * 32)
        enqueue_render_gpu[.MIS](ctx, target, scene, settings)
        var gpu = download_gpu_pixels(ctx, target)
        for i in range(len(gpu)):
            assert_true(gpu[i].is_finite()[0])
            assert_almost_equal(
                gpu[i].x[0], cpu.pixels[i].x[0], atol=1e-3, rtol=1e-3
            )
            assert_almost_equal(
                gpu[i].y[0], cpu.pixels[i].y[0], atol=1e-3, rtol=1e-3
            )
            assert_almost_equal(
                gpu[i].z[0], cpu.pixels[i].z[0], atol=1e-3, rtol=1e-3
            )


def test_absorption_and_iso_cpu_gpu_parity() raises:
    var parsed = parse_pbrt(
        """Film "rgb" "integer xresolution" 4 "integer yresolution" 3 "float iso" 150
LookAt 0 0 3 0 0 0 0 1 0
Camera "perspective" "float fov" 20
Sampler "halton" "integer pixelsamples" 64
Integrator "volpath" "integer maxdepth" 100
WorldBegin
MakeNamedMedium "m" "string type" "homogeneous" "rgb sigma_a" [.01 .2 .5] "rgb sigma_s" [0 0 0]
LightSource "infinite" "rgb L" [1 1 1]
Material "dielectric" "float eta" 1
MediumInterface "m" ""
Shape "sphere"
"""
    )
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    var world = CpuScene[4, 4](parsed^.take_data())
    var cpu = render_wavefront[.MIS, 4](settings, camera, world)
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.SPHERES](ctx, world.scene_data())
        var target = GpuRtRenderTarget(ctx, settings, camera, 3 * 64)
        enqueue_render_gpu[.MIS](ctx, target, scene, settings)
        var gpu = download_gpu_pixels(ctx, target)
        for i in range(len(gpu)):
            assert_true(gpu[i].is_finite()[0])
            assert_almost_equal(
                gpu[i].x[0], cpu.pixels[i].x[0], atol=1e-3, rtol=1e-3
            )
            assert_almost_equal(
                gpu[i].y[0], cpu.pixels[i].y[0], atol=1e-3, rtol=1e-3
            )
            assert_almost_equal(
                gpu[i].z[0], cpu.pixels[i].z[0], atol=1e-3, rtol=1e-3
            )


def test_instance_boundaries_match_analytic_absorption() raises:
    var loader = _CrownLoader(
        """Film "rgb" "integer xresolution" 3 "integer yresolution" 2
LookAt 0 0 3 0 0 0 0 1 0
Camera "perspective" "float fov" .01
Sampler "halton" "integer pixelsamples" 17
Integrator "volpath" "integer maxdepth" 8
WorldBegin
MakeNamedMedium "m" "string type" "homogeneous" "rgb sigma_a" [.01 .2 .5] "rgb sigma_s" [0 0 0]
LightSource "infinite" "rgb L" [1 1 1]
Material "dielectric" "float eta" 1
MediumInterface "m" ""
Scale 2 1 1
Shape "plymesh" "string filename" "cube.ply"
"""
    )
    var parsed = read_pbrt("scene/main.pbrt", loader)
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    var world = CpuScene[4, 4](parsed^.take_data())
    var cpu = render_wavefront[.MIS, 4](settings, camera, world)
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.INSTANCES](ctx, world.scene_data())
        var target = GpuRtRenderTarget(ctx, settings, camera, 2 * 17)
        enqueue_render_gpu[.MIS](ctx, target, scene, settings)
        var gpu = download_gpu_pixels(ctx, target)
        for i in range(len(gpu)):
            assert_almost_equal(
                cpu.pixels[i].x[0], exp(Float32(-0.02)), atol=3e-4
            )
            assert_almost_equal(
                cpu.pixels[i].y[0], exp(Float32(-0.4)), atol=3e-4
            )
            assert_almost_equal(cpu.pixels[i].z[0], exp(Float32(-1)), atol=3e-4)
            assert_almost_equal(gpu[i].x[0], cpu.pixels[i].x[0], atol=1e-4)
            assert_almost_equal(gpu[i].y[0], cpu.pixels[i].y[0], atol=1e-4)
            assert_almost_equal(gpu[i].z[0], cpu.pixels[i].z[0], atol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
