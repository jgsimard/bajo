"""Bistro material and attributed-mesh regressions on an actual device."""

from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_true, assert_almost_equal
from bajo.parser.pbrt import parse_pbrt, read_pbrt
from test.parser.pbrt.test_bistro import _NormalMapLoader
from bajo.rt import CpuScene, render_wavefront
from bajo.rt.material_types import Integrator
from bajo.rt.gpu import (
    prepare_gpu_scene,
    GpuRtRenderTarget,
    enqueue_render_gpu,
    download_gpu_pixels,
)
from test.rt.helpers import assert_pixels_close
from test.rt.cpu.test_coated_reflectance import make_reflectance_scene


def test_rough_dielectric_cpu_gpu_path_nee_and_mis() raises:
    var parsed = parse_pbrt(
        """Film "rgb" "integer xresolution" 8 "integer yresolution" 6
Sampler "halton" "integer pixelsamples" 64
Integrator "path" "integer maxdepth" 12
LookAt 0 0 3 0 0 0 0 1 0
Camera "perspective" "float fov" 35
WorldBegin
LightSource "infinite" "rgb L" [1 .6 .2]
Material "dielectric" "float eta" 1.5 "float roughness" .25
Shape "sphere"
"""
    )
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    var world = CpuScene[4, 4](parsed^.take_data())
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.SPHERES](ctx, world.scene_data())
        comptime for integrator in [
            Integrator.PATH,
            Integrator.NEE,
            Integrator.MIS,
        ]:
            var cpu = render_wavefront[integrator, 4](settings, camera, world)
            var target = GpuRtRenderTarget(ctx, settings, camera, 3 * 64)
            enqueue_render_gpu[integrator](ctx, target, scene, settings)
            var gpu = download_gpu_pixels(ctx, target)
            for pixel in gpu:
                assert_true(pixel.is_finite()[0])
            assert_pixels_close(gpu, cpu.pixels, atol=2e-3, rtol=2e-3)


def test_inline_normal_attributes_and_matrix_render_on_gpu() raises:
    var parsed = parse_pbrt(
        """Film "rgb" "integer xresolution" 4 "integer yresolution" 3
Sampler "halton" "integer pixelsamples" 4
LookAt 1 2 5 1 2 0 0 1 0
Camera "perspective" "float fov" 10
WorldBegin
Transform [2 0 0 0 0 3 0 0 0 0 1 0 1 2 0 1]
Shape "trianglemesh"
 "integer indices" [0 1 2]
 "point3 P" [-5 -5 0 5 -5 0 0 5 0]
 "normal N" [.6 0 .8 .6 0 .8 .6 0 .8]
 "point2 uv" [0 0 1 0 0 1]
"""
    )
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    var world = CpuScene[4, 4](parsed^.take_data())
    var cpu = render_wavefront[.NORMALS, 4](settings, camera, world)
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.INSTANCES](ctx, world.scene_data())
        var target = GpuRtRenderTarget(ctx, settings, camera, 2 * 4)
        enqueue_render_gpu[.NORMALS](ctx, target, scene, settings)
        var gpu = download_gpu_pixels(ctx, target)
        for pixel in gpu:
            assert_true(pixel.is_finite()[0])
            # Normal = normalize((.6 / 2, 0 / 3, .8)).
            assert_almost_equal(pixel.x[0], 0.67556172, atol=1e-5)
            assert_almost_equal(pixel.y[0], 0.5, atol=1e-5)
            assert_almost_equal(pixel.z[0], 0.96816459, atol=1e-5)
        assert_pixels_close(gpu, cpu.pixels, atol=1e-5, rtol=1e-5)


def _check_normal_map(model: String, mirrored: Bool, back: Bool) raises:
    var source = """Film "rgb" "integer xresolution" 4 "integer yresolution" 3
Sampler "halton" "integer pixelsamples" 4
"""
    var x = "6" if mirrored else "-6"
    var z = "-5" if back else "5"
    source += "LookAt " + x + " -9 " + z + " " + x + " -9 0 0 1 0\n"
    source += 'Camera "perspective" "float fov" 5\nWorldBegin\n'
    source += "Scale -2 3 1\n" if mirrored else "Scale 2 3 1\n"
    source += 'Material "' + model + '" "string normalmap" "normal.png"\n'
    source += """Shape "trianglemesh"
 "point3 P" [-10 -10 0 10 -10 0 -10 10 0]
 "normal N" [0 0 1 0 0 1 0 0 1]
 "point2 uv" [0 0 0 1 1 0]
"""
    var loader = _NormalMapLoader(source)
    var parsed = read_pbrt("scene/main.pbrt", loader)
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    var world = CpuScene[4, 4](parsed^.take_data())
    var cpu = render_wavefront[.NORMALS, 4](settings, camera, world)
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.INSTANCES](ctx, world.scene_data())
        var target = GpuRtRenderTarget(ctx, settings, camera, 2 * 4)
        enqueue_render_gpu[.NORMALS](ctx, target, scene, settings)
        var gpu = download_gpu_pixels(ctx, target)
        for pixel in gpu:
            assert_almost_equal(pixel.x[0], 0.5, atol=1e-5)
            assert_almost_equal(
                pixel.y[0],
                Float32(0.2763932) if back else Float32(0.7236068),
                atol=1e-5,
            )
            assert_almost_equal(
                pixel.z[0],
                Float32(0.0527864) if back else Float32(0.9472136),
                atol=1e-5,
            )
        assert_pixels_close(gpu, cpu.pixels, atol=1e-5, rtol=1e-5)


def test_normal_maps_follow_uv_tangents_and_mirrored_instances() raises:
    for model in ["coateddiffuse", "coatedconductor", "dielectric", "diffuse"]:
        for mirrored in [False, True]:
            for back in [False, True]:
                _check_normal_map(model, mirrored, back)


def test_coated_conductor_reflectance_cpu_gpu_path_nee_and_mis() raises:
    for eta in ["1", "1.5"]:
        var parsed = make_reflectance_scene(eta)
        var settings = parsed.settings.copy()
        var camera = parsed.camera
        var world = CpuScene[4, 4](parsed^.take_data())
        with DeviceContext() as ctx:
            var scene = prepare_gpu_scene[kind=.INSTANCES](
                ctx, world.scene_data()
            )
            comptime for integrator in [
                Integrator.PATH,
                Integrator.NEE,
                Integrator.MIS,
            ]:
                var cpu = render_wavefront[integrator, 4](
                    settings, camera, world
                )
                var target = GpuRtRenderTarget(ctx, settings, camera, 3 * 64)
                enqueue_render_gpu[integrator](ctx, target, scene, settings)
                var gpu = download_gpu_pixels(ctx, target)
                for pixel in gpu:
                    assert_true(pixel.is_finite()[0])
                assert_pixels_close(gpu, cpu.pixels, atol=2e-3, rtol=2e-3)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
