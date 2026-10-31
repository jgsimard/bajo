"""Area emitters retain their BSDF and sidedness in GPU path transport."""

from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal, assert_true
from bajo.rt import CpuScene, Color, render_wavefront
from bajo.rt.material_types import Integrator
from bajo.rt.gpu import (
    prepare_gpu_scene,
    GpuRtRenderTarget,
    enqueue_render_gpu,
    download_gpu_pixels,
)
from test.rt.cpu.test_emissive_bsdf import (
    make_emissive_plane,
    make_depth_scene,
    depth_scene_expected,
)
from test.rt.helpers import assert_pixels_close
from bajo.parser.pbrt import parse_pbrt
from bajo.core import Point3f32, Vec3f32
from bajo.core.camera import Camera
from bajo.rt.render_types import RenderSettings


def test_gpu_emissive_diffuse_continues_reflection_and_matches_all_integrators() raises:
    for two_sided in [False, True]:
        for back in [False, True]:
            var parsed = make_emissive_plane(two_sided, back)
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
                    var target = GpuRtRenderTarget(
                        ctx, settings, camera, 3 * 32
                    )
                    enqueue_render_gpu[integrator](ctx, target, scene, settings)
                    var gpu = download_gpu_pixels(ctx, target)
                    assert_pixels_close(gpu, cpu.pixels, atol=1e-5, rtol=1e-5)
                    comptime if integrator == .PATH:
                        var expected = Color(0.1, 0.1, 0.075)
                        if two_sided or not back:
                            expected += Color(0.1, 0.2, 0.3)
                        for pixel in gpu:
                            comptime for channel in range(3):
                                assert_almost_equal(
                                    pixel[channel], expected[channel], atol=1e-6
                                )


def test_back_facing_two_sided_light_illuminates_receiver_on_gpu() raises:
    var source = """Film "rgb" "integer xresolution" 4 "integer yresolution" 3
Sampler "halton" "integer pixelsamples" 32
Integrator "path" "integer maxdepth" 3
LookAt 0 0 1 0 0 0 0 1 0
Camera "perspective" "float fov" 10
WorldBegin
Material "diffuse" "rgb reflectance" [.5 .5 .5]
Shape "trianglemesh" "point3 P" [-5 -5 0 5 -5 0 0 5 0]
 "point2 uv" [0 0 1 0 .5 1]
Material "diffuse" "rgb reflectance" [0 0 0]
AreaLightSource "diffuse" "rgb L" [4 4 4] "bool twosided" "true"
Shape "trianglemesh" "point3 P" [-1 -1 2 1 -1 2 0 1 2]
 "point2 uv" [0 0 1 0 .5 1]
"""
    var parsed = parse_pbrt(source)
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    var world = CpuScene[4, 4](parsed^.take_data())
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.INSTANCES](ctx, world.scene_data())
        comptime for integrator in [Integrator.NEE, Integrator.MIS]:
            var cpu = render_wavefront[integrator, 4](settings, camera, world)
            var target = GpuRtRenderTarget(ctx, settings, camera, 3 * 32)
            enqueue_render_gpu[integrator](ctx, target, scene, settings)
            var gpu = download_gpu_pixels(ctx, target)
            assert_pixels_close(gpu, cpu.pixels, atol=1e-4, rtol=1e-4)
            for pixel in gpu:
                assert_true(pixel.x > 0.05)


def test_emissive_coated_and_dielectric_surfaces_continue_their_bsdfs() raises:
    for model in ["coateddiffuse", "coatedconductor", "dielectric"]:
        var parsed = make_emissive_plane(model=model)
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
                var target = GpuRtRenderTarget(ctx, settings, camera, 3 * 32)
                enqueue_render_gpu[integrator](ctx, target, scene, settings)
                var gpu = download_gpu_pixels(ctx, target)
                assert_pixels_close(gpu, cpu.pixels, atol=1e-3, rtol=1e-3)
                for pixel in gpu:
                    assert_true(
                        pixel.x >= 0.1 and pixel.y >= 0.2 and pixel.z >= 0.3
                    )


def test_mixed_geometry_emission_uses_primitive_kind_and_instance_identity() raises:
    var parsed = parse_pbrt(
        """WorldBegin
Material "diffuse" "rgb reflectance" [0 0 0]
AttributeBegin
 Translate -3 0 0
 AreaLightSource "diffuse" "rgb L" [1 0 0]
 Shape "sphere" "float radius" .5
AttributeEnd
AttributeBegin
 AreaLightSource "diffuse" "rgb L" [0 2 0]
 Shape "trianglemesh" "point3 P" [-1 -1 0 1 -1 0 0 1 0]
AttributeEnd
AttributeBegin
 Translate 3 0 0
 AreaLightSource "diffuse" "rgb L" [0 0 3]
 Shape "trianglemesh" "point3 P" [-1 -1 0 1 -1 0 0 1 0]
  "point2 uv" [0 0 1 0 .5 1]
AttributeEnd
"""
    )
    var settings = RenderSettings(1, 1, 4, UInt64(17), 2)
    var world = CpuScene[4, 4](parsed^.take_data())
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.ALL](ctx, world.scene_data())
        for index in range(3):
            var x = Float32(index * 3 - 3)
            var camera = Camera.from_vfov(
                Point3f32[.WORLD](x, 0, 2),
                Point3f32[.WORLD](x, 0, 0),
                Vec3f32[.WORLD](0, 1, 0),
                5.0,
            )
            var target = GpuRtRenderTarget(ctx, settings, camera, 4)
            enqueue_render_gpu[.PATH](ctx, target, scene, settings)
            var gpu = download_gpu_pixels(ctx, target)
            comptime for channel in range(3):
                var expected = Float32(
                    index + 1
                ) if channel == index else Float32(0.0)
                assert_almost_equal(gpu[0][channel], expected, atol=1e-6)


def test_gpu_depth_limits_preserve_terminal_emission_without_scattering() raises:
    for scenario in range(3):
        var diffuse = scenario == 1
        var miss = scenario == 2
        var parsed = make_depth_scene(diffuse, miss)
        var settings = parsed.settings.copy()
        var camera = parsed.camera
        var world = CpuScene[4, 4](parsed^.take_data())
        with DeviceContext() as ctx:
            var scene = prepare_gpu_scene[kind=.INSTANCES](
                ctx, world.scene_data()
            )
            for depth in range(3 if not diffuse else 1):
                settings.max_depth = depth
                var expected = depth_scene_expected(depth, diffuse, miss)
                comptime for integrator in [
                    Integrator.PATH,
                    Integrator.NEE,
                    Integrator.MIS,
                ]:
                    var cpu = render_wavefront[integrator, 4](
                        settings, camera, world
                    )
                    for capacity in [7, 48]:
                        var target = GpuRtRenderTarget(
                            ctx, settings, camera, capacity
                        )
                        enqueue_render_gpu[integrator](
                            ctx, target, scene, settings
                        )
                        var gpu = download_gpu_pixels(ctx, target)
                        assert_pixels_close(
                            gpu, cpu.pixels, atol=1e-6, rtol=1e-6
                        )
                        for pixel in gpu:
                            comptime for channel in range(3):
                                assert_almost_equal(
                                    pixel[channel], expected[channel], atol=1e-6
                                )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
