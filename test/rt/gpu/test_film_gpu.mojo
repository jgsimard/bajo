"""GPU complete-sample film clamping agrees with independently sampled CPU paths."""

from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal
from bajo.core.color import Color
from bajo.rt import CpuScene, render_wavefront
from bajo.rt.film import FilmResponse
from bajo.rt.render_types import RenderSettings
from bajo.rt.material_types import Integrator
from bajo.rt.gpu import (
    prepare_gpu_scene,
    GpuRtRenderTarget,
    enqueue_render_gpu,
    download_gpu_pixels,
)
from test.rt.cpu.test_film import (
    make_film_path,
    make_varying_film_world,
    film_camera,
)
from test.rt.helpers import assert_pixels_close


def test_gpu_clamps_complete_path_after_emission_and_environment_accumulation() raises:
    var parsed = make_film_path()
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    var world = CpuScene[4, 4](parsed^.take_data())
    var expected = settings.film.apply_sample(Color(36, 12, 12))
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.INSTANCES](ctx, world.scene_data())
        for spp in [5, 8]:
            settings.samples_per_pixel = spp
            settings.sample_sequence_length = spp
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
                    enqueue_render_gpu[integrator](ctx, target, scene, settings)
                    var gpu = download_gpu_pixels(ctx, target)
                    assert_pixels_close(gpu, cpu.pixels, atol=1e-5)
                    for pixel in gpu:
                        comptime for channel in range(3):
                            assert_almost_equal(
                                pixel[channel], expected[channel], atol=1e-5
                            )


def test_gpu_clamps_varying_samples_before_scalar_or_vector_reduction() raises:
    var world = make_varying_film_world()
    var camera = film_camera()
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.SPHERES](ctx, world.scene_data())
        for spp in [5, 8]:
            var settings = RenderSettings(3, 2, spp, 2026, 0, sampler=.HALTON)
            settings.film = FilmResponse(1.7, max_component_value=20)
            settings.film.use_cie1931()
            var expected = List[Color](length=6, fill=Color(0))
            for sample in range(spp):
                var one = settings.copy()
                one.samples_per_pixel = 1
                one.sample_offset = sample
                one.film = FilmResponse()
                var raw = render_wavefront[.PATH, 1](one, camera, world)
                for pixel in range(6):
                    expected[pixel] += settings.film.apply_sample(
                        raw.pixels[pixel]
                    ) / Float32(spp)
            comptime for integrator in [
                Integrator.PATH,
                Integrator.NEE,
                Integrator.MIS,
            ]:
                for capacity in [7, 48]:
                    var target = GpuRtRenderTarget(
                        ctx, settings, camera, capacity
                    )
                    enqueue_render_gpu[integrator](ctx, target, scene, settings)
                    var gpu = download_gpu_pixels(ctx, target)
                    assert_pixels_close(gpu, expected, atol=1e-5)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
