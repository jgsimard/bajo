"""Mixed coated conductors retain instance attributes and chunk invariance."""

from test.rt.helpers import assert_pixels_close, assert_pixels_equal
from std.testing import TestSuite, assert_true
from max.gpu.host import DeviceContext
from bajo.parser import MemoryAssetLoader
from bajo.parser.pbrt import read_pbrt
from bajo.rt import Color, CpuScene, render_wavefront
from bajo.rt.material_types import Integrator
from bajo.core.random import Sampler
from bajo.rt.gpu import (
    GpuRtRenderTarget,
    download_gpu_pixels,
    enqueue_render_gpu,
    prepare_gpu_scene,
)
from test.parser.pbrt.test_pbrt import _triangle_ply


def test_mixed_coated_conductor_instances_match_cpu_and_gpu_chunks() raises:
    var loader = MemoryAssetLoader()
    loader.add_ply_file("scene/mesh.ply", _triangle_ply())
    loader.add_file(
        "scene/main.pbrt",
        """Film "rgb" "integer xresolution" 12 "integer yresolution" 8
Sampler "independent" "integer pixelsamples" 16
LookAt .3 .3 2 .3 .3 0 0 1 0
Camera "perspective" "float fov" 35
WorldBegin
LightSource "infinite" "rgb L" [1 1 1]
MakeNamedMaterial "red" "string type" "diffuse" "rgb reflectance" [.7 .05 .02]
MakeNamedMaterial "rim" "string type" "coatedconductor" "spectrum conductor.eta" "metal-Al-eta" "spectrum conductor.k" "metal-Al-k" "float conductor.roughness" .05
Material "mix" "string materials" ["red" "rim"] "float amount" .3
Shape "plymesh" "string filename" "mesh.ply"
""",
    )
    var parsed = read_pbrt("scene/main.pbrt", loader)
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    var world = CpuScene[4, 4](parsed^.take_data())
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.INSTANCES](ctx, world.scene_data())
        comptime for integrator in [
            Integrator.PATH,
            Integrator.NEE,
            Integrator.MIS,
        ]:
            var cpu = render_wavefront[integrator, 4](settings, camera, world)
            var full = GpuRtRenderTarget(ctx, settings, camera, 96 * 16)
            enqueue_render_gpu[integrator](ctx, full, scene, settings)
            var expected = download_gpu_pixels(ctx, full)
            var chunked = GpuRtRenderTarget(ctx, settings, camera, 2 * 16)
            enqueue_render_gpu[integrator](ctx, chunked, scene, settings)
            var actual = download_gpu_pixels(ctx, chunked)
            for pixel in expected:
                assert_true(pixel.is_finite()[0])
            assert_pixels_equal(actual, expected)
            assert_pixels_close(expected, cpu.pixels, atol=1e-3, rtol=1e-3)

        # An estimator stream belongs to the absolute pixel sample, even
        # when the renderer batches that sample into a different path ID.
        for sampler in [
            Sampler.INDEPENDENT,
            Sampler.HALTON,
            Sampler.OWEN_SOBOL,
        ]:
            var full_settings = settings.copy()
            full_settings.sampler = sampler
            full_settings.sample_sequence_length = 16
            var batch_settings = full_settings.copy()
            batch_settings.samples_per_pixel = 8
            var full = GpuRtRenderTarget(ctx, full_settings, camera, 96 * 16)
            var batch = GpuRtRenderTarget(ctx, batch_settings, camera, 96 * 8)
            enqueue_render_gpu[.MIS](ctx, full, scene, full_settings)
            var expected = download_gpu_pixels(ctx, full)
            enqueue_render_gpu[.MIS](ctx, batch, scene, batch_settings)
            var first = download_gpu_pixels(ctx, batch)
            batch_settings.sample_offset = 8
            enqueue_render_gpu[.MIS](ctx, batch, scene, batch_settings)
            var second = download_gpu_pixels(ctx, batch)
            var averaged = List[Color](capacity=len(expected))
            for i in range(len(expected)):
                averaged.append(0.5 * (first[i] + second[i]))
            assert_pixels_close(averaged, expected, atol=1e-5, rtol=1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
