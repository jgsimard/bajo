"""Deferred coated shading retains instance UVs and global path IDs."""

from std.testing import TestSuite, assert_equal, assert_true
from max.gpu.host import DeviceContext

from bajo.core.random import Sampler
from bajo.rt.gpu import (
    GpuRtRenderTarget,
    download_gpu_pixels,
    enqueue_render_gpu,
    prepare_gpu_scene,
)
from bajo.rt.render_types import RenderSettings
from test.rt.gpu.test_gpu import _camera, _textured_instance_world


def test_gpu_coated_instance_uvs_are_chunk_invariant() raises:
    var world = _textured_instance_world[True]()
    var settings = RenderSettings(8, 5, 8, UInt64(607), 6, Sampler.OWEN_SOBOL)
    var camera = _camera()
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.INSTANCES](ctx, world.scene_data())
        var full = GpuRtRenderTarget(ctx, settings, camera, 40 * 8)
        enqueue_render_gpu[.MIS](ctx, full, scene, settings)
        var expected = download_gpu_pixels(ctx, full)
        var chunked = GpuRtRenderTarget(ctx, settings, camera, 2 * 8)
        enqueue_render_gpu[.MIS](ctx, chunked, scene, settings)
        var actual = download_gpu_pixels(ctx, chunked)
        assert_equal(len(actual), len(expected))
        for i, pixel in enumerate(expected):
            assert_equal(actual[i].x, pixel.x)
            assert_equal(actual[i].y, pixel.y)
            assert_equal(actual[i].z, pixel.z)

        # Different parts of the triangle must retain different texels when
        # UVs cross the deferred queue. These are interior red/green/blue hits.
        assert_true(expected[11].x > 2.0 * max(expected[11].y, expected[11].z))
        assert_true(expected[12].y > 2.0 * max(expected[12].x, expected[12].z))
        assert_true(expected[26].z > 2.0 * max(expected[26].x, expected[26].y))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
