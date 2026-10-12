"""Deferred coated shading retains instance UVs and global path IDs."""

from test.rt.helpers import assert_pixels_equal
from std.testing import TestSuite, assert_almost_equal, assert_true
from max.gpu.host import DeviceContext

from bajo.core.random import Sampler
from bajo.core import Vec3f32
from bajo.rt.material_parameters import CoatedParameters
from bajo.rt.coated_diffuse import _sample_coated_diffuse
from bajo.rt.shading_types import BsdfSample
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
        assert_pixels_equal(actual, expected)

        # Different parts of the triangle must retain different texels when
        # UVs cross the deferred queue. These are interior red/green/blue hits.
        assert_true(expected[11].x > 2.0 * max(expected[11].y, expected[11].z))
        assert_true(expected[12].y > 2.0 * max(expected[12].x, expected[12].z))
        assert_true(expected[26].z > 2.0 * max(expected[26].x, expected[26].y))


def _coated_sample() -> BsdfSample[1]:
    return _sample_coated_diffuse(
        Vec3f32[.WORLD](0.0, 0.0, -1.0),
        Vec3f32[.WORLD](0.0, 0.0, 1.0),
        Vec3f32[.WORLD](0.7, 0.5, 0.25),
        CoatedParameters[1](
            SIMD[.float32, 1](0.2),
            SIMD[.float32, 1](1.5),
            SIMD[.float32, 1](0.01),
            Vec3f32[.WORLD](0.1, 0.05, 0.02),
            SIMD[.float32, 1](0.2),
            SIMD[.float32, 1](10.0),
            SIMD[.float32, 1](2.0),
            Vec3f32[.WORLD](0.0),
            Vec3f32[.WORLD](0.0),
            0.0,
        ),
        SIMD[.float32, 1](0.7),
        SIMD[.float32, 1](0.3),
    )


def _coated_sample_kernel(output: Pointer[Float32, MutAnyOrigin]):
    var sampled = _coated_sample()
    output[unsafe_offset=0] = sampled.direction.x[0]
    output[unsafe_offset=1] = sampled.direction.y[0]
    output[unsafe_offset=2] = sampled.direction.z[0]
    output[unsafe_offset=3] = sampled.weight.x[0]
    output[unsafe_offset=4] = sampled.weight.y[0]
    output[unsafe_offset=5] = sampled.weight.z[0]
    output[unsafe_offset=6] = sampled.pdf[0]
    output[unsafe_offset=7] = Float32(sampled.delta[0])
    output[unsafe_offset=8] = Float32(sampled.ok[0])


def test_layered_coated_diffuse_matches_on_gpu() raises:
    var expected = _coated_sample()
    with DeviceContext() as ctx:
        var output = ctx.enqueue_create_buffer[.float32](9)
        ctx.enqueue_function[_coated_sample_kernel](
            output, grid_dim=1, block_dim=1
        )
        ctx.synchronize()
        with output.map_to_host() as host:
            assert_almost_equal(host[0], expected.direction.x[0], atol=1.0e-6)
            assert_almost_equal(host[1], expected.direction.y[0], atol=1.0e-6)
            assert_almost_equal(host[2], expected.direction.z[0], atol=1.0e-6)
            assert_almost_equal(host[3], expected.weight.x[0], atol=1.0e-6)
            assert_almost_equal(host[4], expected.weight.y[0], atol=1.0e-6)
            assert_almost_equal(host[5], expected.weight.z[0], atol=1.0e-6)
            assert_almost_equal(host[6], expected.pdf[0], atol=1.0e-6)
            assert_true((host[7] != 0.0) == expected.delta[0])
            assert_true((host[8] != 0.0) == expected.ok[0])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
