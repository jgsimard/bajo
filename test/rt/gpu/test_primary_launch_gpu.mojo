"""A trace grid cap must not omit primary samples."""

from test.rt.helpers import assert_pixels_equal
from std.testing import TestSuite, assert_almost_equal
from max.gpu.host import DeviceContext

from bajo.bvh.gpu.utils import upload_list
from bajo.core import Point3f32, Vec3f32
from bajo.rt import Camera, Color, Integrator, RenderSettings, SceneBuilder
from bajo.rt.gpu import (
    GpuRtRenderTarget,
    download_gpu_pixels,
    enqueue_render_gpu,
    prepare_gpu_scene,
)
from bajo.rt.gpu.config import GPU_RT_BVH_WIDE4
from test.rt.gpu.test_gpu import _camera, _textured_instance_world


def _assert_capped_primary[integrator: Integrator]() raises:
    var builder = SceneBuilder()
    var light = builder.add_emissive(Color(0.75, 0.5, 0.25))
    builder.add_quad(
        Point3f32[.WORLD](-10.0, -10.0, -5.0),
        Point3f32[.WORLD](10.0, -10.0, -5.0),
        Point3f32[.WORLD](10.0, 10.0, -5.0),
        Point3f32[.WORLD](-10.0, 10.0, -5.0),
        light,
    )
    var world = builder^.finish()
    var settings = RenderSettings(17, 19, 3, UInt64(2026), 4)
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Point3f32[.WORLD](0.0, 0.0, -5.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        45.0,
    )
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[
            kind=.TRIANGLES, triangle_format=GPU_RT_BVH_WIDE4
        ](ctx, world)
        # Full, partial, and multiple chunks with non-warp-aligned sample counts.
        for capacity in [17 * 19 * 3, 53 * 3, 7 * 3]:
            var target = GpuRtRenderTarget(ctx, settings, camera, capacity)
            # Missing samples must stay zero rather than inherit allocation data.
            var zero = List[Float32](
                length=len(target.arena.sample_radiance), fill=0.0
            )
            target.arena.sample_radiance = upload_list(ctx, zero)
            enqueue_render_gpu[integrator, MAX_BLOCKS=1, SHADOW_MAX_BLOCKS=1](
                ctx, target, scene, settings
            )
            var pixels = download_gpu_pixels(ctx, target)
            for pixel in pixels:
                assert_almost_equal(pixel.x, Float32(0.75), atol=1.0e-6)
                assert_almost_equal(pixel.y, Float32(0.5), atol=1.0e-6)
                assert_almost_equal(pixel.z, Float32(0.25), atol=1.0e-6)


def test_gpu_path_grid_cap_covers_all_samples() raises:
    _assert_capped_primary[.PATH]()


def test_gpu_nee_grid_cap_covers_all_samples() raises:
    _assert_capped_primary[.NEE]()


def test_gpu_mis_grid_cap_covers_all_samples() raises:
    _assert_capped_primary[.MIS]()


def test_gpu_capped_mis_coated_instance_matches_uncapped() raises:
    var world = _textured_instance_world[True]()
    var settings = RenderSettings(8, 5, 8, UInt64(607), 6)
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.INSTANCES](ctx, world.scene_data())
        var full = GpuRtRenderTarget(ctx, settings, _camera())
        enqueue_render_gpu[.MIS](ctx, full, scene, settings)
        var expected = download_gpu_pixels(ctx, full)
        var capped = GpuRtRenderTarget(ctx, settings, _camera())
        var zero = List[Float32](
            length=len(capped.arena.sample_radiance), fill=0.0
        )
        capped.arena.sample_radiance = upload_list(ctx, zero)
        enqueue_render_gpu[.MIS, MAX_BLOCKS=1, SHADOW_MAX_BLOCKS=1](
            ctx, capped, scene, settings
        )
        var actual = download_gpu_pixels(ctx, capped)
        assert_pixels_equal(actual, expected)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
