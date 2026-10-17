"""Scene-sized target allocation preserves explicit bounds and pixel alignment."""

from test.rt.helpers import assert_pixels_equal
from std.testing import TestSuite, assert_equal, assert_true, assert_false
from max.gpu.host import DeviceContext
from bajo.core import Affine3f32, Point3f32, Vec3f32
from bajo.core.random import Sampler
from bajo.rt.gpu.resources import download_gpu_pixels, update_gpu_camera
from bajo.rt.gpu.render import enqueue_render_gpu
from test.rt.gpu.test_gpu import _textured_instance_world
from bajo.bvh.host_utils import compute_bounds
from bajo.rt import Camera, Color, RenderSettings, SceneBuilder
from bajo.rt.scene_data import SceneData
from bajo.rt.wavefront_contract import WavePathFloatAbi, WaveShadowFloatAbi
from bajo.rt.gpu.queue_layout import GPU_SHADE_PLANES
from bajo.rt.gpu import (
    GpuRtPreparedRenderer,
    prepare_gpu_scene,
    render_gpu_configured,
    GPU_RT_BVH_WIDE4,
)


def _camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.0, 3.0),
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        45.0,
    )


def test_gpu_scene_target_bounds_and_alignment() raises:
    var plain_builder = SceneBuilder()
    var matte = plain_builder.add_lambertian(Color(0.6))
    plain_builder.add_sphere(Point3f32[.WORLD](0.0), 1.0, matte)
    var plain = plain_builder^.finish()
    var coated_builder = SceneBuilder()
    var coat = coated_builder.add_coated_diffuse(Color(0.6), 0.2, 1.5)
    coated_builder.add_sphere(Point3f32[.WORLD](0.0), 1.0, coat)
    var coated = coated_builder^.finish()
    var settings = RenderSettings(512, 128, 13, UInt64(653), 4)
    var camera = _camera()
    with DeviceContext() as ctx:
        var plain_scene = prepare_gpu_scene[.SPHERES](ctx, plain)
        var coated_scene = prepare_gpu_scene[.SPHERES](ctx, coated)
        var small = plain_scene.make_render_target(ctx, settings, camera)
        var large = coated_scene.make_render_target(ctx, settings, camera)
        var bounded = coated_scene.make_render_target(
            ctx, settings, camera, 100003
        )
        assert_equal(small.arena.capacity, 262132)
        # The large recommendation is clamped to the full render's 851968 samples.
        assert_equal(large.arena.capacity, 512 * 128 * 13)
        assert_equal(bounded.arena.capacity, 99996)
        assert_equal(
            len(small.arena.path_a.fields),
            WavePathFloatAbi.PLANES * small.arena.capacity,
        )
        assert_equal(
            len(large.arena.shade.fields),
            GPU_SHADE_PLANES * large.arena.capacity,
        )
        assert_equal(
            len(bounded.arena.shadow.fields),
            WaveShadowFloatAbi.PLANES * bounded.arena.capacity,
        )
        var tiny_settings = RenderSettings(2, 3, 17, UInt64(659), 4)
        var tiny = coated_scene.make_render_target(
            ctx, tiny_settings, camera, 1
        )
        assert_equal(tiny.arena.capacity, 17)
        var whole = coated_scene.make_render_target(ctx, tiny_settings, camera)
        assert_equal(whole.arena.capacity, 102)


def test_gpu_instance_and_static_target_recommendations() raises:
    var mesh: List[Point3f32[.LOCAL]] = [
        Point3f32[.LOCAL](-0.5, -0.5, 0.0),
        Point3f32[.LOCAL](0.5, -0.5, 0.0),
        Point3f32[.LOCAL](0.0, 0.5, 0.0),
    ]
    var inst_builder = SceneBuilder()
    var inst_matte = inst_builder.add_lambertian(Color(0.6))
    _ = inst_builder.add_triangle_mesh_instance(
        mesh,
        Affine3f32[.LOCAL, .WORLD].identity(),
        compute_bounds(mesh),
        inst_matte,
    )
    var instances = inst_builder^.finish()
    var tri_builder = SceneBuilder()
    var tri_matte = tri_builder.add_lambertian(Color(0.6))
    tri_builder.add_triangle(
        Point3f32[.WORLD](-0.5, -0.5, 0.0),
        Point3f32[.WORLD](0.5, -0.5, 0.0),
        Point3f32[.WORLD](0.0, 0.5, 0.0),
        tri_matte,
    )
    var triangles = tri_builder^.finish()
    with DeviceContext() as ctx:
        var inst_scene = prepare_gpu_scene[
            .INSTANCES, blas_format=GPU_RT_BVH_WIDE4
        ](ctx, instances)
        var tri_scene = prepare_gpu_scene[
            .TRIANGLES, triangle_format=GPU_RT_BVH_WIDE4
        ](ctx, triangles)
        var tiny_settings = RenderSettings(2, 3, 17, UInt64(661), 4)
        var target = inst_scene.make_render_target(
            ctx, tiny_settings, _camera()
        )
        assert_equal(target.arena.capacity, 102)
        assert_equal(tri_scene.preferred_path_capacity(), 262144)
        assert_equal(inst_scene.preferred_path_capacity(), 4194304)


def _triangle_grid(with_sphere: Bool) raises -> SceneData:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.6))
    if with_sphere:
        builder.add_sphere(Point3f32[.WORLD](0.0), 0.25, matte)
    for i in range(4096):
        var x = Float32(i % 64)
        var y = Float32(i // 64)
        builder.add_triangle(
            Point3f32[.WORLD](x, y, 0.0),
            Point3f32[.WORLD](x + 0.5, y, 0.0),
            Point3f32[.WORLD](x, y + 0.5, 0.0),
            matte,
        )
    return builder^.finish()


def test_gpu_large_static_and_mixed_target_recommendations() raises:
    var static_data = _triangle_grid(False)
    var mixed_data = _triangle_grid(True)
    with DeviceContext() as ctx:
        var static_scene = prepare_gpu_scene[.TRIANGLES](ctx, static_data)
        var mixed_scene = prepare_gpu_scene[.SPHERES_TRIANGLES](ctx, mixed_data)
        assert_equal(static_scene.preferred_path_capacity(), 8388608)
        assert_equal(mixed_scene.preferred_path_capacity(), 1048576)
        var settings = RenderSettings(2, 3, 17, UInt64(673), 4)
        var target = static_scene.make_render_target(ctx, settings, _camera())
        assert_equal(target.arena.capacity, 102)


def test_gpu_prepared_renderer_retains_scene_budget_on_resize() raises:
    var builder = SceneBuilder()
    var coat = builder.add_coated_diffuse(Color(0.6), 0.2, 1.5)
    builder.add_sphere(Point3f32[.WORLD](0.0), 1.0, coat)
    var data = builder^.finish()
    var camera = _camera()
    var initial = RenderSettings(512, 256, 9, UInt64(677), 4)
    var renderer = GpuRtPreparedRenderer[.SPHERES](initial, camera, data)
    assert_equal(renderer.target.arena.capacity, 1048572)
    var resized = RenderSettings(256, 128, 9, UInt64(677), 4)
    var actual = renderer.render[.NORMALS](resized, camera)
    assert_equal(renderer.target.arena.capacity, 256 * 128 * 9)
    var expected = render_gpu_configured[kind=.SPHERES, integrator=.NORMALS](
        resized, camera, data
    )
    var repeated = renderer.render[.NORMALS](resized, camera)
    assert_pixels_equal(actual.pixels, expected.pixels)
    assert_pixels_equal(repeated.pixels, actual.pixels)


def test_gpu_prepared_graph_replay_tracks_camera_sampling_film_and_integrator() raises:
    var builder = SceneBuilder()
    var coat = builder.add_coated_diffuse(Color(0.6, 0.3, 0.2), 0.2, 1.5)
    builder.add_sphere(Point3f32[.WORLD](0.0), 1.0, coat)
    var data = builder^.finish()
    var camera = _camera()
    var settings = RenderSettings(
        8, 5, 2, UInt64(701), 6, sample_sequence_length=8
    )
    var renderer = GpuRtPreparedRenderer[.SPHERES](settings, camera, data)
    var expected_target = renderer.scene.make_render_target(
        renderer.ctx, settings, camera
    )
    for phase in range(12):
        if phase == 1:
            camera = Camera.from_vfov(
                Point3f32[.WORLD](0.5, 0.0, 3.0),
                Point3f32[.WORLD](0.0),
                Vec3f32[.WORLD](0.0, 1.0, 0.0),
                50.0,
            )
        elif phase == 2:
            settings.rng_seed = UInt64(709)
        elif phase == 3:
            settings.max_depth = 3
        elif phase == 4:
            settings.sampler = .OWEN_SOBOL
        elif phase == 5:
            settings.sample_offset = 2
        elif phase == 6:
            settings.sample_sequence_length = 16
        elif phase == 7:
            settings.film.exposure = 0.5
            settings.film.red = Color(0.0, 1.0, 0.0)
        elif phase == 8:
            renderer.target = renderer.scene.make_render_target(
                renderer.ctx, settings, camera, 6
            )
        elif phase == 9:
            renderer.scene.environment_scale_x *= 0.5
        elif phase == 10:
            renderer.scene.environment_world_to_light.m01 = -0.0
        elif phase == 11:
            renderer.invalidate_graph()
        update_gpu_camera(renderer.ctx, renderer.target, camera)
        update_gpu_camera(renderer.ctx, expected_target, camera)
        enqueue_render_gpu[.MIS](
            renderer.ctx, expected_target, renderer.scene, settings
        )
        var expected = download_gpu_pixels(renderer.ctx, expected_target)
        for repeat in range(3):
            renderer.enqueue[.MIS](settings)
            if repeat == 0 and phase != 1:
                assert_false(Bool(renderer._render_graph))
            var actual = download_gpu_pixels(renderer.ctx, renderer.target)
            assert_pixels_equal(actual, expected)
        assert_true(Bool(renderer._render_graph))
    enqueue_render_gpu[.PATH](
        renderer.ctx, expected_target, renderer.scene, settings
    )
    var path_expected = download_gpu_pixels(renderer.ctx, expected_target)
    for _ in range(3):
        renderer.enqueue[.PATH](settings)
        assert_pixels_equal(
            download_gpu_pixels(renderer.ctx, renderer.target), path_expected
        )
    assert_true(Bool(renderer._render_graph))
    # Progressive batches that never repeat do not pay for graph capture.
    for offset in [4, 6, 8]:
        settings.sample_offset = offset
        renderer.enqueue[.PATH](settings)
        _ = download_gpu_pixels(renderer.ctx, renderer.target)
        assert_false(Bool(renderer._render_graph))


def test_gpu_prepared_graph_replays_textured_instances() raises:
    var world = _textured_instance_world[True]()
    var camera = _camera()
    var settings = RenderSettings(8, 5, 2, UInt64(719), 5)
    var renderer = GpuRtPreparedRenderer[.INSTANCES](
        settings, camera, world.scene_data()
    )
    var expected = render_gpu_configured[kind=.INSTANCES, integrator=.MIS](
        settings, camera, world.scene_data()
    )
    for _ in range(3):
        assert_pixels_equal(
            renderer.render[.MIS](settings, camera).pixels, expected.pixels
        )
    assert_true(Bool(renderer._render_graph))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
