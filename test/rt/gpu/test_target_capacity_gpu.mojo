"""Scene-sized target allocation preserves explicit bounds and pixel alignment."""

from test.rt.helpers import assert_pixels_equal
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_true, assert_false
from max.gpu.host import DeviceContext
from bajo.rt.gpu.preview import GpuRtPreviewAccumulator
from bajo.rt.gpu.resources import GpuRtRenderTarget
from bajo.rt.wavefront_contract import WAVE_COUNTER, WAVE_STATUS
from bajo.viewer.accumulation import FrameAccumulator
from bajo.viewer.worker import CompletedFrame
from test.viewer.test_accumulation import _frame
from bajo.benchmark.gpu_harness import gpu_rt_camera
from bajo.rt.film import FilmResponse
from bajo.rt.gpu.render import gpu_rt_apply_film_kernel

from bajo.core import Point3f32, Vec3f32
from bajo.rt.gpu.resources import download_gpu_pixels, update_gpu_camera
from bajo.rt.gpu.render import enqueue_render_gpu
from test.rt.gpu.test_gpu import _textured_instance_world
from bajo.rt import Camera, Color, RenderSettings, SceneBuilder
from bajo.rt.wavefront_contract import WavePathFloatAbi, WaveShadowFloatAbi
from bajo.rt.gpu.queue_layout import GPU_SHADE_PLANES
from bajo.rt.gpu import (
    GpuRtPreparedRenderer,
    prepare_gpu_scene,
    render_gpu_configured,
)


def _camera() -> Camera:
    return Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.0, 3.0),
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        45.0,
    )


def test_gpu_scene_target_bounds_and_alignment() raises:
    var builder = SceneBuilder()
    var coat = builder.add_coated_diffuse(Color(0.6), 0.2, 1.5)
    builder.add_sphere(Point3f32[.WORLD](0.0), 1.0, coat)
    var data = builder^.finish()
    var settings = RenderSettings(17, 19, 3, UInt64(653), 4)
    var camera = _camera()
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[.SPHERES](ctx, data)
        var full = scene.make_render_target(ctx, settings, camera)
        var bounded = scene.make_render_target(ctx, settings, camera, 100)
        assert_equal(full.arena.capacity, 17 * 19 * 3)
        assert_equal(bounded.arena.capacity, 99)
        assert_equal(
            len(full.arena.path_a.fields),
            WavePathFloatAbi.PLANES * full.arena.capacity,
        )
        assert_equal(
            len(full.arena.shade.fields),
            GPU_SHADE_PLANES * full.arena.capacity,
        )
        assert_equal(
            len(bounded.arena.shadow.fields),
            WaveShadowFloatAbi.PLANES * bounded.arena.capacity,
        )
        var tiny_settings = RenderSettings(2, 3, 17, UInt64(659), 4)
        var tiny = scene.make_render_target(ctx, tiny_settings, camera, 1)
        assert_equal(tiny.arena.capacity, 17)
        var whole = scene.make_render_target(ctx, tiny_settings, camera)
        assert_equal(whole.arena.capacity, 102)


def test_gpu_prepared_renderer_resizes_and_replays() raises:
    var builder = SceneBuilder()
    var coat = builder.add_coated_diffuse(Color(0.6), 0.2, 1.5)
    builder.add_sphere(Point3f32[.WORLD](0.0), 1.0, coat)
    var data = builder^.finish()
    var camera = _camera()
    var initial = RenderSettings(17, 13, 3, UInt64(677), 4)
    var renderer = GpuRtPreparedRenderer[.SPHERES](initial, camera, data)
    assert_equal(renderer.target.arena.capacity, 17 * 13 * 3)
    for width, height in [(11, 7), (19, 17)]:
        var resized = RenderSettings(width, height, 3, UInt64(677), 4)
        var actual = renderer.render[.NORMALS](resized, camera)
        assert_equal(renderer.target.arena.capacity, width * height * 3)
        var expected = render_gpu_configured[
            kind=.SPHERES, integrator=.NORMALS
        ](resized, camera, data)
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
            if repeat == 0 and phase != 1 and phase != 5:
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
    # Progressive offsets reuse the captured launch sequence.
    for offset in [4, 6, 8]:
        settings.sample_offset = offset
        renderer.enqueue[.PATH](settings)
        var progressive = download_gpu_pixels(renderer.ctx, renderer.target)
        enqueue_render_gpu[.PATH](
            renderer.ctx, expected_target, renderer.scene, settings
        )
        assert_pixels_equal(
            progressive, download_gpu_pixels(renderer.ctx, expected_target)
        )
        assert_true(Bool(renderer._render_graph))


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


def test_gpu_preview_weights_reset_tails_and_failed_work() raises:
    with DeviceContext() as ctx:
        for count in [1, 3, 7, 17, 65, 4096]:
            var target = GpuRtRenderTarget(
                ctx, RenderSettings(count, 1, 1, 2026, 1), gpu_rt_camera()
            )
            var status = ctx.enqueue_create_host_buffer[.uint32](
                WAVE_COUNTER.COUNT
            )
            status.as_span().fill(0)
            status.enqueue_copy_to(target.arena.counters)
            var gpu = GpuRtPreviewAccumulator(ctx, count)
            var cpu = FrameAccumulator()
            var input = ctx.enqueue_create_host_buffer[.float32](count * 3)
            for phase in range(6):
                if phase == 3:
                    cpu.reset()
                var samples = (phase * 2 + 1) % 7 + 1
                var frame = _frame(count, samples, cpu.generation, phase)
                _ = cpu.add(frame)
                input.as_span().copy_from(
                    ImmSpan(
                        unsafe_ptr=frame.pixels.unsafe_ptr().unsafe_bitcast[
                            Float32
                        ](),
                        length=count * 3,
                    )
                )
                input.enqueue_copy_to(target.pixels)
                var rgba = gpu.add(ctx, target, cpu.generation, samples)
                assert_equal(gpu.samples, cpu.samples)
                assert_equal(gpu.batches, cpu.batches)
                for i in range(len(rgba)):
                    assert_equal(rgba[i], cpu.rgba[i])
                with gpu.sums.map_to_host() as actual:
                    for i in range(count):
                        comptime for axis in range(3):
                            assert_equal(
                                bitcast[.uint32](actual[i * 3 + axis]),
                                bitcast[.uint32](cpu.linear[i][axis]),
                            )
            status.as_span()[WAVE_COUNTER.STATUS] = WAVE_STATUS.SHADOW_OVERFLOW
            status.enqueue_copy_to(target.arena.counters)
            var rejected = False
            try:
                _ = gpu.add(ctx, target, cpu.generation + 1, 1)
            except:
                rejected = True
            assert_true(rejected)
            assert_equal(gpu.generation, cpu.generation)
            assert_equal(gpu.samples, cpu.samples)
            with gpu.sums.map_to_host() as actual:
                for i in range(count):
                    comptime for axis in range(3):
                        assert_equal(
                            bitcast[.uint32](actual[i * 3 + axis]),
                            bitcast[.uint32](cpu.linear[i][axis]),
                        )


def test_gpu_preview_preserves_weights_across_batch_size_changes_and_resize() raises:
    var data = _textured_instance_world[True]()
    var settings = RenderSettings(11, 7, 1, 2026, 3, sample_sequence_length=32)
    var camera = _camera()
    var rgb = GpuRtPreparedRenderer[.INSTANCES](
        settings, camera, data.scene_data()
    )
    var preview = GpuRtPreparedRenderer[.INSTANCES](
        settings, camera, data.scene_data()
    )
    var cpu = FrameAccumulator()
    var display = FrameAccumulator()
    for phase in range(5):
        settings.samples_per_pixel = phase + 1
        settings.sample_offset = phase * 5
        if phase == 3:
            cpu.reset()
            display.reset()
            settings.image_width = 13
        var result = rgb.render[.MIS](settings, camera)
        var frame = CompletedFrame(result^, cpu.generation, settings)
        assert_true(cpu.add(frame))
        var device_result = preview.preview[.MIS](
            settings, camera, cpu.generation
        )
        var gpu_frame = CompletedFrame.from_preview(
            device_result^, cpu.generation, settings
        )
        assert_true(display.add(gpu_frame))
        assert_equal(display.samples, cpu.samples)
        assert_equal(display.batches, cpu.batches)
        assert_equal(len(display.rgba), len(cpu.rgba))
        for i in range(len(cpu.rgba)):
            assert_equal(display.rgba[i], cpu.rgba[i])


def test_fused_gpu_film_matches_separate_pass_across_sample_chunks() raises:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    builder.add_sphere(Point3f32[.WORLD](0, 0, -1), 0.5, matte)
    var data = builder^.finish()
    var camera = _camera()
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[.SPHERES](ctx, data)
        for samples in [1, 3, 4]:
            var settings = RenderSettings(7, 5, samples, 2026, 3)
            var reference = scene.make_render_target(ctx, settings, camera)
            var candidate = scene.make_render_target(ctx, settings, camera, 12)
            enqueue_render_gpu[.MIS](ctx, reference, scene, settings)
            settings.film = FilmResponse(
                1.25,
                Color(0.9, 0.1, 0.03),
                Color(0.04, 0.8, 0.06),
                Color(0.02, 0.07, 1.1),
            )
            ctx.enqueue_function[gpu_rt_apply_film_kernel](
                reference.pixels,
                Int32(reference.pixel_count),
                settings.film,
                grid_dim=1,
                block_dim=128,
            )
            enqueue_render_gpu[.MIS](ctx, candidate, scene, settings)
            var expected = download_gpu_pixels(ctx, reference)
            var actual = download_gpu_pixels(ctx, candidate)
            for i in range(len(expected)):
                comptime for axis in range(3):
                    assert_equal(
                        bitcast[.uint32](actual[i][axis]),
                        bitcast[.uint32](expected[i][axis]),
                    )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
