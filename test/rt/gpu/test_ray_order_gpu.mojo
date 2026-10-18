"""Queue ordering preserves records, indices, partial warps and chunk invariance."""

from test.rt.helpers import assert_pixels_equal
from std.testing import TestSuite, assert_equal, assert_true
from std.memory import bitcast
from max.gpu.host import DeviceContext
from bajo.rt.gpu.wavefront_contract import (
    GpuWavefrontArena,
    enqueue_wavefront_begin,
)
from bajo.rt.gpu.views import gpu_rt_trace_queue_view
from bajo.rt.gpu.queue_order import (
    gpu_rt_queue_order_init_kernel,
    gpu_rt_queue_order_scan_kernel,
)
from bajo.rt.gpu.material_order import gpu_rt_material_order_rays_kernel
from bajo.rt.gpu.light_order import gpu_rt_light_order_rays_kernel
from bajo.rt.gpu.spatial_order import gpu_rt_spatial_order_rays_kernel
from bajo.rt.wavefront_contract import (
    WAVE_COUNTER,
    WavePathFloatAbi,
    WaveShadowFloatAbi,
)
from bajo.core.random import Sampler
from bajo.core import Point3f32
from bajo.rt import SceneBuilder, Color, ImageTexture
from bajo.rt.material_types import Integrator, SurfaceId
from bajo.rt.render_types import RenderSettings
from bajo.rt.gpu import (
    GpuRtRenderTarget,
    prepare_gpu_scene,
    enqueue_render_gpu,
    download_gpu_pixels,
    GPU_RT_BVH_WIDE4,
    GPU_RT_BVH_CWBVH8,
    GPU_RT_BVH_TLAS2,
)
from test.rt.gpu.test_gpu import _camera, _textured_instance_world


def _spread2(n: Int) -> Int:
    return (n & 1) | ((n & 2) << 2)


def _assert_queue_order[
    mode: Int, skip_empty: Bool = False
](coated_count: Int = 63) raises:
    comptime capacity = 1057
    comptime bins = 512 if mode == 0 else 64
    comptime counter = WAVE_COUNTER.ACTIVE if mode == 0 else WAVE_COUNTER.SHADE if mode == 1 else WAVE_COUNTER.SHADOW
    comptime guarded_counter = counter if skip_empty else -1
    var world = _textured_instance_world[True]()
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.INSTANCES](ctx, world.scene_data())
        var scene_view = scene.view()
        # A synthetic TLAS root spans [0,4]^3. Positions at bin centers avoid
        # rounding ambiguity; negative/beyond-root positions exercise clamping.
        var nodes = ctx.enqueue_create_buffer[.float32](14)
        with nodes.map_to_host() as data:
            for i in range(6):
                data[i] = 0.0
                data[i + 6] = 4.0
            data[12] = bitcast[.float32](UInt32(0))
            data[13] = bitcast[.float32](UInt32(0))
        scene_view.instances.value().tlas_nodes = (
            nodes.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmUntrackedOrigin]()
        )
        scene_view.instances.value().tlas_root = 0
        var arena = GpuWavefrontArena(ctx, capacity)
        var keys = List[UInt32](length=capacity, fill=0)
        with arena.path_a.fields.map_to_host() as data:
            for i in range(len(data)):
                data[i] = -123.0
            for idx in range(capacity):
                var direction = (idx * 13 + idx // 7) % 8
                var x = (idx * 17 + idx // 11) % 4
                var y = (idx * 7 + idx // 5) % 4
                var z = (idx * 3 + idx // 17) % 4
                data[WavePathFloatAbi.OX * capacity + idx] = (
                    -4.0 if idx % 17 == 0 else Float32(x) + 0.25
                )
                data[WavePathFloatAbi.OY * capacity + idx] = (
                    8.0 if idx % 23 == 0 else Float32(y) + 0.25
                )
                data[WavePathFloatAbi.OZ * capacity + idx] = Float32(z) + 0.25
                if idx % 17 == 0:
                    x = 0
                if idx % 23 == 0:
                    y = 3
                data[WavePathFloatAbi.DX * capacity + idx] = (
                    -1.0 if direction & 1 else 1.0
                )
                data[WavePathFloatAbi.DY * capacity + idx] = (
                    -1.0 if direction & 2 else 1.0
                )
                data[WavePathFloatAbi.DZ * capacity + idx] = (
                    -1.0 if direction & 4 else 1.0
                )
                comptime if mode == 0:
                    keys[idx] = UInt32(
                        (direction << 6)
                        | _spread2(x)
                        | (_spread2(y) << 1)
                        | (_spread2(z) << 2)
                    )
                elif mode == 1:
                    keys[idx] = UInt32(
                        coated_count if idx % 5
                        == 0 else (idx * 13 + idx // 7) % coated_count
                    )
                else:
                    # Includes the environment bucket (63) and all finite IDs.
                    keys[idx] = UInt32((idx * 13 + idx // 7) % 64)
        with arena.shade.surface_values.map_to_host() as data:
            for idx in range(capacity):
                data[idx] = SurfaceId.pack_raw(
                    .LAMBERTIAN, UInt32(999)
                ) if idx % 5 == 0 else SurfaceId.pack_raw(
                    .COATED_DIFFUSE, keys[idx]
                )
        var queues = gpu_rt_trace_queue_view(
            arena,
            arena.path_a.path_ids,
            arena.path_a.fields,
            arena.path_b.path_ids,
            arena.path_b.fields,
        )
        for active in [1057, 0, 33, 1024, 1023, 1025, 1, 999, 31, 32, 0]:
            with arena.shadow.fields.map_to_host() as data:
                for i in range(len(data)):
                    data[i] = -123.0
            with arena.path_b.fields.map_to_host() as data:
                for i in range(len(data)):
                    data[i] = -123.0
                comptime if mode == 2:
                    for idx in range(capacity):
                        data[WavePathFloatAbi.T_MAX * capacity + idx] = bitcast[
                            .float32
                        ](keys[idx])
            enqueue_wavefront_begin(ctx, arena, active)
            ctx.synchronize()
            with arena.counters.map_to_host() as counters:
                counters[WAVE_COUNTER.SHADE] = UInt32(active)
                counters[WAVE_COUNTER.SHADOW] = UInt32(active)
            ctx.enqueue_function[
                gpu_rt_queue_order_init_kernel[bins, guarded_counter]
            ](queues, grid_dim=1, block_dim=bins)
            # A single block deliberately exercises multiple grid-stride steps.
            comptime for scatter in [False, True]:
                comptime if scatter:
                    ctx.enqueue_function[
                        gpu_rt_queue_order_scan_kernel[bins, guarded_counter]
                    ](queues, grid_dim=1, block_dim=bins)
                comptime if mode == 0:
                    ctx.enqueue_function[
                        gpu_rt_spatial_order_rays_kernel[
                            .INSTANCES,
                            GPU_RT_BVH_WIDE4,
                            GPU_RT_BVH_CWBVH8,
                            GPU_RT_BVH_TLAS2,
                            scatter,
                        ]
                    ](scene_view, queues, grid_dim=1, block_dim=256)
                elif mode == 1:
                    ctx.enqueue_function[
                        gpu_rt_material_order_rays_kernel[scatter]
                    ](queues, Int32(coated_count), grid_dim=1, block_dim=256)
                else:
                    ctx.enqueue_function[
                        gpu_rt_light_order_rays_kernel[scatter]
                    ](queues, grid_dim=1, block_dim=256)
            ctx.synchronize()
            comptime if skip_empty:
                if active == 0:
                    # Empty production queues leave scratch storage untouched.
                    with arena.shadow.fields.map_to_host() as data:
                        for slot in range(capacity):
                            assert_equal(
                                data[
                                    WaveShadowFloatAbi.T_MIN * capacity + slot
                                ],
                                Float32(-123.0),
                            )
                    with arena.path_b.fields.map_to_host() as data:
                        for slot in range(bins):
                            assert_equal(
                                data[WavePathFloatAbi.T_MIN * capacity + slot],
                                Float32(-123.0),
                            )
                    continue
            var seen = List[Bool](length=active, fill=False)
            var last_key = UInt32(0)
            with arena.shadow.fields.map_to_host() as data:
                for slot in range(active):
                    var idx = Int(
                        bitcast[.uint32](
                            data[WaveShadowFloatAbi.T_MIN * capacity + slot]
                        )
                    )
                    assert_true(idx >= 0 and idx < active)
                    assert_true(not seen[idx])
                    seen[idx] = True
                    assert_true(keys[idx] >= last_key)
                    last_key = keys[idx]
                for slot in range(active, capacity):
                    assert_equal(
                        data[WaveShadowFloatAbi.T_MIN * capacity + slot],
                        Float32(-123.0),
                    )
                for plane in range(WaveShadowFloatAbi.PLANES):
                    if plane == WaveShadowFloatAbi.T_MIN:
                        continue
                    for slot in range(capacity):
                        assert_equal(
                            data[plane * capacity + slot], Float32(-123.0)
                        )
            for present in seen:
                assert_true(present)
            with arena.path_b.fields.map_to_host() as data:
                for plane in range(WavePathFloatAbi.PLANES):
                    for slot in range(capacity):
                        if plane == WavePathFloatAbi.T_MIN and slot < bins:
                            continue
                        comptime if mode == 2:
                            if plane == WavePathFloatAbi.T_MAX:
                                assert_equal(
                                    bitcast[.uint32](
                                        data[plane * capacity + slot]
                                    ),
                                    keys[slot],
                                )
                                continue
                        assert_equal(
                            data[plane * capacity + slot], Float32(-123.0)
                        )

        # The scene view borrows a raw pointer. Retain its buffer owner through
        # the final queue use, including the first run's kernel compilation.
        with nodes.map_to_host() as data:
            assert_equal(data[0], Float32(0.0))
            assert_equal(data[6], Float32(4.0))


def test_gpu_spatial_order_preserves_partial_warps_and_queue_reuse() raises:
    _assert_queue_order[0]()


def test_gpu_material_order_handles_noncoatings_and_all_63_materials() raises:
    _assert_queue_order[1](63)
    _assert_queue_order[1](2)


def test_gpu_light_order_preserves_metadata_and_shadow_payloads() raises:
    _assert_queue_order[2]()


def test_gpu_order_skips_empty_queues_and_reuses_nonempty_queues() raises:
    _assert_queue_order[0, True]()
    _assert_queue_order[1, True](2)
    _assert_queue_order[2, True]()


def _assert_ordered_render[integrator: Integrator]() raises:
    var world = _textured_instance_world[True]()
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.INSTANCES](ctx, world.scene_data())
        for sampler in [
            Sampler.HALTON,
            Sampler.INDEPENDENT,
            Sampler.OWEN_SOBOL,
        ]:
            var settings = RenderSettings(17, 19, 8, UInt64(607), 6, sampler)
            var ordered = GpuRtRenderTarget(ctx, settings, _camera(), 2048)
            var reference = GpuRtRenderTarget(ctx, settings, _camera(), 512)
            enqueue_render_gpu[integrator](ctx, ordered, scene, settings)
            var actual = download_gpu_pixels(ctx, ordered)
            enqueue_render_gpu[integrator](ctx, reference, scene, settings)
            var expected = download_gpu_pixels(ctx, reference)
            assert_pixels_equal(actual, expected)


def test_gpu_mis_ordered_textured_instances_match_unordered_chunks() raises:
    _assert_ordered_render[.MIS]()


def test_gpu_nee_ordered_textured_instances_match_unordered_chunks() raises:
    _assert_ordered_render[.NEE]()


def _assert_grouped_lighting_render[integrator: Integrator]() raises:
    var builder = SceneBuilder()
    var texels: List[Float32] = [
        0.9,
        0.2,
        0.1,
        0.1,
        0.9,
        0.2,
        0.2,
        0.1,
        0.9,
        0.8,
        0.8,
        0.8,
    ]
    var texture = builder.surfaces.add_image_texture(
        ImageTexture(2, 2, texels^)
    )
    var coat_a = builder.surfaces.add_coated_diffuse(
        Color(0.8), 0.2, 1.5, texture
    )
    var coat_b = builder.add_coated_diffuse(Color(0.2, 0.4, 0.8), 0.35, 1.4)
    var floor = builder.add_lambertian(Color(0.5))
    var light = builder.add_emissive(Color(5.0, 4.0, 3.0))
    builder.add_sphere(Point3f32[.WORLD](-0.45, 0.0, -1.2), 0.4, coat_a)
    builder.add_sphere(Point3f32[.WORLD](0.45, 0.0, -1.2), 0.4, coat_b)
    builder.add_sphere(Point3f32[.WORLD](-0.8, 1.0, -0.8), 0.2, light)
    builder.add_triangle(
        Point3f32[.WORLD](-0.2, 1.0, -1.5),
        Point3f32[.WORLD](0.2, 1.0, -1.5),
        Point3f32[.WORLD](0.0, 1.0, -0.8),
        light,
    )
    builder.add_quad(
        Point3f32[.WORLD](-2.0, -0.5, -3.0),
        Point3f32[.WORLD](-2.0, -0.5, 1.0),
        Point3f32[.WORLD](2.0, -0.5, 1.0),
        Point3f32[.WORLD](2.0, -0.5, -3.0),
        floor,
    )
    var data = builder^.finish()
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.SPHERES_TRIANGLES](ctx, data)
        assert_equal(scene.lights.count, 2)
        for sampler in [
            Sampler.HALTON,
            Sampler.INDEPENDENT,
            Sampler.OWEN_SOBOL,
        ]:
            var settings = RenderSettings(17, 19, 8, UInt64(607), 6, sampler)
            var ordered = GpuRtRenderTarget(ctx, settings, _camera(), 2048)
            var reference = GpuRtRenderTarget(ctx, settings, _camera(), 512)
            enqueue_render_gpu[integrator](ctx, ordered, scene, settings)
            var actual = download_gpu_pixels(ctx, ordered)
            enqueue_render_gpu[integrator](ctx, reference, scene, settings)
            var expected = download_gpu_pixels(ctx, reference)
            assert_pixels_equal(actual, expected)


def test_gpu_mis_grouped_coatings_and_mixed_lights_match_unordered_chunks() raises:
    _assert_grouped_lighting_render[.MIS]()


def test_gpu_nee_grouped_coatings_and_mixed_lights_match_unordered_chunks() raises:
    _assert_grouped_lighting_render[.NEE]()


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
