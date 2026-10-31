"""Device filter samples and actual reconstruction match CPU and reference values."""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal, assert_equal
from bajo.rt import CpuScene, render_wavefront
from bajo.rt.gpu import (
    prepare_gpu_scene,
    GpuRtRenderTarget,
    enqueue_render_gpu,
    download_gpu_pixels,
)
from bajo.rt.filter import FilterView, FilterSampler, ReconstructionFilter
from bajo.rt.gpu.filter import GpuFilterTables
from bajo.rt.material_types import Integrator
from test.rt.cpu.test_filter import make_filter_edge
from test.rt.helpers import assert_pixels_close


def _filter_kernel[
    _device: Bool = True
](view: FilterView, output: Pointer[Float32, MutAnyOrigin]):
    var index = global_idx.x
    if index >= 257:
        return
    var u = Float32(index) / 257
    var v = Float32(256 - index) / 257
    var sample = view.sample(SIMD[.float32, 1](u), SIMD[.float32, 1](v))
    output[unsafe_offset=3 * index] = sample.pixel_u
    output[unsafe_offset=3 * index + 1] = sample.pixel_v
    output[unsafe_offset=3 * index + 2] = sample.weight


def test_gpu_filter_tables_sample_default_rectangular_and_analytic_filters() raises:
    with DeviceContext() as ctx:
        for config in [
            ReconstructionFilter.gaussian(),
            ReconstructionFilter.gaussian(2.25, 0.75, 0.35),
            ReconstructionFilter.triangle(2, 1),
            ReconstructionFilter(x_radius=1.2, y_radius=0.75),
        ]:
            var host = FilterSampler(config)
            var device = GpuFilterTables(ctx, config)
            var output = ctx.enqueue_create_buffer[.float32](257 * 3)
            ctx.enqueue_function[_filter_kernel[True]](
                device.view(), output, grid_dim=5, block_dim=64
            )
            with output.map_to_host() as values:
                for index in range(257):
                    var sample = host.view().sample(
                        SIMD[.float32, 1](Float32(index) / 257),
                        SIMD[.float32, 1](Float32(256 - index) / 257),
                    )
                    assert_almost_equal(
                        values[3 * index], sample.pixel_u, atol=2e-6
                    )
                    assert_almost_equal(
                        values[3 * index + 1], sample.pixel_v, atol=2e-6
                    )
                    assert_almost_equal(
                        values[3 * index + 2], sample.weight, atol=2e-6
                    )
            _ = host^
            _ = device^


def test_gpu_gaussian_reconstructs_light_outside_the_nominal_pixel() raises:
    var parsed = make_filter_edge()
    var settings = parsed.settings.copy()
    var camera = parsed.camera
    var world = CpuScene[4, 4](parsed^.take_data())
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.TRIANGLES](ctx, world.scene_data())
        var target = GpuRtRenderTarget(ctx, settings, camera)
        comptime for integrator in [
            Integrator.PATH,
            Integrator.NEE,
            Integrator.MIS,
        ]:
            var cpu = render_wavefront[integrator, 4](settings, camera, world)
            enqueue_render_gpu[integrator](ctx, target, scene, settings)
            var gpu = download_gpu_pixels(ctx, target)
            assert_pixels_close(gpu, cpu.pixels, atol=1e-6)
            assert_almost_equal(gpu[0].x, Float32(0.074630116), atol=0.003)
        settings.filter = ReconstructionFilter()
        enqueue_render_gpu[.PATH](ctx, target, scene, settings)
        var box = download_gpu_pixels(ctx, target)
        assert_equal(box[0].x, Float32(0))


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
