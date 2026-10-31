"""Device spectral tables, wavelength PDFs, and physical CIE film conversion."""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.builtin.globals import global_constant
from std.memory import bitcast
from std.testing import TestSuite, assert_almost_equal, assert_equal
from bajo.core.color import Color
from bajo.core.vec import Vec3
from bajo.core.rgb_spectrum import RgbSpectrumTable, RgbSpectrumView
from bajo.core.sampled_spectrum import (
    SampledSpectrum,
    SampledWavelengths,
    sampled_sensor_xyz,
)
from bajo.rt.film import FilmResponse
from bajo.rt.gpu.spectra import GpuRgbSpectrumTable
from test.rt.spectral_reference import REFERENCE_RGB, REFERENCE_SPECTRA
from test.rt.cpu.test_spectra import reference_wavelengths, optical_scene
from bajo.rt.gpu import prepare_gpu_scene
from bajo.rt.gpu.views import GpuRtSceneView


def _spectra_kernel[
    width: SIMDLength, _device: Bool = True
](
    table: RgbSpectrumView,
    film: FilmResponse,
    output: Pointer[Float32, MutAnyOrigin],
):
    var base = Int(global_idx.x) * Int(width)
    if base >= 257:
        return
    ref colors = global_constant[REFERENCE_RGB]()
    var rgb = Vec3[.float32, .WORLD, width](0)
    var u = SIMD[.float32, width](0)
    var scale = SIMD[.float32, width](0)
    var active = SIMD[.bool, width](fill=False)
    for lane in range(Int(width)):
        var index = (base + lane) % 257
        rgb.x[lane] = colors[index % 12].x
        rgb.y[lane] = colors[index % 12].y
        rgb.z[lane] = colors[index % 12].z
        u[lane] = (Float32(index) + 0.5) / 257
        scale[lane] = 1 + Float32(index) / 12
        active[lane] = index % 3 == 0
    var wavelengths = SampledWavelengths[width].visible(u)
    wavelengths.terminate_secondary(active)
    var illuminant = table.sample(
        rgb * scale, wavelengths, RgbSpectrumView.ILLUMINANT
    )
    var albedo = table.sample(rgb, wavelengths, RgbSpectrumView.ALBEDO)
    var unbounded = table.sample(
        rgb * scale, wavelengths, RgbSpectrumView.UNBOUNDED
    )
    var xyz = sampled_sensor_xyz(illuminant, wavelengths)
    var pixel = film.apply_spectral_sample(illuminant, wavelengths)
    var uniform = SampledWavelengths[width].uniform(u)
    var compensated = (
        SampledSpectrum[width](1).safe_divide(wavelengths.pdf).average()
    )
    for lane in range(Int(width)):
        var index = base + lane
        if index >= 257:
            continue
        comptime for i in range(4):
            output[unsafe_offset=index * 32 + i] = wavelengths.lambda_nm[i][
                lane
            ]
            output[unsafe_offset=index * 32 + 4 + i] = wavelengths.pdf[i][lane]
            output[unsafe_offset=index * 32 + 8 + i] = illuminant[i][lane]
            output[unsafe_offset=index * 32 + 12 + i] = albedo[i][lane]
            output[unsafe_offset=index * 32 + 16 + i] = unbounded[i][lane]
            output[unsafe_offset=index * 32 + 26 + i] = table.illuminant(
                wavelengths.lambda_nm[i]
            )[lane]
        comptime for c in range(3):
            output[unsafe_offset=index * 32 + 20 + c] = xyz[c][lane]
            output[unsafe_offset=index * 32 + 23 + c] = pixel[c][lane]
        output[unsafe_offset=index * 32 + 30] = uniform.pdf.s0[lane]
        output[unsafe_offset=index * 32 + 31] = compensated[lane]


def _reference_kernel[
    _device: Bool = True
](table: RgbSpectrumView, output: Pointer[Float32, MutAnyOrigin]):
    var index = Int(global_idx.x)
    if index >= 12:
        return
    ref colors = global_constant[REFERENCE_RGB]()
    for kind in range(3):
        var values = table.sample(
            colors[index], reference_wavelengths(), UInt32(kind)
        )
        comptime for i in range(4):
            output[unsafe_offset=index * 12 + kind * 4 + i] = values[i]


def test_gpu_original_table_bits_and_independent_reference_spectra() raises:
    var owner = RgbSpectrumTable.read("bajo/core/data/srgb_spectra.bin")
    with DeviceContext() as ctx:
        var device = GpuRgbSpectrumTable(ctx, owner)
        with device.data.map_to_host() as words:
            var downloaded = words.as_span()
            var actual = downloaded.unsafe_ptr().unsafe_bitcast[UInt32]()
            var expected = owner.values.unsafe_ptr().unsafe_bitcast[UInt32]()
            var different = False
            var end = len(owner.values) // 64 * 64
            for index in range(0, end, 64):
                different |= (
                    actual.unsafe_load[width=64](index)
                    .ne(expected.unsafe_load[width=64](index))
                    .reduce_or()
                )
            for index in range(end, len(owner.values)):
                different |= (
                    actual[unsafe_offset=index] != expected[unsafe_offset=index]
                )
            assert_equal(different, False)
        var output = ctx.enqueue_create_buffer[.float32](12 * 12)
        ctx.enqueue_function[_reference_kernel[True]](
            device.view(), output, grid_dim=1, block_dim=32
        )
        ref expected = global_constant[REFERENCE_SPECTRA]()
        with output.map_to_host() as values:
            for index in range(12):
                for i in range(12):
                    assert_almost_equal(
                        values[index * 12 + i],
                        expected[index][3 + i],
                        atol=2e-6,
                    )
        _ = device^
    _ = owner^


def test_gpu_wavelengths_spectral_arithmetic_and_cie_film_match_scalar_cpu() raises:
    var owner = RgbSpectrumTable.read("bajo/core/data/srgb_spectra.bin")
    var host = owner.view()
    var film = FilmResponse(1.7, max_component_value=20)
    film.use_cie1931()
    ref colors = global_constant[REFERENCE_RGB]()
    with DeviceContext() as ctx:
        var device = GpuRgbSpectrumTable(ctx, owner)
        comptime for width in [1, 4, 16]:
            var output = ctx.enqueue_create_buffer[.float32](257 * 32)
            ctx.enqueue_function[_spectra_kernel[width, True]](
                device.view(), film, output, grid_dim=5, block_dim=64
            )
            with output.map_to_host() as values:
                for index in range(257):
                    var u = (Float32(index) + 0.5) / 257
                    var wavelengths = SampledWavelengths[1].visible(u)
                    wavelengths.terminate_secondary(
                        SIMD[.bool, 1](index % 3 == 0)
                    )
                    var rgb = colors[index % 12]
                    var scale = 1 + Float32(index) / 12
                    var illuminant = host.sample(
                        rgb * scale, wavelengths, RgbSpectrumView.ILLUMINANT
                    )
                    var albedo = host.sample(
                        rgb, wavelengths, RgbSpectrumView.ALBEDO
                    )
                    var unbounded = host.sample(
                        rgb * scale, wavelengths, RgbSpectrumView.UNBOUNDED
                    )
                    var xyz = sampled_sensor_xyz(illuminant, wavelengths)
                    var pixel = film.apply_spectral_sample(
                        illuminant, wavelengths
                    )
                    comptime for i in range(4):
                        assert_almost_equal(
                            values[index * 32 + i],
                            wavelengths.lambda_nm[i],
                            atol=0.0002,
                        )
                        assert_almost_equal(
                            values[index * 32 + 4 + i],
                            wavelengths.pdf[i],
                            atol=1e-8,
                        )
                        assert_almost_equal(
                            values[index * 32 + 8 + i],
                            illuminant[i],
                            atol=0.0002,
                            rtol=0.00002,
                        )
                        assert_almost_equal(
                            values[index * 32 + 12 + i], albedo[i], atol=0.00001
                        )
                        assert_almost_equal(
                            values[index * 32 + 16 + i],
                            unbounded[i],
                            atol=0.0002,
                            rtol=0.00002,
                        )
                        assert_almost_equal(
                            values[index * 32 + 26 + i],
                            host.illuminant(wavelengths.lambda_nm[i]),
                            atol=3e-7,
                        )
                    comptime for c in range(3):
                        assert_almost_equal(
                            values[index * 32 + 20 + c],
                            xyz[c],
                            atol=0.0003,
                            rtol=0.00003,
                        )
                        assert_almost_equal(
                            values[index * 32 + 23 + c],
                            pixel[c],
                            atol=0.0001,
                            rtol=0.00003,
                        )
                    assert_almost_equal(
                        values[index * 32 + 30], Float32(1.0 / 470), atol=1e-10
                    )
                    assert_almost_equal(
                        values[index * 32 + 31],
                        SampledSpectrum[1](1)
                        .safe_divide(wavelengths.pdf)
                        .average(),
                        atol=0.001,
                    )
        _ = device^
    _ = owner^


def _optical_scene_kernel[
    _device: Bool = True
](scene: GpuRtSceneView, output: Pointer[Float32, MutAnyOrigin]):
    var index = Int(global_idx.x)
    if index >= 256:
        return
    var nm = Float32(360) + Float32(index) * (Float32(470) / 255)
    var wavelengths = SampledWavelengths[1](
        SampledSpectrum[1](nm), SampledSpectrum[1](1)
    )
    var eta_id = scene.spectra.coated_eta_k[unsafe_offset=0]
    var k_id = scene.spectra.coated_eta_k[unsafe_offset=1]
    var glass_id = scene.spectra.dielectric_eta[unsafe_offset=0]
    var constant_id = scene.spectra.dielectric_eta[unsafe_offset=1]
    output[unsafe_offset=index * 4] = scene.spectra.curves.sample(
        eta_id, wavelengths
    ).s0
    output[unsafe_offset=index * 4 + 1] = scene.spectra.curves.sample(
        k_id, wavelengths
    ).s0
    output[unsafe_offset=index * 4 + 2] = scene.spectra.curves.sample(
        glass_id, wavelengths
    ).s0
    output[unsafe_offset=index * 4 + 3] = scene.spectra.curves.sample(
        constant_id, wavelengths, SIMD[.float32, 1](1.4)
    ).s0


def test_gpu_scene_preparation_binds_original_optical_curves_and_material_ids() raises:
    var parsed = optical_scene()
    ref surfaces = parsed.data.surfaces()
    with DeviceContext() as ctx:
        var scene = prepare_gpu_scene[kind=.SPHERES](ctx, parsed.data)
        assert_equal(scene.materials.spectra.count, UInt32(3))
        var output = ctx.enqueue_create_buffer[.float32](256 * 4)
        ctx.enqueue_function[_optical_scene_kernel[True]](
            scene.view(), output, grid_dim=4, block_dim=64
        )
        with output.map_to_host() as values:
            for index in range(256):
                var nm = Float32(360) + Float32(index) * (Float32(470) / 255)
                for curve in range(3):
                    assert_almost_equal(
                        values[index * 4 + curve],
                        surfaces.optical_spectra[curve].evaluate(nm),
                        atol=1e-6,
                    )
                assert_equal(values[index * 4 + 3], Float32(1.4))
        _ = scene^


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
