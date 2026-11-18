"""Four-wavelength primitives and original-table spectral reconstruction."""

from std.builtin.globals import global_constant
from std.memory import bitcast
from std.math import floor
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from bajo.core.color import Color
from bajo.core.vec import Vec3
from bajo.core.cie_data import CIE_XYZ
from bajo.core.sensor_data import CANON_RESPONSE
from bajo.core.sampled_spectrum import (
    SampledSpectrum,
    SampledWavelengths,
    visible_wavelength_pdf,
    cie_xyz,
    sampled_sensor_xyz,
    spectral_to_xyz,
    CIE_Y_INTEGRAL,
)
from bajo.core.rgb_spectrum import (
    RgbSpectrumTable,
    RgbSpectrumView,
    RGB_SPECTRUM_D65_OFFSET,
)
from bajo.rt.sensor import canon_sensor_sample
from bajo.rt.spectrum_sources import (
    SpectrumSource,
    PackedSpectrumSources,
    SpectrumSourcesView,
    NO_SPECTRUM,
)
from bajo.rt.film import FilmResponse
from bajo.parser.pbrt import parse_pbrt
from bajo.rt.scene_description import SceneDescription
from bajo.core.spectrum import PiecewiseLinearSpectrum
from bajo.core.spectrum_tables import PiecewiseSpectrumTable
from test.rt.spectral_reference import (
    REFERENCE_RGB,
    REFERENCE_SPECTRA,
    REFERENCE_VISIBLE,
    REFERENCE_D65,
    REFERENCE_SENSOR,
)


def reference_wavelengths[width: SIMDLength = 1]() -> SampledWavelengths[width]:
    return SampledWavelengths[width](
        SampledSpectrum[width](360, 460.3, 550.7, 830),
        SampledSpectrum[width](0.1, 0.2, 0.3, 0.4),
    )


def test_visible_and_uniform_wavelengths_match_independent_reference() raises:
    var inputs: Array[Float32, 6] = [0, 0.1, 0.25, 0.5, 0.9, 1]
    ref expected_visible = global_constant[REFERENCE_VISIBLE]()
    comptime for width in [1, 4, 16]:
        for index in range(6):
            var wavelengths = SampledWavelengths[width].visible(
                SIMD[.float32, width](inputs[index])
            )
            for lane in range(width):
                assert_almost_equal(
                    wavelengths.lambda_nm.s0[lane],
                    expected_visible[index][0],
                    atol=0.0002,
                )
                assert_almost_equal(
                    wavelengths.pdf.s0[lane],
                    Float32(0) if wavelengths.lambda_nm.s0[lane]
                    > 830 else expected_visible[index][1],
                    atol=1e-8,
                )
            var uniform = SampledWavelengths[width].uniform(
                SIMD[.float32, width](0.9)
            )
            for i in range(4):
                var expected = Float32(783) + Float32(117.5) * Float32(i)
                if expected > 830:
                    expected -= 470
                for lane in range(width):
                    assert_almost_equal(
                        uniform.lambda_nm[i][lane], expected, atol=0.0001
                    )
                    assert_almost_equal(
                        uniform.pdf[i][lane], Float32(1.0 / 470), atol=1e-10
                    )
        var cyclic = SampledWavelengths[width].visible(
            SIMD[.float32, width](0.9)
        )
        for i in range(4):
            var u = Float32(0.9) + Float32(i) * 0.25
            if u > 1:
                u -= 1
            var separate = SampledWavelengths[width].visible(
                SIMD[.float32, width](u)
            )
            for lane in range(width):
                assert_almost_equal(
                    cyclic.lambda_nm[i][lane],
                    separate.lambda_nm.s0[lane],
                    atol=0.0001,
                )
    assert_equal(visible_wavelength_pdf(SIMD[.float32, 1](359)), Float32(0))
    assert_equal(visible_wavelength_pdf(SIMD[.float32, 1](831)), Float32(0))


def test_spectral_arithmetic_and_lane_specific_secondary_termination() raises:
    var a = SampledSpectrum[4](SIMD[.float32, 4](1, 2, 3, 4))
    var b = SampledSpectrum[4](2, 4, 6, 8)
    var result = ((a + b) * b - b).safe_divide(b)
    for i in range(4):
        for lane in range(4):
            assert_equal(
                result[i][lane], Float32(lane + 1) + Float32(2 * (i + 1)) - 1
            )
    assert_equal(b.average()[0], Float32(5))
    assert_equal(b.max_component()[0], Float32(8))
    var wavelengths = SampledWavelengths[4].uniform(
        SIMD[.float32, 4](0.1, 0.3, 0.6, 0.9)
    )
    var active = SIMD[.bool, 4](True, False, True, False)
    wavelengths.terminate_secondary(active)
    var once = wavelengths
    wavelengths.terminate_secondary(active)
    for i in range(4):
        for lane in range(4):
            assert_equal(wavelengths.pdf[i][lane], once.pdf[i][lane])
            var expected = Float32(1.0 / 470)
            if active[lane]:
                expected = expected * 0.25 if i == 0 else 0
            assert_almost_equal(wavelengths.pdf[i][lane], expected, atol=1e-10)
    var weighted = b.safe_divide(wavelengths.pdf)
    assert_equal(weighted.s1[0], Float32(0))
    assert_true(weighted.s1[1] > 0)
    # PDF compensation keeps the average of a constant spectrum unchanged.
    assert_almost_equal(
        a.safe_divide(wavelengths.pdf).average()[0], Float32(470), atol=0.0001
    )


def test_dense_cie_lookup_and_pdf_weighted_sensor_use_original_samples() raises:
    ref observer = global_constant[CIE_XYZ]()
    for nm in [360, 460, 551, 830]:
        var xyz = cie_xyz(SIMD[.float32, 1](Float32(nm)))
        for c in range(3):
            assert_equal(xyz[c], observer[nm - 360][c])
    assert_equal(cie_xyz(SIMD[.float32, 1](550.49)).y, observer[190].y)
    assert_equal(cie_xyz(SIMD[.float32, 1](550.5)).y, observer[191].y)
    for invalid in [
        Float32(359.49),
        Float32(830.5),
        bitcast[.float32](UInt32(0x7FC00000)),
        bitcast[.float32](UInt32(0x7F800000)),
    ]:
        var zero = cie_xyz(SIMD[.float32, 1](invalid))
        assert_equal(zero.max_component(), Float32(0))
    comptime for width in [1, 4, 16]:
        var wavelengths = reference_wavelengths[width]()
        var radiance = SampledSpectrum[width](2, 3, 5, 7)
        var sensor = sampled_sensor_xyz(radiance, wavelengths)
        var normalized = spectral_to_xyz(radiance, wavelengths)
        for lane in range(width):
            for channel in range(3):
                assert_almost_equal(
                    sensor[channel][lane], REFERENCE_SENSOR[channel], atol=1e-6
                )
                assert_almost_equal(
                    normalized[channel][lane] * CIE_Y_INTEGRAL,
                    sensor[channel][lane],
                    atol=1e-6,
                )


def test_original_rgb_table_coefficients_and_spectra_match_independent_cpp() raises:
    var owner = RgbSpectrumTable.read("bajo/core/data/srgb_spectra.bin")
    var table = owner.view()
    ref rgbs = global_constant[REFERENCE_RGB]()
    ref expected = global_constant[REFERENCE_SPECTRA]()
    for index in range(12):
        var c = table.coefficients(rgbs[index])
        if index != 9 and index != 10:
            assert_almost_equal(c.a, expected[index][0], atol=1e-8)
            assert_almost_equal(c.b, expected[index][1], atol=1e-6)
            assert_almost_equal(c.c, expected[index][2], atol=0.0001)
        for kind in range(3):
            var actual = table.sample(
                rgbs[index], reference_wavelengths(), UInt32(kind)
            )
            for i in range(4):
                assert_almost_equal(
                    actual[i], expected[index][3 + kind * 4 + i], atol=2e-6
                )
    # Different colors occupy different SIMD lanes and different table axes.
    comptime for width in [4, 16]:
        var rgb = Vec3[.float32, .WORLD, width](0)
        for lane in range(width):
            rgb.x[lane] = rgbs[lane % 12].x
            rgb.y[lane] = rgbs[lane % 12].y
            rgb.z[lane] = rgbs[lane % 12].z
        for kind in range(3):
            var actual = table.sample(
                rgb, reference_wavelengths[width](), UInt32(kind)
            )
            for i in range(4):
                for lane in range(width):
                    assert_almost_equal(
                        actual[i][lane],
                        expected[lane % 12][3 + kind * 4 + i],
                        atol=2e-6,
                    )
    _ = owner^


def test_d65_normalization_and_spectral_rgb_round_trip() raises:
    var owner = RgbSpectrumTable.read("bajo/core/data/srgb_spectra.bin")
    var table = owner.view()
    var wavelengths = reference_wavelengths()
    for i in range(4):
        assert_almost_equal(
            table.illuminant(wavelengths.lambda_nm[i]),
            REFERENCE_D65[i],
            atol=3e-7,
        )
    var y = Float64(0)
    for nm in range(360, 831):
        y += Float64(
            table.illuminant(SIMD[.float32, 1](Float32(nm)))
        ) * Float64(cie_xyz(SIMD[.float32, 1](Float32(nm))).y)
    assert_almost_equal(y, Float64(CIE_Y_INTEGRAL), atol=0.0001)
    var film = FilmResponse()
    film.use_cie1931()
    ref rgbs = global_constant[REFERENCE_RGB]()
    for index in [0, 1, 2, 11]:
        var xyz = Vec3[.float64, .WORLD](0)
        for nm in range(360, 831):
            var sample = SampledWavelengths[1](
                SampledSpectrum[1](Float32(nm)), SampledSpectrum[1](1)
            )
            var s = table.sample(
                rgbs[index], sample, RgbSpectrumView.ILLUMINANT
            )
            var observer = cie_xyz(SIMD[.float32, 1](Float32(nm)))
            for c in range(3):
                xyz[c] += (
                    Float64(observer[c])
                    * Float64(s.s0)
                    / Float64(CIE_Y_INTEGRAL)
                )
        var rgb = Color(
            Float32(xyz.x) * film.xyz_red.x
            + Float32(xyz.y) * film.xyz_red.y
            + Float32(xyz.z) * film.xyz_red.z,
            Float32(xyz.x) * film.xyz_green.x
            + Float32(xyz.y) * film.xyz_green.y
            + Float32(xyz.z) * film.xyz_green.z,
            Float32(xyz.x) * film.xyz_blue.x
            + Float32(xyz.y) * film.xyz_blue.y
            + Float32(xyz.z) * film.xyz_blue.z,
        )
        for c in range(3):
            # The original PBRT reconstruction is itself a fitted model.
            assert_almost_equal(rgb[c], rgbs[index][c], atol=0.003)
    _ = owner^


def test_spectral_film_clamps_integrated_sensor_before_output_conversion() raises:
    var film = FilmResponse(3, max_component_value=2)
    film.use_cie1931()
    comptime for width in [1, 4, 16]:
        var rgb = film.apply_spectral_sample(
            SampledSpectrum[width](2, 3, 5, 7), reference_wavelengths[width]()
        )
        assert_true(rgb.x[0] < 0)
        var sensor = Vec3[.float32, .WORLD, width](
            rgb.x * film.sensor_red.x
            + rgb.y * film.sensor_red.y
            + rgb.z * film.sensor_red.z,
            rgb.x * film.sensor_green.x
            + rgb.y * film.sensor_green.y
            + rgb.z * film.sensor_green.z,
            rgb.x * film.sensor_blue.x
            + rgb.y * film.sensor_blue.y
            + rgb.z * film.sensor_blue.z,
        )
        for lane in range(width):
            for c in range(3):
                assert_almost_equal(
                    sensor[c][lane],
                    REFERENCE_SENSOR[c] * (Float32(2) / REFERENCE_SENSOR.z),
                    atol=1e-6,
                )
        var raw = FilmResponse(3)
        raw.use_cie1931()
        var unclamped = raw.apply_spectral_sample(
            SampledSpectrum[width](2, 3, 5, 7), reference_wavelengths[width]()
        )
        var recovered_y = (
            unclamped.x * raw.sensor_green.x
            + unclamped.y * raw.sensor_green.y
            + unclamped.z * raw.sensor_green.z
        )
        assert_almost_equal(recovered_y[0], REFERENCE_SENSOR.y * 3, atol=4e-6)
        var zero_pdf = SampledWavelengths[width](
            reference_wavelengths[width]().lambda_nm, SampledSpectrum[width](0)
        )
        assert_equal(
            film.apply_spectral_sample(
                SampledSpectrum[width](1), zero_pdf
            ).max_component()[0],
            Float32(0),
        )


def optical_scene() raises -> SceneDescription:
    return parse_pbrt(
        """WorldBegin
Material "coatedconductor" "spectrum conductor.eta" "metal-CuZn-eta"
 "spectrum conductor.k" "metal-CuZn-k"
Shape "sphere"
Translate 3 0 0
Material "dielectric" "spectrum eta" [400 1.6 700 1.5]
Shape "sphere"
Translate 3 0 0
Material "dielectric" "float eta" 1.4
Shape "sphere"
"""
    )


def test_packed_optical_curves_preserve_original_samples_and_mixed_lane_ids() raises:
    var scene = optical_scene()
    ref surfaces = scene.data.surfaces()
    var owner = PiecewiseSpectrumTable(surfaces.optical_spectra)
    var table = owner.view()
    assert_equal(table.count, UInt32(3))
    var wavelengths = SampledWavelengths[4](
        SampledSpectrum[4](
            SIMD[.float32, 4](555, 555, 475, 550), 550, 700, 890
        ),
        SampledSpectrum[4](1),
    )
    var values = table.sample(
        SIMD[.uint32, 4](0, 1, 2, 0xFFFFFFFF),
        wavelengths,
        SIMD[.float32, 4](1.4),
    )
    assert_almost_equal(values.s0[0], Float32(0.516), atol=1e-7)
    assert_almost_equal(values.s0[1], Float32(2.8125), atol=1e-7)
    assert_almost_equal(values.s0[2], Float32(1.575), atol=1e-7)
    for i in range(4):
        assert_equal(values[i][3], Float32(1.4))
    assert_equal(values.s3[2], Float32(0))
    assert_equal(values.s3[1], Float32(5.575))
    for index in range(3):
        ref curve = surfaces.optical_spectra[index]
        for pair in range(0, len(curve.samples), 2):
            var samples = SampledWavelengths[1](
                SampledSpectrum[1](curve.samples[pair]), SampledSpectrum[1](1)
            )
            var actual = table.sample(SIMD[.uint32, 1](UInt32(index)), samples)
            assert_equal(actual.s0, curve.samples[pair + 1])
    var empty = List[PiecewiseLinearSpectrum]()
    var empty_owner = PiecewiseSpectrumTable(empty)
    var fallback = empty_owner.view().sample(
        SIMD[.uint32, 4](0), wavelengths, SIMD[.float32, 4](1.4)
    )
    assert_equal(fallback.average()[0], Float32(1.4))
    _ = empty_owner^
    _ = owner^


def _linear_curve_reference(
    samples: ImmSpan[Float32, _], nm: Float32
) -> Float32:
    if not (nm >= samples[0] and nm <= samples[len(samples) - 2]):
        return 0
    if nm == samples[len(samples) - 2]:
        return samples[len(samples) - 1]
    for i in range(2, len(samples), 2):
        if nm < samples[i]:
            var t = (nm - samples[i - 2]) / (samples[i] - samples[i - 2])
            return samples[i - 1] * (1 - t) + samples[i + 1] * t
    return 0


def _check_piecewise_packets[width: SIMDLength]() raises:
    var curves = List[PiecewiseLinearSpectrum]()
    for count in [1, 2, 3, 4, 5, 6, 10, 17, 33, 65]:
        var pairs = List[Float32]()
        for i in range(count):
            pairs.append(360 + Float32(i * i))
            pairs.append(Float32((i * 7 + count) % 19) / 8)
        curves.append(PiecewiseLinearSpectrum(pairs^))
    for curve in curves:
        for pair in range(0, len(curve.samples), 2):
            assert_equal(
                curve.evaluate(SIMD[.float32, width](curve.samples[pair])),
                SIMD[.float32, width](curve.samples[pair + 1]),
            )
    var owner = PiecewiseSpectrumTable(curves)
    var table = owner.view()
    for batch in range(20):
        var ids = SIMD[.uint32, width](0)
        var nm = SIMD[.float32, width](0)
        var fallback = SIMD[.float32, width](0)
        for lane in range(Int(width)):
            var index = batch * Int(width) + lane
            var choice = index % (len(curves) + 2)
            ids[lane] = UInt32(choice) if choice <= len(curves) else UInt32.MAX
            fallback[lane] = -Float32(index + 1)
            nm[lane] = Float32(359.5) + Float32(index % 140) * 3
            if index % 5 == 0:
                nm[lane] = 360
            elif index % 5 == 1:
                nm[lane] = 361
            elif index % 5 == 2:
                nm[lane] = bitcast[.float32](UInt32(0x7FC00000))
            elif index % 5 == 3:
                nm[lane] = bitcast[.float32](UInt32(0x7F800000))
        var wavelengths = SampledWavelengths[width](
            SampledSpectrum[width](nm, nm + 0.25, nm + 1, nm + 4096),
            SampledSpectrum[width](1),
        )
        var actual = table.sample(ids, wavelengths, fallback)
        comptime for channel in range(4):
            for lane in range(Int(width)):
                var expected = fallback[lane]
                if ids[lane] < UInt32(len(curves)):
                    expected = _linear_curve_reference(
                        curves[Int(ids[lane])].samples,
                        wavelengths.lambda_nm[channel][lane],
                    )
                assert_almost_equal(actual[channel][lane], expected, atol=1e-6)
        for curve in curves:
            var value = curve.evaluate(nm)
            for lane in range(Int(width)):
                assert_almost_equal(
                    value[lane],
                    _linear_curve_reference(curve.samples, nm[lane]),
                    atol=1e-6,
                )
    var missing = table.sample(
        SIMD[.uint32, width](UInt32.MAX),
        reference_wavelengths[width](),
        SIMD[.float32, width](1.4),
    )
    assert_equal(missing.s0, SIMD[.float32, width](1.4))
    var empty = List[PiecewiseLinearSpectrum]()
    var empty_table = PiecewiseSpectrumTable(empty)
    assert_equal(
        empty_table.view()
        .sample(SIMD[.uint32, width](0), reference_wavelengths[width](), 1.4)
        .s3,
        SIMD[.float32, width](1.4),
    )
    _ = empty_table^
    _ = owner^


def test_piecewise_packets_match_linear_reference_with_mixed_lengths_and_masks() raises:
    comptime for width in [1, 2, 4, 8, 16, 32]:
        _check_piecewise_packets[width]()


def _check_dense_lookup_packets[
    width: SIMDLength
](table: RgbSpectrumView) raises:
    ref observer = global_constant[CIE_XYZ]()
    ref sensor = global_constant[CANON_RESPONSE]()
    var inputs: Array[Float32, 16] = [
        359.49,
        359.5,
        360,
        360.49,
        360.5,
        550.49,
        550.5,
        829.5,
        830,
        830.4999,
        830.5,
        bitcast[.float32](UInt32(0x7FC00000)),
        bitcast[.float32](UInt32(0x7F800000)),
        bitcast[.float32](UInt32(0xFF800000)),
        1e30,
        -1e30,
    ]
    for first in range(0, 32, Int(width)):
        var nm = SIMD[.float32, width](0)
        for lane in range(Int(width)):
            nm[lane] = inputs[(first + lane) % len(inputs)]
        var cie = cie_xyz(nm)
        var d65 = table.illuminant(nm)
        var wavelengths = SampledWavelengths[width](
            SampledSpectrum[width](nm), SampledSpectrum[width](1)
        )
        var canon = canon_sensor_sample(SampledSpectrum[width](1), wavelengths)
        for lane in range(Int(width)):
            var expected_cie = Color(0)
            var expected_canon = Color(0)
            var expected_d65 = Float32(0)
            if nm[lane] >= 359.5 and nm[lane] < 830.5:
                var index = Int(floor(nm[lane] + 0.5)) - 360
                if index >= 0 and index < len(observer):
                    expected_cie = observer[index]
                expected_d65 = table.data[
                    unsafe_offset=RGB_SPECTRUM_D65_OFFSET
                    + min(max(index, 0), 470)
                ]
            if nm[lane] >= 360 and nm[lane] <= 830:
                expected_canon = sensor[Int(floor(nm[lane] + 0.5)) - 360]
            for channel in range(3):
                assert_equal(cie[channel][lane], expected_cie[channel])
                assert_equal(canon[channel][lane], expected_canon[channel])
            assert_equal(d65[lane], expected_d65)


def test_dense_lookup_packets_mask_invalid_wavelengths_before_gathers() raises:
    var owner = RgbSpectrumTable.read_default()
    comptime for width in [1, 2, 4, 8, 16, 32]:
        _check_dense_lookup_packets[width](owner.view())
    _ = owner^


def _check_rgb_lookup_packets[width: SIMDLength](table: RgbSpectrumView) raises:
    for batch in range(12):
        var rgb = Vec3[.float32, .WORLD, width](0)
        for lane in range(Int(width)):
            var index = batch * Int(width) + lane
            rgb.x[lane] = Float32((index * 13) % 257) / 256
            rgb.y[lane] = Float32((index * 37) % 257) / 256
            rgb.z[lane] = Float32((index * 71) % 257) / 256
            if index % 7 < 3:
                var gray = Float32(index % 7) / 2
                rgb.x[lane] = gray
                rgb.y[lane] = gray
                rgb.z[lane] = gray
            elif index % 7 == 3:
                rgb.y[lane] = rgb.x[lane]
            elif index % 7 == 4:
                rgb.z[lane] = rgb.x[lane]
            elif index % 7 == 5:
                rgb.z[lane] = rgb.y[lane]
        var wavelengths = reference_wavelengths[width]()
        for kind in range(3):
            var actual = table.sample(rgb, wavelengths, UInt32(kind))
            for lane in range(Int(width)):
                var scalar = table.sample(
                    Color(rgb.x[lane], rgb.y[lane], rgb.z[lane]),
                    wavelengths.get(lane),
                    UInt32(kind),
                )
                comptime for channel in range(4):
                    assert_almost_equal(
                        actual[channel][lane], scalar[channel], atol=2e-6
                    )


def test_rgb_lookup_packets_preserve_gray_lanes_and_axis_ties() raises:
    var owner = RgbSpectrumTable.read_default()
    comptime for width in [1, 2, 4, 8, 16, 32]:
        _check_rgb_lookup_packets[width](owner.view())
    _ = owner^


def _check_source_packets[width: SIMDLength](view: SpectrumSourcesView) raises:
    for batch in range(8):
        var ids = SIMD[.uint32, width](NO_SPECTRUM)
        var active = SIMD[.bool, width](fill=False)
        var samples = SIMD[.float32, width](0)
        for lane in range(Int(width)):
            var index = batch * Int(width) + lane
            ids[lane] = UInt32(index % (Int(view.count) + 2))
            active[lane] = index % 4 != 1
            if batch == 5:
                ids[lane] = 6
            elif batch == 6:
                ids[lane] = UInt32(5 + lane % 2)
            elif batch == 7:
                active[lane] = False
                ids[lane] = NO_SPECTRUM
            samples[lane] = (Float32(index % 61) + 0.5) / 61
        var wavelengths = SampledWavelengths[width].visible(samples)
        var fallback = SampledSpectrum[width](0.1, 0.2, 0.3, 0.4)
        for role in range(3):
            var actual = view.sample_packet(
                ids, wavelengths, UInt32(role), active, fallback
            )
            for lane in range(Int(width)):
                var expected = fallback.get(lane)
                if active[lane] and ids[lane] < view.count:
                    expected = view.sample(
                        ids[lane], wavelengths.get(lane), UInt32(role)
                    )
                for channel in range(4):
                    assert_almost_equal(
                        actual[channel][lane],
                        expected[channel],
                        atol=2e-6,
                        rtol=2e-5,
                    )
        var emission = view.emission_packet(ids, wavelengths, active)
        for lane in range(Int(width)):
            var expected = SampledSpectrum[1]()
            if active[lane] and ids[lane] < view.count:
                expected = view.emission(ids[lane], wavelengths.get(lane))
            for channel in range(4):
                assert_almost_equal(
                    emission[channel][lane],
                    expected[channel],
                    atol=2e-6,
                    rtol=2e-5,
                )


def test_spectral_source_packets_preserve_mixed_products_roles_and_masks() raises:
    var rgb = RgbSpectrumTable.read_default()
    var curves = List[PiecewiseLinearSpectrum]()
    curves.append(
        PiecewiseLinearSpectrum([360, 0.3, 475, 0.8, 650, 0.5, 830, 0.1])
    )
    var table = PiecewiseSpectrumTable(curves)
    var sources: List[SpectrumSource] = [
        SpectrumSource.rgb(Color(0.2, 0.6, 0.8)),
        SpectrumSource.constant(0.7),
        SpectrumSource.curve(0),
        SpectrumSource.blackbody(5500),
        SpectrumSource.product(0, 1),
        SpectrumSource.product(2, 3),
        SpectrumSource.product(4, 5),
        SpectrumSource.constant(0),
    ]
    var packed = PackedSpectrumSources(sources)
    packed.prepare_photometric(table.view())
    var view = packed.view(table.view(), rgb.view())
    comptime for width in [1, 2, 4, 8, 16, 32]:
        _check_source_packets[width](view)
    _ = packed^
    _ = table^
    _ = rgb^


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
