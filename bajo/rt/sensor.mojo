"""Measured sensor responses and PBRT-compatible host calibration."""

from std.builtin.globals import global_constant
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import floor
from bajo.core import Vec3
from bajo.core.color import Color
from bajo.core.cie_data import CIE_XYZ, D65
from bajo.core.sensor_data import (
    CANON_RESPONSE,
    DAYLIGHT_BASIS,
    SENSOR_SWATCHES,
)
from bajo.core.sampled_spectrum import SampledSpectrum, SampledWavelengths
from bajo.core.mat import Mat, inverse
from bajo.core.colorimetry import srgb_to_xyz_matrix
from bajo.rt.spectrum_sources import blackbody_spectrum


__enum FilmSensorKind(DevicePassable, Equatable, ImplicitlyCopyable, Writable):
    case CIE1931
    case CANON_EOS_5D_MKIV

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "FilmSensorKind"


@always_inline
def canon_sensor_sample[
    width: SIMDLength
](
    radiance: SampledSpectrum[width],
    wavelengths: SampledWavelengths[width],
) -> Vec3[.float32, .WORLD, width]:
    ref responses = global_constant[CANON_RESPONSE]()
    var result = Vec3[.float32, .WORLD, width](0)
    var weighted = radiance.safe_divide(wavelengths.pdf)
    for lane in range(Int(width)):
        comptime for channel in range(4):
            var nm = wavelengths.lambda_nm[channel][lane]
            if nm >= 360 and nm <= 830:
                var response = responses[Int(floor(nm + 0.5)) - 360]
                var value = weighted[channel][lane] * Float32(0.25)
                result.x[lane] += response.x * value
                result.y[lane] += response.y * value
                result.z[lane] += response.z * value
    return result


def daylight(temperature: Float32, nm: Int) -> Color:
    """Illuminant plus original observer packed separately by the caller."""
    var cct = temperature * Float32(1.4388 / 1.4380)
    if cct < 4000:
        var wavelengths = SampledWavelengths[1](
            SampledSpectrum[1](Float32(nm)), SampledSpectrum[1](1)
        )
        return Color(blackbody_spectrum(SIMD[.float32, 1](cct), wavelengths).s0)
    var x: Float32
    if cct <= 7000:
        x = (
            -4.607e9 / (cct * cct * cct)
            + 2.9678e6 / (cct * cct)
            + 99.11 / cct
            + 0.244063
        )
    else:
        x = (
            -2.0064e9 / (cct * cct * cct)
            + 1.9018e6 / (cct * cct)
            + 247.48 / cct
            + 0.23704
        )
    var y = -3 * x * x + 2.870 * x - 0.275
    var m = 0.0241 + 0.2562 * x - 0.7341 * y
    var m1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / m
    var m2 = (0.0300 - 31.4424 * x + 30.0717 * y) / m
    ref basis = global_constant[DAYLIGHT_BASIS]()
    var index = (nm - 300) // 5
    var t = Float32(nm % 5) / 5
    var b = basis[index]
    if t != 0:
        b = b * (1 - t) + basis[index + 1] * t
    return Color((b.x + b.y * m1 + b.z * m2) * 0.01)


def sensor_to_xyz(
    kind: FilmSensorKind, temperature: Float32
) raises -> Mat[.float64, 3, 3, .WORLD]:
    comptime Matrix = Mat[.float64, 3, 3, .WORLD]
    var identity = Matrix.identity()
    if kind == .CIE1931 and temperature == 0:
        return identity^
    var t = temperature if temperature != 0 else Float32(6500)
    ref observer = global_constant[CIE_XYZ]()
    ref response = global_constant[CANON_RESPONSE]()
    ref chart = global_constant[SENSOR_SWATCHES]()
    ref d65 = global_constant[D65]()
    var illumination = List[Float64](length=471, fill=0)
    var reference = List[Float64](length=471, fill=0)
    var sensor_white_g = Float64(0)
    var sensor_white_y = Float64(0)
    var output_white_y = Float64(0)
    var source_white = Array[Float64, 3](fill=0)
    for nm in range(360, 831):
        var index = nm - 360
        illumination[index] = Float64(daylight(t, nm).x)
        var offset = 2 * ((nm - 300) // 5)
        var fraction = Float64(nm % 5) / 5
        reference[index] = (
            Float64(d65[offset + 1]) if fraction
            == 0 else Float64(d65[offset + 1]) * (1 - fraction)
            + Float64(d65[offset + 3]) * fraction
        )
        sensor_white_g += illumination[index] * Float64(response[index].y)
        sensor_white_y += illumination[index] * Float64(observer[index].y)
        output_white_y += reference[index] * Float64(observer[index].y)
        for channel in range(3):
            source_white[channel] += illumination[index] * Float64(
                observer[index][channel]
            )
    if kind == .CIE1931:
        # Bradford adaptation, matching PBRT's WhiteBalance transform.
        var bradford = Matrix(
            0.8951,
            0.2664,
            -0.1614,
            -0.7502,
            1.7135,
            0.0367,
            0.0389,
            -0.0685,
            1.0296,
        )
        var target = srgb_to_xyz_matrix()
        var scales = Array[Float64, 3](fill=0)
        for row in range(3):
            var source_lms = Float64(0)
            var target_lms = Float64(0)
            for column in range(3):
                source_lms += (
                    bradford[row][column]
                    * source_white[column]
                    / source_white[1]
                )
                target_lms += bradford[row][column] * (
                    target[column][0] + target[column][1] + target[column][2]
                )
            scales[row] = target_lms / source_lms
        var inv_bradford = inverse(bradford)
        var result = Matrix(0)
        for row in range(3):
            for column in range(3):
                for k in range(3):
                    result[row][column] += (
                        inv_bradford[row][k] * scales[k] * bradford[k][column]
                    )
        return result^
    var ata = Matrix(0)
    var atb = Matrix(0)
    for swatch in range(24):
        var a = Array[Float64, 3](fill=0)
        var b = Array[Float64, 3](fill=0)
        for nm in range(360, 831):
            var index = nm - 360
            var offset = max(0, min(35, (nm - 380) // 10))
            var reflectance = Float64(chart[swatch][offset])
            if nm >= 380 and nm < 730:
                var f = Float64((nm - 380) % 10) / 10
                reflectance = (
                    reflectance * (1 - f)
                    + Float64(chart[swatch][offset + 1]) * f
                )
            for channel in range(3):
                a[channel] += (
                    reflectance
                    * illumination[index]
                    * Float64(response[index][channel])
                    / sensor_white_g
                )
                b[channel] += (
                    reflectance
                    * reference[index]
                    * Float64(observer[index][channel])
                    / output_white_y
                    * (sensor_white_y / sensor_white_g)
                )
        for row in range(3):
            for column in range(3):
                ata[row][column] += a[row] * a[column]
                atb[row][column] += a[row] * b[column]
    var inv_ata = inverse(ata)
    var result = Matrix(0)
    for row in range(3):
        for column in range(3):
            for k in range(3):
                result[row][column] += inv_ata[column][k] * atb[k][row]
    return result^
