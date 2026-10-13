"""Optical constants for the RGB conductor model used by PBRT ingestion."""

from std.math import exp
from std.utils.numerics import isfinite
from bajo.rt.render_types import Color


def _named_optical_constant(name: ImmStringSpan) raises -> Color:
    # RGB approximation at 610, 550, and 460 nm, linearly interpolated from
    # PBRT v4's Al/Au eta/k wavelength tables (Apache-2.0):
    # https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/spectrum.cpp
    # These are optical constants, not RGB reflectance values.
    match name:
        case "metal-Al-eta":
            return Color(1.24750867, 0.95856902, 0.64627187)
        case "metal-Al-k":
            return Color(7.36845593, 6.68696769, 5.59014846)
        case "metal-Au-eta":
            return Color(0.20605660, 0.35040884, 1.41860777)
        case "metal-Au-k":
            return Color(3.03047363, 2.71406311, 1.84310538)
        case _:
            raise Error("unsupported PBRT optical spectrum: " + String(name))


def _blackbody_rgb(temperature: Float32) raises -> Color:
    """Peak-normalized Planck spectrum integrated to linear sRGB.

    CIE 1931 matching functions sampled at 10 nm from PBRT's tables.
    This preserves PBRT's blackbody intensity normalization in an RGB model.
    """
    if not isfinite(temperature) or temperature <= 0.0:
        raise Error("blackbody temperature must be finite and positive")
    var peak = Float32(2.897771955e6) / temperature
    var xyz = Color(0.0)
    var matching: List[Color] = [
        Color(0.0001299, 3.917e-06, 0.0006061),
        Color(0.0004149, 1.239e-05, 0.001946),
        Color(0.001368, 3.9e-05, 0.006450001),
        Color(0.004243, 0.00012, 0.02005001),
        Color(0.01431, 0.000396, 0.06785001),
        Color(0.04351, 0.00121, 0.2074),
        Color(0.13438, 0.004, 0.6456),
        Color(0.2839, 0.0116, 1.3856),
        Color(0.34828, 0.023, 1.74706),
        Color(0.3362, 0.038, 1.77211),
        Color(0.2908, 0.06, 1.6692),
        Color(0.19536, 0.09098, 1.28764),
        Color(0.09564, 0.13902, 0.8129501),
        Color(0.03201, 0.20802, 0.46518),
        Color(0.0049, 0.323, 0.272),
        Color(0.0093, 0.503, 0.1582),
        Color(0.06327, 0.71, 0.07824999),
        Color(0.1655, 0.862, 0.04216),
        Color(0.2904, 0.954, 0.0203),
        Color(0.4334499, 0.9949501, 0.008749999),
        Color(0.5945, 0.995, 0.0039),
        Color(0.7621, 0.952, 0.0021),
        Color(0.9163, 0.87, 0.001650001),
        Color(1.0263, 0.757, 0.0011),
        Color(1.0622, 0.631, 0.0008),
        Color(1.0026, 0.503, 0.00034),
        Color(0.8544499, 0.381, 0.00019),
        Color(0.6424, 0.265, 4.999999e-05),
        Color(0.4479, 0.175, 2e-05),
        Color(0.2835, 0.107, 0),
        Color(0.1649, 0.061, 0),
        Color(0.0874, 0.032, 0),
        Color(0.04677, 0.017, 0),
        Color(0.0227, 0.00821, 0),
        Color(0.01135916, 0.004102, 0),
        Color(0.005790346, 0.002091, 0),
        Color(0.002899327, 0.001047, 0),
        Color(0.001439971, 0.00052, 0),
        Color(0.0006900786, 0.0002492, 0),
        Color(0.0003323011, 0.00012, 0),
        Color(0.0001661505, 6e-05, 0),
        Color(8.307527e-05, 3e-05, 0),
        Color(4.150994e-05, 1.499e-05, 0),
        Color(2.067383e-05, 7.4657e-06, 0),
        Color(1.025398e-05, 3.7029e-06, 0),
        Color(5.085868e-06, 1.8366e-06, 0),
        Color(2.522525e-06, 9.1093e-07, 0),
        Color(1.251141e-06, 4.5181e-07, 0),
    ]
    for i in range(len(matching)):
        var wavelength = Float32(360 + 10 * i)
        var exponent = Float32(1.438776877e7) / (wavelength * temperature)
        var ratio = peak / wavelength
        var ratio2 = ratio * ratio
        var intensity = ratio2 * ratio2 * ratio * Float32(142.324921)
        intensity /= exp(min(exponent, Float32(80.0))) - 1.0
        xyz += matching[i] * intensity * Float32(10.0 / 106.856895)
    return Color(
        max(
            Float32(0.0),
            3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z,
        ),
        max(
            Float32(0.0),
            -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z,
        ),
        max(
            Float32(0.0),
            0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z,
        ),
    )
