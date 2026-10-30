"""Optical constants for the RGB conductor model used by PBRT ingestion."""

from bajo.core.color import blackbody_xyz, xyz_to_linear_srgb, Color

from std.utils.numerics import isfinite


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
    var rgb = xyz_to_linear_srgb(
        blackbody_xyz(temperature, Float32(142.324921))
    )
    return Color(
        max(Float32(0.0), rgb.x),
        max(Float32(0.0), rgb.y),
        max(Float32(0.0), rgb.z),
    )
