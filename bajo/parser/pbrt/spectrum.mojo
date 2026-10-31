"""Optical constants for the RGB conductor model used by PBRT ingestion."""

from bajo.core.color import blackbody_xyz, xyz_to_linear_srgb, Color

from std.utils.numerics import isfinite
from bajo.parser.pbrt.optical_data import named_optical_spectrum


def _named_optical_constant(name: ImmStringSpan) raises -> Color:
    # Compatibility projection for the existing RGB renderer. The parser also
    # retains the original table in SurfaceStore.optical_spectra; spectral
    # transport must evaluate that table at each path's sampled wavelengths.
    var spectrum = named_optical_spectrum(name)
    return Color(
        spectrum.evaluate(610.0),
        spectrum.evaluate(550.0),
        spectrum.evaluate(460.0),
    )


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
