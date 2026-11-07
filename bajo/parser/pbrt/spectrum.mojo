"""Optical constants for the RGB conductor model used by PBRT ingestion."""

from bajo.core.color import Color, blackbody_rgb as _blackbody_rgb

from bajo.parser.pbrt.optical_data import named_optical_spectrum


def _named_optical_constant(name: StringSpan) raises -> Color:
    # Compatibility projection for the existing RGB renderer. The parser also
    # retains the original table in SurfaceStore.optical_spectra; spectral
    # transport must evaluate that table at each path's sampled wavelengths.
    var spectrum = named_optical_spectrum(name)
    return Color(
        spectrum.evaluate(610.0),
        spectrum.evaluate(550.0),
        spectrum.evaluate(460.0),
    )
