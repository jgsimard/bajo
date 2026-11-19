"""CIE 1931 / D65 colorimetry from the complete original wavelength tables."""

from std.os import abort
from bajo.core.cie_data import CIE_XYZ, D65
from bajo.core.mat import Mat, inverse


comptime _ColorMatrix = Mat[.float64, 3, 3, .WORLD]


def _derive_srgb_transforms() -> Tuple[_ColorMatrix, _ColorMatrix]:
    """Derive sRGB primary scales from PBRT's D65 spectrum and CIE observer.

    Integrate the original observer at every integer wavelength from 360 to
    830 nm. D65 uses linear interpolation between its original 5 nm samples.
    Normalizing the white to Y=1 cancels its arbitrary photometric scale.
    """
    var cie = materialize[CIE_XYZ]()
    var d65 = materialize[D65]()
    var white = Array[Float64, 3](fill=0.0)
    for wavelength in range(360, 831):
        var i = 2 * ((wavelength - 300) // 5)
        var t = Float64(wavelength % 5) / 5.0
        var illuminant = Float64(d65[i + 1])
        if t > 0:
            illuminant = (
                Float64(d65[i + 1]) * (1.0 - t) + Float64(d65[i + 3]) * t
            )
        for channel in range(3):
            white[channel] += illuminant * Float64(
                cie[wavelength - 360][channel]
            )
    var y = white[1]
    for channel in range(3):
        white[channel] /= y
    var primaries = _ColorMatrix(
        0.64 / 0.33,
        0.30 / 0.60,
        0.15 / 0.06,
        1.0,
        1.0,
        1.0,
        (1.0 - 0.64 - 0.33) / 0.33,
        (1.0 - 0.30 - 0.60) / 0.60,
        (1.0 - 0.15 - 0.06) / 0.06,
    )
    try:
        var inv = inverse(primaries)
        var result = primaries.copy()
        for column in range(3):
            var scale = Float64(0.0)
            for row in range(3):
                scale += inv[column][row] * white[row]
            for row in range(3):
                result[row][column] *= scale
        var backward = inverse(result)
        return (result^, backward^)
    except:
        abort("Fixed sRGB colorimetry matrices must be invertible")


comptime _SRGB_TRANSFORMS = _derive_srgb_transforms()
comptime SRGB_TO_XYZ_MATRIX = _SRGB_TRANSFORMS[0]
comptime XYZ_TO_SRGB_MATRIX = _SRGB_TRANSFORMS[1]
