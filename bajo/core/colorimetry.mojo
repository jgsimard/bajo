"""CIE 1931 / D65 colorimetry from the complete original wavelength tables."""

from std.builtin.globals import global_constant
from bajo.core.cie_data import CIE_XYZ, D65
from bajo.core.mat import Mat, inverse


def srgb_to_xyz_matrix() raises -> Mat[.float64, 3, 3, .WORLD]:
    """Derive sRGB primary scales from PBRT's D65 spectrum and CIE observer.

    Integrate the original observer at every integer wavelength from 360 to
    830 nm. D65 uses linear interpolation between its original 5 nm samples.
    Normalizing the white to Y=1 cancels its arbitrary photometric scale.
    """
    ref cie = global_constant[CIE_XYZ]()
    ref d65 = global_constant[D65]()
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
    var primaries = Mat[.float64, 3, 3, .WORLD](
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
    var inv = inverse(primaries)
    var result = primaries.copy()
    for column in range(3):
        var scale = Float64(0.0)
        for row in range(3):
            scale += inv[column][row] * white[row]
        for row in range(3):
            result[row][column] *= scale
    return result^
