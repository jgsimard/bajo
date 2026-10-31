"""Wavelength-domain spectra, independent of RGB color representations."""

from std.utils.numerics import isfinite


struct PiecewiseLinearSpectrum(Copyable):
    """Owned wavelength/value pairs, with wavelengths measured in nanometers.

    Evaluation outside the tabulated domain is zero. No color-space transform,
    normalization, or reduction to representative RGB wavelengths is applied.
    """

    var samples: List[Float32]

    def __init__(out self, var samples: List[Float32]) raises:
        if len(samples) < 2 or len(samples) % 2:
            raise Error("spectra require wavelength/value pairs")
        for index in range(0, len(samples), 2):
            if not isfinite(samples[index]) or samples[index] <= 0.0:
                raise Error("spectrum wavelengths must be finite and positive")
            if not isfinite(samples[index + 1]) or samples[index + 1] < 0.0:
                raise Error("spectrum values must be finite and non-negative")
            if index and samples[index] <= samples[index - 2]:
                raise Error("spectrum wavelengths must strictly increase")
        self.samples = samples^

    @always_inline
    def evaluate[
        length: SIMDLength = 1
    ](self, wavelength: SIMD[.float32, length]) -> SIMD[.float32, length]:
        return evaluate_piecewise_spectrum(self.samples, wavelength)


@always_inline
def evaluate_piecewise_spectrum[
    length: SIMDLength = 1
](samples: ImmSpan[Float32, _], wavelength: SIMD[.float32, length]) -> SIMD[
    .float32, length
]:
    """Host/device evaluation of validated, interleaved spectral samples."""
    var result = SIMD[.float32, length](0.0)
    var count = len(samples) // 2
    for lane in range(Int(length)):
        var nm = wavelength[lane]
        if not (nm >= samples[0] and nm <= samples[len(samples) - 2]):
            continue
        if count == 1 or nm == samples[len(samples) - 2]:
            result[lane] = samples[len(samples) - 1]
            continue
        var lo = 0
        var hi = count - 1
        while lo + 1 < hi:
            var mid = (lo + hi) // 2
            if samples[2 * mid] <= nm:
                lo = mid
            else:
                hi = mid
        var t = (nm - samples[2 * lo]) / (samples[2 * lo + 2] - samples[2 * lo])
        result[lane] = samples[2 * lo + 1] * (1 - t) + samples[2 * lo + 3] * t
    return result
