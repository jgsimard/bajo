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

    @inline(.always)
    def evaluate[
        length: SIMDLength = 1
    ](self, wavelength: SIMD[.float32, length]) -> SIMD[.float32, length]:
        return evaluate_piecewise_spectrum(self.samples, wavelength)


@inline(.always)
def evaluate_piecewise_spectrum[
    length: SIMDLength = 1
](samples: ImmSpan[Float32, _], wavelength: SIMD[.float32, length]) -> SIMD[
    .float32, length
]:
    """Host/device evaluation of validated, interleaved spectral samples."""
    return _evaluate_piecewise_spectrum(
        samples.unsafe_ptr(),
        SIMD[.int64, length](0),
        SIMD[.int64, length](len(samples) // 2),
        wavelength,
        SIMD[.bool, length](fill=True),
    )


@inline(.always)
def _evaluate_piecewise_spectrum[
    length: SIMDLength
](
    samples: ImmPointer[Float32, _],
    offsets: SIMD[.int64, length],
    counts: SIMD[.int64, length],
    wavelength: SIMD[.float32, length],
    active: SIMD[.bool, length],
) -> SIMD[.float32, length]:
    """Evaluate independently sized packed curves with one masked search.

    Active lanes reference validated, nonempty curves. Inactive lanes load
    only word zero, which packed tables allocate even when logically empty.
    """
    var first = active.select(offsets, 0)
    var count = active.select(counts, 1)
    var last = first + 2 * (count - 1)
    var first_nm = samples.unsafe_gather(first)
    var last_nm = samples.unsafe_gather(last)
    var valid = active & wavelength.ge(first_nm) & wavelength.le(last_nm)
    if not valid.reduce_or():
        return SIMD[.float32, length](0)
    first = valid.select(first, 0)
    count = valid.select(count, 1)
    var lo = SIMD[.int64, length](0)
    var remaining = count - 1
    # Both halves retain ceil(remaining / 2) intervals. This overlapping
    # search avoids per-lane upper bounds and convergence masks.
    while remaining.gt(1).reduce_or():
        var half = remaining // 2
        var mid = lo + half
        var right = samples.unsafe_gather(first + 2 * mid).le(wavelength)
        lo += right.select(half, 0)
        remaining -= half
    var lower = first + 2 * lo
    var upper = first + 2 * min(lo + 1, count - 1)
    var lower_nm = samples.unsafe_gather(lower)
    var upper_nm = samples.unsafe_gather(upper)
    var low_value = samples.unsafe_gather(lower + valid.cast[.int64]())
    var high_value = samples.unsafe_gather(upper + valid.cast[.int64]())
    var t = (valid.select(wavelength, lower_nm) - lower_nm) / count.gt(
        1
    ).select(upper_nm - lower_nm, 1)
    return valid.select(low_value * (1 - t) + high_value * t, 0)
