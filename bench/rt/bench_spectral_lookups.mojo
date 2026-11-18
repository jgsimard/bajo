"""Spectral lookup throughput at scalar and packet widths.

Build with `mojo build -O3 -I . bench/rt/bench_spectral_lookups.mojo -o /tmp/spectra`
and run `/tmp/spectra`. Timings exclude input preparation and table loading.
"""

from std.benchmark import Unit, keep, run
from bajo.core import Vec3
from bajo.core.rgb_spectrum import RgbSpectrumTable, RgbSpectrumView
from bajo.core.sampled_spectrum import (
    SampledSpectrum,
    SampledWavelengths,
    cie_xyz,
)
from bajo.core.spectrum import PiecewiseLinearSpectrum
from bajo.core.spectrum_tables import (
    PiecewiseSpectrumView,
    PiecewiseSpectrumTable,
)
from bajo.rt.sensor import canon_sensor_sample


def benchmark[
    width: SIMDLength
](rgb: RgbSpectrumView[_], curves: PiecewiseSpectrumView[_]) raises:
    comptime count = 4096
    var colors = List[Vec3[.float32, .WORLD, width]]()
    var wavelengths = List[SampledWavelengths[width]]()
    var ids = List[SIMD[.uint32, width]]()
    for first in range(0, count, Int(width)):
        var color = Vec3[.float32, .WORLD, width](0)
        var nm = SIMD[.float32, width](0)
        var indices = SIMD[.uint32, width](0)
        for lane in range(Int(width)):
            var index = first + lane
            color.x[lane] = Float32((index * 13) % 257) / 256
            color.y[lane] = Float32((index * 37) % 257) / 256
            color.z[lane] = Float32((index * 71) % 257) / 256
            nm[lane] = 350 + Float32((index * 43) % 491)
            indices[lane] = UInt32(index % 5)
        colors.append(color)
        wavelengths.append(
            SampledWavelengths[width](
                SampledSpectrum[width](nm, nm + 0.3, nm + 0.6, nm + 0.9),
                SampledSpectrum[width](1),
            )
        )
        ids.append(indices)

    def coefficients() {imm}:
        for color in colors:
            var result = rgb.coefficients(color)
            keep(result.a)
            keep(result.b)
            keep(result.c)

    def piecewise() {imm}:
        for i in range(len(wavelengths)):
            var result = curves.sample(ids[i], wavelengths[i], 1)
            keep(result.s0)
            keep(result.s1)
            keep(result.s2)
            keep(result.s3)

    def illuminant() {imm}:
        for sample in wavelengths:
            keep(rgb.illuminant(sample.lambda_nm.s0))

    def observer() {imm}:
        for sample in wavelengths:
            var result = cie_xyz(sample.lambda_nm.s0)
            keep(result.x)
            keep(result.y)
            keep(result.z)

    def sensor() {imm}:
        for sample in wavelengths:
            var result = canon_sensor_sample(SampledSpectrum[width](1), sample)
            keep(result.x)
            keep(result.y)
            keep(result.z)

    print("width", Int(width), "ns/path (mean, 4096 paths per iteration)")
    print(
        "  coefficients", run(coefficients, max_iters=20).mean(Unit.ns) / count
    )
    print("  piecewise", run(piecewise, max_iters=20).mean(Unit.ns) / count)
    print("  illuminant", run(illuminant, max_iters=20).mean(Unit.ns) / count)
    print("  cie", run(observer, max_iters=20).mean(Unit.ns) / count)
    print("  canon", run(sensor, max_iters=20).mean(Unit.ns) / count)


def main() raises:
    var rgb = RgbSpectrumTable.read_default()
    var spectra = List[PiecewiseLinearSpectrum]()
    for count in [1, 2, 5, 17, 65]:
        var pairs = List[Float32]()
        for i in range(count):
            pairs.append(360 + Float32(i) * 470 / Float32(max(count - 1, 1)))
            pairs.append(Float32((i * 7) % 11) / 10)
        spectra.append(PiecewiseLinearSpectrum(pairs^))
    var curves = PiecewiseSpectrumTable(spectra)
    benchmark[1](rgb.view(), curves.view())
    benchmark[4](rgb.view(), curves.view())
    benchmark[16](rgb.view(), curves.view())
    _ = curves^
    _ = rgb^
