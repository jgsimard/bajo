"""Authored source expressions: packet evaluation versus direct scalar leaves.

Both paths use the same tables and original source order. Setup is excluded.
"""
from std.benchmark import run, keep, Unit
from bajo.core import Vec3W
from bajo.core.rgb_spectrum import RgbSpectrumTable, RgbSpectrumView
from bajo.core.spectrum import PiecewiseLinearSpectrum
from bajo.core.spectrum_tables import PiecewiseSpectrumTable
from bajo.core.sampled_spectrum import SampledSpectrum, SampledWavelengths
from bajo.rt.spectrum_sources import (
    SpectrumSource,
    PackedSpectrumSources,
    SpectrumSourcesView,
)


def benchmark[packet: Bool](sources: SpectrumSourcesView, pattern: Int) raises:
    var wavelengths = List[SampledWavelengths[16]]()
    var indices = List[SIMD[.uint32, 16]]()
    for block in range(256):
        var u = SIMD[.float32, 16](0)
        var ids = SIMD[.uint32, 16](6)
        for lane in range(16):
            u[lane] = Float32((block * 16 + lane) % 257) / 257
            if pattern == 1:
                ids[lane] = UInt32(5 + lane % 2)
            elif pattern == 2:
                ids[lane] = UInt32((block * 16 + lane) % 7)
        wavelengths.append(SampledWavelengths[16].visible(u))
        indices.append(ids)

    def sample() {imm}:
        for block in range(256):
            comptime if packet:
                var value = sources.sample_packet(
                    indices[block],
                    wavelengths[block],
                    RgbSpectrumView.UNBOUNDED,
                )
                keep(value.s0)
                keep(value.s1)
                keep(value.s2)
                keep(value.s3)
            else:
                # Match the original material setup: scalar leaves are stored
                # into a complete packet before the BSDF consumes them.
                var combined = SampledSpectrum[16]()
                for lane in range(16):
                    var value = sources.sample(
                        indices[block][lane],
                        wavelengths[block].get(lane),
                        RgbSpectrumView.UNBOUNDED,
                    )
                    combined.set(lane, value)
                keep(combined.s0)
                keep(combined.s1)
                keep(combined.s2)
                keep(combined.s3)

    print(
        "packet",
        packet,
        "pattern",
        pattern,
        "ns/path",
        run(sample, max_iters=30).mean(Unit.ns) / 4096,
    )


def main() raises:
    var rgb = RgbSpectrumTable.read_default()
    var curves = List[PiecewiseLinearSpectrum]()
    curves.append(
        PiecewiseLinearSpectrum(
            [360, 0.2, 450, 0.7, 550, 0.3, 650, 0.9, 830, 0.1]
        )
    )
    var table = PiecewiseSpectrumTable(curves)
    var roots = List[SpectrumSource]()
    roots.append(SpectrumSource.rgb(Vec3W(0.3, 0.7, 0.4)))
    roots.append(SpectrumSource.constant(0.6))
    roots.append(SpectrumSource.curve(0))
    roots.append(SpectrumSource.blackbody(6100))
    roots.append(SpectrumSource.product(0, 1))
    roots.append(SpectrumSource.product(2, 3))
    roots.append(SpectrumSource.product(4, 5))
    var packed = PackedSpectrumSources(roots)
    var view = packed.view(table.view(), rgb.view())
    for pattern in range(3):
        benchmark[False](view, pattern)
        benchmark[True](view, pattern)
    _ = packed^
    _ = table^
    _ = rgb^
