"""Authored spectral expressions, packed into host/device multiplication lists.

RGB leaves are reconstructed separately: reconstructing a product of RGB
values is not equivalent to multiplying their reconstructed spectra.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import exp, pow
from bajo.core import Vec3
from bajo.core.color import Color
from bajo.core.rgb_spectrum import RgbSpectrumView
from bajo.core.sampled_spectrum import (
    SampledSpectrum,
    SampledWavelengths,
    CIE_Y_INTEGRAL,
    cie_xyz,
)
from bajo.core.spectrum_tables import PiecewiseSpectrumView

comptime NO_SPECTRUM = UInt32.MAX


@fieldwise_init
struct SpectrumSource(Copyable, Writable):
    comptime RGB = UInt32(0)
    comptime CONSTANT = UInt32(1)
    comptime CURVE = UInt32(2)
    comptime BLACKBODY = UInt32(3)
    comptime PRODUCT = UInt32(4)

    var kind: UInt32
    var value: Color
    var first: UInt32
    var second: UInt32

    @staticmethod
    def rgb(value: Color) -> Self:
        return Self(Self.RGB, value, 0, 0)

    @staticmethod
    def constant(value: Float32) -> Self:
        return Self(Self.CONSTANT, Color(value), 0, 0)

    @staticmethod
    def curve(index: UInt32) -> Self:
        return Self(Self.CURVE, Color(0), index, 0)

    @staticmethod
    def blackbody(temperature: Float32) -> Self:
        return Self(Self.BLACKBODY, Color(temperature), 0, 0)

    @staticmethod
    def product(first: UInt32, second: UInt32) -> Self:
        return Self(Self.PRODUCT, Color(0), first, second)


@always_inline
def blackbody_spectrum[
    width: SIMDLength
](
    temperature: SIMD[.float32, width],
    wavelengths: SampledWavelengths[width],
) -> SampledSpectrum[width]:
    """Planck spectrum normalized at Wien's peak, as in PBRT."""
    var result = SampledSpectrum[width]()
    var valid = temperature.gt(0)
    var t = valid.select(temperature, 1)
    var peak = Float32(2.8977721e-3) / t
    comptime for channel in range(4):
        var wavelength = wavelengths.lambda_nm[channel] * Float32(1e-9)
        var exponent = Float32(1.438776877e-2) / (wavelength * t)
        var ratio = peak / wavelength
        result[channel] = (valid & exponent.lt(80)).select(
            pow(ratio, Float32(5))
            * Float32(142.324921)
            / (exp(min(exponent, 80)) - 1),
            0,
        )
    return result


@fieldwise_init
struct SpectrumSourcesView(DevicePassable, TrivialRegisterPassable):
    var sources: ImmPointer[SpectrumSource, ImmUntrackedOrigin]
    var leaves: ImmPointer[UInt32, ImmUntrackedOrigin]
    var ranges: ImmPointer[UInt32, ImmUntrackedOrigin]
    var count: UInt32
    var curves: PiecewiseSpectrumView
    var rgb: RgbSpectrumView
    var photometric: ImmPointer[Float32, ImmUntrackedOrigin]

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "SpectrumSourcesView"

    @always_inline
    def sample[
        width: SIMDLength
    ](
        self,
        index: UInt32,
        wavelengths: SampledWavelengths[width],
        role: UInt32,
    ) -> SampledSpectrum[width]:
        debug_assert(index < self.count, "invalid spectrum source")
        var begin = Int(self.ranges[unsafe_offset=2 * Int(index)])
        var end = Int(self.ranges[unsafe_offset=2 * Int(index) + 1])
        var result = SampledSpectrum[width](1)
        for offset in range(begin, end):
            var leaf = self.sources[
                unsafe_offset=Int(self.leaves[unsafe_offset=offset])
            ].copy()
            var value: SampledSpectrum[width]
            if leaf.kind == SpectrumSource.RGB:
                var rgb = leaf.value.broadcast[width]()
                if role == RgbSpectrumView.ALBEDO:
                    rgb = rgb.clamp(0, 1)
                value = self.rgb.sample(rgb, wavelengths, role)
            elif leaf.kind == SpectrumSource.CONSTANT:
                value = SampledSpectrum[width](leaf.value.x)
            elif leaf.kind == SpectrumSource.CURVE:
                value = self.curves.sample(
                    SIMD[.uint32, width](leaf.first), wavelengths
                )
            else:
                value = blackbody_spectrum(
                    SIMD[.float32, width](leaf.value.x), wavelengths
                )
            result = result * value
        return result

    @always_inline
    def emission[
        width: SIMDLength
    ](
        self, index: UInt32, wavelengths: SampledWavelengths[width]
    ) -> SampledSpectrum[width]:
        var source = self.sources[unsafe_offset=Int(index)].copy()
        # PBRT source roots keep the authored light scale outside normalization.
        var normalized = (
            source.first if source.kind == SpectrumSource.PRODUCT else index
        )
        return (
            self.sample(index, wavelengths, RgbSpectrumView.ILLUMINANT)
            * self.photometric[unsafe_offset=Int(normalized)]
        )


struct PackedSpectrumSources:
    var sources: List[SpectrumSource]
    var leaves: List[UInt32]
    # Half-open leaf spans: runtime evaluation needs no recursion or stack.
    var ranges: List[UInt32]
    var count: UInt32
    var photometric: List[Float32]

    def __init__(out self, sources: ImmSpan[SpectrumSource, _]) raises:
        self.sources = List[SpectrumSource](sources)
        self.leaves = List[UInt32]()
        self.ranges = List[UInt32]()
        for source in sources:
            if source.kind > SpectrumSource.PRODUCT:
                raise Error("unsupported spectrum source kind")
            if (
                not source.value.is_finite()[0]
                or source.value.x < 0
                or source.value.y < 0
                or source.value.z < 0
            ):
                raise Error(
                    "spectrum source values must be finite and non-negative"
                )
            if source.kind == SpectrumSource.BLACKBODY and source.value.x <= 0:
                raise Error("blackbody temperature must be positive")
        self.count = UInt32(len(sources))
        self.photometric = List[Float32](
            length=max(1, len(sources)), fill=1 / CIE_Y_INTEGRAL
        )
        for index in range(len(sources)):
            var begin = UInt32(len(self.leaves))
            var pending = List[UInt32]([UInt32(index)])
            while pending:
                var current = pending.pop()
                var node = sources[Int(current)].copy()
                if node.kind == SpectrumSource.PRODUCT:
                    if node.first >= current or node.second >= current:
                        raise Error(
                            "spectrum products must reference earlier sources"
                        )
                    pending.append(node.second)
                    pending.append(node.first)
                else:
                    self.leaves.append(current)
                if len(self.leaves) - Int(begin) + len(pending) > 4096:
                    raise Error("spectrum expression exceeds 4096 factors")
            self.ranges.extend([begin, UInt32(len(self.leaves))])
        if not self.sources:
            self.sources.append(SpectrumSource.constant(0))
            self.leaves.append(0)
            self.ranges.extend([0, 0])

    def prepare_photometric(mut self, curves: PiecewiseSpectrumView) raises:
        for index in range(Int(self.count)):
            ref source = self.sources[index]
            if (
                source.kind == SpectrumSource.CURVE
                and source.first >= curves.count
            ):
                raise Error("spectrum curve binding is out of range")
            if (
                source.kind == SpectrumSource.RGB
                or source.kind == SpectrumSource.PRODUCT
            ):
                continue
            var integral = Float32(0)
            if source.kind == SpectrumSource.CONSTANT:
                integral = source.value.x * CIE_Y_INTEGRAL
            else:
                for nm in range(360, 831):
                    var wavelengths = SampledWavelengths[1](
                        SampledSpectrum[1](Float32(nm)), SampledSpectrum[1](1)
                    )
                    var value = blackbody_spectrum(
                        SIMD[.float32, 1](source.value.x), wavelengths
                    )
                    if source.kind == SpectrumSource.CURVE:
                        value = curves.sample(
                            SIMD[.uint32, 1](source.first), wavelengths
                        )
                    integral += value.s0 * cie_xyz(Float32(nm)).y
            self.photometric[index] = 1 / integral if integral > 0 else 0

    def view(
        self, curves: PiecewiseSpectrumView, rgb: RgbSpectrumView
    ) -> SpectrumSourcesView:
        return SpectrumSourcesView(
            self.sources.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
            self.leaves.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
            self.ranges.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
            self.count,
            curves,
            rgb,
            self.photometric.unsafe_ptr().unsafe_origin_cast[
                ImmUntrackedOrigin
            ](),
        )
