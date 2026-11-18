"""Authored spectral expressions, packed into host/device multiplication lists.

RGB leaves are reconstructed separately: reconstructing a product of RGB
values is not equivalent to multiplying their reconstructed spectra.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import exp, pow
from std.bit import count_trailing_zeros
from std.memory import pack_bits
from std.sys.info import size_of
from bajo.core.color import Color
from bajo.core.rgb_spectrum import RgbSpectrumView
from bajo.core.sampled_spectrum import (
    SampledSpectrum,
    SampledWavelengths,
    CIE_Y_INTEGRAL,
    cie_xyz,
)
from bajo.core.spectrum_tables import PiecewiseSpectrumView
from bajo.core.vec import Vec3

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
struct SpectrumSourcesView[origin: ImmOrigin = ImmUntrackedOrigin](
    DevicePassable, TrivialRegisterPassable
):
    var sources: ImmPointer[SpectrumSource, Self.origin]
    var leaves: ImmPointer[UInt32, Self.origin]
    var ranges: ImmPointer[UInt32, Self.origin]
    var count: UInt32
    var curves: PiecewiseSpectrumView[Self.origin]
    var rgb: RgbSpectrumView[Self.origin]
    var photometric: ImmPointer[Float32, Self.origin]

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "SpectrumSourcesView"

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

    def sample_packet[
        width: SIMDLength
    ](
        self,
        indices: SIMD[.uint32, width],
        wavelengths: SampledWavelengths[width],
        role: UInt32,
        active: SIMD[.bool, width] = SIMD[.bool, width](fill=True),
        fallback: SampledSpectrum[width] = {},
    ) -> SampledSpectrum[width]:
        """Evaluate shared and paired source expressions across wavelengths.

        Products retain their original leaf order. Divergent roots use direct
        scalar leaves; invalid or inactive IDs keep the supplied fallback.
        """
        var valid = active & indices.lt(self.count)
        if not valid.reduce_or():
            return fallback
        comptime if width == 1:
            return self.sample(indices[0], wavelengths, role)
        var first = Int(count_trailing_zeros(pack_bits(valid)))
        var first_group = valid & indices.eq(indices[first])
        var remaining = valid & ~first_group
        if not remaining.reduce_or():
            return SampledSpectrum[width].select(
                valid, self.sample(indices[first], wavelengths, role), fallback
            )
        var second = Int(count_trailing_zeros(pack_bits(remaining)))
        if (~remaining | indices.eq(indices[second])).reduce_and():
            var value = SampledSpectrum[width].select(
                first_group,
                self.sample(indices[first], wavelengths, role),
                self.sample(indices[second], wavelengths, role),
            )
            return SampledSpectrum[width].select(valid, value, fallback)
        # Measurements favor direct leaves for highly divergent expressions:
        # otherwise every leaf kind runs across mostly unrelated wavelengths.
        var result = fallback
        comptime for lane in range(Int(width)):
            if valid[lane]:
                result.set(
                    lane,
                    self.sample(indices[lane], wavelengths.get(lane), role),
                )
        return result

    def emission_packet[
        width: SIMDLength
    ](
        self,
        indices: SIMD[.uint32, width],
        wavelengths: SampledWavelengths[width],
        active: SIMD[.bool, width] = SIMD[.bool, width](fill=True),
    ) -> SampledSpectrum[width]:
        var valid = active & indices.lt(self.count)
        if not valid.reduce_or():
            return SampledSpectrum[width]()
        comptime if width == 1:
            return self.emission(indices[0], wavelengths)
        comptime assert size_of[SpectrumSource]() == 6 * size_of[UInt32]()
        var roots = valid.select(indices, 0).cast[.int64]() * 6
        var words = self.sources.unsafe_bitcast[UInt32]()
        var normalized = (
            words.unsafe_gather(roots)
            .eq(SpectrumSource.PRODUCT)
            .select(words.unsafe_gather(roots + 4), indices)
        )
        var factor = self.photometric.unsafe_gather(
            valid.select(normalized, 0).cast[.int64]()
        )
        return self.sample_packet(
            indices, wavelengths, RgbSpectrumView.ILLUMINANT, valid
        ) * valid.select(factor, 0)


struct PackedSpectrumSources:
    var sources: List[SpectrumSource]
    var leaves: List[UInt32]
    # Half-open leaf spans: runtime evaluation needs no recursion or stack.
    var ranges: List[UInt32]
    var count: UInt32
    var photometric: List[Float32]

    def __init__(out self, sources: ImmSpan[SpectrumSource, _]) raises:
        self.sources = List[SpectrumSource](sources)
        self.leaves = {}
        self.ranges = {}
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

    def prepare_photometric(mut self, curves: PiecewiseSpectrumView[_]) raises:
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

    def view[
        curves_origin: ImmOrigin, rgb_origin: ImmOrigin
    ](
        self,
        curves: PiecewiseSpectrumView[curves_origin],
        rgb: RgbSpectrumView[rgb_origin],
    ) -> SpectrumSourcesView[
        origin_of(
            type_of(self.sources.unsafe_ptr()).origin,
            type_of(self.leaves.unsafe_ptr()).origin,
            type_of(self.ranges.unsafe_ptr()).origin,
            type_of(self.photometric.unsafe_ptr()).origin,
            curves_origin,
            rgb_origin,
        )
    ]:
        # Bind the raw buffers to their storage, including the borrowed
        # curve/RGB tables. These casts join borrows rather than erase their lifetimes.
        var sources = self.sources.unsafe_ptr()
        var leaves = self.leaves.unsafe_ptr()
        var ranges = self.ranges.unsafe_ptr()
        var photometric = self.photometric.unsafe_ptr()
        comptime origin = origin_of(
            type_of(sources).origin,
            type_of(leaves).origin,
            type_of(ranges).origin,
            type_of(photometric).origin,
            curves_origin,
            rgb_origin,
        )
        return SpectrumSourcesView[origin](
            sources.unsafe_origin_cast[origin](),
            leaves.unsafe_origin_cast[origin](),
            ranges.unsafe_origin_cast[origin](),
            self.count,
            PiecewiseSpectrumView[origin](
                curves.samples.unsafe_origin_cast[origin](),
                curves.ranges.unsafe_origin_cast[origin](),
                curves.count,
            ),
            RgbSpectrumView[origin](rgb.data.unsafe_origin_cast[origin]()),
            photometric.unsafe_origin_cast[origin](),
        )
