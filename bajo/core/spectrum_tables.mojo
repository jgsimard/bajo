"""Packed original wavelength curves with the same view on host and device."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from bajo.core.spectrum import (
    PiecewiseLinearSpectrum,
    evaluate_piecewise_spectrum,
)
from bajo.core.sampled_spectrum import SampledSpectrum, SampledWavelengths


@fieldwise_init
struct PiecewiseSpectrumView(Copyable, DevicePassable, TrivialRegisterPassable):
    var samples: ImmPointer[Float32, ImmUntrackedOrigin]
    # Each descriptor is an offset followed by an interleaved word count.
    var ranges: ImmPointer[UInt32, ImmUntrackedOrigin]
    var count: UInt32

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "PiecewiseSpectrumView"

    @always_inline
    def sample[
        width: SIMDLength
    ](
        self,
        indices: SIMD[.uint32, width],
        wavelengths: SampledWavelengths[width],
        fallback: SIMD[.float32, width] = 0,
    ) -> SampledSpectrum[width]:
        """Missing spectrum IDs use a wavelength-independent caller fallback."""
        var result = SampledSpectrum[width](fallback)
        for lane in range(Int(width)):
            var index = indices[lane]
            if index >= self.count:
                continue
            var offset = Int(self.ranges[unsafe_offset=2 * Int(index)])
            var size = Int(self.ranges[unsafe_offset=2 * Int(index) + 1])
            var curve = ImmSpan(
                unsafe_ptr=self.samples.unsafe_offset(offset), length=size
            )
            comptime for i in range(4):
                result[i][lane] = evaluate_piecewise_spectrum(
                    curve, SIMD[.float32, 1](wavelengths.lambda_nm[i][lane])
                )
        return result


struct PiecewiseSpectrumTable:
    var samples: List[Float32]
    var ranges: List[UInt32]
    var count: UInt32

    def __init__(out self, spectra: ImmSpan[PiecewiseLinearSpectrum, _]) raises:
        self.samples = List[Float32]()
        self.ranges = List[UInt32]()
        if len(spectra) > 0xFFFFFFFF:
            raise Error("too many optical spectra")
        self.count = UInt32(len(spectra))
        for curve in spectra:
            if len(self.samples) + len(curve.samples) > 0xFFFFFFFF:
                raise Error("optical spectrum storage exceeds UInt32 offsets")
            self.ranges.append(UInt32(len(self.samples)))
            self.ranges.append(UInt32(len(curve.samples)))
            self.samples.extend(Span(curve.samples))
        # Device APIs and nonnullable views need allocated storage even when
        # the logical table is empty; count still prevents every dereference.
        if not self.samples:
            self.samples.append(0)
            self.ranges.extend([0, 0])

    def view(self) -> PiecewiseSpectrumView:
        """Borrow the packed curves; keep this owner alive through all sampling.
        """
        return PiecewiseSpectrumView(
            self.samples.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
            self.ranges.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
            self.count,
        )
