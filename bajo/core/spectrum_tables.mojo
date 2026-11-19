"""Packed original wavelength curves with the same view on host and device."""

from layout import TileTensor, Coord, Idx, row_major
from bajo.core.tensor_gather import gather_record_field
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from bajo.core.spectrum import (
    PiecewiseLinearSpectrum,
    _evaluate_piecewise_spectrum,
)
from bajo.core.sampled_spectrum import SampledSpectrum, SampledWavelengths


@fieldwise_init
struct PiecewiseSpectrumView[origin: ImmOrigin = ImmUntrackedOrigin](
    Copyable, DevicePassable, TrivialRegisterPassable
):
    var samples: ImmPointer[Float32, Self.origin]
    # Each descriptor is an offset followed by an interleaved word count.
    var ranges: ImmPointer[UInt32, Self.origin]
    var count: UInt32

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "PiecewiseSpectrumView"

    @inline(.always)
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
        var valid = indices.lt(self.count)
        if not valid.reduce_or():
            return result
        var descriptors = TileTensor(
            ptr=self.ranges, layout=row_major(Coord(Int(self.count), Idx[2]))
        )
        var ids = valid.select(indices, 0)
        var offsets = gather_record_field[0](descriptors, ids).cast[.int64]()
        var counts = (
            gather_record_field[1](descriptors, ids).cast[.int64]() // 2
        )
        comptime if width <= 4:
            # Small packets keep independent searches: sparse or short curves
            # do not pay for the longest search in another path's lane.
            comptime for lane in range(Int(width)):
                var value = SampledSpectrum[1](fallback[lane])
                comptime for channel in range(4):
                    value[channel] = valid[lane].select(
                        _evaluate_piecewise_spectrum(
                            self.samples,
                            SIMD[.int64, 1](offsets[lane]),
                            SIMD[.int64, 1](counts[lane]),
                            SIMD[.float32, 1](
                                wavelengths.lambda_nm[channel][lane]
                            ),
                            SIMD[.bool, 1](valid[lane]),
                        ),
                        fallback[lane],
                    )
                result.set(lane, value)
            return result
        # Search all four wavelengths together, sharing loop control across
        # channels as well as paths. Packing is resolved to shuffles at compile time.
        var packed_nm = SIMD[.float32, 4 * width](0)
        var packed_offsets = SIMD[.int64, 4 * width](0)
        var packed_counts = SIMD[.int64, 4 * width](0)
        var packed_valid = SIMD[.bool, 4 * width](fill=False)
        comptime for channel in range(4):
            comptime for lane in range(Int(width)):
                packed_nm[channel * Int(width) + lane] = wavelengths.lambda_nm[
                    channel
                ][lane]
                packed_offsets[channel * Int(width) + lane] = offsets[lane]
                packed_counts[channel * Int(width) + lane] = counts[lane]
                packed_valid[channel * Int(width) + lane] = valid[lane]
        var values = _evaluate_piecewise_spectrum(
            self.samples, packed_offsets, packed_counts, packed_nm, packed_valid
        )
        comptime for channel in range(4):
            result[channel] = valid.select(
                values.slice[width, offset=channel * Int(width)](), fallback
            )
        return result


struct PiecewiseSpectrumTable:
    var samples: List[Float32]
    var ranges: List[UInt32]
    var count: UInt32

    def __init__(out self, spectra: ImmSpan[PiecewiseLinearSpectrum, _]) raises:
        self.samples = {}
        self.ranges = {}
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

    def view(
        self,
    ) -> PiecewiseSpectrumView[
        origin_of(
            type_of(self.samples.unsafe_ptr()).origin,
            type_of(self.ranges.unsafe_ptr()).origin,
        )
    ]:
        """Borrow the buffers, preventing reallocation while the view is live.
        """
        var samples = self.samples.unsafe_ptr()
        var ranges = self.ranges.unsafe_ptr()
        comptime origin = origin_of(
            type_of(samples).origin, type_of(ranges).origin
        )
        return PiecewiseSpectrumView[origin](
            samples.unsafe_origin_cast[origin](),
            ranges.unsafe_origin_cast[origin](),
            self.count,
        )
