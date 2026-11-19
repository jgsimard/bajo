"""Per-camera-sample spectral state, independent of compacted ray slots."""

from layout import TileTensor, Coord, Idx, row_major
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu.host import DeviceBuffer, DeviceContext
from bajo.core.sampled_spectrum import SampledSpectrum, SampledWavelengths


@fieldwise_init
struct GpuSpectralStateView(DevicePassable, TrivialRegisterPassable):
    comptime PLANES = 17

    var fields: Pointer[Float32, MutUntrackedOrigin]
    var capacity: Int32
    var sample_base: UInt32

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "GpuSpectralStateView"

    def load(self, field: Int, index: Int) -> SampledSpectrum[1]:
        var fields = TileTensor(
            ptr=self.fields,
            layout=row_major(Coord(Idx[Self.PLANES], Int(self.capacity))),
        )
        # Share the sample's base pointer across its four strided channels.
        var sample = fields[:, index]
        return SampledSpectrum[1](
            sample[field],
            sample[field + 1],
            sample[field + 2],
            sample[field + 3],
        )

    def store(self, field: Int, index: Int, value: SampledSpectrum[1]):
        var fields = TileTensor(
            ptr=self.fields,
            layout=row_major(Coord(Idx[Self.PLANES], Int(self.capacity))),
        )
        var sample = fields[:, index]
        comptime for channel in range(4):
            sample[field + channel] = value[channel]

    def wavelengths(self, index: Int) -> SampledWavelengths[1]:
        return SampledWavelengths[1](self.load(0, index), self.load(4, index))

    def accumulate(self, index: Int, value: SampledSpectrum[1]):
        self.store(12, index, self.load(12, index) + value)


struct GpuSpectralState:
    var fields: DeviceBuffer[.float32]
    var capacity: Int

    def __init__(out self, mut ctx: DeviceContext, capacity: Int) raises:
        self.fields = ctx.enqueue_create_buffer[.float32](
            GpuSpectralStateView.PLANES * capacity
        )
        self.capacity = capacity

    def view(self, sample_base: UInt32) -> GpuSpectralStateView:
        return GpuSpectralStateView(
            self.fields.unsafe_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            Int32(self.capacity),
            sample_base,
        )
