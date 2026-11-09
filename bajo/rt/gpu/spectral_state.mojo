"""Per-camera-sample spectral state, independent of compacted ray slots."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu.host import DeviceBuffer, DeviceContext
from bajo.core.sampled_spectrum import SampledSpectrum, SampledWavelengths


@fieldwise_init
struct GpuSpectralStateView(DevicePassable, TrivialRegisterPassable):
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

    @always_inline
    def load(self, field: Int, index: Int) -> SampledSpectrum[1]:
        return SampledSpectrum[1](
            self.fields[unsafe_offset=field * Int(self.capacity) + index],
            self.fields[unsafe_offset=(field + 1) * Int(self.capacity) + index],
            self.fields[unsafe_offset=(field + 2) * Int(self.capacity) + index],
            self.fields[unsafe_offset=(field + 3) * Int(self.capacity) + index],
        )

    @always_inline
    def store(self, field: Int, index: Int, value: SampledSpectrum[1]):
        comptime for channel in range(4):
            self.fields[
                unsafe_offset=(field + channel) * Int(self.capacity) + index
            ] = value[channel]

    @always_inline
    def wavelengths(self, index: Int) -> SampledWavelengths[1]:
        return SampledWavelengths[1](self.load(0, index), self.load(4, index))

    @always_inline
    def accumulate(self, index: Int, value: SampledSpectrum[1]):
        self.store(12, index, self.load(12, index) + value)


struct GpuSpectralState:
    var fields: DeviceBuffer[.float32]
    var capacity: Int

    def __init__(out self, mut ctx: DeviceContext, capacity: Int) raises:
        self.fields = ctx.enqueue_create_buffer[.float32](17 * capacity)
        self.capacity = capacity

    def view(self, sample_base: UInt32) -> GpuSpectralStateView:
        return GpuSpectralStateView(
            self.fields.unsafe_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            Int32(self.capacity),
            sample_base,
        )
