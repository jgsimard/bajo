"""GPU launch sampling values, with optional stream-ordered graph offset."""
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from bajo.rt.render_types import SamplingConfig


@fieldwise_init
struct GpuRtSamplingConfig(Copyable, DevicePassable):
    var fixed: SamplingConfig
    var offset: Optional[Pointer[UInt32, ImmUntrackedOrigin]]

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "GpuRtSamplingConfig"

    @always_inline
    def load(self) -> SamplingConfig:
        var value = self.fixed.copy()
        if self.offset:
            value.sample_offset = self.offset.value()[unsafe_offset=0]
        return value
