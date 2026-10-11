"""Linear RGB film response, applied once after sample accumulation."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.utils.numerics import isfinite
from bajo.core import dot
from bajo.core import Vec3f32

comptime Color = Vec3f32[.WORLD]


struct FilmResponse(Copyable, DevicePassable, Writable):
    var exposure: Float32
    var red: Color
    var green: Color
    var blue: Color

    def __init__(
        out self,
        exposure: Float32 = 1.0,
        red: Color = Color(1.0, 0.0, 0.0),
        green: Color = Color(0.0, 1.0, 0.0),
        blue: Color = Color(0.0, 0.0, 1.0),
    ):
        self.exposure = exposure
        self.red = red
        self.green = green
        self.blue = blue

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "FilmResponse"

    @always_inline
    def apply(self, color: Color) -> Color:
        return (
            Color(
                dot(self.red, color),
                dot(self.green, color),
                dot(self.blue, color),
            )
            * self.exposure
        )

    def is_identity(self) -> Bool:
        return (
            self.exposure == 1.0
            and self.red.x == 1.0
            and self.red.y == 0.0
            and self.red.z == 0.0
            and self.green.x == 0.0
            and self.green.y == 1.0
            and self.green.z == 0.0
            and self.blue.x == 0.0
            and self.blue.y == 0.0
            and self.blue.z == 1.0
        )

    def validate(self) raises:
        if (
            not isfinite(self.exposure)
            or self.exposure < 0.0
            or not self.red.is_finite()[0]
            or not self.green.is_finite()[0]
            or not self.blue.is_finite()[0]
        ):
            raise Error(
                "film response must be finite with non-negative exposure"
            )
