"""Linear Float32 RGB image output for CPU/GPU benchmark comparisons."""

from std.memory import bitcast
from bajo.rt.render_types import Color


def write_linear(path: String, pixels: ImmSpan[Color, _]) raises:
    var bytes = List[UInt8](length=len(pixels) * 12, fill=0)
    var index = 0
    for pixel in pixels:
        var channels = SIMD[.float32, 4](pixel.x, pixel.y, pixel.z, 0.0)
        comptime for channel in range(3):
            var word = bitcast[.uint32](channels[channel])
            comptime for byte in range(4):
                bytes[index + byte] = UInt8(word >> UInt32(byte * 8))
            index += 4
    with open(path, "w") as f:
        f.write_bytes(bytes)
