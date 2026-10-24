"""Native file output for viewer frames, without Python or NumPy."""

from std.memory import bitcast
from bajo.core import Vec3W


def write_linear_frame(path: String, pixels: ImmSpan[Vec3W, _]) raises:
    var bytes = List[UInt8](length=len(pixels) * 12, fill=0)
    var out_idx = 0
    for pixel in pixels:
        var values = SIMD[.float32, 4](pixel.x, pixel.y, pixel.z, 0.0)
        for channel in range(3):
            var value = values[channel]
            var word = bitcast[.uint32](value)
            bytes[out_idx + 0] = UInt8(word & UInt32(0xFF))
            bytes[out_idx + 1] = UInt8((word >> UInt32(8)) & UInt32(0xFF))
            bytes[out_idx + 2] = UInt8((word >> UInt32(16)) & UInt32(0xFF))
            bytes[out_idx + 3] = UInt8(word >> UInt32(24))
            out_idx += 4
    with open(path, "w") as f:
        f.write_bytes(bytes)
