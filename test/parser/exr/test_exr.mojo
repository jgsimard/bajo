from std.memory import bitcast
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

from bajo.parser.exr import parse_exr


def _append_u32(mut bytes: List[UInt8], value: UInt32):
    bytes.append(UInt8(value & UInt32(255)))
    bytes.append(UInt8((value >> UInt32(8)) & UInt32(255)))
    bytes.append(UInt8((value >> UInt32(16)) & UInt32(255)))
    bytes.append(UInt8(value >> UInt32(24)))


def _append_u64(mut bytes: List[UInt8], value: UInt64):
    for shift in range(8):
        bytes.append(UInt8(value >> UInt64(8 * shift)))


def _append_f32(mut bytes: List[UInt8], value: Float32):
    _append_u32(bytes, bitcast[.uint32](value))


def _append_string(mut bytes: List[UInt8], value: String):
    var span = StringSpan(value).as_bytes()
    for byte in span:
        bytes.append(byte)
    bytes.append(UInt8(0))


def _append_attribute(
    mut bytes: List[UInt8],
    name: String,
    attribute_type: String,
    payload: List[UInt8],
):
    _append_string(bytes, name)
    _append_string(bytes, attribute_type)
    _append_u32(bytes, UInt32(len(payload)))
    for byte in payload:
        bytes.append(byte)


def _channel_list() -> List[UInt8]:
    var channels = List[UInt8]()
    for name in ["B", "G", "R"]:
        _append_string(channels, name)
        _append_u32(channels, UInt32(2))  # FLOAT
        channels.append(UInt8(0))  # pLinear
        channels.extend([UInt8(0), UInt8(0), UInt8(0)])
        _append_u32(channels, UInt32(1))
        _append_u32(channels, UInt32(1))
    channels.append(UInt8(0))
    return channels^


def _box2i(max_x: UInt32, max_y: UInt32) -> List[UInt8]:
    var payload = List[UInt8]()
    _append_u32(payload, UInt32(0))
    _append_u32(payload, UInt32(0))
    _append_u32(payload, max_x)
    _append_u32(payload, max_y)
    return payload^


def _tiny_uncompressed_exr() -> List[UInt8]:
    var bytes = List[UInt8]()
    _append_u32(bytes, UInt32(20000630))
    _append_u32(bytes, UInt32(2))
    _append_attribute(bytes, "channels", "chlist", _channel_list())
    _append_attribute(bytes, "compression", "compression", [UInt8(0)])
    _append_attribute(bytes, "dataWindow", "box2i", _box2i(1, 0))
    bytes.append(UInt8(0))  # End of header.

    var chunk_offset = UInt64(len(bytes) + 8)
    _append_u64(bytes, chunk_offset)
    _append_u32(bytes, UInt32(0))  # Scanline y.
    _append_u32(bytes, UInt32(24))
    # OpenEXR scanline order is B, G, R for this channel list.
    _append_f32(bytes, 0.3)
    _append_f32(bytes, 0.6)
    _append_f32(bytes, 0.2)
    _append_f32(bytes, 0.5)
    _append_f32(bytes, 0.1)
    _append_f32(bytes, 0.4)
    return bytes^


def test_decodes_float_rgb_scanline() raises:
    var image = parse_exr(_tiny_uncompressed_exr())
    assert_equal(image.width, 2)
    assert_equal(image.height, 1)
    assert_equal(len(image.pixels), 6)
    assert_almost_equal(image.pixels[0], 0.1)
    assert_almost_equal(image.pixels[1], 0.2)
    assert_almost_equal(image.pixels[2], 0.3)
    assert_almost_equal(image.pixels[3], 0.4)
    assert_almost_equal(image.pixels[4], 0.5)
    assert_almost_equal(image.pixels[5], 0.6)


def test_rejects_invalid_exr() raises:
    with assert_raises():
        _ = parse_exr([UInt8(1), UInt8(2), UInt8(3), UInt8(4)])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
