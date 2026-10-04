from std.memory import bitcast
from std.testing import TestSuite, assert_raises, assert_true

from bajo.core import Vec3W
from bajo.viewer.common import _linear_frame


def test_owned_rgb_frame_preserves_float_bits() raises:
    var words = List[UInt32]()
    for word in [
        0x00000000,
        0x80000000,
        0x3F800000,
        0x40000000,
        0xC0400000,
        0x3E800000,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0x00000001,
        0x00800000,
        0x7F7FFFFF,
    ]:
        words.append(UInt32(word))
    var pixels = List[Vec3W]()
    for base in range(0, len(words), 3):
        pixels.append(
            Vec3W(
                bitcast[.float32](words[base]),
                bitcast[.float32](words[base + 1]),
                bitcast[.float32](words[base + 2]),
            )
        )
    var frame = _linear_frame(2, 2, pixels)
    assert_true(Int(py=frame.shape[0]) == 2)
    assert_true(Int(py=frame.shape[1]) == 2)
    assert_true(Int(py=frame.shape[2]) == 3)
    assert_true(String(py=frame.dtype.name) == "float32")
    assert_true(Bool(py=frame.flags.c_contiguous))
    assert_true(Bool(py=frame.flags.owndata))
    # Neither later source writes nor destruction may change the returned array.
    for index in range(len(pixels)):
        pixels[index] = Vec3W(42.0)
    pixels.clear()
    var packed = frame.reshape(-1).view("uint32")
    for index in range(len(words)):
        assert_true(UInt32(Int(py=packed[index])) == words[index])


def test_frame_rejects_inconsistent_dimensions() raises:
    var pixels = List[Vec3W](length=1, fill=Vec3W(1.0))
    with assert_raises():
        _ = _linear_frame(2, 1, pixels)
    with assert_raises():
        _ = _linear_frame(0, 1, pixels)
    with assert_raises():
        _ = _linear_frame(1, -1, pixels)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
