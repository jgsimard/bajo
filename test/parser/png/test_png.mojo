from std.memory import bitcast
from std.testing import TestSuite, assert_equal

from bajo.parser.png import _linearize_srgb_u8, _srgb_to_linear


def _assert_exact(
    encoded: List[UInt8], actual: List[Float32], index: Int
) raises:
    var expected = _srgb_to_linear(Float32(encoded[index]) / 255.0)
    assert_equal(
        bitcast[.uint32](actual[index]),
        bitcast[.uint32](expected),
    )


def test_simd_conversion_is_bit_exact() raises:
    # Include every possible channel and a non-SIMD-aligned tail.
    var encoded = List[UInt8](length=259, fill=0)
    for index in range(len(encoded)):
        encoded[index] = UInt8(index % 256)

    var actual = _linearize_srgb_u8(encoded)
    assert_equal(len(actual), len(encoded))
    for index in range(len(encoded)):
        _assert_exact(encoded, actual, index)


def test_threaded_simd_conversion_is_bit_exact() raises:
    comptime channels_per_task = 256
    var encoded = List[UInt8](length=channels_per_task + 17, fill=0)
    for index in range(len(encoded)):
        encoded[index] = UInt8(index % 256)

    var actual = _linearize_srgb_u8[channels_per_task](encoded)
    assert_equal(len(actual), len(encoded))
    for index in range(256):
        _assert_exact(encoded, actual, index)
    for index in range(channels_per_task - 32, channels_per_task + 17):
        _assert_exact(encoded, actual, index)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
