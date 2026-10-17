"""Binary-corpus driver for test_deflate_interop.py; not a standalone suite."""

from std.sys.arg import argv
from std.testing import assert_equal

from bajo.parser.deflate import inflate_zlib
from bajo.parser.mmap import MMap


def _u32(bytes: ImmSpan[UInt8, _], mut offset: Int) -> Int:
    var result = 0
    for shift in range(0, 32, 8):
        result |= Int(bytes[offset]) << shift
        offset += 1
    return result


def main() raises:
    var args = argv()
    var mapped = MMap(String(args[1]))
    var bytes = mapped.as_bytes_span()
    var offset = 0
    var cases = 0
    while offset < len(bytes):
        var error_expected = _u32(bytes, offset) != 0
        var capacity = _u32(bytes, offset)
        var expected_size = _u32(bytes, offset)
        var compressed_size = _u32(bytes, offset)
        var expected = bytes[offset : offset + expected_size]
        offset += expected_size
        var compressed = bytes[offset : offset + compressed_size]
        offset += compressed_size
        var output = List[UInt8](length=capacity + 32, fill=0xA5)
        var rejected = False
        var count = 0
        try:
            count = inflate_zlib(compressed, Span(output)[16 : 16 + capacity])
        except error:
            rejected = True
            if not error_expected:
                raise Error(t"case {cases}: rejected valid input: {error}")
        var label = "case " + String(cases)
        assert_equal(rejected, error_expected, label)
        if not rejected:
            assert_equal(count, expected_size, label)
            for i in range(count):
                assert_equal(output[16 + i], expected[i], label)
        for i in range(16):
            assert_equal(output[i], UInt8(0xA5), label)
            assert_equal(output[16 + capacity + i], UInt8(0xA5), label)
        cases += 1
    print("DEFLATE differential cases passed:", cases)
