"""Independent permutation/order/stability checks for spatial Morton records."""
from std.testing import TestSuite, assert_equal, assert_true
from bajo.bvh.cpu.builder.lbvh import MortonItem
from bajo.bvh.spatial.radix import sort_morton30


def test_spatial_morton_sort_boundaries_duplicates_and_odd_lengths() raises:
    var keys: List[UInt32] = [
        0,
        1,
        32767,
        32768,
        65535,
        536870911,
        536870912,
        1073741822,
        1073741823,
    ]
    for count in [0, 1, 2, 257, 16384, 65537]:
        for mode in range(2):
            var input = List[MortonItem]()
            var state = UInt32(0x78AE2913)
            for i in range(count):
                state = state * UInt32(1664525) + UInt32(1013904223)
                var code = keys[
                    (i * 7) % len(keys)
                ] if mode == 0 else state & UInt32(0x3FFFFFFF)
                input.append(MortonItem(code, UInt32(i) ^ UInt32(0xA5A5A5A5)))
            var output = input.copy()
            sort_morton30(output)
            assert_equal(len(output), count)
            var seen = List[Bool](length=count, fill=False)
            var previous_code = UInt32(0)
            var previous_position = 0
            for i in range(count):
                var position = Int(output[i].item_idx ^ UInt32(0xA5A5A5A5))
                assert_true(position < count)
                assert_true(not seen[position])
                seen[position] = True
                assert_equal(output[i].code, input[position].code)
                if i > 0:
                    assert_true(previous_code <= output[i].code)
                    if previous_code == output[i].code:
                        assert_true(previous_position < position)
                previous_code = output[i].code
                previous_position = position
            for present in seen:
                assert_true(present)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
