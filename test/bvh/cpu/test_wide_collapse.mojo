"""Sequential and parallel emission must preserve the selected frontier."""

from std.testing import TestSuite, assert_equal

from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.cpu.bounds_bvh import BoundsBvh
from bajo.bvh.cpu.builder import BinaryBoundsBvh, BoundsItem
from bajo.bvh.tagged_ref import decode_ref_index, is_leaf_ref
from bajo.core import AABB, Point3f32


def _assert_same_subtree[
    width: SIMDLength
](
    sequential: BoundsBvh[.WORLD, width],
    parallel: BoundsBvh[.WORLD, width],
    sequential_idx: UInt32,
    parallel_idx: UInt32,
) raises:
    ref a = sequential.nodes[Int(sequential_idx)]
    ref b = parallel.nodes[Int(parallel_idx)]
    comptime for lane in range(width):
        assert_equal(a.aabb._min.x[lane], b.aabb._min.x[lane])
        assert_equal(a.aabb._min.y[lane], b.aabb._min.y[lane])
        assert_equal(a.aabb._min.z[lane], b.aabb._min.z[lane])
        assert_equal(a.aabb._max.x[lane], b.aabb._max.x[lane])
        assert_equal(a.aabb._max.y[lane], b.aabb._max.y[lane])
        assert_equal(a.aabb._max.z[lane], b.aabb._max.z[lane])
        if a.data[lane] == EMPTY_LANE or is_leaf_ref(a.data[lane]):
            assert_equal(a.data[lane], b.data[lane])
        else:
            assert_equal(is_leaf_ref(b.data[lane]), False)
            _assert_same_subtree(
                sequential,
                parallel,
                decode_ref_index(a.data[lane]),
                decode_ref_index(b.data[lane]),
            )


def _check_emission[width: SIMDLength, use_dp: Bool](count: Int) raises:
    var items = List[BoundsItem[.WORLD]](capacity=count)
    for i in range(count):
        var x = Float32((i * 17) % 257)
        var y = Float32((i * 29) % 251)
        var z = Float32((i * 11) % 127)
        items.append(
            BoundsItem(
                AABB[.WORLD](
                    Point3f32[.WORLD](x, y, z),
                    Point3f32[.WORLD](x + 0.5, y + 0.5, z + 0.5),
                ),
                UInt32(i),
            )
        )
    var binary = BinaryBoundsBvh[.WORLD, 4, .LBVH](items^)

    def pack_leaf(first: UInt32, _count: UInt32) -> UInt32:
        return first

    var sequential = BoundsBvh[.WORLD, width].__init__[use_dp_collapse=use_dp](
        binary, pack_leaf
    )
    var parallel = BoundsBvh[.WORLD, width].__init__[
        parallel_emit=True, use_dp_collapse=use_dp
    ](binary, pack_leaf)
    assert_equal(len(sequential.nodes), len(parallel.nodes))
    if count > 0:
        _assert_same_subtree(sequential, parallel, UInt32(0), UInt32(0))


def test_wide_collapse_parallel_matches_sequential() raises:
    comptime for width in [2, 4, 8, 16]:
        comptime for use_dp in [False, True]:
            for count in [1, 2, 3, 257, 4097]:
                _check_emission[width, use_dp](count)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
