"""Local Morton refinement preserves payloads, coarse ordering and stable ties."""
from std.testing import TestSuite, assert_equal
from std.math import min
from bajo.bvh.cpu.builder.lbvh import (
    MortonItem,
    refine_morton_pairs,
    _sorted_morton_pairs,
)
from bajo.bvh.cpu.builder.types import BoundsItem
from bajo.core import AABB, Point3f32
from bajo.bvh.cpu.builder.builder import BinaryBoundsBvh
from bajo.bvh.cpu.build_method import CpuBvhBuildMethod


def check[tile: Int](count: Int, ties: Bool, separate_codes: Bool) raises:
    var items = List[BoundsItem[.WORLD]]()
    for i in range(count):
        var p = Point3f32[.WORLD](Float32(i // 3 if ties else i), 0, 0)
        items.append(BoundsItem(AABB[.WORLD](p, p), UInt32(i)))
    var pairs = List[MortonItem]()
    for i in range(count):
        pairs.append(
            MortonItem(
                UInt32(i // 19 if separate_codes else 0), UInt32(count - 1 - i)
            )
        )
    var before = pairs.copy()
    refine_morton_pairs[tile](pairs, items)
    var seen = List[Bool](length=count, fill=False)
    for i in range(count):
        assert_equal(pairs[i].code, before[i].code)
        assert_equal(seen[Int(pairs[i].item_idx)], False)
        seen[Int(pairs[i].item_idx)] = True
    # Along X only, Morton order must agree with coordinate order. Equal
    # coordinates must retain their incoming order, independently of payload ID.
    for first in range(0, count, tile):
        var end = min(first + tile, count)
        var head = first
        while head < end:
            var last = head + 1
            while last < end and before[last].code == before[head].code:
                last += 1
            if last - head >= 8:
                var expected = List[UInt64]()
                for i in range(head, last):
                    var id = before[i].item_idx
                    var x = id // UInt32(3) if ties else id
                    expected.append((UInt64(x) << 32) | UInt64(i))
                sort(expected)
                for i in range(len(expected)):
                    assert_equal(
                        pairs[head + i].item_idx,
                        before[Int(UInt32(expected[i]))].item_idx,
                    )
            else:
                for i in range(head, last):
                    assert_equal(pairs[i].item_idx, before[i].item_idx)
            head = last


def test_morton_refine_cpu_short_runs_and_tail() raises:
    check[128](65, False, True)


def test_morton_refine_cpu_large_cluster_and_tail() raises:
    check[256](513, False, False)


def test_morton_refine_cpu_stable_centroid_ties() raises:
    check[256](513, True, False)


def test_morton_refine_cpu_parallel_tiles() raises:
    check[128](4096, False, False)


def test_morton_refine_cpu_empty_and_short_input() raises:
    check[128](0, False, False)
    check[128](7, False, False)


def test_morton_refine_cpu_vector_precheck_distinct_codes() raises:
    var items = List[BoundsItem[.WORLD]]()
    var pairs = List[MortonItem]()
    for i in range(4096):
        var p = Point3f32[.WORLD](Float32(i), 0, 0)
        items.append(BoundsItem(AABB[.WORLD](p, p), UInt32(i)))
        pairs.append(MortonItem(UInt32(i), UInt32(4095 - i)))
    refine_morton_pairs[256](pairs, items)
    for i in range(4096):
        assert_equal(pairs[i].code, UInt32(i))
        assert_equal(pairs[i].item_idx, UInt32(4095 - i))


def check_default_pipeline[method: CpuBvhBuildMethod]() raises:
    var items = List[BoundsItem[.WORLD]]()
    for i in range(257):
        var x = Float32(255 - i) * 0.001 if i < 256 else Float32(100000)
        var p = Point3f32[.WORLD](x, 0, 0)
        items.append(BoundsItem(AABB[.WORLD](p, p), UInt32(i)))
    var builder = BinaryBoundsBvh[.WORLD, 2, method](items^)
    var pairs = _sorted_morton_pairs(builder, AABB[.WORLD].invalid())
    for i in range(256):
        assert_equal(pairs[i].item_idx, UInt32(255 - i))
        assert_equal(pairs[i].code, UInt32(0))
    assert_equal(pairs[256].item_idx, UInt32(256))


def test_default_lbvh_pipeline_refines_collisions() raises:
    check_default_pipeline[.LBVH]()


def test_default_hploc_pipeline_refines_collisions() raises:
    check_default_pipeline[.HPLOC]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
