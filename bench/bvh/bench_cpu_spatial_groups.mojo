"""Spatial forest construction across item/group counts.

Build with `mojo build -O3 -I . bench/bvh/bench_cpu_spatial_groups.mojo`.
Reports the median of seven builds after warmup. Full output hashes, outside
the timed region, permit comparison with the same benchmark on other revisions.
"""

from std.memory import bitcast
from std.testing import assert_equal
from std.time import perf_counter_ns
from bajo.bvh.spatial import SpatialBoundsBvh
from bajo.core import AABB, Point3f32


def _forest_hash(tree: SpatialBoundsBvh[.WORLD]) -> UInt64:
    var digest = UInt64(14695981039346656037)
    for word in tree.nodes:
        digest = (digest ^ UInt64(word)) * UInt64(1099511628211)
    for value in tree.bounds:
        digest = (digest ^ UInt64(bitcast[.uint32](value))) * UInt64(
            1099511628211
        )
    for label in tree.group_ids:
        digest = (digest ^ UInt64(label)) * UInt64(1099511628211)
    for root in tree.group_roots:
        digest = (digest ^ UInt64(root)) * UInt64(1099511628211)
    return digest


def main() raises:
    print("items,groups,median_ms,nodes,hash")
    for count in [4096, 32768]:
        var boxes = List[AABB[.WORLD]](capacity=count)
        for i in range(count):
            var x = Float32(i % 256)
            var y = Float32(i // 256)
            boxes.append(
                AABB(
                    Point3f32[.WORLD](x, y, 0.0),
                    Point3f32[.WORLD](x + 0.5, y + 0.5, 0.5),
                )
            )
        for group_count in [0, 1, 2, 4, 8, 9, 16, 64, 256, 1024]:
            var groups = List[UInt32](capacity=count if group_count else 0)
            if group_count:
                for i in range(count):
                    groups.append(UInt32(i % group_count))
            var times = List[Float64](capacity=7)
            var expected = UInt64(0)
            var nodes = 0
            for repeat in range(8):
                var start = perf_counter_ns()
                var tree = SpatialBoundsBvh(boxes, groups)
                var elapsed = Float64(perf_counter_ns() - start) * 1e-6
                nodes = len(tree.nodes) // 2
                var digest = _forest_hash(tree)
                if repeat == 0:
                    expected = digest
                else:
                    assert_equal(digest, expected)
                    times.append(elapsed)
            sort(times)
            print(count, group_count, times[3], nodes, expected, sep=",")
