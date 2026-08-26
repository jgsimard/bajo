"""Controlled CPU binary-build and wide-collapse phase benchmark."""

from std.benchmark import keep
from std.math import round
from std.time import perf_counter_ns

from bajo.benchmark.bvh_fixtures import make_grid_triangles
from bajo.bvh.cpu.bounds_bvh import BoundsBvh
from bajo.bvh.cpu.builder import BinaryBoundsBvh, BoundsItem
from bajo.bvh.cpu.build_method import CpuBvhBuildMethod
from bajo.bvh.host_utils import compute_bounds
from bajo.core import AABB, Point3f32
from bajo.core.utils import ns_to_ms
from bajo.parser.obj.pack import pack_obj_triangles


comptime OBJ_PATH = "./assets/dragon/dragon.obj"
comptime LEAF_SIZE = 16
comptime WIDE_WIDTH = 16
comptime REPEATS = 7


def _median(values: List[Int]) -> Int:
    var ordered = values.copy()
    sort(ordered)
    return ordered[(len(ordered) - 1) >> 1]


def _make_items(
    vertices: List[Point3f32[.WORLD]],
) -> Tuple[List[BoundsItem[.WORLD]], AABB[.WORLD], AABB[.WORLD]]:
    var count = len(vertices) / 3
    var items = List[BoundsItem[.WORLD]](capacity=count)
    var root_bounds = AABB[.WORLD].invalid()
    var centroid_bounds = AABB[.WORLD].invalid()
    for i in range(count):
        var bounds = AABB(
            vertices[i * 3 + 0],
            vertices[i * 3 + 1],
            vertices[i * 3 + 2],
        )
        var item = BoundsItem(bounds, UInt32(i))
        root_bounds.grow(bounds)
        centroid_bounds.grow(item.centroid)
        items.append(item)
    return (items^, root_bounds, centroid_bounds)


def _bench_method[
    method: CpuBvhBuildMethod,
](
    label: String,
    items: List[BoundsItem[.WORLD]],
    root_bounds: AABB[.WORLD],
    centroid_bounds: AABB[.WORLD],
):
    var binary_times = List[Int](capacity=REPEATS)
    var collapse_times = List[Int](capacity=REPEATS)
    var binary_nodes = 0
    var wide_nodes = 0
    var leaves = 0

    for _ in range(REPEATS):
        var build_items = items.copy()
        var start = perf_counter_ns()
        var builder = BinaryBoundsBvh[.WORLD, LEAF_SIZE, method](
            build_items^, root_bounds, centroid_bounds
        )
        binary_times.append(Int(perf_counter_ns() - start))

        var leaf_count = [0]

        @always_inline
        def record_leaf(
            _first_item: UInt32,
            _item_count: UInt32,
        ) {mut leaf_count} -> UInt32:
            var idx = UInt32(leaf_count[0])
            leaf_count[0] += 1
            return idx

        start = perf_counter_ns()
        var tree = BoundsBvh[.WORLD, WIDE_WIDTH](builder, record_leaf)
        collapse_times.append(Int(perf_counter_ns() - start))
        binary_nodes = Int(builder.nodes_used)
        wide_nodes = len(tree.nodes)
        leaves = leaf_count[0]
        keep(wide_nodes)

    print(
        t"{label}\t{method.name()}\t{len(items)}\t"
        t"{round(ns_to_ms(_median(binary_times)), 3)}\t"
        t"{round(ns_to_ms(_median(collapse_times)), 3)}\t"
        t"{binary_nodes}\t{wide_nodes}\t{leaves}"
    )


def _bench_scene(
    label: String,
    vertices: List[Point3f32[.WORLD]],
):
    var prepared = _make_items(vertices)
    var items = prepared[0].copy()
    print("")
    print(t"{label}: {len(items)} triangles")
    print(
        "Scene\tBuilder\tTriangles\tBinary ms\tCollapse ms\tBinary nodes\tWide"
        " nodes\tLeaves"
    )
    _bench_method[.MEDIAN](label, items, prepared[1], prepared[2])
    _bench_method[.SAH](label, items, prepared[1], prepared[2])
    _bench_method[.LBVH](label, items, prepared[1], prepared[2])
    _bench_method[.HPLOC](label, items, prepared[1], prepared[2])


def main() raises:
    print("CPU BVH build phase benchmark; BVH16 leaf16; median of 7")
    _bench_scene("grid", make_grid_triangles())
    _bench_scene("dragon", pack_obj_triangles[.WORLD](OBJ_PATH))
