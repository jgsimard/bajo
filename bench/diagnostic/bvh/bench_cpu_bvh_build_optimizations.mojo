"""Controlled CPU binary-build and wide-collapse phase benchmark."""

from std.benchmark import keep
from std.math import round
from std.time import perf_counter_ns

from bajo.benchmark.bvh_fixtures import make_grid_triangles
from bajo.bvh.cpu.bounds_bvh import BoundsBvh
from bajo.bvh.cpu.builder import BinaryBoundsBvh, BoundsItem
from bajo.bvh.cpu.builder.lbvh import (
    MortonItem,
    _radix_sort_morton_pairs_parallel,
    _radix_sort_morton_pairs_inline_scratch,
)
from bajo.bvh.cpu.parallel import _worker_count
from bajo.bvh.cpu.build_method import CpuBvhBuildMethod
from bajo.bvh.host_utils import compute_bounds
from bajo.benchmark.timing import summarize_timings
from bajo.core import AABB, Point3f32
from bajo.core.morton import morton3
from bajo.core.utils import ns_to_ms
from bajo.parser.obj.pack import pack_obj_triangles


comptime OBJ_PATH = "./assets/dragon/dragon.obj"
comptime LEAF_SIZE = 16
comptime WIDE_WIDTH = 16
comptime REPEATS = 7
comptime EXPERIMENT_REPEATS = 31
comptime HPLOC_BUILD_BATCH = 4


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


def _make_morton_pairs(
    items: List[BoundsItem[.WORLD]], centroid_bounds: AABB[.WORLD]
) -> List[MortonItem]:
    var extent = centroid_bounds.extent()
    var inv = extent.safe_inv()
    var pairs = List[MortonItem](capacity=len(items))
    for i, item in enumerate(items):
        var c = (item.centroid - centroid_bounds._min) * inv
        pairs.append(MortonItem(morton3(c.x, c.y, c.z), UInt32(i)))
    return pairs^


def _bench_radix_scratch_ab(
    label: String,
    items: List[BoundsItem[.WORLD]],
    centroid_bounds: AABB[.WORLD],
):
    var source = _make_morton_pairs(items, centroid_bounds)
    var workers = _worker_count(len(source))
    var warm_baseline = source.copy()
    _radix_sort_morton_pairs_parallel(warm_baseline, workers)
    var warm_single = List[MortonItem](capacity=len(source) * 2)
    warm_single.resize(unsafe_uninit_length=len(source) * 2)
    for i in range(len(source)):
        warm_single.unsafe_get(i) = source.unsafe_get(i)
    _radix_sort_morton_pairs_inline_scratch(warm_single, len(source), workers)

    var baseline_times = List[Int](capacity=EXPERIMENT_REPEATS)
    var single_times = List[Int](capacity=EXPERIMENT_REPEATS)
    var final_baseline = warm_baseline^
    var final_single = warm_single^
    for sample in range(EXPERIMENT_REPEATS):
        if sample % 2 == 0:
            var start = perf_counter_ns()
            var baseline = source.copy()
            _radix_sort_morton_pairs_parallel(baseline, workers)
            baseline_times.append(Int(perf_counter_ns() - start))
            final_baseline = baseline^

            start = perf_counter_ns()
            var single = List[MortonItem](capacity=len(source) * 2)
            single.resize(unsafe_uninit_length=len(source) * 2)
            for i in range(len(source)):
                single.unsafe_get(i) = source.unsafe_get(i)
            _radix_sort_morton_pairs_inline_scratch(
                single, len(source), workers
            )
            single_times.append(Int(perf_counter_ns() - start))
            final_single = single^
        else:
            var start = perf_counter_ns()
            var single = List[MortonItem](capacity=len(source) * 2)
            single.resize(unsafe_uninit_length=len(source) * 2)
            for i in range(len(source)):
                single.unsafe_get(i) = source.unsafe_get(i)
            _radix_sort_morton_pairs_inline_scratch(
                single, len(source), workers
            )
            single_times.append(Int(perf_counter_ns() - start))
            final_single = single^

            start = perf_counter_ns()
            var baseline = source.copy()
            _radix_sort_morton_pairs_parallel(baseline, workers)
            baseline_times.append(Int(perf_counter_ns() - start))
            final_baseline = baseline^

    var exact = len(final_baseline) == len(final_single)
    if exact:
        for i in range(len(final_baseline)):
            exact &= final_baseline[i].code == final_single[i].code
            exact &= final_baseline[i].item_idx == final_single[i].item_idx
    var baseline = summarize_timings(baseline_times)
    var single = summarize_timings(single_times)
    print(
        t"{label} LBVH radix allocation A/B:"
        t" two={round(ns_to_ms(baseline.median_ns), 3)} ms"
        t" ({round(ns_to_ms(baseline.min_ns), 3)}..{round(ns_to_ms(baseline.max_ns), 3)}),"
        t" one={round(ns_to_ms(single.median_ns), 3)} ms"
        t" ({round(ns_to_ms(single.min_ns), 3)}..{round(ns_to_ms(single.max_ns), 3)}),"
        t" delta={round(Float64(single.median_ns - baseline.median_ns) * 100.0 / Float64(baseline.median_ns), 3)}%,"
        t" exact={exact}"
    )


def _bench_method[
    method: CpuBvhBuildMethod,
    hploc_balance_tasks: Int = 0,
](
    method_label: String,
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
        var builder = BinaryBoundsBvh[.WORLD, LEAF_SIZE, method].__init__[
            hploc_balance_tasks
        ](build_items^, root_bounds, centroid_bounds)
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
        t"{label}\t{method_label}\t{len(items)}\t"
        t"{round(ns_to_ms(_median(binary_times)), 3)}\t"
        t"{round(ns_to_ms(_median(collapse_times)), 3)}\t"
        t"{binary_nodes}\t{wide_nodes}\t{leaves}"
    )


def _timed_lbvh_collapse[
    use_dp_collapse: Bool,
](builder: BinaryBoundsBvh[.WORLD, LEAF_SIZE, .LBVH]) -> Tuple[Int, Int, Int]:
    var leaf_count = [0]

    @always_inline
    def record_leaf(
        _first_item: UInt32,
        _item_count: UInt32,
    ) {mut leaf_count} -> UInt32:
        var idx = UInt32(leaf_count[0])
        leaf_count[0] += 1
        return idx

    var start = perf_counter_ns()
    var tree = BoundsBvh[.WORLD, WIDE_WIDTH].__init__[
        use_dp_collapse=use_dp_collapse
    ](builder, record_leaf)
    var elapsed = Int(perf_counter_ns() - start)
    keep(len(tree.nodes))
    return (elapsed, len(tree.nodes), leaf_count[0])


def _bench_lbvh_collapse_ab(
    label: String,
    items: List[BoundsItem[.WORLD]],
    root_bounds: AABB[.WORLD],
    centroid_bounds: AABB[.WORLD],
):
    var build_items = items.copy()
    var builder = BinaryBoundsBvh[.WORLD, LEAF_SIZE, .LBVH](
        build_items^, root_bounds, centroid_bounds
    )

    _ = _timed_lbvh_collapse[True](builder)
    _ = _timed_lbvh_collapse[False](builder)
    var dp_times = List[Int](capacity=EXPERIMENT_REPEATS)
    var greedy_times = List[Int](capacity=EXPERIMENT_REPEATS)
    var dp_nodes = 0
    var greedy_nodes = 0
    var dp_leaves = 0
    var greedy_leaves = 0
    for sample in range(EXPERIMENT_REPEATS):
        if sample % 2 == 0:
            var dp = _timed_lbvh_collapse[True](builder)
            dp_times.append(dp[0])
            dp_nodes = dp[1]
            dp_leaves = dp[2]
            var greedy = _timed_lbvh_collapse[False](builder)
            greedy_times.append(greedy[0])
            greedy_nodes = greedy[1]
            greedy_leaves = greedy[2]
        else:
            var greedy = _timed_lbvh_collapse[False](builder)
            greedy_times.append(greedy[0])
            greedy_nodes = greedy[1]
            greedy_leaves = greedy[2]
            var dp = _timed_lbvh_collapse[True](builder)
            dp_times.append(dp[0])
            dp_nodes = dp[1]
            dp_leaves = dp[2]
    var dp = summarize_timings(dp_times)
    var greedy = summarize_timings(greedy_times)
    print(
        t"{label} LBVH collapse A/B: DP={round(ns_to_ms(dp.median_ns), 3)} ms"
        t" ({round(ns_to_ms(dp.min_ns), 3)}..{round(ns_to_ms(dp.max_ns), 3)}),"
        t" greedy={round(ns_to_ms(greedy.median_ns), 3)} ms"
        t" ({round(ns_to_ms(greedy.min_ns), 3)}..{round(ns_to_ms(greedy.max_ns), 3)}),"
        t" delta={round(Float64(greedy.median_ns - dp.median_ns) * 100.0 / Float64(dp.median_ns), 3)}%,"
        t" nodes={dp_nodes}/{greedy_nodes}, leaves={dp_leaves}/{greedy_leaves}"
    )


def _bench_hploc_repeated(
    label: String,
    items: List[BoundsItem[.WORLD]],
    root_bounds: AABB[.WORLD],
    centroid_bounds: AABB[.WORLD],
):
    var times = List[Int](capacity=EXPERIMENT_REPEATS)
    var nodes = 0
    var quality = Float32(0)
    for _ in range(EXPERIMENT_REPEATS):
        var start = perf_counter_ns()
        for _ in range(HPLOC_BUILD_BATCH):
            var build_items = items.copy()
            var builder = BinaryBoundsBvh[.WORLD, LEAF_SIZE, .HPLOC](
                build_items^, root_bounds, centroid_bounds
            )
            nodes = Int(builder.nodes_used)
            quality = builder.tree_quality()
            keep(nodes)
        times.append(Int(perf_counter_ns() - start) // HPLOC_BUILD_BATCH)
    var summary = summarize_timings(times)
    print(
        t"{label} H-PLOC repeated: {round(ns_to_ms(summary.median_ns), 3)} ms"
        t" ({round(ns_to_ms(summary.min_ns), 3)}..{round(ns_to_ms(summary.max_ns), 3)}),"
        t" nodes={nodes}, quality={quality}"
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
    _bench_method[.MEDIAN]("median", label, items, prepared[1], prepared[2])
    _bench_method[.SAH]("sah", label, items, prepared[1], prepared[2])
    _bench_method[.LBVH]("lbvh", label, items, prepared[1], prepared[2])
    _bench_method[.HPLOC]("hploc-fixed", label, items, prepared[1], prepared[2])
    _bench_method[.HPLOC, 8](
        "hploc-balanced8", label, items, prepared[1], prepared[2]
    )
    _bench_method[.HPLOC, 16](
        "hploc-balanced16", label, items, prepared[1], prepared[2]
    )
    _bench_method[.HPLOC, 32](
        "hploc-balanced32", label, items, prepared[1], prepared[2]
    )
    _bench_lbvh_collapse_ab(label, items, prepared[1], prepared[2])
    _bench_radix_scratch_ab(label, items, prepared[2])
    _bench_hploc_repeated(label, items, prepared[1], prepared[2])


def main() raises:
    print("CPU BVH build phase benchmark; BVH16 leaf16; median of 7")
    _bench_scene("grid", make_grid_triangles())
    _bench_scene("dragon", pack_obj_triangles[.WORLD](OBJ_PATH))
