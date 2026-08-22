"""Long-workload benchmark for the unified segmented GPU BVH pipeline."""

from std.math import max, round
from std.sys import has_accelerator
from std.sys.arg import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.bvh.constants import (
    Primitive,
    TRI_LEAF_PACKED_STRIDE,
    WideNode,
)
from bajo.bvh.gpu.compressed_bounds_bvh import (
    CWBVH_NODE_WORDS,
    CWBVH_TRIANGLE_WORDS,
)
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.tlas import build_triangle_tlas
from bajo.bvh.gpu.triangle_bvh import (
    build_triangle_blas_set,
    build_triangle_bvh,
)
from bajo.bvh.gpu.utils import upload_vertices
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import BlasDescLayout, GpuBlasSet, Instance
from bajo.core import AABB, Affine3f32, Frame, Point3f32
from bajo.core.utils import ns_to_ms
from bajo.parser.obj.pack import pack_obj_triangles
from bajo.rt.gpu.triangle_geometry import GpuRtTriangleGeometry


comptime BUNNY_PATH = "./assets/bunny/bunny.obj"
comptime BUDDHA_PATH = "./assets/buddha/buddha.obj"
comptime DRAGON_PATH = "./assets/dragon/dragon.obj"
comptime BENCH_REPEATS = 11
comptime SHAPE_REPEATS = 7


def _make_mesh(triangle_count: Int, seed: Int) -> List[Point3f32[Frame.LOCAL]]:
    var vertices = List[Point3f32[Frame.LOCAL]](capacity=triangle_count * 3)
    for i in range(triangle_count):
        var x = Float32((i * 17 + seed * 13) % 257) * 0.25
        var y = Float32((i * 29 + seed * 7) % 251) * 0.25
        var z = Float32((i * 11 + seed * 19) % 127) * 0.125
        vertices.append(Point3f32[Frame.LOCAL](x - 0.2, y - 0.1, z))
        vertices.append(Point3f32[Frame.LOCAL](x + 0.2, y - 0.1, z))
        vertices.append(Point3f32[Frame.LOCAL](x, y + 0.2, z + 0.05))
    return vertices^


def _make_uniform_sets(
    blas_count: Int, triangles_per_blas: Int
) -> List[List[Point3f32[Frame.LOCAL]]]:
    var sets = List[List[Point3f32[Frame.LOCAL]]](capacity=blas_count)
    for blas_idx in range(blas_count):
        sets.append(_make_mesh(triangles_per_blas, blas_idx))
    return sets^


def _make_instances(bounds: List[AABB[Frame.LOCAL]]) -> List[Instance]:
    var instances = List[Instance](capacity=108)
    for copy_idx in range(36):
        for blas_idx in range(3):
            var transform = Affine3f32[Frame.LOCAL, Frame.WORLD].identity()
            transform.tx = Float32(copy_idx % 6) * 3.0
            transform.tz = Float32(copy_idx / 6) * 3.0
            instances.append(
                Instance(
                    transform,
                    UInt32(blas_idx),
                    bounds[blas_idx],
                    Primitive.TRIANGLE,
                )
            )
    return instances^


def _median_ns(timings: List[Int]) -> Int:
    var values = timings.copy()
    sort(values)
    return values[(len(values) - 1) >> 1]


def _print_row(label: String, timings: List[Int]):
    var values = timings.copy()
    sort(values)
    var middle = (len(values) - 1) >> 1
    print(
        t"{label}\t{round(ns_to_ms(values[middle]), 3)}\t"
        t"{round(ns_to_ms(values[0]), 3)}.."
        t"{round(ns_to_ms(values[len(values) - 1]), 3)}"
    )


@fieldwise_init
struct BlasSetSummary(Copyable, Writable):
    var nodes: UInt64
    var leaf_blocks: UInt64
    var bytes: Int


def _summarize_blas_set[
    node_width: SIMDLength, leaf_width: SIMDLength
](blases: GpuBlasSet[node_width, leaf_width]) raises -> BlasSetSummary:
    var nodes = UInt64(0)
    var leaf_blocks = UInt64(0)
    with blases.descs.map_to_host() as descs:
        for blas_idx in range(blases.blas_count):
            var base = BlasDescLayout.base(blas_idx)
            nodes += UInt64(descs[base + BlasDescLayout.NODE_COUNT])
            leaf_blocks += UInt64(descs[base + BlasDescLayout.LEAF_BLOCK_COUNT])
    return BlasSetSummary(
        nodes,
        leaf_blocks,
        4 * (len(blases.descs) + len(blases.nodes) + len(blases.leaves)),
    )


def _bench_wide_shape[
    build_method: GpuBvhBuildMethod,
](
    mut ctx: DeviceContext,
    label: String,
    vertex_sets: List[List[Point3f32[Frame.LOCAL]]],
) raises:
    var triangle_count = 0
    for vertices in vertex_sets:
        triangle_count += len(vertices) / 3
    var warm = build_triangle_blas_set[4, 4, build_method](ctx, vertex_sets)
    ctx.synchronize()
    var times = List[Int](capacity=SHAPE_REPEATS)
    for _ in range(SHAPE_REPEATS):
        var start = perf_counter_ns()
        var blases = build_triangle_blas_set[4, 4, build_method](
            ctx, vertex_sets
        )
        ctx.synchronize()
        times.append(Int(perf_counter_ns() - start))
    var packed = build_triangle_blas_set[4, 4, build_method](ctx, vertex_sets)
    ctx.synchronize()
    var summary = _summarize_blas_set(packed)
    var capacity_nodes = 0
    for vertices in vertex_sets:
        var count = len(vertices) / 3
        if count > 0:
            capacity_nodes += max(count - 1, 1)
    var capacity_bytes = 4 * (
        len(vertex_sets) * BlasDescLayout.STRIDE
        + capacity_nodes * 4 * WideNode.CHILD_STRIDE
        + triangle_count * 4 * TRI_LEAF_PACKED_STRIDE
    )
    var saved = 100.0 * (1.0 - Float64(summary.bytes) / Float64(capacity_bytes))
    print(
        t"{label}\t{len(vertex_sets)}\t{triangle_count}\t"
        t"{round(ns_to_ms(_median_ns(times)), 3)}\t"
        t"{summary.nodes}\t{summary.leaf_blocks}\t{capacity_bytes}\t"
        t"{summary.bytes}\t{round(saved, 1)}"
    )


def _bench_cwbvh_shape(
    mut ctx: DeviceContext,
    label: String,
    vertex_sets: List[List[Point3f32[Frame.LOCAL]]],
) raises:
    var triangle_count = 0
    for vertices in vertex_sets:
        triangle_count += len(vertices) / 3
    var warm = build_triangle_blas_set[8, 4, GpuBvhBuildMethod.HPLOC, True](
        ctx, vertex_sets
    )
    ctx.synchronize()
    var times = List[Int](capacity=SHAPE_REPEATS)
    for _ in range(SHAPE_REPEATS):
        var start = perf_counter_ns()
        var blases = build_triangle_blas_set[
            8, 4, GpuBvhBuildMethod.HPLOC, True
        ](ctx, vertex_sets)
        ctx.synchronize()
        times.append(Int(perf_counter_ns() - start))
    var packed = build_triangle_blas_set[8, 4, GpuBvhBuildMethod.HPLOC, True](
        ctx, vertex_sets
    )
    ctx.synchronize()
    var summary = _summarize_blas_set(packed)
    var capacity_nodes = 0
    for vertices in vertex_sets:
        var count = len(vertices) / 3
        if count > 0:
            capacity_nodes += max(count - 1, 1)
    var capacity_bytes = 4 * (
        len(vertex_sets) * BlasDescLayout.STRIDE
        + capacity_nodes * CWBVH_NODE_WORDS
        + triangle_count * CWBVH_TRIANGLE_WORDS
    )
    var saved = 100.0 * (1.0 - Float64(summary.bytes) / Float64(capacity_bytes))
    print(
        t"{label}\t{len(vertex_sets)}\t{triangle_count}\t"
        t"{round(ns_to_ms(_median_ns(times)), 3)}\t"
        t"{summary.nodes}\t{summary.leaf_blocks}\t{capacity_bytes}\t"
        t"{summary.bytes}\t{round(saved, 1)}"
    )


def _run_dispatch_probe(mode: String) raises:
    """Run one unified many-tiny workload for external GPU profiling."""
    var many_tiny = _make_uniform_sets(128, 4)
    with DeviceContext() as ctx:
        if mode == "lbvh-tiny":
            var result = build_triangle_blas_set[4, 4, GpuBvhBuildMethod.LBVH](
                ctx, many_tiny
            )
            ctx.synchronize()
            print(t"{mode}: {result.blas_count} BLASes")
            return
        if mode == "hploc-tiny":
            var result = build_triangle_blas_set[4, 4, GpuBvhBuildMethod.HPLOC](
                ctx, many_tiny
            )
            ctx.synchronize()
            print(t"{mode}: {result.blas_count} BLASes")
            return
        if mode == "cwbvh-tiny":
            var result = build_triangle_blas_set[
                8, 4, GpuBvhBuildMethod.HPLOC, True
            ](ctx, many_tiny)
            ctx.synchronize()
            print(t"{mode}: {result.blas_count} BLASes")
            return
    raise Error(t"unknown dispatch probe: {mode}")


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    var args = argv()
    if len(args) > 1:
        _run_dispatch_probe(args[1])
        return

    var vertex_sets = List[List[Point3f32[Frame.LOCAL]]](capacity=3)
    vertex_sets.append(pack_obj_triangles[Frame.LOCAL](BUNNY_PATH))
    vertex_sets.append(pack_obj_triangles[Frame.LOCAL](BUDDHA_PATH))
    vertex_sets.append(pack_obj_triangles[Frame.LOCAL](DRAGON_PATH))
    var bounds = List[AABB[Frame.LOCAL]](capacity=3)
    var triangle_count = 0
    for vertices in vertex_sets:
        bounds.append(compute_bounds(vertices))
        triangle_count += len(vertices) / 3
    var instances = _make_instances(bounds)

    print("Unified segmented GPU acceleration-structure build benchmark")
    print(
        t"3 BLASes, {triangle_count} triangles, {len(instances)} instances, "
        t"median of {BENCH_REPEATS}"
    )
    var cwbvh_times = List[Int](capacity=BENCH_REPEATS)
    var tlas_times = List[Int](capacity=BENCH_REPEATS)
    var geometry_times = List[Int](capacity=BENCH_REPEATS)
    var wide_times = List[Int](capacity=BENCH_REPEATS)
    with DeviceContext() as ctx:
        var dragon_vertices = upload_vertices(ctx, vertex_sets[2])
        var warm_cwbvh = build_triangle_blas_set[
            8, 4, GpuBvhBuildMethod.HPLOC, True
        ](ctx, vertex_sets)
        var warm_tlas = build_triangle_tlas[
            2, 8, 2, 4, GpuBvhBuildMethod.LBVH, True
        ](ctx, instances)
        var warm_geometry = GpuRtTriangleGeometry[
            Frame.LOCAL, 8, 4, GpuBvhBuildMethod.HPLOC, True
        ](ctx, vertex_sets[2])
        var warm_wide = build_triangle_bvh[
            Frame.LOCAL, 4, 4, GpuBvhBuildMethod.HPLOC
        ](ctx, dragon_vertices)
        ctx.synchronize()

        for _ in range(BENCH_REPEATS):
            var start = perf_counter_ns()
            var cwbvh = build_triangle_blas_set[
                8, 4, GpuBvhBuildMethod.HPLOC, True
            ](ctx, vertex_sets)
            ctx.synchronize()
            cwbvh_times.append(Int(perf_counter_ns() - start))

            start = perf_counter_ns()
            var tlas = build_triangle_tlas[
                2, 8, 2, 4, GpuBvhBuildMethod.LBVH, True
            ](ctx, instances)
            ctx.synchronize()
            tlas_times.append(Int(perf_counter_ns() - start))

            start = perf_counter_ns()
            var geometry = GpuRtTriangleGeometry[
                Frame.LOCAL, 8, 4, GpuBvhBuildMethod.HPLOC, True
            ](ctx, vertex_sets[2])
            ctx.synchronize()
            geometry_times.append(Int(perf_counter_ns() - start))

            start = perf_counter_ns()
            var wide = build_triangle_bvh[
                Frame.LOCAL, 4, 4, GpuBvhBuildMethod.HPLOC
            ](ctx, dragon_vertices)
            ctx.synchronize()
            wide_times.append(Int(perf_counter_ns() - start))

    print("Case\tBuild median ms\tBuild min..max ms")
    _print_row("BLAS CWBVH8 segmented", cwbvh_times)
    _print_row("TLAS LBVH2 one segment", tlas_times)
    _print_row("Dragon geometry H-PLOC CWBVH8", geometry_times)
    _print_row("Dragon H-PLOC wide4 one segment", wide_times)

    var one_large = _make_uniform_sets(1, 16384)
    var many_tiny = _make_uniform_sets(128, 4)
    var many_medium = _make_uniform_sets(16, 1024)
    var mixed = List[List[Point3f32[Frame.LOCAL]]]()
    mixed.append(_make_mesh(1, 0))
    mixed.append(_make_mesh(17, 1))
    mixed.append(_make_mesh(257, 2))
    mixed.append(_make_mesh(4096, 3))

    with DeviceContext() as ctx:
        print("")
        print(t"Segmented LBVH wide4; median of {SHAPE_REPEATS}")
        print(
            "Case\tBLASes\tTriangles\tBuild ms\tNodes\tLeaf blocks\t"
            "Capacity bytes\tFinal bytes\tSaved %"
        )
        _bench_wide_shape[GpuBvhBuildMethod.LBVH](ctx, "one large", one_large)
        _bench_wide_shape[GpuBvhBuildMethod.LBVH](ctx, "many tiny", many_tiny)
        _bench_wide_shape[GpuBvhBuildMethod.LBVH](
            ctx, "many medium", many_medium
        )
        _bench_wide_shape[GpuBvhBuildMethod.LBVH](ctx, "mixed", mixed)

        print("")
        print(t"Segmented H-PLOC wide4; median of {SHAPE_REPEATS}")
        print(
            "Case\tBLASes\tTriangles\tBuild ms\tNodes\tLeaf blocks\t"
            "Capacity bytes\tFinal bytes\tSaved %"
        )
        _bench_wide_shape[GpuBvhBuildMethod.HPLOC](ctx, "one large", one_large)
        _bench_wide_shape[GpuBvhBuildMethod.HPLOC](ctx, "many tiny", many_tiny)
        _bench_wide_shape[GpuBvhBuildMethod.HPLOC](
            ctx, "many medium", many_medium
        )
        _bench_wide_shape[GpuBvhBuildMethod.HPLOC](ctx, "mixed", mixed)

        print("")
        print(t"Segmented H-PLOC CWBVH8; median of {SHAPE_REPEATS}")
        print(
            "Case\tBLASes\tTriangles\tBuild ms\tNodes\tTriangles\t"
            "Capacity bytes\tFinal bytes\tSaved %"
        )
        _bench_cwbvh_shape(ctx, "one large", one_large)
        _bench_cwbvh_shape(ctx, "many tiny", many_tiny)
        _bench_cwbvh_shape(ctx, "many medium", many_medium)
        _bench_cwbvh_shape(ctx, "mixed", mixed)
