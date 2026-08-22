"""Benchmark canonical packed CPU BLAS construction and traversal."""

from std.benchmark import keep
from std.math import max, round
from std.time import perf_counter_ns

from bajo.bvh.constants import (
    SPHERE_LEAF_PACKED_STRIDE,
    TRACE,
    TRI_LEAF_PACKED_STRIDE,
    WideNode,
    f32_max,
)
from bajo.bvh.cpu.blas_set import (
    build_sphere_blases,
    build_triangle_blases,
    trace_blas_set,
)
from bajo.bvh.types import Sphere
from bajo.core import Frame, Point3f32, Rayf32, Vec3f32
from bajo.core.utils import ns_to_ms


comptime WIDTH = 4
comptime REPEATS = 7


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


def _make_spheres(sphere_count: Int, seed: Int) -> List[Sphere[Frame.LOCAL]]:
    var spheres = List[Sphere[Frame.LOCAL]](capacity=sphere_count)
    for i in range(sphere_count):
        var x = Float32((i * 17 + seed * 13) % 257) * 0.25
        var y = Float32((i * 29 + seed * 7) % 251) * 0.25
        var z = Float32((i * 11 + seed * 19) % 127) * 0.125
        spheres.append(
            Sphere[Frame.LOCAL](Point3f32[Frame.LOCAL](x, y, z), 0.2)
        )
    return spheres^


def _make_uniform_sphere_sets(
    blas_count: Int, spheres_per_blas: Int
) -> List[List[Sphere[Frame.LOCAL]]]:
    var sets = List[List[Sphere[Frame.LOCAL]]](capacity=blas_count)
    for blas_idx in range(blas_count):
        sets.append(_make_spheres(spheres_per_blas, blas_idx))
    return sets^


def _median(values: List[Int]) -> Int:
    var sorted = values.copy()
    sort(sorted)
    return sorted[(len(sorted) - 1) >> 1]


def _bench_case(
    label: String,
    vertex_sets: List[List[Point3f32[Frame.LOCAL]]],
) raises:
    var triangle_count = 0
    for vertices in vertex_sets:
        triangle_count += len(vertices) / 3

    var warm_batch = build_triangle_blases[WIDTH](vertex_sets)
    keep(warm_batch.blas_count)

    var batch_times = List[Int](capacity=REPEATS)
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        var batch = build_triangle_blases[WIDTH](vertex_sets)
        batch_times.append(Int(perf_counter_ns() - start))
        keep(batch.blas_count)

    var batch_ns = _median(batch_times)
    var capacity_f32 = 0
    for vertices in vertex_sets:
        var count = len(vertices) / 3
        if count > 0:
            capacity_f32 += max(count - 1, 1) * WIDTH * WideNode.CHILD_STRIDE
        capacity_f32 += count * WIDTH * TRI_LEAF_PACKED_STRIDE
    var final_f32 = len(warm_batch.nodes) + len(warm_batch.leaves)
    var saved = 100.0 * (1.0 - Float64(final_f32) / Float64(capacity_f32))
    print(
        t"{label}\t{len(vertex_sets)}\t{triangle_count}\t"
        t"{round(ns_to_ms(batch_ns), 3)}\t{capacity_f32 * 4}\t"
        t"{final_f32 * 4}\t{round(saved, 1)}"
    )


def _bench_sphere_case(
    label: String,
    sphere_sets: List[List[Sphere[Frame.LOCAL]]],
) raises:
    var sphere_count = 0
    for spheres in sphere_sets:
        sphere_count += len(spheres)

    var warm_batch = build_sphere_blases[WIDTH](sphere_sets)
    keep(warm_batch.blas_count)

    var batch_times = List[Int](capacity=REPEATS)
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        var batch = build_sphere_blases[WIDTH](sphere_sets)
        batch_times.append(Int(perf_counter_ns() - start))
        keep(batch.blas_count)

    var batch_ns = _median(batch_times)
    var capacity_f32 = 0
    for spheres in sphere_sets:
        var count = len(spheres)
        if count > 0:
            capacity_f32 += max(count - 1, 1) * WIDTH * WideNode.CHILD_STRIDE
        capacity_f32 += count * WIDTH * SPHERE_LEAF_PACKED_STRIDE
    var final_f32 = len(warm_batch.nodes) + len(warm_batch.leaves)
    var saved = 100.0 * (1.0 - Float64(final_f32) / Float64(capacity_f32))
    print(
        t"{label}\t{len(sphere_sets)}\t{sphere_count}\t"
        t"{round(ns_to_ms(batch_ns), 3)}\t{capacity_f32 * 4}\t"
        t"{final_f32 * 4}\t{round(saved, 1)}"
    )


def _make_rays(count: Int) -> List[Rayf32[Frame.LOCAL]]:
    var rays = List[Rayf32[Frame.LOCAL]](capacity=count)
    for i in range(count):
        var x = Float32(i % 256) * 0.25
        var y = Float32((i / 256) % 256) * 0.25
        rays.append(
            Rayf32[Frame.LOCAL](
                Point3f32[Frame.LOCAL](x, y, -1.0),
                Vec3f32[Frame.LOCAL](0.0, 0.0, 1.0),
            )
        )
    return rays^


def _bench_trace(vertices: List[Point3f32[Frame.LOCAL]]) raises:
    comptime RAY_COUNT = 65536
    var rays = _make_rays(RAY_COUNT)
    var packed = build_triangle_blases[WIDTH]([vertices.copy()])
    var packed_times = List[Int](capacity=REPEATS)
    var packed_checksum = Float64(0.0)

    for _ in range(REPEATS):
        var start = perf_counter_ns()
        var checksum = Float64(0.0)
        for ray in rays:
            var hit = trace_blas_set(packed, UInt32(0), ray)
            if hit.t < f32_max:
                checksum += Float64(hit.t)
        packed_times.append(Int(perf_counter_ns() - start))
        packed_checksum = checksum

    keep(packed_checksum)
    var packed_ns = _median(packed_times)
    var packed_mrays = Float64(RAY_COUNT) * 1000.0 / Float64(packed_ns)
    print("")
    print("Trace path\tMRay/s\tChecksum")
    print(t"CpuBlasSet\t{round(packed_mrays, 3)}\t{round(packed_checksum, 6)}")


def run_benchmark() raises:
    var one_large = _make_uniform_sets(1, 16384)
    var many_tiny = _make_uniform_sets(128, 4)
    var many_medium = _make_uniform_sets(16, 1024)
    var mixed = List[List[Point3f32[Frame.LOCAL]]]()
    mixed.append(_make_mesh(1, 0))
    mixed.append(_make_mesh(17, 1))
    mixed.append(_make_mesh(257, 2))
    mixed.append(_make_mesh(4096, 3))

    print("CPU triangle BLAS build benchmark; SAH width4; median of 7")
    print(
        "Case\tBLASes\tTriangles\tBuild ms\tCapacity bytes\tFinal"
        " bytes\tSaved %"
    )
    _bench_case("one large", one_large)
    _bench_case("many tiny", many_tiny)
    _bench_case("many medium", many_medium)
    _bench_case("mixed", mixed)
    _bench_trace(one_large[0].copy())

    var one_large_spheres = _make_uniform_sphere_sets(1, 16384)
    var many_tiny_spheres = _make_uniform_sphere_sets(128, 4)
    var many_medium_spheres = _make_uniform_sphere_sets(16, 1024)
    var mixed_spheres = List[List[Sphere[Frame.LOCAL]]]()
    mixed_spheres.append(_make_spheres(1, 0))
    mixed_spheres.append(_make_spheres(17, 1))
    mixed_spheres.append(_make_spheres(257, 2))
    mixed_spheres.append(_make_spheres(4096, 3))

    print("")
    print("CPU sphere BLAS build benchmark; SAH width4; median of 7")
    print(
        "Case\tBLASes\tSpheres\tBuild ms\tCapacity bytes\tFinal bytes\tSaved %"
    )
    _bench_sphere_case("one large", one_large_spheres)
    _bench_sphere_case("many tiny", many_tiny_spheres)
    _bench_sphere_case("many medium", many_medium_spheres)
    _bench_sphere_case("mixed", mixed_spheres)


def main() raises:
    run_benchmark()
