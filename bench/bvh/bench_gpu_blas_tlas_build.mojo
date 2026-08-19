"""Long-workload benchmark for packed GPU BLAS-set and TLAS construction."""

from std.math import round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.bvh.constants import Primitive
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.tlas import build_triangle_tlas
from bajo.bvh.gpu.triangle_bvh import (
    build_triangle_blas_set,
    build_triangle_bvh,
)
from bajo.bvh.gpu.utils import upload_vertices
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Instance
from bajo.core import AABB, Affine3f32, Frame, Point3f32, Vec3f32
from bajo.core.utils import ns_to_ms
from bajo.obj.pack import pack_obj_triangles
from bajo.rt.gpu.triangle_geometry import GpuRtTriangleGeometry


comptime BUNNY_PATH = "./assets/bunny/bunny.obj"
comptime BUDDHA_PATH = "./assets/buddha/buddha.obj"
comptime DRAGON_PATH = "./assets/dragon/dragon.obj"
comptime BENCH_REPEATS = 11


def _make_instances(
    bounds: List[AABB[Frame.LOCAL]],
) -> List[Instance]:
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


def _print_row(label: String, timings: List[Int]):
    var values = timings.copy()
    sort(values)
    var middle = (len(values) - 1) >> 1
    print(
        t"{label}\t{round(ns_to_ms(values[middle]), 3)}\t"
        t"{round(ns_to_ms(values[0]), 3)}.."
        t"{round(ns_to_ms(values[len(values) - 1]), 3)}"
    )


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"

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

    print("GPU acceleration-structure build benchmark")
    print(
        t"3 BLASes, {triangle_count} triangles, {len(instances)} instances, "
        t"median of {BENCH_REPEATS}"
    )

    var blas_times = List[Int](capacity=BENCH_REPEATS)
    var tlas_times = List[Int](capacity=BENCH_REPEATS)
    var geometry_times = List[Int](capacity=BENCH_REPEATS)
    var wide_times = List[Int](capacity=BENCH_REPEATS)
    with DeviceContext() as ctx:
        var dragon_vertices = upload_vertices(ctx, vertex_sets[2])
        # Warm the runtime, allocator, and compiled kernels.
        var warm_blases = build_triangle_blas_set[
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
            var t0 = perf_counter_ns()
            var blases = build_triangle_blas_set[
                8, 4, GpuBvhBuildMethod.HPLOC, True
            ](ctx, vertex_sets)
            ctx.synchronize()
            blas_times.append(Int(perf_counter_ns() - t0))

            t0 = perf_counter_ns()
            var tlas = build_triangle_tlas[
                2, 8, 2, 4, GpuBvhBuildMethod.LBVH, True
            ](ctx, instances)
            ctx.synchronize()
            tlas_times.append(Int(perf_counter_ns() - t0))

            t0 = perf_counter_ns()
            var geometry = GpuRtTriangleGeometry[
                Frame.LOCAL, 8, 4, GpuBvhBuildMethod.HPLOC, True
            ](ctx, vertex_sets[2])
            ctx.synchronize()
            geometry_times.append(Int(perf_counter_ns() - t0))

            t0 = perf_counter_ns()
            var wide = build_triangle_bvh[
                Frame.LOCAL, 4, 4, GpuBvhBuildMethod.HPLOC
            ](ctx, dragon_vertices)
            ctx.synchronize()
            wide_times.append(Int(perf_counter_ns() - t0))

    print("Case\tBuild median ms\tBuild min..max ms")
    _print_row("BLAS H-PLOC CWBVH8", blas_times)
    _print_row("TLAS LBVH2", tlas_times)
    _print_row("Dragon geometry H-PLOC CWBVH8", geometry_times)
    _print_row("Dragon H-PLOC wide4", wide_times)
