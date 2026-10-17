"""CPU mesh refit throughput, including validation, packing and both BVHs.

Build with `mojo build -O3 -I . bench/bvh/bench_cpu_mesh_updates.mojo`.
Reports seven-trial medians after warmup; topology construction is excluded.
"""
from std.time import perf_counter_ns
from std.benchmark import keep
from bajo.core import Point3f32
from bajo.bvh.spatial import SpatialMesh


def main() raises:
    print("faces,winding_cache,median_refit_ms,checksum")
    for side in [16, 64, 256]:
        var points = List[Point3f32[.WORLD]]()
        var indices = List[UInt32]()
        for y in range(side + 1):
            for x in range(side + 1):
                points.append(Point3f32[.WORLD](Float32(x), Float32(y), 0.0))
        for y in range(side):
            for x in range(side):
                var a = UInt32(y * (side + 1) + x)
                var b = a + UInt32(side + 1)
                indices.extend([a, a + 1, b, b, a + 1, b + 1])
        for cached in [False, True]:
            var mesh = SpatialMesh(points, indices, cache_winding=cached)
            var times = List[Float64]()
            for trial in range(8):
                var start = perf_counter_ns()
                for repeat in range(16):
                    points[0].z = Float32(repeat % 2) * 0.1
                    mesh.refit(points)
                    keep(mesh.bvh.bounds.unsafe_ptr())
                var elapsed = Float64(perf_counter_ns() - start) / 16e6
                if trial > 0:
                    times.append(elapsed)
            sort(times)
            var checksum = Float64(0)
            for value in mesh.bvh.bounds:
                checksum += Float64(value)
            for value in mesh.packed_triangles:
                checksum += Float64(value)
            print(len(indices) // 3, cached, times[3], checksum, sep=",")
