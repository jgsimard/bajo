"""Full surface-hit timings with and without mesh normals and UVs.

Patterns: 0 shares an instance; 1 mixes instances; 2 shares an instance with
roughly one third of lanes active. Each sample traces 16,384 packets; report
the median of seven samples after warmup, excluding scene and ray setup.
"""

from std.time import perf_counter_ns
from bajo.bvh.host_utils import compute_bounds
from bajo.core import Affine3f32, Point3, Point3f32, Ray, Vec3
from bajo.rt import Color, CpuScene, SceneBuilder


def make_world[attributes: Bool]() raises -> CpuScene[16, 16]:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    var vertices = List[Point3f32[.LOCAL]]()
    var normals = List[Float32]()
    var uvs = List[Float32]()
    for y in range(4):
        for x in range(4):
            for triangle in range(2):
                for corner in range(3):
                    var px = Float32(x) + Float32(Int(corner == 1))
                    var py = Float32(y) + Float32(Int(corner == 2))
                    if triangle != 0:
                        px = Float32(2 * x + 1) - px
                        py = Float32(2 * y + 1) - py
                    vertices.append(Point3f32[.LOCAL](px, py, 0.0))
                    comptime if attributes:
                        normals.extend([0.1 * px, 0.1 * py, 1.0])
                        uvs.extend([px * 0.25, py * 0.25])
    var bounds = compute_bounds(vertices)
    var mesh = builder.add_owned_triangle_mesh(vertices^, normals^, uvs^)
    for instance in range(4):
        var transform = Affine3f32[.LOCAL, .WORLD].identity()
        transform.m00 = 1.3
        transform.m01 = 0.1
        transform.m11 = 0.8
        transform.m22 = 1.1
        transform.tx = Float32(instance) * 8.0
        builder.add_triangle_instance(mesh, transform, bounds, matte)
    return CpuScene[16, 16](builder^.finish())


def run_case[length: SIMDLength](world: CpuScene[16, 16], pattern: Int):
    var packets = List[Ray[.float32, .WORLD, length]]()
    var valid = SIMD[.bool, length](fill=True)
    if pattern == 2:
        comptime for lane in range(length):
            valid[lane] = lane % 3 == 1
    for packet in range(128):
        var rays = Ray[.float32, .WORLD, length](
            Point3[.float32, .WORLD, length](0.0, 0.0, 3.0),
            Vec3[.float32, .WORLD, length](0.0, 0.0, -1.0),
        )
        comptime for lane in range(length):
            var x = Float32((packet + lane) % 4) + 0.2
            var y = Float32((packet // 4 + lane // 4) % 4) + 0.3
            var instance = lane % 4 if pattern == 1 else packet % 4
            rays.o.x[lane] = 1.3 * x + 0.1 * y + Float32(instance) * 8.0
            rays.o.y[lane] = 0.8 * y
        packets.append(rays)
    var times = List[Int]()
    var checksum = Float64(0.0)
    for trial in range(8):
        var sum = SIMD[.float64, length](0.0)
        var start = perf_counter_ns()
        for _ in range(128):
            for rays in packets:
                var hit = world.trace_surface(rays, valid)
                sum += valid.select(
                    (
                        hit.normal.x
                        + 2.0 * hit.normal.y
                        + 3.0 * hit.normal.z
                        + hit.uv_u
                        + 2.0 * hit.uv_v
                        + hit.t
                    ),
                    0.0,
                ).cast[.float64]()
        var elapsed = Int(perf_counter_ns() - start)
        checksum = sum.reduce_add()
        if trial != 0:
            times.append(elapsed)
    sort(times)
    print(
        "CASE",
        Int(length),
        pattern,
        "MS",
        Float64(times[3]) * 1e-6,
        "CHECKSUM",
        checksum,
    )


def main() raises:
    comptime for attributes in [False, True]:
        var world = make_world[attributes]()
        print("ATTRIBUTES", attributes)
        comptime for length in [4, 8, 16]:
            comptime for pattern in range(3):
                run_case[length](world, pattern)
