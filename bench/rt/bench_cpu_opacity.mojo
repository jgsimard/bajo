"""Alpha visibility through 2048 spheres; median of seven warmed runs."""
from std.time import perf_counter_ns
from std.benchmark import keep
from bajo.benchmark.timing import summarize_timings
from bajo.core import Point3, Point3W, Ray, Vec3
from bajo.rt import SceneBuilder, CpuScene, Color
from bajo.rt.material_types import PrimitiveId, NO_TEXTURE
from bajo.rt.opacity import Opacity


def measure[Fn: def() -> Float64](label: String, ref action: Fn):
    var checksum = action()
    var times = List[Int]()
    for _ in range(7):
        var start = perf_counter_ns()
        var result = action()
        times.append(Int(perf_counter_ns() - start))
        keep(result)
    print(
        label,
        "median_ms",
        Float64(summarize_timings(times).median_ns) / 1e6,
        "checksum",
        checksum,
    )


def main() raises:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    for layer in range(2):
        for y in range(32):
            for x in range(32):
                var index = UInt32(layer * 1024 + y * 32 + x)
                builder.add_sphere(
                    Point3W(Float32(x), Float32(y), Float32(-2 * layer)),
                    0.4,
                    matte,
                )
                builder.opacity.records[
                    PrimitiveId(.SPHERE, index).value
                ] = Opacity(
                    Float32(0.3) if layer == 0 else Float32(1), NO_TEXTURE, 1, 1
                )
    var world = CpuScene[16, 16](builder^.finish())
    var packets = List[Ray[.float32, .WORLD, 16]]()
    for block in range(256):
        var ray = Ray[.float32, .WORLD, 16](
            Point3[.float32, .WORLD, 16](0),
            Vec3[.float32, .WORLD, 16](0, 0, -1),
        )
        for lane in range(16):
            var index = block * 16 + lane
            ray.o.x[lane] = Float32(index % 32) + Float32(index % 3) * 0.07
            ray.o.y[lane] = Float32((index // 32) % 32) + 0.1
            ray.o.z[lane] = 3
        packets.append(ray)

    def primary() {imm} -> Float64:
        var checksum = Float64(0)
        for ray in packets:
            var hit = world.trace_surface(ray)
            checksum += Float64(hit.t.reduce_add())
        return checksum

    def shadow() {imm} -> Float64:
        var checksum = Float64(0)
        for ray in packets:
            checksum += Float64(world.occluded(ray).reduce_bit_count())
        return checksum

    measure("alpha sphere primary 4096 rays", primary)
    measure("alpha sphere shadow 4096 rays", shadow)
