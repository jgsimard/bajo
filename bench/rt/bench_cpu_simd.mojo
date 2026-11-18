"""CPU packet kernels, median of seven warmed runs; setup excluded.

Build with -O3 -I .; select a previous library with -I /path/to/snapshot
instead of -I . to measure the same fixtures and public entry points.
Each timed run averages sixteen iterations after four warmups.
"""
from std.time import perf_counter_ns
from std.benchmark import keep
from bajo.benchmark.timing import summarize_timings
from bajo.core import Vec3, Point3W, Vec3W
from bajo.rt import CpuScene, Camera
from bajo.rt.material_types import SurfaceId
from bajo.core.image import ImageTexture
from bajo.rt.cpu.scene import _triangle_instance_shading
from bajo.rt.cpu.denoise_guides import make_cpu_denoise_guides
from bajo.postprocess import (
    DenoiseStatistics,
    reconstruct_frame,
    atrous_denoise,
    AtrousWorkspace,
)
from test.rt.cpu.test_texture_storage import make_textures
from test.rt.cpu.test_instance_shading_simd import make_scene
from test.postprocess.cpu.test_atrous import flat_guides, sample_fixture
from examples.cornell_box import make_cornell_scene


def measure[Fn: def() raises -> Float64](label: String, ref action: Fn) raises:
    var checksum = action()
    for _ in range(3):
        keep(action())
    var times = List[Int]()
    for _ in range(7):
        var start = perf_counter_ns()
        for _ in range(16):
            var result = action()
            keep(result)
            if result != checksum:
                raise Error("unstable benchmark output")
        times.append(Int(perf_counter_ns() - start) // 16)
    var timing = summarize_timings(times)
    print(
        label,
        "median_ms",
        Float64(timing.median_ns) / 1e6,
        "checksum",
        checksum,
    )


def main() raises:
    var textures = make_textures()
    for i in range(4, 16):
        _ = textures.add_image_texture(
            ImageTexture(1, 1, [Float32(i) / 16, 0.2, 0.4])
        )
    for i in range(16):
        var surface = textures.add_lambertian(Vec3W(0.5))
        textures.normal_maps[surface.value[0]] = UInt32(i)
    var data = make_scene()
    for groups in [1, 2, 4, 16]:
        var ids = List[SIMD[.uint32, 16]]()
        var coordinates = List[SIMD[.float32, 16]]()
        for block in range(4096):
            var index = SIMD[.uint32, 16](0)
            var u = SIMD[.float32, 16](0)
            for lane in range(16):
                index[lane] = UInt32(lane % groups)
                u[lane] = Float32((block * 17 + lane * 7) % 257) / 256
            ids.append(index)
            coordinates.append(u)

        def texture_samples() raises {imm} -> Float64:
            var checksum = Float64(0)
            for block in range(len(ids)):
                var c = textures.sample_image_textures(
                    ids[block], coordinates[block], coordinates[block]
                )
                keep(c.x)
                checksum += Float64(c.x.reduce_add())
            return checksum

        measure("textures groups=" + String(groups), texture_samples)

        def normals() raises {imm} -> Float64:
            var checksum = Float64(0)
            for block in range(len(ids)):
                var surface = SurfaceId[16](
                    SurfaceId.pack_raw(.LAMBERTIAN, 0) | ids[block]
                )
                var n = textures.shading_normal(
                    surface,
                    Vec3[.float32, .WORLD, 16](0, 0, 1),
                    coordinates[block],
                    coordinates[block],
                    dpdu=Vec3[.float32, .WORLD, 16](1, 0, 0),
                )
                keep(n.x)
                checksum += Float64(n.z.reduce_add())
            return checksum

        measure("normal maps groups=" + String(groups), normals)

        def attributes() raises {imm} -> Float64:
            var checksum = Float64(0)
            for block in range(len(ids)):
                var result = _triangle_instance_shading(
                    data,
                    ids[block] % UInt32(4),
                    SIMD[.uint32, 16](0),
                    coordinates[block] * 0.4,
                    SIMD[.float32, 16](0.3),
                    Vec3[.float32, .WORLD, 16](0, 0, 1),
                )
                keep(result[0].x)
                checksum += Float64(
                    result[0].z.reduce_add() + result[1].reduce_add()
                )
            return checksum

        if groups <= 4:
            measure("attributes groups=" + String(groups), attributes)
    var guides = flat_guides(257, 129)
    var batch = sample_fixture(257, 129, 16)
    for i in range(len(batch.colors)):
        batch.colors[i] = batch.colors[i] * (Float32(1 + i % 13) / 7)
        batch.components[i] = UInt32(i % 3)
    var stats = DenoiseStatistics(257, 129)
    stats.add(batch)

    def reconstruct() raises {imm} -> Float64:
        var frame = reconstruct_frame(stats, guides)
        return Float64(frame.diffuse[180].x + frame.variance[28])

    measure("reconstruction 257x129", reconstruct)
    var frame = reconstruct_frame(stats, guides)
    var workspace = AtrousWorkspace()

    def filter() raises {imm, mut workspace} -> Float64:
        atrous_denoise(frame, guides, workspace)
        return Float64(workspace.ping[180].x + workspace.variance[28])

    measure("atrous 257x129", filter)
    var world = CpuScene[16, 16](make_cornell_scene())
    var camera = Camera.from_vfov(
        Point3W(0, 1, 3.2), Point3W(0, 1, -1), Vec3W(0, 1, 0), 40
    )

    def trace_guides() raises {imm} -> Float64:
        var result = make_cpu_denoise_guides(world, camera, 257, 129)
        var checksum = Float64(0)
        for field in range(13):
            checksum += Float64(result.fields[(64 * 257 + 128) * 13 + field])
        return checksum

    measure("guides 257x129", trace_guides)
