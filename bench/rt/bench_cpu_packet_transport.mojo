"""Absorption-heavy CPU transport with scalar and SIMD packet widths.

Build with `mojo build -O3 -I . bench/rt/bench_cpu_packet_transport.mojo`.
Reports seven-trial medians after warmup; preparation is outside the timer.
"""
from bajo.bvh import Camera
from bajo.core import Point3f32, Vec3f32
from bajo.rt import CpuScene, SceneBuilder, render_wavefront
from bajo.rt.media import SceneMedia
from bajo.rt.render_types import Color, RenderSettings
from bajo.benchmark.cpu_harness import pixel_checksum


def _run[width: SIMDLength](world: CpuScene[16, 16], camera: Camera) raises:
    var settings = RenderSettings(128, 128, 4, 2026, 12, .HALTON)
    var times = List[Int]()
    var checksum = Float64(0)
    for trial in range(8):
        var image = render_wavefront[.PATH, width, 8192, False](
            settings, camera, world
        )
        checksum = pixel_checksum(image.pixels)
        if trial > 0:
            times.append(image.timings.render_ns)
    sort(times)
    print(Int(width), Float64(times[3]) * 1e-6, checksum, sep=",")


def main() raises:
    var builder = SceneBuilder()
    builder.set_environment(.uniform(Color(1)))
    for i in range(5):
        var surface = builder.add_dielectric(1.1 + Float32(i) * 0.1)
        builder.add_sphere(
            Point3f32[.WORLD](Float32(i - 2) * 1.8, 0, 0), 0.85, surface
        )
        builder.media.absorption.append(
            Color(0.01, 0.2, 0.5) * (1 + Float32(i) * 0.25)
        )
        var medium = UInt32(i + 1)
        SceneMedia.set_boundary(builder.media.spheres, i, medium)
    var world = CpuScene[16, 16](builder^.finish())
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0, 0, 3),
        Point3f32[.WORLD](0),
        Vec3f32[.WORLD](0, 1, 0),
        45,
    )
    print("width,median_render_ms,checksum")
    _run[1](world, camera)
    _run[4](world, camera)
    _run[8](world, camera)
    _run[16](world, camera)
