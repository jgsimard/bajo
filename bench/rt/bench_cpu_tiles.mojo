"""CPU wavefront A/B benchmark, including exact linear image output.

Arguments: scene parallel repeats output.f32 [width height spp packet_width].
Scenes: weekend, mixed, cornell, coated. Packet widths: 4 and 16.
Setup and output writes are excluded from reported render timings.
"""

from std.sys.arg import argv
from bajo.benchmark.cpu_harness import pixel_checksum
from bajo.benchmark.rt_fixtures import (
    make_mixed_triangle_world,
    mixed_triangle_camera,
    weekend_camera,
)
from bajo.core import Point3W, Vec3W
from bajo.parser.number import parse_int_f32
from bajo.rt import (
    Camera,
    Color,
    CpuScene,
    Integrator,
    RenderSettings,
    SceneBuilder,
)
from bajo.rt.cpu import render_wavefront
from bajo.benchmark.image_dump import write_linear
from examples.cornell_box import make_cornell_world
from examples.rtiaw import make_weekend_world


def _coated_world() raises -> CpuScene[16, 16]:
    var builder = SceneBuilder()
    var ground = builder.add_lambertian(Color(0.5))
    var coat = builder.add_coated_diffuse(Color(0.5, 0.25, 0.12), 0.2, 1.5)
    builder.add_sphere(Point3W(0.0, -100.5, -1.0), 100.0, ground)
    for x in range(-2, 3):
        builder.add_sphere(Point3W(Float32(x) * 1.1, 0.0, -2.0), 0.5, coat)
    return CpuScene[16, 16](builder^.finish())


def _bench[
    length: SIMDLength, parallel: Bool
](
    settings: RenderSettings,
    camera: Camera,
    world: CpuScene[16, 16],
    repeats: Int,
    output: String,
) raises:
    comptime chunk = 1024 if parallel else 8192
    var warmup = render_wavefront[.MIS, length, chunk, parallel](
        settings, camera, world
    )
    var expected = pixel_checksum(warmup.pixels)
    write_linear(output, warmup.pixels)
    var times = List[Int](capacity=repeats)
    for i in range(repeats):
        var result = render_wavefront[.MIS, length, chunk, parallel](
            settings, camera, world
        )
        if pixel_checksum(result.pixels) != expected:
            raise Error("CPU image changed between benchmark runs")
        times.append(result.timings.render_ns)
        print("TRIAL", i, Float64(result.timings.render_ns) * 1.0e-6)
    sort(times)
    print("MEDIAN_MS", Float64(times[(repeats - 1) / 2]) * 1.0e-6)
    print("CHECKSUM", expected)


def main() raises:
    var args = argv()
    if len(args) < 5:
        raise Error(
            "expected scene parallel repeats output.f32 [width height spp"
            " packet_width]"
        )
    var name = String(args[1])
    var parallel = parse_int_f32(StringSpan(args[2]), "parallel") != 0
    var repeats = parse_int_f32(StringSpan(args[3]), "repeats")
    var settings = RenderSettings(320, 180, 4, UInt64(2026), 8)
    if len(args) > 5:
        settings.image_width = parse_int_f32(StringSpan(args[5]), "width")
    if len(args) > 6:
        settings.image_height = parse_int_f32(StringSpan(args[6]), "height")
    if len(args) > 7:
        settings.samples_per_pixel = parse_int_f32(StringSpan(args[7]), "spp")
    var packet_width = 16
    if len(args) > 8:
        packet_width = parse_int_f32(StringSpan(args[8]), "packet_width")
    if repeats < 1 or (packet_width != 4 and packet_width != 16):
        raise Error("repeats must be positive; packet_width must be 4 or 16")
    var camera = weekend_camera()
    var world: CpuScene[16, 16]
    match name:
        case "weekend":
            world = make_weekend_world()
        case "mixed":
            world = make_mixed_triangle_world()
            camera = mixed_triangle_camera(world)
        case "cornell":
            world = make_cornell_world()
            camera = Camera.from_vfov(
                Point3W(0.0, 1.0, 3.2),
                Point3W(0.0, 1.0, -1.0),
                Vec3W(0.0, 1.0, 0.0),
                28.0,
                4.2,
            )
        case "coated":
            world = _coated_world()
            camera = Camera.from_vfov(
                Point3W(0.0, 1.0, 5.0),
                Point3W(0.0, 0.0, -2.0),
                Vec3W(0.0, 1.0, 0.0),
                40.0,
            )
        case _:
            raise Error("unknown benchmark scene")
    print(
        "CONFIG",
        name,
        settings.image_width,
        settings.image_height,
        settings.samples_per_pixel,
        packet_width,
        parallel,
    )
    comptime for length in [4, 16]:
        if packet_width == length:
            if parallel:
                _bench[length, True](
                    settings, camera, world, repeats, String(args[4])
                )
            else:
                _bench[length, False](
                    settings, camera, world, repeats, String(args[4])
                )
