from std.math import max, round
from std.sys import argv
from std.time import perf_counter_ns

from bajo.bvh.constants import TRACE
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.core import (
    AABB,
    Frame,
    Point3f32,
    Rayf32,
    Vec3f32,
    cross,
    dot,
    normalize,
)
from bajo.obj.mmap import MMap


comptime RAY_BATCH_SIZE = 1024 * 1024
comptime SCREEN_SIZE = 1024
comptime TINYBVH_FAR = Float32(1.0e30)
comptime MIN_RUNS = 5
comptime MIN_TRACE_NS = Int(1_500_000_000)


@fieldwise_init
struct ViewPyramid(Copyable):
    var eye: Point3f32[Frame.WORLD]
    var p1: Point3f32[Frame.WORLD]
    var p2: Point3f32[Frame.WORLD]
    var p3: Point3f32[Frame.WORLD]


@fieldwise_init
struct TraceSummary(Copyable):
    var checksum: Float64
    var hits: Int


@fieldwise_init
struct TraceMeasurement(Copyable):
    var elapsed_ns: Int
    var runs: Int
    var checksum: Float64
    var hits: Int


struct Xor32:
    var state: UInt32

    def __init__(out self):
        self.state = UInt32(0x12345678)

    def random_uint(mut self) -> UInt32:
        self.state ^= self.state << 13
        self.state ^= self.state >> 17
        self.state ^= self.state << 5
        return self.state

    def random_float(mut self) -> Float32:
        return Float32(self.random_uint()) * Float32(2.3283064365387e-10)


def load_tinybvh_triangle_file(
    path: String,
) raises -> List[Point3f32[Frame.WORLD]]:
    var mapped = MMap[ImmutAnyOrigin](path)
    var bytes = mapped.as_bytes_span()
    if len(bytes) < 4:
        raise Error("TinyBVH triangle file is missing its header: " + path)

    var byte_ptr = bytes.unsafe_ptr()
    var triangle_count = Int(byte_ptr.unsafe_bitcast[UInt32]()[unsafe_offset=0])
    var expected_bytes = 4 + triangle_count * 48
    if len(bytes) != expected_bytes:
        raise Error(
            t"TinyBVH triangle file has {len(bytes)} bytes; expected"
            t" {expected_bytes}"
        )

    var data = byte_ptr.unsafe_offset(4).unsafe_bitcast[Float32]()
    var vertices = List[Point3f32[Frame.WORLD]](capacity=triangle_count * 3)
    for triangle in range(triangle_count):
        var base = triangle * 12
        vertices.append(
            Point3f32[Frame.WORLD](
                data[unsafe_offset=base + 0],
                data[unsafe_offset=base + 1],
                data[unsafe_offset=base + 2],
            )
        )
        vertices.append(
            Point3f32[Frame.WORLD](
                data[unsafe_offset=base + 4],
                data[unsafe_offset=base + 5],
                data[unsafe_offset=base + 6],
            )
        )
        vertices.append(
            Point3f32[Frame.WORLD](
                data[unsafe_offset=base + 8],
                data[unsafe_offset=base + 9],
                data[unsafe_offset=base + 10],
            )
        )
    return vertices^


def dragon_camera(
    view: Int,
) -> Tuple[Point3f32[Frame.WORLD], Vec3f32[Frame.WORLD]]:
    if view == 0:
        var eye = Point3f32[Frame.WORLD](-4.3, 3.2, -5.8)
        var target = Point3f32[Frame.WORLD](-0.39, 2.09, -2.48)
        return (eye, normalize(target - eye))
    if view == 1:
        var eye = Point3f32[Frame.WORLD](-4.8, 2.1, -7.2)
        var target = Point3f32[Frame.WORLD](-0.53, 0.80, -1.88)
        return (eye, normalize(target - eye))

    var eye = Point3f32[Frame.WORLD](-0.7, 4.9, 0.4)
    var target = Point3f32[Frame.WORLD](0.13, 3.28, -1.36)
    return (eye, normalize(target - eye))


def make_view_pyramid_from_camera(
    eye: Point3f32[Frame.WORLD],
    direction: Vec3f32[Frame.WORLD],
) -> ViewPyramid:
    var right = normalize(cross(Vec3f32[Frame.WORLD](0.0, 1.0, 0.0), direction))
    var up = cross(direction, right) * 0.8
    var center = eye + direction * 2.0
    return ViewPyramid(
        eye,
        center - right + up,
        center + right + up,
        center - right - up,
    )


def make_view_pyramid(view: Int) -> ViewPyramid:
    var camera = dragon_camera(view)
    return make_view_pyramid_from_camera(camera[0], camera[1])


def pyramid_ray(
    pyramid: ViewPyramid,
    x: Int,
    y: Int,
) -> Rayf32[Frame.WORLD]:
    var uv_scale = Float32(1.0 / Float32(SCREEN_SIZE))
    var u = Float32(x) * uv_scale
    var v = Float32(y) * uv_scale
    var point = (
        pyramid.p1
        + (pyramid.p2 - pyramid.p1) * u
        + (pyramid.p3 - pyramid.p1) * v
    )
    return Rayf32[Frame.WORLD](
        pyramid.eye,
        normalize(point - pyramid.eye),
        0.0,
        TINYBVH_FAR,
    )


def make_primary_rays_from_pyramid(
    pyramid: ViewPyramid,
) -> List[Rayf32[Frame.WORLD]]:
    # TinyBVH stores the square image as consecutive 4x4 tiles.
    var tiles_per_row = SCREEN_SIZE // 4
    var rays = List[Rayf32[Frame.WORLD]](capacity=RAY_BATCH_SIZE)
    for tile_y in range(tiles_per_row):
        for tile_x in range(tiles_per_row):
            for y in range(4):
                for x in range(4):
                    rays.append(
                        pyramid_ray(
                            pyramid,
                            x + tile_x * 4,
                            y + tile_y * 4,
                        )
                    )
    return rays^


def make_primary_rays(view: Int) -> List[Rayf32[Frame.WORLD]]:
    return make_primary_rays_from_pyramid(make_view_pyramid(view))


def diffuse_reflection(
    mut rng: Xor32,
    normal: Vec3f32[Frame.WORLD],
) -> Vec3f32[Frame.WORLD]:
    while True:
        var candidate = Vec3f32[Frame.WORLD](
            rng.random_float() * 2.0 - 1.0,
            rng.random_float() * 2.0 - 1.0,
            rng.random_float() * 2.0 - 1.0,
        )
        if dot(candidate, candidate) <= 1.0:
            if dot(candidate, normal) < 0.0:
                candidate = -candidate
            return normalize(candidate)


def hit_normal(
    vertices: List[Point3f32[Frame.WORLD]],
    primitive: UInt32,
) -> Vec3f32[Frame.WORLD]:
    var base = Int(primitive) * 3
    ref a = vertices[base + 0]
    ref b = vertices[base + 1]
    ref c = vertices[base + 2]
    # Match TinyBVH's benchmark orientation exactly.
    return normalize(cross(b - a, a - c))


def make_first_bounce_rays[
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
](
    bvh: TriangleBvh[Frame.WORLD, bounds_width, leaf_width],
    vertices: List[Point3f32[Frame.WORLD]],
    bounds: AABB[Frame.WORLD],
    mut rng: Xor32,
) -> List[Rayf32[Frame.WORLD]]:
    var pyramid = make_view_pyramid(0)
    var extent = bounds.extent()
    var scene_size = max(max(extent.x, extent.y), extent.z)
    var epsilon = scene_size * 1.0e-6
    var rays = List[Rayf32[Frame.WORLD]](capacity=RAY_BATCH_SIZE)

    while len(rays) < RAY_BATCH_SIZE:
        for y in range(0, SCREEN_SIZE, 3):
            for x in range(0, SCREEN_SIZE, 3):
                if len(rays) == RAY_BATCH_SIZE:
                    break
                var primary = pyramid_ray(pyramid, x, y)
                var hit = bvh.trace[TRACE.CLOSEST_HIT](primary)
                if not hit.is_hit():
                    continue

                var normal = hit_normal(vertices, hit.prim)
                if dot(primary.d, normal) > 0.0:
                    normal = -normal
                var intersection = primary.o + primary.d * hit.t
                var direction = diffuse_reflection(rng, normal)
                rays.append(
                    Rayf32[Frame.WORLD](
                        intersection + normal * epsilon,
                        direction,
                        0.0,
                        TINYBVH_FAR,
                    )
                )
    return rays^


def make_ao_rays[
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
](
    bvh: TriangleBvh[Frame.WORLD, bounds_width, leaf_width],
    vertices: List[Point3f32[Frame.WORLD]],
    bounds: AABB[Frame.WORLD],
    mut rng: Xor32,
) -> List[Rayf32[Frame.WORLD]]:
    # RaySet::AO_RAYS is enum value 5; TinyBVH selects camera 5 % 3 = 2.
    var pyramid = make_view_pyramid(2)
    var extent = bounds.extent()
    var scene_size = max(max(extent.x, extent.y), extent.z)
    var epsilon = scene_size * 1.0e-6
    var rays = List[Rayf32[Frame.WORLD]](capacity=RAY_BATCH_SIZE)

    while len(rays) < RAY_BATCH_SIZE:
        for y in range(0, SCREEN_SIZE, 4):
            for x in range(0, SCREEN_SIZE, 4):
                if len(rays) == RAY_BATCH_SIZE:
                    break
                var primary = pyramid_ray(pyramid, x, y)
                var hit = bvh.trace[TRACE.CLOSEST_HIT](primary)
                if not hit.is_hit():
                    continue

                var normal = hit_normal(vertices, hit.prim)
                if dot(primary.d, normal) > 0.0:
                    normal = -normal
                var origin = primary.o + primary.d * hit.t + normal * epsilon
                for _ in range(4):
                    if len(rays) == RAY_BATCH_SIZE:
                        break
                    rays.append(
                        Rayf32[Frame.WORLD](
                            origin,
                            diffuse_reflection(rng, normal),
                            0.0,
                            scene_size * 0.05,
                        )
                    )
    return rays^


def trace_once[
    mode: TRACE,
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
](
    bvh: TriangleBvh[Frame.WORLD, bounds_width, leaf_width],
    rays: List[Rayf32[Frame.WORLD]],
) -> TraceSummary:
    var checksum = 0.0
    var hits = 0
    for ray in rays:
        var hit = bvh.trace[mode](ray)
        comptime if mode == TRACE.CLOSEST_HIT:
            if hit.is_hit():
                checksum += (
                    Float64(hit.t)
                    + Float64(hit.u)
                    + Float64(hit.v)
                    + Float64(hit.prim)
                )
                hits += 1
        else:
            # A miss retains this ray's finite t_max; shadow hits return t=0.
            if hit.t < ray.t_max:
                hits += 1
    return TraceSummary(checksum, hits)


def measure_trace[
    mode: TRACE,
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
](
    bvh: TriangleBvh[Frame.WORLD, bounds_width, leaf_width],
    rays: List[Rayf32[Frame.WORLD]],
) -> TraceMeasurement:
    var summary = trace_once[mode](bvh, rays)
    var runs = 0
    var start = perf_counter_ns()
    var elapsed = 0
    while runs < MIN_RUNS or elapsed < MIN_TRACE_NS:
        summary = trace_once[mode](bvh, rays)
        runs += 1
        elapsed = Int(perf_counter_ns() - start)
    return TraceMeasurement(elapsed, runs, summary.checksum, summary.hits)


def print_trace_experiment[
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
](
    bvh: TriangleBvh[Frame.WORLD, bounds_width, leaf_width],
    rays: List[Rayf32[Frame.WORLD]],
    description: String,
    short_name: String,
    triangle_count: Int,
):
    var closest = measure_trace[TRACE.CLOSEST_HIT](bvh, rays)
    var any_hit = measure_trace[TRACE.ANY_HIT](bvh, rays)
    var closest_rate = (
        Float64(len(rays) * closest.runs)
        / (Float64(closest.elapsed_ns) * 1.0e-9)
        / 1.0e6
    )
    var any_rate = (
        Float64(len(rays) * any_hit.runs)
        / (Float64(any_hit.elapsed_ns) * 1.0e-9)
        / 1.0e6
    )

    print(
        t"BVH TRACE - Bajo SAH BVH{Int(bounds_width)}/leaf{Int(leaf_width)} -"
        t" Stanford Dragon ({triangle_count // 1000}k tris) - {description}"
    )
    print(
        "find nearest:",
        round(closest_rate, 2),
        "M, any hit:",
        round(any_rate, 2),
        "M",
    )
    print(
        "csv: cpu,dragon,",
        triangle_count,
        t",bajo-bvh{Int(bounds_width)}-leaf{Int(leaf_width)},sah,{short_name},",
        round(closest_rate, 2),
        ",",
        closest.runs,
        ",",
        round(any_rate, 2),
        ",",
        any_hit.runs,
        ",M",
    )
    print(
        "validation: closest_hits=",
        closest.hits,
        "closest_checksum=",
        closest.checksum,
        "any_hits=",
        any_hit.hits,
    )


def main() raises:
    var args = argv()
    var scene_path = "external/tinybvh/testdata/dragon.bin"
    if len(args) > 1:
        scene_path = args[1]

    print("BAJO ADDITION TO THE TINY_BVH BENCHMARK TOOL")
    print("methodology: 1,048,576 rays; warm-up; >=5 runs and >=1.5 s")
    print("scene:", scene_path)

    var vertices = load_tinybvh_triangle_file(scene_path)
    var bvh = TriangleBvh[Frame.WORLD, 16, 16].__init__["sah"](vertices)
    var bounds = bvh.bounds()
    var rng = Xor32()
    var triangle_count = len(vertices) // 3

    for view in range(3):
        var primary = make_primary_rays(view)
        print_trace_experiment(
            bvh,
            primary,
            String(t"cam{view + 1} primary"),
            String(t"cam{view + 1}pri"),
            triangle_count,
        )

    var first_bounce = make_first_bounce_rays(bvh, vertices, bounds, rng)
    print_trace_experiment(
        bvh,
        first_bounce,
        "first bounce",
        "1 bounce",
        triangle_count,
    )

    var ao = make_ao_rays(bvh, vertices, bounds, rng)
    print_trace_experiment(
        bvh,
        ao,
        "AO rays",
        "AO rays",
        triangle_count,
    )
