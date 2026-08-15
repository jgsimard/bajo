from std.math import max, round
from std.time import perf_counter_ns

from bajo.bvh.constants import f32_max
from bajo.bvh.camera import Camera
from bajo.bvh.host_utils import compute_bounds
from bajo.core import (
    AABB,
    Affine3f32,
    Frame,
    Point3f32,
    Point3W,
    Rayf32,
    Vec3f32,
    Vec3W,
    normalize,
)
from bajo.core.random import Rng, random_in_unit_disk, random_on_hemisphere
from bajo.core.utils import ns_to_ms
from bajo.rt import (
    RENDER,
    ShadingPoint,
    Camera,
    Color,
    Instance,
    RenderSettings,
    Sphere,
    SurfaceId,
    SurfaceStore,
    World,
    add_triangle,
    add_triangle_instance,
    add_triangle_mesh_instance,
    render_depth_first,
)
from bajo.rt.cpu import sample_bsdf
from bajo.rt.cpu.common import _russian_roulette
from bajo.rt.types import MAT, PRIM
from examples.rtiaw import make_weekend_world


comptime TIMING_WIDTH = 960
comptime TIMING_HEIGHT = 540
comptime TIMING_SPP = 8
comptime TIMING_REPEATS = 7
comptime COUNTER_WIDTH = 160
comptime COUNTER_HEIGHT = 90
comptime COUNTER_SPP = 2
comptime MAX_DEPTH = 8
comptime RNG_SEED = UInt64(2026)
comptime TRIANGLE_GRID = 64


@fieldwise_init
struct TimingResult(Copyable):
    var median_total_ns: Int
    var median_init_ns: Int
    var median_render_ns: Int
    var min_render_ns: Int
    var max_render_ns: Int
    var checksum: Float64


struct TraceCounters:
    var primary_paths: Int
    var primary_hits: Int
    var rays: Int
    var hits: Int
    var misses: Int
    var sphere_hits: Int
    var triangle_hits: Int
    var instance_hits: Int
    var lambertian_hits: Int
    var metal_hits: Int
    var dielectric_hits: Int
    var escaped_paths: Int
    var absorbed_paths: Int
    var roulette_paths: Int
    var max_depth_paths: Int
    var shadow_rays: Int
    var shadow_occluded: Int
    var bounce_rays: List[Int]

    def __init__(out self):
        self.primary_paths = 0
        self.primary_hits = 0
        self.rays = 0
        self.hits = 0
        self.misses = 0
        self.sphere_hits = 0
        self.triangle_hits = 0
        self.instance_hits = 0
        self.lambertian_hits = 0
        self.metal_hits = 0
        self.dielectric_hits = 0
        self.escaped_paths = 0
        self.absorbed_paths = 0
        self.roulette_paths = 0
        self.max_depth_paths = 0
        self.shadow_rays = 0
        self.shadow_occluded = 0
        self.bounce_rays = List[Int](length=MAX_DEPTH, fill=0)


def ratio(numerator: Int, denominator: Int) -> Float64:
    if denominator == 0:
        return 0.0
    return Float64(numerator) / Float64(denominator)


def pixel_checksum(pixels: List[Vec3W]) -> Float64:
    var checksum = 0.0
    for i in range(len(pixels)):
        ref p = pixels[i]
        var weight = Float64((i % 251) + 1)
        checksum += weight * (
            Float64(p.x) + 2.0 * Float64(p.y) + 3.0 * Float64(p.z)
        )
    return checksum


def sort_timings(mut values: List[Int]):
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[j] < values[i]:
                var tmp = values[i]
                values[i] = values[j]
                values[j] = tmp


def time_render[
    ALGORITHM: RENDER
](settings: RenderSettings, camera: Camera, world: World) -> TimingResult:
    # Warm all compiled code and renderer-owned allocations before measuring.
    var warmup = render_depth_first[ALGORITHM, MAX_DEPTH](
        settings, camera, world
    )
    var checksum = pixel_checksum(warmup.pixels)
    var total_times = List[Int](capacity=TIMING_REPEATS)
    var init_times = List[Int](capacity=TIMING_REPEATS)
    var render_times = List[Int](capacity=TIMING_REPEATS)

    for _ in range(TIMING_REPEATS):
        var result = render_depth_first[ALGORITHM, MAX_DEPTH](
            settings, camera, world
        )
        var current_checksum = pixel_checksum(result.pixels)
        debug_assert["safe", _use_compiler_assume=True](
            current_checksum == checksum, "render checksum changed between runs"
        )
        total_times.append(result.timings.total_ns)
        init_times.append(result.timings.init_ns)
        render_times.append(result.timings.render_ns)

    sort_timings(total_times)
    sort_timings(init_times)
    sort_timings(render_times)
    var middle = (TIMING_REPEATS - 1) >> 1
    return TimingResult(
        total_times[middle],
        init_times[middle],
        render_times[middle],
        render_times[0],
        render_times[TIMING_REPEATS - 1],
        checksum,
    )


def print_timing(label: String, result: TimingResult, sample_count: Int):
    var primary_msamples_s = (
        Float64(sample_count) / Float64(result.median_render_ns) * 1.0e3
    )
    var spread_percent = 100.0 * (
        Float64(result.max_render_ns - result.min_render_ns)
        / Float64(result.median_render_ns)
    )
    print(
        t"  {label}: {round(ns_to_ms(result.median_total_ns), 3)} ms total, "
        t"{round(ns_to_ms(result.median_render_ns), 3)} ms kernel, "
        t"{round(primary_msamples_s, 3)} Mprimary/s, "
        t"{round(ns_to_ms(result.median_init_ns), 3)} ms init"
    )
    print(
        t"    kernel min..max: {round(ns_to_ms(result.min_render_ns), 3)}.."
        t"{round(ns_to_ms(result.max_render_ns), 3)} ms "
        t"({round(spread_percent, 2)}% span), "
        t"checksum={round(result.checksum, 3)}"
    )


def make_primary_ray(
    settings: RenderSettings,
    camera: Camera,
    px: Int,
    py: Int,
    mut rng: Rng,
) -> Rayf32[Frame.WORLD]:
    # Keep this byte-for-byte equivalent to the production depth-first path.
    var lens = random_in_unit_disk[Frame.WORLD](rng)
    return camera.make_ray_sampled(
        px,
        py,
        settings.image_width,
        settings.image_height,
        rng.f32(),
        rng.f32(),
        lens.x,
        lens.y,
        0.001,
    )


def count_hit(mut counters: TraceCounters, primitive: PRIM, material: MAT):
    counters.hits += 1
    if primitive == PRIM.SPHERE:
        counters.sphere_hits += 1
    elif primitive == PRIM.TRIANGLE:
        counters.triangle_hits += 1
    elif primitive == PRIM.TRIANGLE_INSTANCE:
        counters.instance_hits += 1

    if material == MAT.LAMBERTIAN:
        counters.lambertian_hits += 1
    elif material == MAT.METAL:
        counters.metal_hits += 1
    elif material == MAT.DIELECTRIC:
        counters.dielectric_hits += 1


def count_path[
    DEPTH: Int
](settings: RenderSettings, camera: Camera, world: World) -> TraceCounters:
    var counters = TraceCounters()
    for py in range(settings.image_height):
        for px in range(settings.image_width):
            var pixel_idx = py * settings.image_width + px
            var rng = Rng(seed=settings.rng_seed, id=UInt64(pixel_idx))
            for sample_idx in range(settings.samples_per_pixel):
                counters.primary_paths += 1
                var ray = make_primary_ray(settings, camera, px, py, rng)
                var path_id = UInt32(
                    pixel_idx * settings.samples_per_pixel + sample_idx
                )
                var throughput = Color(1.0)
                var terminated = False
                for bounce in range(DEPTH):
                    counters.rays += 1
                    counters.bounce_rays[bounce] += 1
                    var hit = world.trace(ray)
                    if not hit:
                        counters.misses += 1
                        counters.escaped_paths += 1
                        terminated = True
                        break

                    ref record = hit.value()
                    if bounce == 0:
                        counters.primary_hits += 1
                    count_hit(
                        counters,
                        record.primitive.kind(),
                        record.surface.kind(),
                    )
                    var scattered = sample_bsdf(
                        record.surface,
                        world.surfaces,
                        ray,
                        ShadingPoint(
                            record.p, record.normal, record.front_face
                        ),
                        rng,
                    )
                    if not scattered.ok:
                        counters.absorbed_paths += 1
                        terminated = True
                        break
                    throughput *= scattered.weight
                    var roulette = _russian_roulette(
                        settings,
                        path_id,
                        UInt32(bounce + 1),
                        throughput,
                    )
                    if not roulette.survived:
                        counters.roulette_paths += 1
                        terminated = True
                        break
                    throughput = roulette.throughput
                    ray = Rayf32[Frame.WORLD](
                        record.p, scattered.direction, 0.001, f32_max
                    )

                if not terminated:
                    counters.max_depth_paths += 1

    return counters^


def count_ao(
    settings: RenderSettings, camera: Camera, world: World
) -> TraceCounters:
    var counters = TraceCounters()
    for py in range(settings.image_height):
        for px in range(settings.image_width):
            var pixel_idx = py * settings.image_width + px
            var rng = Rng(seed=settings.rng_seed, id=UInt64(pixel_idx))
            for _sample in range(settings.samples_per_pixel):
                counters.primary_paths += 1
                counters.rays += 1
                counters.bounce_rays[0] += 1
                var ray = make_primary_ray(settings, camera, px, py, rng)
                var hit = world.trace(ray)
                if not hit:
                    counters.misses += 1
                    counters.escaped_paths += 1
                    continue

                ref record = hit.value()
                counters.primary_hits += 1
                count_hit(
                    counters,
                    record.primitive.kind(),
                    record.surface.kind(),
                )
                var ao_dir = random_on_hemisphere[Frame.WORLD](
                    rng, record.normal
                )
                var ao_ray = Rayf32[Frame.WORLD](
                    record.p, normalize(ao_dir), 0.001, 4.0
                )
                counters.shadow_rays += 1
                if world.occluded(ao_ray):
                    counters.shadow_occluded += 1

    return counters^


def print_common_counters(label: String, counters: TraceCounters):
    print(t"  {label}")
    print(
        t"    primary={counters.primary_paths}, traced rays={counters.rays}, "
        t"rays/primary={round(ratio(counters.rays, counters.primary_paths), 3)}"
    )
    print(
        t"    primary hit"
        t" rate={round(100.0 * ratio(counters.primary_hits, counters.primary_paths), 2)}%,"
        t" hits={counters.primary_hits}"
    )
    print(
        t"    trace hit"
        t" rate={round(100.0 * ratio(counters.hits, counters.rays), 2)}%,"
        t" misses={counters.misses}"
    )
    print(
        t"    geometry hits: sphere={counters.sphere_hits}, "
        t"standalone triangle={counters.triangle_hits}, "
        t"instance triangle={counters.instance_hits}"
    )
    print(
        t"    material hits: diffuse={counters.lambertian_hits}, "
        t"metal={counters.metal_hits}, glass={counters.dielectric_hits}"
    )


def print_path_counters(label: String, counters: TraceCounters):
    print_common_counters(label, counters)
    print(
        t"    termination: escaped={counters.escaped_paths},"
        t" absorbed={counters.absorbed_paths},"
        t" roulette={counters.roulette_paths},"
        t" depth-limit={counters.max_depth_paths}"
    )
    print(
        t"    rays by bounce: {counters.bounce_rays[0]}, "
        t"{counters.bounce_rays[1]}, {counters.bounce_rays[2]}, "
        t"{counters.bounce_rays[3]}, {counters.bounce_rays[4]}, "
        t"{counters.bounce_rays[5]}, {counters.bounce_rays[6]}, "
        t"{counters.bounce_rays[7]}"
    )


def print_ao_counters(label: String, counters: TraceCounters):
    print_common_counters(label, counters)
    print(
        t"    AO shadow rays={counters.shadow_rays}, "
        t"occluded={counters.shadow_occluded} "
        t"({round(100.0 * ratio(counters.shadow_occluded, counters.shadow_rays), 2)}%)"
    )


def make_triangle_mesh() -> List[Point3f32[Frame.LOCAL]]:
    var vertices = List[Point3f32[Frame.LOCAL]](
        capacity=TRIANGLE_GRID * TRIANGLE_GRID * 6
    )
    var inv_grid = 1.0 / Float32(TRIANGLE_GRID)
    for z in range(TRIANGLE_GRID):
        for x in range(TRIANGLE_GRID):
            var x0 = -0.9 + 1.8 * Float32(x) * inv_grid
            var x1 = -0.9 + 1.8 * Float32(x + 1) * inv_grid
            var z0 = -0.9 + 1.8 * Float32(z) * inv_grid
            var z1 = -0.9 + 1.8 * Float32(z + 1) * inv_grid
            var y00 = Float32(0.08) if (x + z) % 7 == 0 else Float32(0.0)
            var y10 = Float32(0.08) if (x + 1 + z) % 7 == 0 else Float32(0.0)
            var y01 = Float32(0.08) if (x + z + 1) % 7 == 0 else Float32(0.0)
            var y11 = Float32(0.08) if (x + z + 2) % 7 == 0 else Float32(0.0)
            var p00 = Point3f32[Frame.LOCAL](x0, y00, z0)
            var p10 = Point3f32[Frame.LOCAL](x1, y10, z0)
            var p01 = Point3f32[Frame.LOCAL](x0, y01, z1)
            var p11 = Point3f32[Frame.LOCAL](x1, y11, z1)
            vertices.append(p00)
            vertices.append(p11)
            vertices.append(p10)
            vertices.append(p00)
            vertices.append(p01)
            vertices.append(p11)
    return vertices^


def make_triangle_world() -> World:
    var surfaces = SurfaceStore()
    var diffuse = surfaces.add_lambertian(Color(0.55, 0.32, 0.18))
    var ground = surfaces.add_lambertian(Color(0.35, 0.38, 0.32))
    var metal = surfaces.add_metal(Color(0.75, 0.78, 0.82), 0.12)
    var glass = surfaces.add_dielectric(1.45)
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()

    add_triangle(
        triangle_vertices,
        triangle_surfaces,
        Point3W(-7.0, -0.2, -7.0),
        Point3W(7.0, -0.2, 7.0),
        Point3W(7.0, -0.2, -7.0),
        ground,
    )
    add_triangle(
        triangle_vertices,
        triangle_surfaces,
        Point3W(-7.0, -0.2, -7.0),
        Point3W(-7.0, -0.2, 7.0),
        Point3W(7.0, -0.2, 7.0),
        ground,
    )

    var mesh = make_triangle_mesh()
    var mesh_bounds = compute_bounds(mesh)
    var first_transform = Affine3f32[Frame.LOCAL, Frame.WORLD].from_translation(
        Vec3f32[Frame.WORLD](-4.0, 0.0, -4.0)
    )
    var mesh_idx = add_triangle_mesh_instance(
        triangle_meshes,
        triangle_instances,
        triangle_instance_surfaces,
        mesh,
        first_transform,
        mesh_bounds,
        diffuse,
    )
    for iz in range(5):
        for ix in range(5):
            if ix == 0 and iz == 0:
                continue
            var transform = Affine3f32[
                Frame.LOCAL, Frame.WORLD
            ].from_translation(
                Vec3f32[Frame.WORLD](
                    Float32(ix) * 2.0 - 4.0,
                    0.0,
                    Float32(iz) * 2.0 - 4.0,
                )
            )
            var selector = (ix + 2 * iz) % 5
            var surface = diffuse.copy()
            if selector == 1:
                surface = metal.copy()
            elif selector == 2:
                surface = glass.copy()
            add_triangle_instance(
                triangle_instances,
                triangle_instance_surfaces,
                mesh_idx,
                transform,
                mesh_bounds,
                surface,
            )

    return World(
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )


def triangle_camera(world: World) -> Camera:
    var bounds = AABB[Frame.WORLD].invalid()
    for inst in world.triangle_instances:
        bounds.grow(inst.bounds)
    var center = bounds.centroid()
    var extent = bounds.extent()
    var scene_w = max(extent.x, extent.z)
    if scene_w < 1.0:
        scene_w = 1.0
    var eye = Point3W(center.x, center.y + scene_w * 0.78, center.z + scene_w)
    var target = Point3W(center.x, center.y, center.z)
    return Camera.from_vfov(eye, target, Vec3W(0.0, 1.0, 0.0), 44.0)


def main() raises:
    print("CPU ray tracer end-to-end benchmark and deterministic counters")
    print(
        t"timing: {TIMING_WIDTH}x{TIMING_HEIGHT} x {TIMING_SPP} spp, "
        t"depth {MAX_DEPTH}, median of {TIMING_REPEATS} after warmup"
    )
    print(
        t"counters: {COUNTER_WIDTH}x{COUNTER_HEIGHT} x {COUNTER_SPP} spp, "
        t"single-thread deterministic replay"
    )

    var sphere_build_t0 = perf_counter_ns()
    var sphere_world = make_weekend_world()
    var sphere_build_ns = Int(perf_counter_ns() - sphere_build_t0)
    var sphere_camera = Camera.from_vfov(
        Point3W(13.0, 2.0, 3.0),
        Point3W(0.0, 0.0, 0.0),
        Vec3W(0.0, 1.0, 0.0),
        20.0,
        10.0,
        0.6,
    )

    var triangle_build_t0 = perf_counter_ns()
    var triangle_world = make_triangle_world()
    var triangle_build_ns = Int(perf_counter_ns() - triangle_build_t0)
    var triangle_cam = triangle_camera(triangle_world)

    var timing_settings = RenderSettings(
        TIMING_WIDTH, TIMING_HEIGHT, TIMING_SPP, RNG_SEED
    )
    var sample_count = TIMING_WIDTH * TIMING_HEIGHT * TIMING_SPP
    print("\nEnd-to-end production renderer timings")
    print(
        t"  scene build: spheres={round(ns_to_ms(sphere_build_ns), 3)} ms, "
        t"mixed triangles={round(ns_to_ms(triangle_build_ns), 3)} ms"
    )
    print_timing(
        "sphere / path",
        time_render[RENDER.PATH](timing_settings, sphere_camera, sphere_world),
        sample_count,
    )
    print_timing(
        "sphere / AO",
        time_render[RENDER.AO](timing_settings, sphere_camera, sphere_world),
        sample_count,
    )
    print_timing(
        "sphere / normals",
        time_render[RENDER.NORMALS](
            timing_settings, sphere_camera, sphere_world
        ),
        sample_count,
    )
    print_timing(
        "triangles / path",
        time_render[RENDER.PATH](timing_settings, triangle_cam, triangle_world),
        sample_count,
    )
    print_timing(
        "triangles / AO",
        time_render[RENDER.AO](timing_settings, triangle_cam, triangle_world),
        sample_count,
    )
    print_timing(
        "triangles / normals",
        time_render[RENDER.NORMALS](
            timing_settings, triangle_cam, triangle_world
        ),
        sample_count,
    )

    var counter_settings = RenderSettings(
        COUNTER_WIDTH, COUNTER_HEIGHT, COUNTER_SPP, RNG_SEED
    )
    print("\nDeterministic workload counters (not included in timings)")
    print_path_counters(
        "sphere / path",
        count_path[MAX_DEPTH](counter_settings, sphere_camera, sphere_world),
    )
    print_ao_counters(
        "sphere / AO",
        count_ao(counter_settings, sphere_camera, sphere_world),
    )
    print_path_counters(
        "triangles / path",
        count_path[MAX_DEPTH](counter_settings, triangle_cam, triangle_world),
    )
    print_ao_counters(
        "triangles / AO",
        count_ao(counter_settings, triangle_cam, triangle_world),
    )
