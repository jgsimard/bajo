from std.math import max, round
from std.time import perf_counter_ns

from bajo.bvh.constants import f32_max
from bajo.core import (
    Frame,
    Rayf32,
    normalize,
)
from bajo.core.random import Rng, random_in_unit_disk, random_on_hemisphere
from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Integrator,
    ShadingPoint,
    Camera,
    Color,
    RenderSettings,
    CpuScene,
    render_depth_first,
)
from bajo.rt.cpu import sample_bsdf
from bajo.rt.common import russian_roulette
from bajo.rt.types import MaterialKind, PrimitiveKind
from examples.rtiaw import make_weekend_world
from bajo.benchmark.cpu_harness import pixel_checksum
from bajo.benchmark.rt_fixtures import (
    make_mixed_triangle_world,
    mixed_triangle_camera,
    weekend_camera,
)
from bajo.benchmark.timing import ratio


comptime TIMING_WIDTH = 960
comptime TIMING_HEIGHT = 540
comptime TIMING_SPP = 8
comptime TIMING_REPEATS = 7
comptime COUNTER_WIDTH = 160
comptime COUNTER_HEIGHT = 90
comptime COUNTER_SPP = 2
comptime MAX_DEPTH = 8
comptime RNG_SEED = UInt64(2026)


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


def time_render[
    ALGORITHM: Integrator
](settings: RenderSettings, camera: Camera, world: CpuScene[]) -> TimingResult:
    # Warm all compiled code and renderer-owned allocations before measuring.
    var warmup = render_depth_first[ALGORITHM](settings, camera, world)
    var checksum = pixel_checksum(warmup.pixels)
    var total_times = List[Int](capacity=TIMING_REPEATS)
    var init_times = List[Int](capacity=TIMING_REPEATS)
    var render_times = List[Int](capacity=TIMING_REPEATS)

    for _ in range(TIMING_REPEATS):
        var result = render_depth_first[ALGORITHM](settings, camera, world)
        var current_checksum = pixel_checksum(result.pixels)
        debug_assert["safe", _use_compiler_assume=True](
            current_checksum == checksum, "render checksum changed between runs"
        )
        total_times.append(result.timings.total_ns)
        init_times.append(result.timings.init_ns)
        render_times.append(result.timings.render_ns)

    sort(total_times)
    sort(init_times)
    sort(render_times)
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
) -> Rayf32[.WORLD]:
    # Keep this byte-for-byte equivalent to the production depth-first path.
    var lens = random_in_unit_disk[.WORLD](rng)
    return camera.make_ray_sampled(
        Float32(px),
        Float32(py),
        Float32(settings.image_width),
        Float32(settings.image_height),
        rng.f32(),
        rng.f32(),
        lens.x,
        lens.y,
        0.001,
    )


def count_hit(mut counters: TraceCounters, primitive: PrimitiveKind, material: MaterialKind):
    counters.hits += 1
    if primitive == PrimitiveKind.SPHERE:
        counters.sphere_hits += 1
    elif primitive == PrimitiveKind.TRIANGLE:
        counters.triangle_hits += 1
    elif primitive == PrimitiveKind.TRIANGLE_INSTANCE:
        counters.instance_hits += 1

    if material == .LAMBERTIAN:
        counters.lambertian_hits += 1
    elif material == .METAL:
        counters.metal_hits += 1
    elif material == .DIELECTRIC:
        counters.dielectric_hits += 1


def count_path[
    DEPTH: Int
](settings: RenderSettings, camera: Camera, world: CpuScene[]) -> TraceCounters:
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
                        world.scene_data().surfaces(),
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
                    var roulette = russian_roulette(
                        settings.rng_seed,
                        path_id,
                        UInt32(bounce + 1),
                        throughput,
                    )
                    if not roulette.survived:
                        counters.roulette_paths += 1
                        terminated = True
                        break
                    throughput = roulette.throughput
                    ray = Rayf32[.WORLD](
                        record.p, scattered.direction, 0.001, f32_max
                    )

                if not terminated:
                    counters.max_depth_paths += 1

    return counters^


def count_ao(
    settings: RenderSettings, camera: Camera, world: CpuScene[]
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
                var ao_dir = random_on_hemisphere[.WORLD](
                    rng, record.normal
                )
                var ao_ray = Rayf32[.WORLD](
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


def run_benchmark() raises:
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
    var sphere_camera = weekend_camera()

    var triangle_build_t0 = perf_counter_ns()
    var triangle_world = make_mixed_triangle_world()
    var triangle_build_ns = Int(perf_counter_ns() - triangle_build_t0)
    var triangle_cam = mixed_triangle_camera(triangle_world)

    var timing_settings = RenderSettings(
        TIMING_WIDTH, TIMING_HEIGHT, TIMING_SPP, RNG_SEED, MAX_DEPTH
    )
    var sample_count = TIMING_WIDTH * TIMING_HEIGHT * TIMING_SPP
    print("\nEnd-to-end production renderer timings")
    print(
        t"  scene build: spheres={round(ns_to_ms(sphere_build_ns), 3)} ms, "
        t"mixed triangles={round(ns_to_ms(triangle_build_ns), 3)} ms"
    )
    print_timing(
        "sphere / path",
        time_render[.PATH](timing_settings, sphere_camera, sphere_world),
        sample_count,
    )
    print_timing(
        "sphere / AO",
        time_render[.AO](timing_settings, sphere_camera, sphere_world),
        sample_count,
    )
    print_timing(
        "sphere / normals",
        time_render[.NORMALS](
            timing_settings, sphere_camera, sphere_world
        ),
        sample_count,
    )
    print_timing(
        "triangles / path",
        time_render[.PATH](timing_settings, triangle_cam, triangle_world),
        sample_count,
    )
    print_timing(
        "triangles / AO",
        time_render[.AO](timing_settings, triangle_cam, triangle_world),
        sample_count,
    )
    print_timing(
        "triangles / normals",
        time_render[.NORMALS](
            timing_settings, triangle_cam, triangle_world
        ),
        sample_count,
    )

    var counter_settings = RenderSettings(
        COUNTER_WIDTH, COUNTER_HEIGHT, COUNTER_SPP, RNG_SEED, MAX_DEPTH
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


def main() raises:
    run_benchmark()
