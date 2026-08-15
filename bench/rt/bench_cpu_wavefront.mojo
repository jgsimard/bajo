"""Controlled serial CPU wavefront timing and workload counters."""

from std.math import round

from bajo.core import Point3W, Vec3W
from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Camera,
    RENDER,
    RenderSettings,
    World,
)
from bajo.rt.cpu import (
    _make_initial_path_packets_range,
    _path_stage_rng,
    _russian_roulette,
    render_wavefront,
    sample_bsdf,
)
from bajo.rt.wavefront_queue import PacketPathQueue, WavePath
from bajo.rt.types import MAT
from bench.rt.bench_cpu_end_to_end import (
    make_triangle_world,
    pixel_checksum,
    sort_timings,
    triangle_camera,
)
from examples.rtiaw import make_weekend_world


comptime TIMING_WIDTH = 320
comptime TIMING_HEIGHT = 180
comptime TIMING_SPP = 4
comptime TIMING_REPEATS = 7
comptime COUNTER_WIDTH = 160
comptime COUNTER_HEIGHT = 90
comptime COUNTER_SPP = 2
comptime MAX_DEPTH = 8
comptime RNG_SEED = UInt64(2026)
comptime CHUNK_PATHS = 8192


@fieldwise_init
struct WaveTiming(Copyable):
    var median_total_ns: Int
    var median_init_ns: Int
    var median_render_ns: Int
    var min_render_ns: Int
    var max_render_ns: Int
    var checksum: Float64


struct WaveCounters:
    var primary_paths: Int
    var rays: Int
    var hits: Int
    var misses: Int
    var escaped: Int
    var absorbed: Int
    var roulette_terminated: Int
    var depth_limited: Int
    var lambertian_hits: Int
    var metal_hits: Int
    var dielectric_hits: Int
    var peak_material_queue: Int
    var active_by_bounce: List[Int]

    def __init__(out self):
        self.primary_paths = 0
        self.rays = 0
        self.hits = 0
        self.misses = 0
        self.escaped = 0
        self.absorbed = 0
        self.roulette_terminated = 0
        self.depth_limited = 0
        self.lambertian_hits = 0
        self.metal_hits = 0
        self.dielectric_hits = 0
        self.peak_material_queue = 0
        self.active_by_bounce = List[Int](capacity=MAX_DEPTH)


def time_wavefront[
    PACKET_LANES: SIMDLength
](settings: RenderSettings, camera: Camera, world: World) -> WaveTiming:
    var warmup = render_wavefront[
        RENDER.PATH, MAX_DEPTH, PACKET_LANES, CHUNK_PATHS, False
    ](settings, camera, world)
    var checksum = pixel_checksum(warmup.pixels)
    var total_times = List[Int](capacity=TIMING_REPEATS)
    var init_times = List[Int](capacity=TIMING_REPEATS)
    var render_times = List[Int](capacity=TIMING_REPEATS)

    for _ in range(TIMING_REPEATS):
        var result = render_wavefront[
            RENDER.PATH, MAX_DEPTH, PACKET_LANES, CHUNK_PATHS, False
        ](settings, camera, world)
        var current_checksum = pixel_checksum(result.pixels)
        debug_assert["safe", _use_compiler_assume=True](
            current_checksum == checksum,
            "wavefront checksum changed between runs",
        )
        total_times.append(result.timings.total_ns)
        init_times.append(result.timings.init_ns)
        render_times.append(result.timings.render_ns)

    sort_timings(total_times)
    sort_timings(init_times)
    sort_timings(render_times)
    var middle = (TIMING_REPEATS - 1) >> 1
    return WaveTiming(
        total_times[middle],
        init_times[middle],
        render_times[middle],
        render_times[0],
        render_times[TIMING_REPEATS - 1],
        checksum,
    )


def count_wavefront(
    settings: RenderSettings, camera: Camera, world: World
) -> WaveCounters:
    var counters = WaveCounters()
    var path_count = (
        settings.image_width
        * settings.image_height
        * settings.samples_per_pixel
    )
    var active_paths = _make_initial_path_packets_range[1](
        settings, camera, 0, path_count
    )
    counters.primary_paths = len(active_paths)

    for _bounce in range(MAX_DEPTH):
        if len(active_paths) == 0:
            break

        counters.active_by_bounce.append(len(active_paths))
        var next_paths = PacketPathQueue[1](len(active_paths))
        var lambertian_queue = 0
        var metal_queue = 0
        var dielectric_queue = 0

        for path_idx in range(len(active_paths)):
            var path = active_paths.get(path_idx)
            counters.rays += 1
            var hit = world.trace(path.ray)
            if not hit:
                counters.misses += 1
                counters.escaped += 1
                continue

            counters.hits += 1
            ref record = hit.value()
            var material_kind = record.surface.kind()
            if material_kind == MAT.LAMBERTIAN:
                counters.lambertian_hits += 1
                lambertian_queue += 1
            elif material_kind == MAT.METAL:
                counters.metal_hits += 1
                metal_queue += 1
            elif material_kind == MAT.DIELECTRIC:
                counters.dielectric_hits += 1
                dielectric_queue += 1

            var rng = _path_stage_rng(
                settings, path.path_id, UInt32(_bounce + 1)
            )
            var scattered = sample_bsdf(
                record.surface,
                world.surfaces,
                path.ray,
                record,
                rng,
            )
            if scattered.ok:
                var roulette = _russian_roulette(
                    settings,
                    path.path_id,
                    UInt32(_bounce + 1),
                    path.throughput * scattered.weight,
                )
                if roulette.survived:
                    next_paths.append(
                        WavePath(
                            path.path_id,
                            scattered.ray,
                            roulette.throughput,
                        )
                    )
                else:
                    counters.roulette_terminated += 1
            else:
                counters.absorbed += 1

        counters.peak_material_queue = max(
            counters.peak_material_queue,
            max(lambertian_queue, max(metal_queue, dielectric_queue)),
        )
        active_paths = next_paths^

    counters.depth_limited = len(active_paths)
    return counters^


def print_timing(label: String, timing: WaveTiming):
    var primary_count = TIMING_WIDTH * TIMING_HEIGHT * TIMING_SPP
    var primary_mray_s = (
        Float64(primary_count) / Float64(timing.median_render_ns) * 1.0e3
    )
    print(t"  {label}")
    print(
        t"    total={round(ns_to_ms(timing.median_total_ns), 3)} ms, "
        t"init={round(ns_to_ms(timing.median_init_ns), 3)} ms, "
        t"render={round(ns_to_ms(timing.median_render_ns), 3)} ms"
    )
    print(
        t"    render range={round(ns_to_ms(timing.min_render_ns), 3)}.."
        t"{round(ns_to_ms(timing.max_render_ns), 3)} ms, "
        t"primary={round(primary_mray_s, 3)} M/s, "
        t"checksum={round(timing.checksum, 3)}"
    )


def print_counters(label: String, counters: WaveCounters):
    print(t"  {label}")
    print(
        t"    primary={counters.primary_paths}, rays={counters.rays}, "
        t"hits={counters.hits}, misses={counters.misses}"
    )
    print(
        t"    materials: diffuse={counters.lambertian_hits}, "
        t"metal={counters.metal_hits}, glass={counters.dielectric_hits}"
    )
    print(
        t"    termination: escaped={counters.escaped},"
        t" absorbed={counters.absorbed},"
        t" roulette={counters.roulette_terminated},"
        t" depth-limit={counters.depth_limited}, peak material"
        t" queue={counters.peak_material_queue}"
    )
    print("    active by bounce:", counters.active_by_bounce)


def benchmark_scene(
    label: String,
    timing_settings: RenderSettings,
    counter_settings: RenderSettings,
    camera: Camera,
    world: World,
):
    print_timing(
        label + " / packet width 1 (scalar)",
        time_wavefront[1](timing_settings, camera, world),
    )
    print_timing(
        label + " / packet width 4",
        time_wavefront[4](timing_settings, camera, world),
    )
    print_timing(
        label + " / packet width 8",
        time_wavefront[8](timing_settings, camera, world),
    )
    print_timing(
        label + " / packet width 16",
        time_wavefront[16](timing_settings, camera, world),
    )
    print_counters(label, count_wavefront(counter_settings, camera, world))


def main():
    print("CPU serial wavefront benchmark")
    print(
        t"timing: {TIMING_WIDTH}x{TIMING_HEIGHT} x {TIMING_SPP} spp, "
        t"depth {MAX_DEPTH}, median of {TIMING_REPEATS} after warmup"
    )
    print(t"counters: {COUNTER_WIDTH}x{COUNTER_HEIGHT} x {COUNTER_SPP} spp")

    var timing_settings = RenderSettings(
        TIMING_WIDTH, TIMING_HEIGHT, TIMING_SPP, RNG_SEED
    )
    var counter_settings = RenderSettings(
        COUNTER_WIDTH, COUNTER_HEIGHT, COUNTER_SPP, RNG_SEED
    )

    var sphere_world = make_weekend_world()
    var sphere_camera = Camera.from_vfov(
        Point3W(13.0, 2.0, 3.0),
        Point3W(0.0, 0.0, 0.0),
        Vec3W(0.0, 1.0, 0.0),
        20.0,
        10.0,
        0.6,
    )
    benchmark_scene(
        "Weekend spheres",
        timing_settings,
        counter_settings,
        sphere_camera,
        sphere_world,
    )

    var triangle_world = make_triangle_world()
    var triangle_cam = triangle_camera(triangle_world)
    benchmark_scene(
        "Mixed triangles",
        timing_settings,
        counter_settings,
        triangle_cam,
        triangle_world,
    )
