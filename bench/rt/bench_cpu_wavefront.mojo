"""Controlled serial CPU wavefront timing and workload counters."""

from std.math import round

from bajo.bvh.constants import f32_max
from bajo.core import Frame, Point3W, Rayf32, Vec3W
from bajo.core.utils import ns_to_ms
from bajo.rt import (
    Camera,
    Color,
    Integrator,
    RenderSettings,
    ShadingPoint,
    CpuScene,
)
from bajo.rt.cpu import render_wavefront, sample_bsdf
from bajo.rt.common import path_stage_rng, russian_roulette
from bajo.rt.cpu.wavefront.primary import _initialize_path_packets_range
from bajo.rt.wavefront_queue import PacketPathQueue, PathPacket
from bajo.benchmark.cpu_harness import pixel_checksum
from bajo.benchmark.rt_fixtures import (
    make_mixed_triangle_world,
    mixed_triangle_camera,
    weekend_camera,
)
from examples.cornell_box import make_cornell_world
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
    length: SIMDLength, ALGORITHM: Integrator = .PATH
](settings: RenderSettings, camera: Camera, world: CpuScene[]) -> WaveTiming:
    var warmup = render_wavefront[ALGORITHM, length, CHUNK_PATHS, False](
        settings, camera, world
    )
    var checksum = pixel_checksum(warmup.pixels)
    var total_times = List[Int](capacity=TIMING_REPEATS)
    var init_times = List[Int](capacity=TIMING_REPEATS)
    var render_times = List[Int](capacity=TIMING_REPEATS)

    for _ in range(TIMING_REPEATS):
        var result = render_wavefront[ALGORITHM, length, CHUNK_PATHS, False](
            settings, camera, world
        )
        var current_checksum = pixel_checksum(result.pixels)
        debug_assert["safe", _use_compiler_assume=True](
            current_checksum == checksum,
            "wavefront checksum changed between runs",
        )
        total_times.append(result.timings.total_ns)
        init_times.append(result.timings.init_ns)
        render_times.append(result.timings.render_ns)

    sort(total_times)
    sort(init_times)
    sort(render_times)
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
    settings: RenderSettings, camera: Camera, world: CpuScene[]
) -> WaveCounters:
    var counters = WaveCounters()
    var path_count = (
        settings.image_width
        * settings.image_height
        * settings.samples_per_pixel
    )
    var active_paths = PacketPathQueue[1](path_count)
    var next_paths = PacketPathQueue[1](path_count)
    _initialize_path_packets_range[1](
        active_paths, settings, camera, 0, path_count
    )
    counters.primary_paths = len(active_paths)

    for _bounce in range(MAX_DEPTH):
        if len(active_paths) == 0:
            break

        counters.active_by_bounce.append(len(active_paths))
        next_paths.clear()
        var lambertian_queue = 0
        var metal_queue = 0
        var dielectric_queue = 0

        for packet in active_paths.packets:
            var path_id = packet.path_ids[0]
            var ray = Rayf32[.WORLD](
                Point3W(packet.ox[0], packet.oy[0], packet.oz[0]),
                Vec3W(packet.dx[0], packet.dy[0], packet.dz[0]),
                packet.t_min[0],
                packet.t_max[0],
            )
            var throughput = Color(packet.tx[0], packet.ty[0], packet.tz[0])
            counters.rays += 1
            var hit = world.trace(ray)
            if not hit:
                counters.misses += 1
                counters.escaped += 1
                continue

            counters.hits += 1
            ref record = hit.value()
            var material_kind = record.surface.kind()
            if material_kind == .LAMBERTIAN:
                counters.lambertian_hits += 1
                lambertian_queue += 1
            elif material_kind == .METAL:
                counters.metal_hits += 1
                metal_queue += 1
            elif material_kind == .DIELECTRIC:
                counters.dielectric_hits += 1
                dielectric_queue += 1

            var rng = path_stage_rng(
                settings.rng_seed, path_id, UInt32(_bounce + 1)
            )
            var scattered = sample_bsdf(
                record.surface,
                world.scene_data().surfaces(),
                ray,
                ShadingPoint(record.p, record.normal, record.front_face),
                rng,
            )
            if scattered.ok:
                var roulette = russian_roulette(
                    settings.rng_seed,
                    path_id,
                    UInt32(_bounce + 1),
                    throughput * scattered.weight,
                )
                if roulette.survived:
                    var next = PathPacket[1]()
                    next.path_ids[0] = path_id
                    next.ox[0] = record.p.x
                    next.oy[0] = record.p.y
                    next.oz[0] = record.p.z
                    next.t_min[0] = 0.001
                    next.dx[0] = scattered.direction.x
                    next.dy[0] = scattered.direction.y
                    next.dz[0] = scattered.direction.z
                    next.t_max[0] = f32_max
                    next.tx[0] = roulette.throughput.x
                    next.ty[0] = roulette.throughput.y
                    next.tz[0] = roulette.throughput.z
                    next.bsdf_pdfs[0] = scattered.pdf
                    next.deltas[0] = scattered.delta
                    next_paths.append_packet(next^, 1)
                else:
                    counters.roulette_terminated += 1
            else:
                counters.absorbed += 1

        counters.peak_material_queue = max(
            counters.peak_material_queue,
            max(lambertian_queue, max(metal_queue, dielectric_queue)),
        )
        swap(active_paths, next_paths)

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
    world: CpuScene[],
):
    print_timing(
        label + " / packet width 1",
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


def benchmark_direct_lighting(
    settings: RenderSettings, camera: Camera, world: CpuScene[]
):
    print_timing(
        "Cornell NEE / packet width 1",
        time_wavefront[1, .NEE](settings, camera, world),
    )
    print_timing(
        "Cornell NEE / packet width 8",
        time_wavefront[8, .NEE](settings, camera, world),
    )
    print_timing(
        "Cornell NEE / packet width 16",
        time_wavefront[16, .NEE](settings, camera, world),
    )
    print_timing(
        "Cornell MIS / packet width 1",
        time_wavefront[1, .MIS](settings, camera, world),
    )
    print_timing(
        "Cornell MIS / packet width 8",
        time_wavefront[8, .MIS](settings, camera, world),
    )
    print_timing(
        "Cornell MIS / packet width 16",
        time_wavefront[16, .MIS](settings, camera, world),
    )


def main() raises:
    print("CPU serial wavefront benchmark")
    print(
        t"timing: {TIMING_WIDTH}x{TIMING_HEIGHT} x {TIMING_SPP} spp, "
        t"depth {MAX_DEPTH}, median of {TIMING_REPEATS} after warmup"
    )
    print(t"counters: {COUNTER_WIDTH}x{COUNTER_HEIGHT} x {COUNTER_SPP} spp")

    var timing_settings = RenderSettings(
        TIMING_WIDTH, TIMING_HEIGHT, TIMING_SPP, RNG_SEED, MAX_DEPTH
    )
    var counter_settings = RenderSettings(
        COUNTER_WIDTH, COUNTER_HEIGHT, COUNTER_SPP, RNG_SEED, MAX_DEPTH
    )

    var sphere_world = make_weekend_world()
    var sphere_camera = weekend_camera()
    benchmark_scene(
        "Weekend spheres",
        timing_settings,
        counter_settings,
        sphere_camera,
        sphere_world,
    )

    var triangle_world = make_mixed_triangle_world()
    var triangle_cam = mixed_triangle_camera(triangle_world)
    benchmark_scene(
        "Mixed triangles",
        timing_settings,
        counter_settings,
        triangle_cam,
        triangle_world,
    )

    var cornell_world = make_cornell_world()
    var cornell_camera = Camera.from_vfov(
        Point3W(0.0, 1.0, 3.2),
        Point3W(0.0, 1.0, -1.0),
        Vec3W(0.0, 1.0, 0.0),
        28.0,
        4.2,
    )
    benchmark_direct_lighting(timing_settings, cornell_camera, cornell_world)
