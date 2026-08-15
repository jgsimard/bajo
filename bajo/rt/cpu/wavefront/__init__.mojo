"""CPU wavefront renderer orchestration and public entry points."""

from max.algorithm import parallelize
from std.math import ceildiv
from std.sys import num_logical_cores
from std.time import perf_counter_ns

from bajo.bvh.camera import Camera
from bajo.rt.types import (
    Color,
    RENDER,
    RenderResult,
    RenderSettings,
    RenderTimings,
    World,
)
from bajo.rt.wavefront_queue import PacketShadeQueue


from .primary import _make_initial_path_packets_range
from .packet import _trace_path_packets


comptime CPU_WAVEFRONT_SERIAL_CHUNK_PATHS = 8192
comptime CPU_WAVEFRONT_PARALLEL_CHUNK_PATHS = 1024

comptime WAVE_PARALLEL_RUNTIME_DEFAULT = 0
comptime WAVE_PARALLEL_LOGICAL_CORES = 1
comptime WAVE_PARALLEL_TASK_PARTITIONS = 2


def _whole_pixel_chunk_paths(samples_per_pixel: Int, target_paths: Int) -> Int:
    return max(
        samples_per_pixel,
        (target_paths / samples_per_pixel) * samples_per_pixel,
    )


struct _PacketMaterialQueues[PACKET_LANES: SIMDLength]:
    var lambertian: PacketShadeQueue[Self.PACKET_LANES]
    var metal: PacketShadeQueue[Self.PACKET_LANES]
    var dielectric: PacketShadeQueue[Self.PACKET_LANES]

    def __init__(out self):
        self.lambertian = PacketShadeQueue[Self.PACKET_LANES](0)
        self.metal = PacketShadeQueue[Self.PACKET_LANES](0)
        self.dielectric = PacketShadeQueue[Self.PACKET_LANES](0)


def _trace_packet_range[
    PACKET_LANES: SIMDLength,
    ALGORITHM: RENDER,
    MAX_DEPTH: Int,
](
    settings: RenderSettings,
    camera: Camera,
    world: World,
    mut pixels: List[Color],
    path_begin: Int,
    path_end: Int,
    mut queues: _PacketMaterialQueues[PACKET_LANES],
):
    var active_paths = _make_initial_path_packets_range[PACKET_LANES](
        settings, camera, path_begin, path_end
    )
    _trace_path_packets[MAX_DEPTH, PACKET_LANES, ALGORITHM](
        settings,
        world,
        pixels,
        active_paths^,
        queues.lambertian,
        queues.metal,
        queues.dielectric,
    )


def render_wavefront[
    ALGORITHM: RENDER = RENDER.PATH,
    MAX_DEPTH: Int = 8,
    PACKET_LANES: SIMDLength = 1,
    CHUNK_PATHS: Int = CPU_WAVEFRONT_PARALLEL_CHUNK_PATHS,
    PARALLEL: Bool = True,
    SCHEDULER_MODE: Int = WAVE_PARALLEL_TASK_PARTITIONS,
](settings: RenderSettings, camera: Camera, world: World) -> RenderResult:
    """Render with compile-time packet, chunk, and CPU scheduling choices."""
    comptime assert CHUNK_PATHS > 0, "wavefront chunk size must be positive"
    comptime assert MAX_DEPTH >= 0, "max depth must be non-negative"
    comptime assert ALGORITHM in (RENDER.PATH, RENDER.NEE, RENDER.MIS)
    comptime if PARALLEL:
        comptime assert (
            WAVE_PARALLEL_RUNTIME_DEFAULT
            <= SCHEDULER_MODE
            <= WAVE_PARALLEL_TASK_PARTITIONS
        ), "unknown parallel wavefront scheduler mode"

    var total_t0 = perf_counter_ns()
    var pixel_count = settings.image_width * settings.image_height
    var path_count = pixel_count * settings.samples_per_pixel
    var paths_per_chunk = _whole_pixel_chunk_paths(
        settings.samples_per_pixel, CHUNK_PATHS
    )
    var init_t0 = perf_counter_ns()
    var pixels = List[Color](length=pixel_count, fill=Color(0.0))
    var init_t1 = perf_counter_ns()

    var render_t0 = perf_counter_ns()
    comptime if PARALLEL:
        # Whole-pixel boundaries give every task exclusive output pixels.
        # Global path IDs retain their Philox streams regardless of task order.
        var chunk_count = ceildiv(path_count, paths_per_chunk)

        def worker(chunk_idx: Int) {imm, mut pixels}:
            var path_begin = chunk_idx * paths_per_chunk
            var path_end = min(path_begin + paths_per_chunk, path_count)
            var queues = _PacketMaterialQueues[PACKET_LANES]()
            _trace_packet_range[PACKET_LANES, ALGORITHM, MAX_DEPTH](
                settings,
                camera,
                world,
                pixels,
                path_begin,
                path_end,
                queues,
            )

        comptime if SCHEDULER_MODE == WAVE_PARALLEL_LOGICAL_CORES:
            parallelize(
                worker, chunk_count, min(num_logical_cores(), chunk_count)
            )
        elif SCHEDULER_MODE == WAVE_PARALLEL_TASK_PARTITIONS:
            parallelize(worker, chunk_count, chunk_count)
        else:
            parallelize(worker, chunk_count)
    else:
        var queues = _PacketMaterialQueues[PACKET_LANES]()
        var path_begin = 0
        while path_begin < path_count:
            var path_end = min(path_begin + paths_per_chunk, path_count)
            _trace_packet_range[PACKET_LANES, ALGORITHM, MAX_DEPTH](
                settings,
                camera,
                world,
                pixels,
                path_begin,
                path_end,
                queues,
            )
            path_begin = path_end
    var render_t1 = perf_counter_ns()

    var scale_factor = 1.0 / Float32(settings.samples_per_pixel)
    for ref pixel in pixels:
        pixel = pixel * scale_factor
    var total_t1 = perf_counter_ns()
    return RenderResult(
        pixels^,
        RenderTimings(
            Int(total_t1 - total_t0),
            Int(init_t1 - init_t0),
            Int(render_t1 - render_t0),
            pixel_count,
            path_count,
            MAX_DEPTH,
        ),
    )
