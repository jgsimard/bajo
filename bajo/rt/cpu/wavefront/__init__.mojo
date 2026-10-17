"""CPU wavefront renderer orchestration and public entry points."""

from max.algorithm import parallelize
from std.atomic import Atomic, Ordering
from std.math import ceildiv
from std.sys import num_logical_cores
from std.time import perf_counter_ns

from bajo.bvh import Camera
from bajo.bvh.cpu.traversal_mode import CpuTraversalMode
from bajo.rt.render_types import (
    Color,
    RenderResult,
    RenderSettings,
    RenderTimings,
    SamplingConfig,
)
from bajo.rt.material_types import Integrator, SurfaceStore
from ..scene import CpuScene
from ..scheduler_mode import CpuSchedulerMode
from bajo.rt.wavefront_queue import PacketPathQueue, PacketShadeQueue


from .primary import _initialize_path_packets_range
from .packet import _trace_packets


comptime CPU_WAVEFRONT_SERIAL_CHUNK_PATHS = 8192
comptime CPU_WAVEFRONT_PARALLEL_CHUNK_PATHS = 1024


def _whole_pixel_chunk_paths(samples_per_pixel: Int, target_paths: Int) -> Int:
    return max(
        samples_per_pixel,
        (target_paths / samples_per_pixel) * samples_per_pixel,
    )


struct _PacketQueueArena[length: SIMDLength, integrator: Integrator]:
    var active_paths: PacketPathQueue[Self.length]
    var next_paths: PacketPathQueue[Self.length]
    var lambertian: PacketShadeQueue[Self.length]
    var metal: PacketShadeQueue[Self.length]
    var dielectric: PacketShadeQueue[Self.length]
    var coated_diffuse: PacketShadeQueue[Self.length]

    def __init__(out self, capacity: Int, surfaces: SurfaceStore):
        # Two path buffers ping-pong between bounces. Material queues reserve
        # their worst-case size once so appends never grow storage mid-bounce.
        self.active_paths = PacketPathQueue[Self.length](capacity)
        var shade_capacity = capacity if Integrator.is_path_tracing[
            Self.integrator
        ] else 0
        self.next_paths = PacketPathQueue[Self.length](shade_capacity)
        self.lambertian = PacketShadeQueue[Self.length](
            shade_capacity if len(surfaces.lambertians) > 0 else 0
        )
        self.metal = PacketShadeQueue[Self.length](
            shade_capacity if len(surfaces.metals) > 0 else 0
        )
        self.dielectric = PacketShadeQueue[Self.length](
            shade_capacity if len(surfaces.dielectrics) > 0 else 0
        )
        self.coated_diffuse = PacketShadeQueue[Self.length](
            shade_capacity if len(surfaces.coated_diffuses) > 0 else 0
        )


def _trace_packet_range[
    length: SIMDLength,
    integrator: Integrator,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
    traversal_mode: CpuTraversalMode,
    *adaptive_packet_sizes: SIMDLength,
](
    settings: RenderSettings,
    sampling: SamplingConfig,
    camera: Camera,
    world: CpuScene[world_bvh_width, instance_bvh_width],
    pixels: MutSpan[Color, _],
    path_begin: Int,
    path_end: Int,
    mut queues: _PacketQueueArena[length, integrator],
):
    _initialize_path_packets_range[length](
        queues.active_paths, settings, sampling, camera, path_begin, path_end
    )
    _trace_packets[
        length,
        integrator,
        world_bvh_width,
        instance_bvh_width,
        traversal_mode,
        *adaptive_packet_sizes,
    ](
        settings,
        sampling,
        world,
        pixels,
        queues.active_paths,
        queues.next_paths,
        queues.lambertian,
        queues.metal,
        queues.dielectric,
        queues.coated_diffuse,
    )


def render_wavefront_configured[
    traversal_mode: CpuTraversalMode,
    integrator: Integrator = .PATH,
    length: SIMDLength = 16,
    CHUNK_PATHS: Int = CPU_WAVEFRONT_PARALLEL_CHUNK_PATHS,
    PARALLEL: Bool = True,
    scheduler_mode: CpuSchedulerMode = .TASK_PARTITIONS,
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
    *adaptive_packet_sizes: SIMDLength,
](
    settings: RenderSettings,
    camera: Camera,
    world: CpuScene[world_bvh_width, instance_bvh_width],
) raises -> RenderResult:
    """Render with compile-time packet, chunk, and CPU scheduling choices."""
    comptime assert CHUNK_PATHS > 0, "wavefront chunk size must be positive"
    comptime assert (
        Integrator.is_path_tracing[integrator]
        or integrator == .NORMALS
        or integrator == .AO
    )
    comptime if PARALLEL:
        comptime assert CpuSchedulerMode.is_valid[
            scheduler_mode
        ], "unknown parallel wavefront scheduler mode"

    settings.validate()
    var sampling = SamplingConfig.from_settings(settings)
    var total_t0 = perf_counter_ns()
    var pixel_count = settings.image_width * settings.image_height
    var path_count = pixel_count * settings.samples_per_pixel
    var paths_per_chunk = _whole_pixel_chunk_paths(
        settings.samples_per_pixel, CHUNK_PATHS
    )
    var arena_capacity = min(paths_per_chunk, path_count)
    var init_t0 = perf_counter_ns()
    var pixels = List[Color](length=pixel_count, fill=Color(0.0))
    var init_t1 = perf_counter_ns()

    var render_t0 = perf_counter_ns()
    comptime if PARALLEL:
        # Whole-pixel boundaries give every task exclusive output pixels.
        # Global path IDs retain their Philox streams regardless of task order.
        var chunk_count = ceildiv(path_count, paths_per_chunk)

        # Bounded workers claim whole-pixel chunks dynamically and retain a
        # private arena. The atomic counter publishes no data; relaxed ordering
        # suffices to assign distinct chunks and therefore distinct pixels.
        var worker_count = min(num_logical_cores(), chunk_count)
        var next_chunk = List[Int](length=1, fill=0)
        var next_chunk_ptr = next_chunk.unsafe_ptr()

        def worker(_worker_idx: Int) {imm, mut pixels}:
            var chunk_idx = Atomic.fetch_add[ordering=Ordering.RELAXED](
                next_chunk_ptr, 1
            )
            if chunk_idx >= chunk_count:
                return
            var queues = _PacketQueueArena[length, integrator](
                arena_capacity, world.scene_data().surfaces()
            )
            while chunk_idx < chunk_count:
                var path_begin = chunk_idx * paths_per_chunk
                var path_end = min(path_begin + paths_per_chunk, path_count)
                _trace_packet_range[
                    length,
                    integrator,
                    world_bvh_width,
                    instance_bvh_width,
                    traversal_mode,
                    *adaptive_packet_sizes,
                ](
                    settings,
                    sampling,
                    camera,
                    world,
                    pixels,
                    path_begin,
                    path_end,
                    queues,
                )
                chunk_idx = Atomic.fetch_add[ordering=Ordering.RELAXED](
                    next_chunk_ptr, 1
                )

        # Explicit core/task partitions both map one callback per arena;
        # runtime-default mode retains the runtime's partitioning choice.
        comptime if scheduler_mode == .RUNTIME_DEFAULT:
            parallelize(worker, worker_count)
        else:
            parallelize(worker, worker_count, worker_count)

    else:
        var queues = _PacketQueueArena[length, integrator](
            arena_capacity, world.scene_data().surfaces()
        )
        var path_begin = 0
        while path_begin < path_count:
            var path_end = min(path_begin + paths_per_chunk, path_count)
            _trace_packet_range[
                length,
                integrator,
                world_bvh_width,
                instance_bvh_width,
                traversal_mode,
                *adaptive_packet_sizes,
            ](
                settings,
                sampling,
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
        pixel = settings.film.apply(pixel * scale_factor)
    var total_t1 = perf_counter_ns()
    return RenderResult(
        pixels^,
        RenderTimings(
            Int(total_t1 - total_t0),
            Int(init_t1 - init_t0),
            Int(render_t1 - render_t0),
            pixel_count,
            path_count,
            settings.max_depth,
        ),
    )


def render_wavefront[
    integrator: Integrator = .PATH,
    length: SIMDLength = 16,
    CHUNK_PATHS: Int = CPU_WAVEFRONT_PARALLEL_CHUNK_PATHS,
    PARALLEL: Bool = True,
    scheduler_mode: CpuSchedulerMode = .TASK_PARTITIONS,
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
](
    settings: RenderSettings,
    camera: Camera,
    world: CpuScene[world_bvh_width, instance_bvh_width],
) raises -> RenderResult:
    """Render with the default auto-coherent CPU scene traversal."""
    return render_wavefront_configured[
        CpuTraversalMode.AUTO_COHERENT,
        integrator,
        length,
        CHUNK_PATHS,
        PARALLEL,
        scheduler_mode,
        world_bvh_width,
        instance_bvh_width,
        16,
        8,
        4,
    ](settings, camera, world)
