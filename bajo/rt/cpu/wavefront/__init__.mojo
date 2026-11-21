"""CPU wavefront renderer orchestration and public entry points."""

from max.algorithm import parallelize
from std.atomic import Atomic, Ordering
from std.math import ceildiv
from std.sys import num_logical_cores
from std.time import perf_counter_ns
from std.memory import ArcPointer

from bajo.core.camera import Camera
from bajo.bvh.cpu.traversal_mode import CpuTraversalMode
from bajo.core.color import Color
from bajo.postprocess.samples import DenoiseSampleBatch
from bajo.rt.render_types import (
    RenderResult,
    RenderSettings,
    RenderColorMode,
    RenderTimings,
    SamplingConfig,
)
from bajo.rt.material_types import Integrator, SurfaceStore
from ..scene import CpuScene
from ..scheduler_mode import CpuSchedulerMode
from bajo.rt.wavefront_queue import PacketPathQueue, PacketShadeQueue
from bajo.rt.filter import FilterSampler


from .primary import _initialize_path_packets_range
from .packet import _trace_packets
from .spectral import SpectralSampleState, trace_spectral_packets
from ..spectral_resources import CpuSpectralResources
from bajo.rt.spectrum_sources import SpectrumSourcesView
from bajo.rt.film import FilmResponse
from bajo.rt.spectral_light_distribution import SpectralLightDistributionView


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
    var samples: List[Color]
    var weights: List[Float32]
    var spectral: SpectralSampleState

    def __init__(out self, capacity: Int, surfaces: SurfaceStore):
        # Two path buffers ping-pong between bounces. Material queues reserve
        # their worst-case size once so appends never grow storage mid-bounce.
        self.active_paths = PacketPathQueue[Self.length](capacity)
        self.samples = List[Color](length=capacity, fill=Color(0.0))
        self.weights = List[Float32](length=capacity, fill=1)
        self.spectral = {}
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


struct CpuWavefrontWorkspace[length: SIMDLength, integrator: Integrator]:
    """Private worker queues retained between render calls; one owner per caller.
    """

    var arenas: List[_PacketQueueArena[Self.length, Self.integrator]]
    var spectral: Optional[CpuSpectralResources]
    var spectral_scene: Optional[ArcPointer[UInt8]]

    def __init__(out self):
        self.arenas = {}
        self.spectral = {}
        self.spectral_scene = {}

    def prepare(mut self, workers: Int, capacity: Int, surfaces: SurfaceStore):
        while len(self.arenas) < workers:
            self.arenas.append(
                _PacketQueueArena[Self.length, Self.integrator](
                    capacity, surfaces
                )
            )
        var packets = ceildiv(capacity, Int(Self.length))
        for worker in range(workers):
            ref arena = self.arenas[worker]
            arena.active_paths.packets.reserve(packets)
            arena.samples.resize(capacity, fill=Color(0.0))
            arena.weights.resize(capacity, fill=1)
            comptime if Integrator.is_path_tracing[Self.integrator]:
                arena.next_paths.packets.reserve(packets)
                if len(surfaces.lambertians):
                    arena.lambertian.packets.reserve(packets)
                if len(surfaces.metals):
                    arena.metal.packets.reserve(packets)
                if len(surfaces.dielectrics):
                    arena.dielectric.packets.reserve(packets)
                if len(surfaces.coated_diffuses):
                    arena.coated_diffuse.packets.reserve(packets)


def _trace_packet_range[
    _rgb_only: Bool,
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
    spectra: Optional[SpectrumSourcesView[_]],
    film: FilmResponse,
    lighting: Optional[SpectralLightDistributionView[_]],
    components: Optional[MutPointer[UInt32, _]],
    sample_colors: Optional[MutPointer[Color, _]],
    sample_weights: Optional[MutPointer[Float32, _]],
):
    _initialize_path_packets_range[length](
        queues.active_paths,
        settings,
        sampling,
        camera,
        path_begin,
        path_end,
        queues.weights,
    )
    var count = path_end - path_begin
    comptime if not _rgb_only and Integrator.is_path_tracing[integrator]:
        if spectra:
            trace_spectral_packets[
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
                spectra.value(),
                lighting.value(),
                queues.spectral,
                queues.active_paths,
                queues.next_paths,
                path_begin,
                count,
                components,
            )
            var spp = settings.samples_per_pixel
            for first in range(0, count, spp):
                var total = Color(0)
                var weight = Float32(0)
                for index in range(first, first + spp):
                    var sample = film.apply_spectral_sample(
                        queues.spectral.radiance[index],
                        queues.spectral.wavelengths[index],
                    )
                    total += sample * queues.weights[index]
                    weight += queues.weights[index]
                    if sample_colors:
                        var id = path_begin + index
                        sample_colors.value()[unsafe_offset=id] = sample
                        sample_weights.value()[
                            unsafe_offset=id
                        ] = queues.weights[index]
                pixels[
                    (path_begin + first) // spp
                ] = total / weight if weight != 0 else Color(0)
            return
    for index in range(count):
        queues.samples[index] = Color(0.0)
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
        queues.samples[0:count],
        queues.active_paths,
        queues.next_paths,
        queues.lambertian,
        queues.metal,
        queues.dielectric,
        queues.coated_diffuse,
        path_begin,
        components,
    )
    # Each worker owns complete pixels. Retain complete path radiance until
    # sample-level sensor conversion/clamping, then reduce in sample order.
    var spp = settings.samples_per_pixel
    for first in range(0, count, spp):
        var sum = Color(0.0)
        var weight = Float32(0)
        for index in range(first, first + spp):
            var sample = settings.film.apply_sample(queues.samples[index])
            sum += sample * queues.weights[index]
            weight += queues.weights[index]
            if sample_colors:
                var id = path_begin + index
                sample_colors.value()[unsafe_offset=id] = sample
                sample_weights.value()[unsafe_offset=id] = queues.weights[index]
        pixels[
            (path_begin + first) // spp
        ] = sum / weight if weight != 0 else Color(0)


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
    var workspace = CpuWavefrontWorkspace[length, integrator]()
    return render_wavefront_configured[
        traversal_mode,
        integrator,
        length,
        CHUNK_PATHS,
        PARALLEL,
        scheduler_mode,
        world_bvh_width,
        instance_bvh_width,
        *adaptive_packet_sizes,
    ](settings, camera, world, workspace)


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
    mut workspace: CpuWavefrontWorkspace[length, integrator],
) raises -> RenderResult:
    """Render RGB or spectral transport with retained worker queues."""
    return _render_wavefront_configured[
        False,
        traversal_mode,
        integrator,
        length,
        CHUNK_PATHS,
        PARALLEL,
        scheduler_mode,
        world_bvh_width,
        instance_bvh_width,
        *adaptive_packet_sizes,
    ](settings, camera, world, workspace)


def _trace_packet_batches[
    _rgb_only: Bool,
    PARALLEL: Bool,
    scheduler_mode: CpuSchedulerMode,
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
    output: MutSpan[Color, _],
    arenas: MutSpan[_PacketQueueArena[length, integrator], _],
    worker_count: Int,
    chunk_count: Int,
    paths_per_chunk: Int,
    path_count: Int,
    spectra: Optional[SpectrumSourcesView[_]],
    film: FilmResponse,
    lighting: Optional[SpectralLightDistributionView[_]],
    components: Optional[MutPointer[UInt32, _]],
    sample_colors: Optional[MutPointer[Color, _]],
    sample_weights: Optional[MutPointer[Float32, _]],
):
    # Accept already-borrowed views so worker captures contain no owning
    # containers. Queue/output partitions remain disjoint for each worker.
    comptime if PARALLEL:
        # Whole-pixel chunks keep output partitions disjoint and preserve the
        # global Philox streams. Relaxed atomics only assign work, not data.
        var next_chunk = List[Int](length=1, fill=0)
        var next_chunk_ptr = next_chunk.unsafe_ptr()

        def worker(_worker_idx: Int) {imm}:
            var chunk_idx = Atomic.fetch_add[ordering=Ordering.relaxed](
                next_chunk_ptr, 1
            )
            if chunk_idx >= chunk_count:
                return
            ref queues = arenas[_worker_idx]
            while chunk_idx < chunk_count:
                var path_begin = chunk_idx * paths_per_chunk
                var path_end = min(path_begin + paths_per_chunk, path_count)
                _trace_packet_range[
                    _rgb_only,
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
                    output,
                    path_begin,
                    path_end,
                    queues,
                    spectra,
                    film,
                    lighting,
                    components,
                    sample_colors,
                    sample_weights,
                )
                chunk_idx = Atomic.fetch_add[ordering=Ordering.relaxed](
                    next_chunk_ptr, 1
                )

        # Explicit partitions map one callback per arena;
        # runtime-default mode retains the runtime's partitioning choice.
        comptime if scheduler_mode == .RUNTIME_DEFAULT:
            parallelize(worker, worker_count)
        else:
            parallelize(worker, worker_count, worker_count)

    else:
        ref queues = arenas[0]
        var path_begin = 0
        while path_begin < path_count:
            var path_end = min(path_begin + paths_per_chunk, path_count)
            _trace_packet_range[
                _rgb_only,
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
                output,
                path_begin,
                path_end,
                queues,
                spectra,
                film,
                lighting,
                components,
                sample_colors,
                sample_weights,
            )
            path_begin = path_end


def _render_wavefront_configured[
    _rgb_only: Bool,
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
    mut workspace: CpuWavefrontWorkspace[length, integrator],
) raises -> RenderResult:
    """Render with compile-time packet, chunk, and CPU scheduling choices."""
    comptime assert CHUNK_PATHS > 0, "wavefront chunk size must be positive"
    comptime assert (
        Integrator.is_path_tracing[integrator]
        or integrator == .NORMALS
        or integrator == .AO
    )

    settings.validate()
    comptime if _rgb_only:
        if settings.color_mode != .RGB:
            raise Error("CPU RGB entry points require RGB settings")
    var sampling = SamplingConfig.from_settings(settings)
    var filter_sampler = FilterSampler(settings.filter)
    sampling.filter = filter_sampler.view()
    var total_t0 = perf_counter_ns()
    var pixel_count = settings.image_width * settings.image_height
    var path_count = pixel_count * settings.samples_per_pixel
    var paths_per_chunk = _whole_pixel_chunk_paths(
        settings.samples_per_pixel, CHUNK_PATHS
    )
    var arena_capacity = min(paths_per_chunk, path_count)
    var init_t0 = perf_counter_ns()
    var film = settings.film.copy()
    comptime if not _rgb_only and Integrator.is_path_tracing[integrator]:
        if settings.color_mode == .SPECTRAL:
            var scene_identity = world.snapshot_identity()
            var same_scene = False
            if workspace.spectral_scene:
                same_scene = Int(
                    Pointer(to=workspace.spectral_scene.value()[])
                ) == Int(Pointer(to=scene_identity[]))
            if not workspace.spectral:
                workspace.spectral = Optional(
                    CpuSpectralResources(world.scene_data().surfaces())
                )
            elif not same_scene:
                # A failed preparation must not leave the previous identity
                # pointing at partially replaced tables.
                workspace.spectral_scene = Optional[ArcPointer[UInt8]]()
                workspace.spectral.value().prepare(
                    world.scene_data().surfaces()
                )
            if not workspace.spectral.value().lighting:
                workspace.spectral.value().prepare_lighting(world.scene_data())
            workspace.spectral_scene = Optional(scene_identity^)
            film = settings.film.prepared_spectral()
    var pixels = List[Color](length=pixel_count, fill=Color(0.0))
    # Keep each buffer's owner initialized even when collection is disabled,
    # so optional worker pointers retain their separate tracked lifetimes.
    var samples = DenoiseSampleBatch(
        pixel_count if settings.collect_denoise else 0,
        settings.samples_per_pixel,
    )
    var components_ptr = samples.components.unsafe_ptr()
    var colors_ptr = samples.colors.unsafe_ptr()
    var weights_ptr = samples.weights.unsafe_ptr()
    var components = Optional(components_ptr)
    var sample_colors = Optional(colors_ptr)
    var sample_weights = Optional(weights_ptr)
    if not settings.collect_denoise:
        components = Optional[type_of(components_ptr)]()
        sample_colors = Optional[type_of(colors_ptr)]()
        sample_weights = Optional[type_of(weights_ptr)]()
    var init_t1 = perf_counter_ns()

    var render_t0 = perf_counter_ns()
    var chunk_count = ceildiv(path_count, paths_per_chunk)
    var worker_count = 1
    comptime if PARALLEL:
        worker_count = min(num_logical_cores(), chunk_count)
    # Allocate queues before borrowing spectral buffers. Workers then mutate
    # only arenas, so the spectral owners stay immutably borrowed throughout.
    workspace.prepare(
        worker_count, arena_capacity, world.scene_data().surfaces()
    )
    var output = Span(pixels)
    var arenas = Span(workspace.arenas)
    var use_spectral = False
    comptime if not _rgb_only and Integrator.is_path_tracing[integrator]:
        use_spectral = settings.color_mode == .SPECTRAL
    if use_spectral:
        ref resources = workspace.spectral.value()
        var spectra = resources.view()
        var lighting = resources.lighting.value().view(resources.rgb.view())
        _trace_packet_batches[
            _rgb_only,
            PARALLEL,
            scheduler_mode,
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
            output,
            arenas,
            worker_count,
            chunk_count,
            paths_per_chunk,
            path_count,
            Optional(spectra),
            film,
            Optional(lighting),
            components,
            sample_colors,
            sample_weights,
        )
    else:
        _trace_packet_batches[
            _rgb_only,
            PARALLEL,
            scheduler_mode,
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
            output,
            arenas,
            worker_count,
            chunk_count,
            paths_per_chunk,
            path_count,
            Optional[SpectrumSourcesView[]](),
            film,
            Optional[SpectralLightDistributionView[]](),
            components,
            sample_colors,
            sample_weights,
        )
    var render_t1 = perf_counter_ns()
    # SamplingConfig carries an untracked view shared with device code. Keep
    # its host owner alive until all workers have finished using the tables.
    _ = filter_sampler^

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
        Optional(samples^) if settings.collect_denoise else Optional[
            DenoiseSampleBatch
        ](),
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
