"""Persistent render resources and asynchronous GPU RT submission helpers."""

from bajo.rt.gpu.spectral_state import GpuSpectralState

from std.math import ceildiv, min
from std.sys import size_of, align_of
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from bajo.gpu.buffers import upload_list
from bajo.core.camera import Camera
from bajo.bvh.gpu.utils import upload_camera
from bajo.core.color import Color
from bajo.rt.film import FilmResponse
from bajo.rt.gpu.filter import GpuFilterTables
from bajo.rt.render_types import RenderSettings, SamplingConfig
from bajo.rt.gpu.sampling import GpuRtSamplingConfig
from bajo.rt.material_types import Integrator, SurfaceId
from bajo.rt.wavefront_contract import WAVE_PATH_ID_MASK
from bajo.rt.gpu.common_kernels import (
    GPU_RT_BLOCK_SIZE,
    gpu_rt_primary_kernel,
    gpu_rt_resolve_kernel,
    gpu_rt_resolve_film_kernel,
)
from bajo.rt.gpu.wavefront_contract import (
    GpuWavefrontArena,
    WAVE_COUNTER,
    WAVE_STATUS,
    enqueue_wavefront_advance,
    enqueue_wavefront_begin,
)


comptime _GPU_RT_DEFAULT_PATH_CAPACITY = 262144


def upload_surface_ids[
    width: SIMDLength,
](
    mut ctx: DeviceContext,
    surfaces: ImmSpan[SurfaceId[width], _],
) raises -> DeviceBuffer[.uint32]:
    """Upload the packed scalar surface sidecar shared by GPU geometries."""
    comptime assert width == 1
    return upload_list(ctx, [surface.value[0] for surface in surfaces])


struct GpuRtRenderTarget:
    """Reusable path queues, camera state, and device-resident pixel output.

    The caller owns the `DeviceContext`; keeping both the target and a GPU scene
    alive amortizes all buffer allocation, scene upload, and BVH construction.
    `pixels` remains device-resident after submission so downstream GPU work can
    consume it without a synchronization or host copy.
    """

    var image_width: Int
    var image_height: Int
    var samples_per_pixel: Int
    var pixel_count: Int
    var sample_count: Int
    var arena: GpuWavefrontArena
    var camera: DeviceBuffer[.float32]
    var pixels: DeviceBuffer[.float32]
    var filter: GpuFilterTables
    var spectral: Optional[GpuSpectralState]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        settings: RenderSettings,
        camera: Camera,
        path_capacity: Int = 0,
    ) raises:
        settings.validate()
        self.filter = GpuFilterTables(ctx, settings.filter)
        self.spectral = Optional[GpuSpectralState]()
        self.image_width = settings.image_width
        self.image_height = settings.image_height
        self.samples_per_pixel = settings.samples_per_pixel
        self.pixel_count = self.image_width * self.image_height
        self.sample_count = self.pixel_count * self.samples_per_pixel
        if UInt64(self.sample_count) > UInt64(WAVE_PATH_ID_MASK):
            raise Error(
                "GPU RT compact path IDs support at most 2^31-1 samples"
            )
        var requested_capacity = (
            path_capacity if path_capacity
            > 0 else _GPU_RT_DEFAULT_PATH_CAPACITY
        )
        var arena_capacity = min(requested_capacity, self.sample_count)
        arena_capacity = (
            arena_capacity / self.samples_per_pixel * self.samples_per_pixel
        )
        if arena_capacity == 0:
            arena_capacity = self.samples_per_pixel
        self.arena = GpuWavefrontArena(ctx, arena_capacity)
        self.camera = upload_camera(ctx, camera)
        self.pixels = ctx.enqueue_create_buffer[.float32](self.pixel_count * 3)

    def ensure_spectral(mut self, ctx: DeviceContext) raises:
        if not self.spectral:
            var context = ctx.copy()
            self.spectral = Optional(
                GpuSpectralState(context, self.arena.capacity)
            )

    def validate_compatible(self, settings: RenderSettings) raises:
        settings.validate()
        if (
            settings.image_width != self.image_width
            or settings.image_height != self.image_height
            or settings.samples_per_pixel != self.samples_per_pixel
        ):
            raise Error(
                "GPU RT settings must match the persistent render target"
            )


def update_gpu_camera(
    mut ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    camera: Camera,
) raises:
    """Queue a camera upload while retaining all render allocations."""
    var values = camera.flatten()
    debug_assert["safe", _use_compiler_assume=True](
        len(values) == len(target.camera)
    )
    var host_camera = ctx.enqueue_create_host_buffer[.float32](len(values))
    host_camera.enqueue_copy_from(values)
    host_camera.enqueue_copy_to(target.camera)


def _enqueue_gpu_primary[
    integrator: Integrator,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    sampling: GpuRtSamplingConfig,
    sample_begin: Int = 0,
    chunk_sample_count: Int = -1,
) raises:
    """Reset queues and enqueue primary rays without synchronizing."""
    var active_count = chunk_sample_count
    if active_count < 0:
        active_count = min(target.arena.capacity, target.sample_count)
    debug_assert["safe", _use_compiler_assume=True](
        sample_begin >= 0
        and active_count > 0
        and active_count <= target.arena.capacity
        and sample_begin + active_count <= target.sample_count,
        "GPU RT primary chunk is outside the render sample range",
    )
    target.arena.sample_base = UInt32(sample_begin)
    enqueue_wavefront_begin(ctx, target.arena, active_count)
    ctx.enqueue_function[gpu_rt_primary_kernel[integrator]](
        target.camera,
        target.arena.path_a.path_ids,
        target.arena.path_a.fields,
        target.arena.sample_radiance,
        Int32(target.arena.capacity),
        Int32(active_count),
        UInt32(sample_begin),
        Int32(target.image_width),
        Int32(target.image_height),
        Int32(target.samples_per_pixel),
        sampling,
        grid_dim=ceildiv(active_count, GPU_RT_BLOCK_SIZE),
        block_dim=GPU_RT_BLOCK_SIZE,
    )


def _enqueue_gpu_resolve(
    ctx: DeviceContext,
    target: GpuRtRenderTarget,
    sample_begin: Int = 0,
    chunk_sample_count: Int = -1,
    film: FilmResponse = FilmResponse(),
) raises:
    """Enqueue sample reduction into `target.pixels` without synchronizing."""
    var active_count = chunk_sample_count
    if active_count < 0:
        active_count = min(target.arena.capacity, target.sample_count)
    debug_assert["safe", _use_compiler_assume=True](
        sample_begin % target.samples_per_pixel == 0
        and active_count % target.samples_per_pixel == 0,
        "GPU RT chunks must contain complete pixels",
    )
    var pixel_begin = sample_begin / target.samples_per_pixel
    var chunk_pixel_count = active_count / target.samples_per_pixel
    if film.is_identity():
        ctx.enqueue_function[gpu_rt_resolve_kernel](
            target.arena.sample_radiance,
            target.pixels,
            Int32(target.arena.capacity),
            Int32(pixel_begin),
            Int32(chunk_pixel_count),
            Int32(target.samples_per_pixel),
            grid_dim=ceildiv(chunk_pixel_count, GPU_RT_BLOCK_SIZE),
            block_dim=GPU_RT_BLOCK_SIZE,
        )
    else:
        ctx.enqueue_function[gpu_rt_resolve_film_kernel](
            target.arena.sample_radiance,
            target.pixels,
            Int32(target.arena.capacity),
            Int32(pixel_begin),
            Int32(chunk_pixel_count),
            Int32(target.samples_per_pixel),
            film,
            grid_dim=ceildiv(chunk_pixel_count, GPU_RT_BLOCK_SIZE),
            block_dim=GPU_RT_BLOCK_SIZE,
        )


def enqueue_gpu_wavefront[
    Scene: AnyType,
    //,
    integrator: Integrator,
    primary_bounce_fn: def(
        DeviceContext,
        GpuWavefrontArena,
        DeviceBuffer[.float32],
        Scene,
        DeviceBuffer[.uint32],
        DeviceBuffer[.float32],
        DeviceBuffer[.uint32],
        DeviceBuffer[.float32],
        Int,
        Int,
        Int,
        Int,
        GpuRtSamplingConfig,
    ) raises thin -> None,
    bounce_fn: def(
        DeviceContext,
        GpuWavefrontArena,
        Scene,
        DeviceBuffer[.uint32],
        DeviceBuffer[.float32],
        DeviceBuffer[.uint32],
        DeviceBuffer[.float32],
        GpuRtSamplingConfig,
        UInt32,
        Bool,
    ) raises thin -> None,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    scene: Scene,
    settings: RenderSettings,
    sample_offset: Optional[Pointer[UInt32, ImmUntrackedOrigin]] = Optional[
        Pointer[UInt32, ImmUntrackedOrigin]
    ](),
) raises:
    """Compile-time scheduler shared by every geometry specialization."""
    target.validate_compatible(settings)
    target.filter.update(ctx, settings.filter)
    var fixed_sampling = SamplingConfig.from_settings(settings)
    fixed_sampling.filter = target.filter.view()
    var sampling = GpuRtSamplingConfig(fixed_sampling, sample_offset)
    var sample_begin = 0
    while sample_begin < target.sample_count:
        var chunk_sample_count = min(
            target.arena.capacity, target.sample_count - sample_begin
        )
        comptime if integrator == Integrator.PATH:
            # The register-heavier fused primary traversal loses on PATH.
            # Keep its lean primary queue producer and start tracing at zero.
            _enqueue_gpu_primary[integrator](
                ctx, target, sampling, sample_begin, chunk_sample_count
            )
        else:
            if (
                Integrator.uses_direct_lighting[integrator]
                and settings.max_depth == 0
            ):
                _enqueue_gpu_primary[integrator](
                    ctx, target, sampling, sample_begin, chunk_sample_count
                )
            else:
                target.arena.sample_base = UInt32(sample_begin)
                enqueue_wavefront_begin(ctx, target.arena, chunk_sample_count)
                primary_bounce_fn(
                    ctx,
                    target.arena,
                    target.camera,
                    scene,
                    target.arena.path_a.path_ids,
                    target.arena.path_a.fields,
                    target.arena.path_b.path_ids,
                    target.arena.path_b.fields,
                    chunk_sample_count,
                    target.image_width,
                    target.image_height,
                    target.samples_per_pixel,
                    sampling,
                )
                enqueue_wavefront_advance(ctx, target.arena)
        comptime if Integrator.is_path_tracing[integrator]:
            var first_bounce = 0 if integrator == .PATH else 1
            if settings.max_depth == 0:
                first_bounce = 0
            for bounce in range(first_bounce, settings.max_depth + 1):
                if bounce % 2 == 0:
                    bounce_fn(
                        ctx,
                        target.arena,
                        scene,
                        target.arena.path_a.path_ids,
                        target.arena.path_a.fields,
                        target.arena.path_b.path_ids,
                        target.arena.path_b.fields,
                        sampling,
                        UInt32(bounce),
                        bounce < settings.max_depth,
                    )
                else:
                    bounce_fn(
                        ctx,
                        target.arena,
                        scene,
                        target.arena.path_b.path_ids,
                        target.arena.path_b.fields,
                        target.arena.path_a.path_ids,
                        target.arena.path_a.fields,
                        sampling,
                        UInt32(bounce),
                        bounce < settings.max_depth,
                    )
                enqueue_wavefront_advance(ctx, target.arena)
        _enqueue_gpu_resolve(
            ctx, target, sample_begin, chunk_sample_count, settings.film
        )
        sample_begin += chunk_sample_count


struct _GpuRtReadback:
    var pixels: HostBuffer[.float32]
    var counters: HostBuffer[.uint32]
    var pixel_count: Int

    def __init__(out self, ctx: DeviceContext, pixel_count: Int) raises:
        self.pixel_count = pixel_count
        self.pixels = ctx.enqueue_create_host_buffer[.float32](pixel_count * 3)
        self.counters = ctx.enqueue_create_host_buffer[.uint32](
            WAVE_COUNTER.COUNT
        )

    def download(
        self, ctx: DeviceContext, target: GpuRtRenderTarget
    ) raises -> List[Color]:
        if target.pixel_count != self.pixel_count:
            raise Error("GPU readback requires matching dimensions")
        self.pixels.enqueue_copy_from(target.pixels)
        self.counters.enqueue_copy_from(target.arena.counters)
        ctx.synchronize()
        if self.counters.as_span()[WAVE_COUNTER.STATUS] != WAVE_STATUS.OK:
            raise Error("GPU RT wavefront queue overflow")
        comptime assert size_of[Color]() == 3 * size_of[Float32]()
        comptime assert align_of[Color]() >= align_of[Float32]()
        var pixels = List[Color](unsafe_uninit_length=self.pixel_count)
        var destination = MutSpan(
            unsafe_ptr=pixels.unsafe_ptr().unsafe_bitcast[Float32](),
            length=self.pixel_count * 3,
        )
        destination.copy_from(self.pixels.as_span())
        return pixels^


def download_gpu_pixels(
    ctx: DeviceContext, target: GpuRtRenderTarget
) raises -> List[Color]:
    """Validate queue status and download pixels without uploading them again.
    """
    var readback = _GpuRtReadback(ctx, target.pixel_count)
    return readback.download(ctx, target)
