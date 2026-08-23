"""Persistent render resources and asynchronous GPU RT submission helpers."""

from std.math import ceildiv, min
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh import Camera
from bajo.bvh.gpu.utils import upload_camera, upload_list
from bajo.rt.types import Color, RENDER, RenderSettings, SurfaceId
from bajo.rt.wavefront_contract import WAVE_PATH_ID_MASK
from bajo.rt.gpu.common_kernels import (
    GPU_RT_BLOCK_SIZE,
    gpu_rt_primary_kernel,
    gpu_rt_resolve_kernel,
)
from bajo.rt.gpu.wavefront_contract import (
    GpuWavefrontArena,
    WAVE_COUNTER,
    WAVE_STATUS,
    enqueue_wavefront_advance,
    enqueue_wavefront_begin,
)


comptime GPU_RT_DEFAULT_PATH_CAPACITY = 262144


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

    def __init__(
        out self,
        mut ctx: DeviceContext,
        settings: RenderSettings,
        camera: Camera,
        path_capacity: Int = 0,
    ) raises:
        debug_assert["safe", _use_compiler_assume=True](
            settings.image_width > 0
            and settings.image_height > 0
            and settings.samples_per_pixel > 0,
            "GPU RT render dimensions and sample count must be positive",
        )
        self.image_width = settings.image_width
        self.image_height = settings.image_height
        self.samples_per_pixel = settings.samples_per_pixel
        self.pixel_count = self.image_width * self.image_height
        self.sample_count = self.pixel_count * self.samples_per_pixel
        debug_assert["safe", _use_compiler_assume=True](
            UInt64(self.sample_count) <= UInt64(WAVE_PATH_ID_MASK),
            "GPU RT compact path IDs support at most 2^31-1 samples",
        )
        var requested_capacity = (
            path_capacity if path_capacity > 0 else GPU_RT_DEFAULT_PATH_CAPACITY
        )
        var arena_capacity = min(requested_capacity, self.sample_count)
        arena_capacity = (
            arena_capacity / self.samples_per_pixel * self.samples_per_pixel
        )
        if arena_capacity == 0:
            arena_capacity = self.samples_per_pixel
        self.arena = GpuWavefrontArena(ctx, arena_capacity)
        self.camera = upload_camera(ctx, camera)
        self.pixels = ctx.enqueue_create_buffer[.float32](
            self.pixel_count * 3
        )

    def validate(self, settings: RenderSettings):
        debug_assert["safe", _use_compiler_assume=True](
            settings.image_width == self.image_width
            and settings.image_height == self.image_height
            and settings.samples_per_pixel == self.samples_per_pixel,
            "GPU RT settings must match the persistent render target",
        )


def update_gpu_camera(
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    camera: Camera,
) raises:
    """Update camera parameters while retaining all render allocations."""
    var values = camera.flatten()
    debug_assert["safe", _use_compiler_assume=True](
        len(values) == len(target.camera)
    )
    ctx.synchronize()
    with target.camera.map_to_host() as mapped:
        for i, value in enumerate(values):
            mapped[i] = value


def enqueue_gpu_primary[
    ALGORITHM: RENDER,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    settings: RenderSettings,
    sample_begin: Int = 0,
    chunk_sample_count: Int = -1,
) raises:
    """Reset queues and enqueue primary rays without synchronizing."""
    target.validate(settings)
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
    ctx.enqueue_function[gpu_rt_primary_kernel[ALGORITHM]](
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
        settings.rng_seed,
        grid_dim=ceildiv(active_count, GPU_RT_BLOCK_SIZE),
        block_dim=GPU_RT_BLOCK_SIZE,
    )


def enqueue_gpu_resolve(
    ctx: DeviceContext,
    target: GpuRtRenderTarget,
    sample_begin: Int = 0,
    chunk_sample_count: Int = -1,
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


def enqueue_gpu_wavefront[
    Scene: AnyType,
    //,
    ALGORITHM: RENDER,
    bounce_fn: def(
        DeviceContext,
        GpuWavefrontArena,
        Scene,
        DeviceBuffer[.uint32],
        DeviceBuffer[.float32],
        DeviceBuffer[.uint32],
        DeviceBuffer[.float32],
        UInt64,
        UInt32,
    ) raises thin -> None,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    scene: Scene,
    settings: RenderSettings,
) raises:
    """Compile-time scheduler shared by every geometry specialization."""
    comptime assert ALGORITHM.is_valid()

    var sample_begin = 0
    while sample_begin < target.sample_count:
        var chunk_sample_count = min(
            target.arena.capacity, target.sample_count - sample_begin
        )
        enqueue_gpu_primary[ALGORITHM](
            ctx, target, settings, sample_begin, chunk_sample_count
        )
        comptime if ALGORITHM in (RENDER.NORMALS, RENDER.AO):
            bounce_fn(
                ctx,
                target.arena,
                scene,
                target.arena.path_a.path_ids,
                target.arena.path_a.fields,
                target.arena.path_b.path_ids,
                target.arena.path_b.fields,
                settings.rng_seed,
                UInt32(0),
            )
            enqueue_wavefront_advance(ctx, target.arena)
        else:
            for bounce in range(settings.max_depth):
                if bounce % 2 == 0:
                    bounce_fn(
                        ctx,
                        target.arena,
                        scene,
                        target.arena.path_a.path_ids,
                        target.arena.path_a.fields,
                        target.arena.path_b.path_ids,
                        target.arena.path_b.fields,
                        settings.rng_seed,
                        UInt32(bounce),
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
                        settings.rng_seed,
                        UInt32(bounce),
                    )
                enqueue_wavefront_advance(ctx, target.arena)
        enqueue_gpu_resolve(ctx, target, sample_begin, chunk_sample_count)
        sample_begin += chunk_sample_count


def download_gpu_pixels(
    ctx: DeviceContext,
    target: GpuRtRenderTarget,
) raises -> List[Color]:
    """Synchronize, validate queue status, and copy resolved pixels to host."""
    ctx.synchronize()
    with target.arena.counters.map_to_host() as counters:
        if counters[WAVE_COUNTER.STATUS] != WAVE_STATUS.OK:
            raise "GPU RT wavefront queue overflow"
    var pixels = List[Color](length=target.pixel_count, fill=Color(0.0))
    with target.pixels.map_to_host() as mapped:
        for pixel_idx in range(target.pixel_count):
            pixels[pixel_idx] = Color(
                mapped[3 * pixel_idx + 0],
                mapped[3 * pixel_idx + 1],
                mapped[3 * pixel_idx + 2],
            )
    return pixels^
