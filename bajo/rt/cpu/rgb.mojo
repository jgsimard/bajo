"""CPU RGB-only rendering without instantiating spectral transport.

These functions mirror the mode-aware CPU entry points and use the same
workspace and renderer implementation. Non-RGB settings raise before modifying
caller-owned workspace state. Use bajo.rt.cpu for spectral or mode-switching
renderers.
"""

from bajo.bvh.cpu.traversal_mode import CpuTraversalMode
from bajo.core.camera import Camera
from bajo.rt.material_types import Integrator
from bajo.rt.render_types import RenderResult, RenderSettings
from .scene import CpuScene
from .scheduler_mode import CpuSchedulerMode
from .wavefront import (
    CPU_WAVEFRONT_PARALLEL_CHUNK_PATHS,
    CpuWavefrontWorkspace,
    _render_wavefront_configured,
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
    """Render RGB with an explicitly selected traversal policy."""
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
    """Render RGB while retaining caller-owned worker queues."""
    return _render_wavefront_configured[
        True,
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
    """Render RGB with the default auto-coherent CPU scene traversal."""
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
