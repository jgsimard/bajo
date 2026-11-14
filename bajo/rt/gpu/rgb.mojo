"""RGB-only GPU entry points that do not instantiate spectral transport.

These interfaces mirror bajo.rt.gpu and reject non-RGB settings. Use the
mode-aware interfaces when a renderer must switch between RGB and spectral.
"""

from max.gpu.host import DeviceContext

from bajo.bvh.gpu import GpuBvhBuildMethod
from bajo.core.camera import Camera
from bajo.rt.material_types import Integrator
from bajo.rt.render_types import RenderResult, RenderSettings
from bajo.rt.scene_data import SceneData
from .common_kernels import GPU_RT_MAX_BLOCKS
from .config import (
    GpuRtBvhFormat,
    GpuRtSceneKind,
    GPU_RT_BVH_WIDE4,
    GPU_RT_BVH_CWBVH8,
    GPU_RT_BVH_TLAS4,
)
from .prepared_renderer import GpuRtPreparedRenderer as _PreparedRenderer
from .render import (
    _enqueue_render_gpu_rgb,
    _validate_rgb_settings,
    render_gpu as _render_gpu,
    render_gpu_configured as _render_gpu_configured,
)
from .resources import GpuRtRenderTarget
from .scene import GpuRtScene


comptime GpuRtPreparedRenderer[
    kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat = GPU_RT_BVH_WIDE4,
    triangle_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
    tlas_format: GpuRtBvhFormat = GPU_RT_BVH_TLAS4,
    blas_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
] = _PreparedRenderer[
    kind,
    sphere_format,
    triangle_format,
    tlas_format,
    blas_format,
    _rgb_only=True,
]


def enqueue_render_gpu[
    integrator: Integrator = .PATH,
    kind: GpuRtSceneKind = .ALL,
    sphere_format: GpuRtBvhFormat = GPU_RT_BVH_WIDE4,
    triangle_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
    tlas_format: GpuRtBvhFormat = GPU_RT_BVH_TLAS4,
    blas_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
    MAX_BLOCKS: Int = GPU_RT_MAX_BLOCKS,
    SHADOW_MAX_BLOCKS: Int = MAX_BLOCKS,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtScene[
        kind, sphere_format, triangle_format, tlas_format, blas_format
    ],
    settings: RenderSettings,
    sample_offset: Optional[Pointer[UInt32, ImmUntrackedOrigin]] = {},
) raises:
    """Submit RGB transport without compiling the spectral scheduler."""
    _validate_rgb_settings(settings)
    _enqueue_render_gpu_rgb[
        integrator,
        kind,
        sphere_format,
        triangle_format,
        tlas_format,
        blas_format,
        MAX_BLOCKS,
        SHADOW_MAX_BLOCKS,
    ](ctx, target, world, settings, sample_offset)


def render_gpu_configured[
    kind: GpuRtSceneKind,
    integrator: Integrator = .PATH,
    sphere_format: GpuRtBvhFormat = GPU_RT_BVH_WIDE4,
    triangle_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
    tlas_format: GpuRtBvhFormat = GPU_RT_BVH_TLAS4,
    blas_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
    triangle_build_method: GpuBvhBuildMethod = .HPLOC,
    tlas_build_method: GpuBvhBuildMethod = .HPLOC,
    blas_build_method: GpuBvhBuildMethod = .HPLOC,
](
    settings: RenderSettings,
    camera: Camera,
    world: SceneData,
) raises -> RenderResult:
    """Render RGB with an explicitly selected geometry and builder set."""
    return _render_gpu_configured[
        kind,
        integrator,
        sphere_format,
        triangle_format,
        tlas_format,
        blas_format,
        triangle_build_method,
        tlas_build_method,
        blas_build_method,
        _rgb_only=True,
    ](settings, camera, world)


def render_gpu[
    integrator: Integrator = .PATH,
](
    settings: RenderSettings,
    camera: Camera,
    world: SceneData,
) raises -> RenderResult:
    """Render RGB with automatic geometry selection and default BVH formats."""
    return _render_gpu[integrator, _rgb_only=True](settings, camera, world)
