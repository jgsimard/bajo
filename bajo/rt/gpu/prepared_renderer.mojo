"""Persistent prepared GPU renderer ownership."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.bvh import Camera
from bajo.bvh.gpu import GpuBvhBuildMethod
from bajo.rt.gpu.config import (
    GpuRtBvhFormat,
    GpuRtSceneKind,
    GPU_RT_BVH_CWBVH8,
    GPU_RT_BVH_TLAS2,
    GPU_RT_BVH_WIDE4,
)
from bajo.rt.gpu.render import enqueue_render_gpu
from bajo.rt.gpu.resources import (
    GpuRtRenderTarget,
    download_gpu_pixels,
    update_gpu_camera,
)
from bajo.rt.gpu.scene import GpuRtScene, prepare_gpu_scene
from bajo.rt.types import (
    Integrator,
    RenderResult,
    RenderSettings,
    RenderTimings,
    SceneData,
)


struct GpuRtPreparedRenderer[
    kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat = GPU_RT_BVH_WIDE4,
    triangle_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
    tlas_format: GpuRtBvhFormat = GPU_RT_BVH_TLAS2,
    blas_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
](Movable):
    """Own a device context, prepared scene, and reusable render target."""

    var ctx: DeviceContext
    var scene: GpuRtScene[
        Self.kind,
        Self.sphere_format,
        Self.triangle_format,
        Self.tlas_format,
        Self.blas_format,
    ]
    var target: GpuRtRenderTarget

    def __init__[
        triangle_build_method: GpuBvhBuildMethod = .HPLOC,
        tlas_build_method: GpuBvhBuildMethod = .LBVH,
        blas_build_method: GpuBvhBuildMethod = .HPLOC,
    ](
        out self,
        settings: RenderSettings,
        camera: Camera,
        data: SceneData,
    ) raises:
        self.ctx = DeviceContext()
        self.scene = prepare_gpu_scene[
            Self.kind,
            Self.sphere_format,
            Self.triangle_format,
            Self.tlas_format,
            Self.blas_format,
            triangle_build_method,
            tlas_build_method,
            blas_build_method,
        ](self.ctx, data)
        self.target = GpuRtRenderTarget(self.ctx, settings, camera)

    def render[
        integrator: Integrator,
    ](
        mut self,
        settings: RenderSettings,
        camera: Camera,
    ) raises -> RenderResult:
        """Render while retaining the scene and every compatible allocation."""
        comptime assert integrator.is_valid()
        var total_t0 = perf_counter_ns()
        var init_t0 = perf_counter_ns()
        if (
            settings.image_width != self.target.image_width
            or settings.image_height != self.target.image_height
            or settings.samples_per_pixel != self.target.samples_per_pixel
        ):
            self.target = GpuRtRenderTarget(self.ctx, settings, camera)
        else:
            update_gpu_camera(self.ctx, self.target, camera)
        var init_ns = Int(perf_counter_ns() - init_t0)

        var render_t0 = perf_counter_ns()
        enqueue_render_gpu[integrator](
            self.ctx, self.target, self.scene, settings
        )
        var pixels = download_gpu_pixels(self.ctx, self.target)
        var render_ns = Int(perf_counter_ns() - render_t0)
        return RenderResult(
            pixels^,
            RenderTimings(
                Int(perf_counter_ns() - total_t0),
                init_ns,
                render_ns,
                settings.image_width * settings.image_height,
                settings.image_width
                * settings.image_height
                * settings.samples_per_pixel,
                settings.max_depth,
            ),
        )
