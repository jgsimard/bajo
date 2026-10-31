"""Persistent prepared GPU renderer ownership."""

from std.time import perf_counter_ns
from std.memory import bitcast
from std.reflection import reflect
from std.builtin.rebind import downcast, rebind
from max.gpu.host import (
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
    DeviceBuffer,
    HostBuffer,
    DeviceEvent,
)

from bajo.core.camera import Camera
from bajo.bvh.gpu import GpuBvhBuildMethod
from bajo.rt.gpu.config import (
    GpuRtBvhFormat,
    GpuRtSceneKind,
    GPU_RT_BVH_CWBVH8,
    GPU_RT_BVH_TLAS4,
    GPU_RT_BVH_WIDE4,
)
from bajo.rt.gpu.render import enqueue_render_gpu
from bajo.rt.gpu.resources import (
    GpuRtRenderTarget,
    _GpuRtReadback,
    download_gpu_pixels,
    update_gpu_camera,
)
from bajo.rt.gpu.preview import GpuRtPreviewAccumulator, GpuRtPreviewResult
from bajo.rt.gpu.scene import GpuRtScene, prepare_gpu_scene
from bajo.rt.gpu.views import (
    GpuRtSceneView,
    GpuRtTraceQueueView,
    gpu_rt_trace_queue_view,
)
from bajo.rt.render_types import RenderResult, RenderSettings, RenderTimings
from bajo.rt.material_types import Integrator
from bajo.rt.scene_data import SceneData
from bajo.rt.filter import FilterView


@always_inline
def _same_captured_fields[T: AnyType](a: T, b: T) -> Bool:
    # Compare initialized fields rather than struct padding. Float constants
    # compare by bits so a changed signed zero also invalidates the capture.
    comptime if T == Float32:
        return bitcast[.uint32](rebind[Float32](a)) == bitcast[.uint32](
            rebind[Float32](b)
        )
    elif T == Float64:
        return bitcast[.uint64](rebind[Float64](a)) == bitcast[.uint64](
            rebind[Float64](b)
        )
    elif conforms_to(T, Equatable):
        return rebind[downcast[T, Equatable]](a) == rebind[
            downcast[T, Equatable]
        ](b)
    else:
        comptime for field in range(reflect[T].field_count()):
            if not _same_captured_fields(
                __struct_field_ref(field, a), __struct_field_ref(field, b)
            ):
                return False
        return True


def _same_optional_capture[T: Copyable](a: Optional[T], b: Optional[T]) -> Bool:
    if Bool(a) != Bool(b):
        return False
    return not a or _same_captured_fields(a.value(), b.value())


def _same_scene_capture(a: GpuRtSceneView, b: GpuRtSceneView) -> Bool:
    # Optional payloads are read only when present. All other scene fields
    # participate automatically, including buffer addresses and environment.
    if not (
        _same_optional_capture(a.spheres, b.spheres)
        and _same_optional_capture(a.triangles, b.triangles)
        and _same_optional_capture(a.instances, b.instances)
    ):
        return False
    comptime names = reflect[GpuRtSceneView].field_names()
    comptime for field in range(reflect[GpuRtSceneView].field_count()):
        comptime if names[field] != "spheres" and names[
            field
        ] != "triangles" and names[field] != "instances":
            if not _same_captured_fields(
                __struct_field_ref(field, a), __struct_field_ref(field, b)
            ):
                return False
    return True


@fieldwise_init
struct _RenderGraphArguments(Copyable):
    var settings: RenderSettings
    var scene: GpuRtSceneView
    var queues: GpuRtTraceQueueView
    var camera: Int
    var pixels: Int
    var integrator: Integrator
    var filter: FilterView

    def matches(self, other: Self) -> Bool:
        return (
            self.integrator == other.integrator
            and self.camera == other.camera
            and self.pixels == other.pixels
            and _same_captured_fields(self.settings, other.settings)
            and _same_scene_capture(self.scene, other.scene)
            and _same_captured_fields(self.queues, other.queues)
            and _same_captured_fields(self.filter, other.filter)
        )


struct GpuRtPreparedRenderer[
    kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat = GPU_RT_BVH_WIDE4,
    triangle_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
    tlas_format: GpuRtBvhFormat = GPU_RT_BVH_TLAS4,
    blas_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
]:
    """Own a device context, prepared scene, and reusable render target."""

    var ctx: DeviceContext[]
    var scene: GpuRtScene[
        Self.kind,
        Self.sphere_format,
        Self.triangle_format,
        Self.tlas_format,
        Self.blas_format,
    ]
    var target: GpuRtRenderTarget
    var _sample_offset: DeviceBuffer[.uint32]
    var _sample_staging: HostBuffer[.uint32]
    var _sample_complete: DeviceEvent
    var _sample_pending: Bool
    var _last_offset: UInt32
    var _readback: Optional[_GpuRtReadback]
    var _preview: Optional[GpuRtPreviewAccumulator]
    var _render_graph: Optional[DeviceGraph]
    var _graph_arguments: Optional[_RenderGraphArguments]

    def __init__[
        triangle_build_method: GpuBvhBuildMethod = .HPLOC,
        tlas_build_method: GpuBvhBuildMethod = .HPLOC,
        blas_build_method: GpuBvhBuildMethod = .HPLOC,
    ](
        out self,
        settings: RenderSettings,
        camera: Camera,
        data: SceneData,
    ) raises:
        self._readback = Optional[_GpuRtReadback]()
        self._preview = Optional[GpuRtPreviewAccumulator]()
        self._render_graph = Optional[DeviceGraph]()
        self._graph_arguments = Optional[_RenderGraphArguments]()
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
        self.target = self.scene.make_render_target(self.ctx, settings, camera)
        self._sample_offset = self.ctx.enqueue_create_buffer[.uint32](1)
        self._sample_staging = self.ctx.enqueue_create_host_buffer[.uint32](1)
        self._sample_complete = self.ctx.create_event()
        self._sample_pending = False
        self._last_offset = 0

    def invalidate_graph(mut self):
        """Release the retained graph; the next frame uses direct submission."""
        self._render_graph = Optional[DeviceGraph]()
        self._graph_arguments = Optional[_RenderGraphArguments]()

    def enqueue[
        integrator: Integrator
    ](mut self, settings: RenderSettings) raises:
        """Submit a frame without a download; repeated configurations replay a graph.

        Camera uploads and downstream device operations can use the retained
        context/target. The sample offset is uploaded separately; other sampling and film
        values remain part of the graph configuration.
        """
        self.target.validate_compatible(settings)
        if not self._sample_pending or self._last_offset != UInt32(
            settings.sample_offset
        ):
            if self._sample_pending:
                self._sample_complete.synchronize()
            self._sample_staging.as_span()[0] = UInt32(settings.sample_offset)
            self._sample_staging.enqueue_copy_to(self._sample_offset)
            self.ctx.stream().record_event(self._sample_complete)
            self._sample_pending = True
            self._last_offset = UInt32(settings.sample_offset)
        var captured_settings = settings.copy()
        captured_settings.sample_offset = 0
        var offset = (
            self._sample_offset.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmUntrackedOrigin]()
        )
        self.target.filter.update(self.ctx, settings.filter)
        var queues = gpu_rt_trace_queue_view(
            self.target.arena,
            self.target.arena.path_a.path_ids,
            self.target.arena.path_a.fields,
            self.target.arena.path_b.path_ids,
            self.target.arena.path_b.fields,
        )
        queues.sample_base = (
            0  # The scheduler sets this independently per chunk.
        )
        var arguments = _RenderGraphArguments(
            captured_settings.copy(),
            self.scene.view(),
            queues,
            Int(self.target.camera.unsafe_ptr()),
            Int(self.target.pixels.unsafe_ptr()),
            integrator,
            self.target.filter.view(),
        )
        if (
            not self._graph_arguments
            or not self._graph_arguments.value().matches(arguments)
        ):
            # Structural settings and buffer changes still submit directly;
            # changing only the sample offset reuses the recorded sequence.
            self.invalidate_graph()
            enqueue_render_gpu[integrator](
                self.ctx,
                self.target,
                self.scene,
                captured_settings,
                Optional(offset),
            )
            self._graph_arguments = arguments^
            return
        if not self._render_graph:
            # Capture the existing scheduler, keeping one implementation of
            # the bounce sequence. The owner retains all referenced buffers.
            var ctx = self.ctx.copy()
            ref target = self.target
            ref scene = self.scene

            def record(
                mut builder: DeviceGraphBuilder,
            ) raises {mut target, imm scene, captured_settings, offset}:
                var recording = builder.recording_context()
                enqueue_render_gpu[integrator](
                    recording,
                    target,
                    scene,
                    captured_settings,
                    Optional(offset),
                )

            self._render_graph = DeviceGraph.create(ctx, record)
        self._render_graph.value().replay()

    def _prepare_frame(
        mut self, settings: RenderSettings, camera: Camera
    ) raises:
        settings.validate()
        var resized = (
            settings.image_width != self.target.image_width
            or settings.image_height != self.target.image_height
        )
        if (
            settings.image_width != self.target.image_width
            or settings.image_height != self.target.image_height
            or settings.samples_per_pixel != self.target.samples_per_pixel
        ):
            self.invalidate_graph()
            if resized:
                self._readback = Optional[_GpuRtReadback]()
                self._preview = Optional[GpuRtPreviewAccumulator]()
            self.target = self.scene.make_render_target(
                self.ctx, settings, camera
            )
        else:
            update_gpu_camera(self.ctx, self.target, camera)

    def render[
        integrator: Integrator,
    ](
        mut self,
        settings: RenderSettings,
        camera: Camera,
    ) raises -> RenderResult:
        """Render while retaining the scene and every compatible allocation."""
        var total_t0 = perf_counter_ns()
        var init_t0 = perf_counter_ns()
        self._prepare_frame(settings, camera)
        if not self._readback:
            self._readback = _GpuRtReadback(self.ctx, self.target.pixel_count)
        var init_ns = Int(perf_counter_ns() - init_t0)

        var render_t0 = perf_counter_ns()
        self.enqueue[integrator](settings)
        var pixels = self._readback.value().download(self.ctx, self.target)
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

    def preview[
        integrator: Integrator
    ](
        mut self, settings: RenderSettings, camera: Camera, generation: UInt64
    ) raises -> GpuRtPreviewResult:
        """Render a weighted progressive preview; advance generation on reset.

        Only RGBA is downloaded. Full Float32 output remains available through
        render(); changed image dimensions clear the retained preview accumulation.
        """
        var total_t0 = perf_counter_ns()
        self._prepare_frame(settings, camera)
        if not self._preview:
            self._preview = GpuRtPreviewAccumulator(
                self.ctx, self.target.pixel_count
            )
        var init_ns = Int(perf_counter_ns() - total_t0)
        var render_t0 = perf_counter_ns()
        self.enqueue[integrator](settings)
        var rgba = self._preview.value().add(
            self.ctx, self.target, generation, settings.samples_per_pixel
        )
        return GpuRtPreviewResult(
            rgba^,
            RenderTimings(
                Int(perf_counter_ns() - total_t0),
                init_ns,
                Int(perf_counter_ns() - render_t0),
                self.target.pixel_count,
                self.target.sample_count,
                settings.max_depth,
            ),
            self._preview.value().samples,
            self._preview.value().batches,
        )
