"""Persistent prepared GPU renderer ownership."""

from std.time import perf_counter_ns
from std.memory import bitcast
from std.reflection import reflect
from std.builtin.rebind import downcast, rebind
from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder

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
from bajo.rt.gpu.views import (
    GpuRtSceneView,
    GpuRtTraceQueueView,
    gpu_rt_trace_queue_view,
)
from bajo.rt.render_types import (
    RenderResult,
    RenderSettings,
    RenderTimings,
)
from bajo.rt.material_types import Integrator
from bajo.rt.scene_data import SceneData


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

    def matches(self, other: Self) -> Bool:
        return (
            self.integrator == other.integrator
            and self.camera == other.camera
            and self.pixels == other.pixels
            and _same_captured_fields(self.settings, other.settings)
            and _same_scene_capture(self.scene, other.scene)
            and _same_captured_fields(self.queues, other.queues)
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

    def invalidate_graph(mut self):
        """Release the retained graph; the next frame uses direct submission."""
        self._render_graph = Optional[DeviceGraph]()
        self._graph_arguments = Optional[_RenderGraphArguments]()

    def enqueue[
        integrator: Integrator
    ](mut self, settings: RenderSettings) raises:
        """Submit a frame without a download; repeated configurations replay a graph.

        Camera uploads and downstream device operations can use the retained
        context/target. Captured settings include all sampling and film values.
        """
        self.target.validate_compatible(settings)
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
            settings.copy(),
            self.scene.view(),
            queues,
            Int(self.target.camera.unsafe_ptr()),
            Int(self.target.pixels.unsafe_ptr()),
            integrator,
        )
        if (
            not self._graph_arguments
            or not self._graph_arguments.value().matches(arguments)
        ):
            # Changing sample offsets are common in progressive rendering.
            # Submit a new configuration directly; capture only when reused.
            self.invalidate_graph()
            enqueue_render_gpu[integrator](
                self.ctx, self.target, self.scene, settings
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
            ) raises {mut target, imm scene, settings}:
                var recording = builder.recording_context()
                enqueue_render_gpu[integrator](
                    recording, target, scene, settings
                )

            self._render_graph = DeviceGraph.create(ctx, record)
        self._render_graph.value().replay()

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
        if (
            settings.image_width != self.target.image_width
            or settings.image_height != self.target.image_height
            or settings.samples_per_pixel != self.target.samples_per_pixel
        ):
            self.invalidate_graph()
            self.target = self.scene.make_render_target(
                self.ctx, settings, camera
            )
        else:
            update_gpu_camera(self.ctx, self.target, camera)
        var init_ns = Int(perf_counter_ns() - init_t0)

        var render_t0 = perf_counter_ns()
        self.enqueue[integrator](settings)
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
