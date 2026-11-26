"""A native render worker keeps scene ownership and GPU contexts off the UI thread.

One POSIX thread serializes preparation and rendering. Release/acquire atomics
publish immutable requests and completed frames; raylib is never called here.
"""

from std.atomic import Atomic, Ordering
from std.time import perf_counter_ns
from std.ffi import external_call, c_int
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.sys import has_accelerator
from bajo.core.camera import Camera
from bajo.rt.material_types import Integrator
from bajo.core.color import Color
from bajo.rt.render_types import RenderSettings, RenderResult
from bajo.rt.cpu.wavefront import (
    render_wavefront_configured,
    CpuWavefrontWorkspace,
)
from bajo.bvh.cpu import CpuTraversalMode
from bajo.rt.cpu.scene import CpuScene
from bajo.rt.gpu.prepared_renderer import GpuRtPreparedRenderer
from bajo.rt.preview_types import PreviewResult
from bajo.rt.gpu.config import GpuRtSceneKind
from bajo.viewer.core import ViewerScene
from .debug import DebugBox, cpu_bvh_boxes
from .preview import CpuPreviewAccumulator
from bajo.rt.render_types import RenderTimings
from .notification import create_notification, notify, wait_notification


def _render_backend[
    integrator: Integrator, kind: GpuRtSceneKind
](
    mut renderer: GpuRtPreparedRenderer[kind],
    settings: RenderSettings,
    camera: Camera,
    preview: Bool,
    generation: UInt64,
    mut output: Optional[PreviewResult],
    denoise: Bool = False,
    refresh: Bool = False,
    preparation_ns: Int = 0,
) raises -> RenderResult:
    if preview:
        if refresh:
            output = renderer.refresh_preview(camera, generation, denoise)
        else:
            output = renderer.preview[integrator](
                settings, camera, generation, denoise
            )
        output.value().preparation_ns += preparation_ns
        return RenderResult(List[Color](), output.value().timings)
    return renderer.render[integrator](settings, camera)


struct GpuBackend:
    var preview_result: Optional[PreviewResult]
    var spheres: Optional[GpuRtPreparedRenderer[.SPHERES]]
    var triangles: Optional[GpuRtPreparedRenderer[.TRIANGLES]]
    var spheres_triangles: Optional[GpuRtPreparedRenderer[.SPHERES_TRIANGLES]]
    var instances: Optional[GpuRtPreparedRenderer[.INSTANCES]]
    var spheres_instances: Optional[GpuRtPreparedRenderer[.SPHERES_INSTANCES]]
    var triangles_instances: Optional[
        GpuRtPreparedRenderer[.TRIANGLES_INSTANCES]
    ]
    var all: Optional[GpuRtPreparedRenderer[.ALL]]

    def __init__(out self):
        self.preview_result = {}
        self.spheres = {}
        self.triangles = {}
        self.spheres_triangles = {}
        self.instances = {}
        self.spheres_instances = {}
        self.triangles_instances = {}
        self.all = {}

    def render[
        integrator: Integrator
    ](
        mut self,
        scene: ViewerScene[],
        settings: RenderSettings,
        camera: Camera,
        preview: Bool = False,
        generation: UInt64 = 0,
        denoise: Bool = False,
        refresh: Bool = False,
    ) raises -> RenderResult:
        self.preview_result = {}
        var preparation_t0 = perf_counter_ns()
        ref data = scene.scene_data()
        var mask = (
            Int(len(data.spheres()) > 0)
            + 2 * Int(len(data.triangle_vertices()) > 0)
            + 4 * Int(len(data.triangle_instances()) > 0)
        )
        match mask:
            case 1:
                if not self.spheres:
                    self.spheres = GpuRtPreparedRenderer[.SPHERES](
                        settings, camera, data
                    )
                return _render_backend[integrator](
                    self.spheres.value(),
                    settings,
                    camera,
                    preview,
                    generation,
                    self.preview_result,
                    denoise,
                    refresh,
                    Int(perf_counter_ns() - preparation_t0),
                )
            case 2:
                if not self.triangles:
                    self.triangles = GpuRtPreparedRenderer[.TRIANGLES](
                        settings, camera, data
                    )
                return _render_backend[integrator](
                    self.triangles.value(),
                    settings,
                    camera,
                    preview,
                    generation,
                    self.preview_result,
                    denoise,
                    refresh,
                    Int(perf_counter_ns() - preparation_t0),
                )
            case 3:
                if not self.spheres_triangles:
                    self.spheres_triangles = GpuRtPreparedRenderer[
                        .SPHERES_TRIANGLES
                    ](settings, camera, data)
                return _render_backend[integrator](
                    self.spheres_triangles.value(),
                    settings,
                    camera,
                    preview,
                    generation,
                    self.preview_result,
                    denoise,
                    refresh,
                    Int(perf_counter_ns() - preparation_t0),
                )
            case 4:
                if not self.instances:
                    self.instances = GpuRtPreparedRenderer[.INSTANCES](
                        settings, camera, data
                    )
                return _render_backend[integrator](
                    self.instances.value(),
                    settings,
                    camera,
                    preview,
                    generation,
                    self.preview_result,
                    denoise,
                    refresh,
                    Int(perf_counter_ns() - preparation_t0),
                )
            case 5:
                if not self.spheres_instances:
                    self.spheres_instances = GpuRtPreparedRenderer[
                        .SPHERES_INSTANCES
                    ](settings, camera, data)
                return _render_backend[integrator](
                    self.spheres_instances.value(),
                    settings,
                    camera,
                    preview,
                    generation,
                    self.preview_result,
                    denoise,
                    refresh,
                    Int(perf_counter_ns() - preparation_t0),
                )
            case 6:
                if not self.triangles_instances:
                    self.triangles_instances = GpuRtPreparedRenderer[
                        .TRIANGLES_INSTANCES
                    ](settings, camera, data)
                return _render_backend[integrator](
                    self.triangles_instances.value(),
                    settings,
                    camera,
                    preview,
                    generation,
                    self.preview_result,
                    denoise,
                    refresh,
                    Int(perf_counter_ns() - preparation_t0),
                )
            case 7:
                if not self.all:
                    self.all = GpuRtPreparedRenderer[.ALL](
                        settings, camera, data
                    )
                return _render_backend[integrator](
                    self.all.value(),
                    settings,
                    camera,
                    preview,
                    generation,
                    self.preview_result,
                    denoise,
                    refresh,
                    Int(perf_counter_ns() - preparation_t0),
                )
            case _:
                raise Error("scene has no supported geometry")


struct CompletedFrame:
    var rgba: List[UInt8]
    var preview_samples: Int
    var preview_batches: Int
    var pixels: List[Color]
    var generation: UInt64
    var presentation_revision: UInt64
    var samples: Int
    var width: Int
    var height: Int
    var render_ms: Float64
    var preparation_ns: Int
    var batch_ns: Int
    var postprocess_ns: Int
    var work_ns: Int

    def __init__(
        out self,
        var result: RenderResult,
        generation: UInt64,
        settings: RenderSettings,
    ):
        self.rgba = {}
        self.preview_samples = 0
        self.preview_batches = 0
        self.render_ms = Float64(result.timings.render_ns) / 1_000_000
        self.preparation_ns = result.timings.init_ns
        self.batch_ns = result.timings.render_ns
        self.postprocess_ns = 0
        self.work_ns = result.timings.total_ns
        self.pixels = result.pixels^
        result.pixels = List[Color]()
        self.generation = generation
        self.presentation_revision = 0
        self.samples = settings.samples_per_pixel
        self.width = settings.image_width
        self.height = settings.image_height

    @staticmethod
    def from_preview(
        var result: PreviewResult,
        generation: UInt64,
        settings: RenderSettings,
    ) -> Self:
        var frame = Self(
            RenderResult(List[Color](), result.timings), generation, settings
        )
        frame.rgba = result.rgba^
        result.rgba = List[UInt8]()
        frame.preview_samples = result.samples
        frame.preview_batches = result.batches
        frame.preparation_ns = result.preparation_ns
        frame.batch_ns = result.batch_ns
        frame.postprocess_ns = result.postprocess_ns
        return frame^


def _render_cpu[
    traversal: CpuTraversalMode, integrator: Integrator
](
    settings: RenderSettings,
    camera: Camera,
    world: CpuScene[16, 16],
    mut workspace: CpuWavefrontWorkspace[16, integrator],
) raises -> RenderResult:
    var preview_settings = settings.copy()
    preview_settings.collect_denoise = integrator != Integrator.NORMALS
    return render_wavefront_configured[
        traversal,
        integrator,
        16,
        1024,
        True,
        .TASK_PARTITIONS,
        16,
        16,
        16,
        8,
        4,
    ](preview_settings, camera, world, workspace)


struct _WorkerState[gpu_enabled: Bool]:
    var cpu_path: CpuWavefrontWorkspace[16, .PATH]
    var cpu_nee: CpuWavefrontWorkspace[16, .NEE]
    var cpu_mis: CpuWavefrontWorkspace[16, .MIS]
    var cpu_normals: CpuWavefrontWorkspace[16, .NORMALS]
    var cpu_ao: CpuWavefrontWorkspace[16, .AO]
    var scene: ViewerScene[]
    var gpu: GpuBackend
    var camera: Camera
    var settings: RenderSettings
    var integrator: Int
    var backend: Int
    var generation: UInt64
    var status: UInt32
    var stop: UInt32
    var notification: c_int
    var result: Optional[CompletedFrame]
    var error: String
    var build_method: Int
    var traversal: Int
    var debug_requested: Bool
    var trace_requested: Bool
    var boxes: List[DebugBox]
    var cpu_preview: Optional[CpuPreviewAccumulator]
    var preview_requested: Bool
    var refresh_requested: Bool
    var denoise_enabled: Bool
    var presentation_revision: UInt64
    var operation_preparation_ns: Int
    var operation_ns: Int

    def __init__(
        out self, var scene: ViewerScene[], build_method: Int, traversal: Int
    ):
        self.cpu_path = {}
        self.cpu_nee = {}
        self.cpu_mis = {}
        self.cpu_normals = {}
        self.cpu_ao = {}
        self.build_method = build_method
        self.traversal = traversal
        self.camera = scene.state.camera
        self.settings = scene.state.settings.copy()
        self.scene = scene^
        self.gpu = {}
        self.integrator = 0
        self.backend = 0
        self.generation = 0
        self.status = 0
        self.stop = 0
        self.notification = -1
        self.result = {}
        self.error = ""
        self.debug_requested = False
        self.trace_requested = True
        self.boxes = {}
        self.cpu_preview = {}
        self.preview_requested = False
        self.refresh_requested = False
        self.denoise_enabled = False
        self.presentation_revision = 0
        self.operation_preparation_ns = 0
        self.operation_ns = 0

    def take_boxes(mut self) -> List[DebugBox]:
        var boxes = self.boxes^
        self.boxes = {}
        return boxes^

    def render_integrator[
        integrator: Integrator
    ](mut self) raises -> RenderResult:
        if self.backend == 1:
            comptime if not Self.gpu_enabled:
                raise Error("GPU tracing is disabled for this CPU-only worker")
            elif has_accelerator():
                return self.gpu.render[integrator](
                    self.scene,
                    self.settings,
                    self.camera,
                    True,
                    self.generation,
                    self.denoise_enabled and self.integrator != 3,
                    self.refresh_requested,
                )
            else:
                raise Error(
                    "GPU tracing is unavailable in this build; launch"
                    " viewer_gpu"
                )
        self.prepare_cpu()
        if self.traversal == 1:
            return self.render_cpu[.FIXED_PACKET, integrator]()
        if self.traversal == 2:
            return self.render_cpu[.ADAPTIVE, integrator]()
        return self.render_cpu[.AUTO_COHERENT, integrator]()

    def render_cpu[
        traversal: CpuTraversalMode, integrator: Integrator
    ](mut self) raises -> RenderResult:
        comptime if integrator == .PATH:
            return _render_cpu[traversal, .PATH](
                self.settings,
                self.camera,
                self.scene.cpu_scene(),
                self.cpu_path,
            )
        elif integrator == .NEE:
            return _render_cpu[traversal, .NEE](
                self.settings, self.camera, self.scene.cpu_scene(), self.cpu_nee
            )
        elif integrator == .MIS:
            return _render_cpu[traversal, .MIS](
                self.settings, self.camera, self.scene.cpu_scene(), self.cpu_mis
            )
        elif integrator == .NORMALS:
            return _render_cpu[traversal, .NORMALS](
                self.settings,
                self.camera,
                self.scene.cpu_scene(),
                self.cpu_normals,
            )
        else:
            return _render_cpu[traversal, .AO](
                self.settings, self.camera, self.scene.cpu_scene(), self.cpu_ao
            )

    def prepare_cpu(mut self) raises:
        var already_prepared = self.scene.has_cpu_scene()
        var t0 = perf_counter_ns()
        match self.build_method:
            case 1:
                self.scene.prepare_cpu[.LBVH]()
            case 2:
                self.scene.prepare_cpu[.HPLOC]()
            case 3:
                self.scene.prepare_cpu[.MEDIAN]()
            case _:
                self.scene.prepare_cpu()

        if not already_prepared:
            self.operation_preparation_ns += Int(perf_counter_ns() - t0)

    def render(mut self) raises:
        var t0 = perf_counter_ns()
        self.operation_preparation_ns = 0
        self.operation_ns = 0
        try:
            self._render()
        except error:
            self.operation_ns = Int(perf_counter_ns() - t0)
            raise error
        self.operation_ns = Int(perf_counter_ns() - t0)
        if case .Some(ref result) = self.result:
            result.preparation_ns += self.operation_preparation_ns
            result.work_ns = self.operation_ns

    def _render(mut self) raises:
        if self.debug_requested:
            self.prepare_cpu()
            self.boxes = cpu_bvh_boxes(self.scene.cpu_scene())
        if not self.trace_requested:
            return
        if self.refresh_requested and self.backend == 0:
            guard case .Some(ref preview) = self.cpu_preview;
              preview.generation == self.generation else:
                raise Error("CPU preview generation is no longer available")
            self._publish_cpu_preview(
                RenderTimings(
                    0,
                    0,
                    0,
                    self.settings.image_width * self.settings.image_height,
                    0,
                    self.settings.max_depth,
                )
            )
            return
        var result: RenderResult
        match self.integrator:
            case 0:
                result = self.render_integrator[.PATH]()
            case 1:
                result = self.render_integrator[.NEE]()
            case 2:
                result = self.render_integrator[.MIS]()
            case 3:
                result = self.render_integrator[.NORMALS]()
            case 4:
                result = self.render_integrator[.AO]()
            case _:
                raise Error("unknown integrator")
        if self.backend == 1; case .Some(ref preview) = self.gpu.preview_result:
            self.error = preview.postprocess_error.copy()
            self.result = CompletedFrame.from_preview(
                self.gpu.preview_result.take(), self.generation, self.settings
            )
        elif self.preview_requested:
            if (
                not self.cpu_preview
                or self.cpu_preview.value().width != self.settings.image_width
                or self.cpu_preview.value().height != self.settings.image_height
            ):
                self.cpu_preview = CpuPreviewAccumulator(
                    self.settings.image_width, self.settings.image_height
                )
            var post_t0 = perf_counter_ns()
            self.cpu_preview.value().add(
                result.pixels,
                self.generation,
                self.settings.samples_per_pixel,
                result.denoise_samples,
            )
            self._publish_cpu_preview(result.timings)
            self.result.value().postprocess_ns = Int(
                perf_counter_ns() - post_t0
            )
        else:
            self.result = CompletedFrame(
                result^, self.generation, self.settings
            )
        if case .Some(ref completed) = self.result:
            completed.presentation_revision = self.presentation_revision
            if self.refresh_requested:
                completed.samples = 0

    def _publish_cpu_preview(mut self, timings: RenderTimings) raises:
        var post_t0 = perf_counter_ns()
        ref preview = self.cpu_preview.value()
        var rgba: List[UInt8]
        try:
            rgba = preview.present(
                self.scene.cpu_scene(),
                self.camera,
                self.denoise_enabled and self.integrator != 3,
            )
        except error:
            if not self.denoise_enabled:
                raise Error(String(error))
            rgba = preview.present(self.scene.cpu_scene(), self.camera, False)
            self.error = "Denoising failed: " + String(error)
        self.result = CompletedFrame.from_preview(
            PreviewResult(
                rgba^,
                timings.copy(),
                preview.samples,
                preview.batches,
                postprocess_ns=Int(perf_counter_ns() - post_t0),
            ),
            self.generation,
            self.settings,
        )
        self.result.value().presentation_revision = self.presentation_revision
        if self.refresh_requested:
            self.result.value().samples = 0


def _run_render_thread[gpu_enabled: Bool](address: UInt) -> UInt:
    var state = Pointer[_WorkerState[gpu_enabled], MutUntrackedOrigin](
        unsafe_from_address=Int(address)
    )
    while not Atomic.load[ordering=Ordering.acquire](Pointer(to=state[].stop)):
        if (
            Atomic.load[ordering=Ordering.acquire](Pointer(to=state[].status))
            == 1
        ):
            try:
                state[].render()
            except error:
                state[].error = String(error)
            Atomic.store[ordering=Ordering.release](
                Pointer(to=state[].status), UInt32(2)
            )
        else:
            wait_notification(state[].notification)
    return 0


# C ABI is sufficient for pthread_create. Exporting these callbacks would force
# both rendering paths to compile even when only CpuRenderWorker is used.
def _bajo_viewer_render_thread(address: UInt) abi("C") -> UInt:
    return _run_render_thread[True](address)


def _bajo_viewer_cpu_render_thread(address: UInt) abi("C") -> UInt:
    return _run_render_thread[False](address)


struct _RenderWorker[gpu_enabled: Bool]:
    var state: Pointer[_WorkerState[Self.gpu_enabled], MutUntrackedOrigin]
    var thread: UInt

    def __init__(
        out self,
        var scene: ViewerScene[],
        build_method: Int = 0,
        traversal: Int = 0,
    ) raises:
        var state = unsafe_alloc[_WorkerState[Self.gpu_enabled]](1)
        state.unsafe_write(
            _WorkerState[Self.gpu_enabled](scene^, build_method, traversal)
        )
        state[].notification = create_notification()
        var thread = UInt(0)
        var code: c_int
        comptime if Self.gpu_enabled:
            code = external_call["pthread_create", c_int](
                Pointer(to=thread),
                UInt(0),
                _bajo_viewer_render_thread,
                UInt(Int(state)),
            )
        else:
            code = external_call["pthread_create", c_int](
                Pointer(to=thread),
                UInt(0),
                _bajo_viewer_cpu_render_thread,
                UInt(Int(state)),
            )
        if code:
            if state[].notification >= 0:
                _ = external_call["close", c_int](state[].notification)
            state.unsafe_deinit_pointee()
            state.unsafe_free()
            raise Error("could not start native rendering thread")
        self.state = state.unsafe_origin_cast[MutUntrackedOrigin]()
        self.thread = thread

    def idle(self) -> Bool:
        return (
            Atomic.load[ordering=Ordering.acquire](
                Pointer(to=self.state[].status)
            )
            == 0
        )

    def ready(self) -> Bool:
        return (
            Atomic.load[ordering=Ordering.acquire](
                Pointer(to=self.state[].status)
            )
            == 2
        )

    def submit(
        self,
        camera: Camera,
        settings: RenderSettings,
        generation: UInt64,
        integrator: Int,
        backend: Int,
        trace: Bool,
        debug: Bool,
        preview: Bool = False,
        refresh: Bool = False,
        denoise: Bool = False,
        presentation_revision: UInt64 = 0,
    ) raises:
        if not self.idle():
            raise Error("render worker is busy")
        self.state[].camera = camera
        self.state[].settings = settings.copy()
        self.state[].generation = generation
        self.state[].integrator = integrator
        self.state[].backend = backend
        self.state[].trace_requested = trace
        self.state[].debug_requested = debug
        self.state[].preview_requested = preview
        self.state[].refresh_requested = refresh
        self.state[].denoise_enabled = denoise
        self.state[].presentation_revision = presentation_revision
        self.state[].error = ""
        Atomic.store[ordering=Ordering.release](
            Pointer(to=self.state[].status), UInt32(1)
        )
        notify(self.state[].notification)

    def submit_preview(
        self,
        camera: Camera,
        settings: RenderSettings,
        generation: UInt64,
        integrator: Int,
        backend: Int,
        trace: Bool,
        debug: Bool,
        denoise: Bool,
        presentation_revision: UInt64,
    ) raises:
        self.submit(
            camera,
            settings,
            generation,
            integrator,
            backend,
            trace,
            debug,
            True,
            False,
            denoise,
            presentation_revision,
        )

    def refresh_preview(
        self,
        camera: Camera,
        settings: RenderSettings,
        generation: UInt64,
        integrator: Int,
        backend: Int,
        denoise: Bool,
        presentation_revision: UInt64,
    ) raises:
        self.submit(
            camera,
            settings,
            generation,
            integrator,
            backend,
            True,
            False,
            True,
            True,
            denoise,
            presentation_revision,
        )

    def acknowledge(self):
        # Call only after consuming the published result/error/debug snapshot.
        Atomic.store[ordering=Ordering.release](
            Pointer(to=self.state[].status), UInt32(0)
        )

    def __deinit__(deinit self):
        Atomic.store[ordering=Ordering.release](
            Pointer(to=self.state[].stop), UInt32(1)
        )
        notify(self.state[].notification)
        _ = external_call["pthread_join", c_int](self.thread, UInt(0))
        if self.state[].notification >= 0:
            _ = external_call["close", c_int](self.state[].notification)
        self.state.unsafe_deinit_pointee()
        self.state.unsafe_free()


# Preserve the default worker interface and GPU support. CPU-only clients use
# the same native thread/queues without instantiating the GPU render dispatch.
comptime WorkerState = _WorkerState[True]
comptime RenderWorker = _RenderWorker[True]
comptime CpuRenderWorker = _RenderWorker[False]
