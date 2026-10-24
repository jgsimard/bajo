"""A native render worker keeps scene ownership and GPU contexts off the UI thread.

One POSIX thread serializes preparation and rendering. Release/acquire atomics
publish immutable requests and completed frames; raylib is never called here.
"""

from std.atomic import Atomic, Ordering
from std.ffi import external_call, c_int
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.sys import has_accelerator
from bajo.bvh import Camera
from bajo.rt.material_types import Integrator
from bajo.rt.render_types import RenderSettings, RenderResult, Color
from bajo.rt.cpu.wavefront import render_wavefront_configured
from bajo.bvh.cpu import CpuTraversalMode
from bajo.rt.gpu.prepared_renderer import GpuRtPreparedRenderer
from bajo.viewer.core import ViewerScene
from .debug import DebugBox, cpu_bvh_boxes


struct GpuBackend:
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
        self.spheres = Optional[GpuRtPreparedRenderer[.SPHERES]]()
        self.triangles = Optional[GpuRtPreparedRenderer[.TRIANGLES]]()
        self.spheres_triangles = Optional[
            GpuRtPreparedRenderer[.SPHERES_TRIANGLES]
        ]()
        self.instances = Optional[GpuRtPreparedRenderer[.INSTANCES]]()
        self.spheres_instances = Optional[
            GpuRtPreparedRenderer[.SPHERES_INSTANCES]
        ]()
        self.triangles_instances = Optional[
            GpuRtPreparedRenderer[.TRIANGLES_INSTANCES]
        ]()
        self.all = Optional[GpuRtPreparedRenderer[.ALL]]()

    def render[
        integrator: Integrator
    ](
        mut self, scene: ViewerScene[], settings: RenderSettings, camera: Camera
    ) raises -> RenderResult:
        ref data = scene.scene_data()
        var mask = (
            Int(len(data.spheres()) > 0)
            + 2 * Int(len(data.triangle_vertices()) > 0)
            + 4 * Int(len(data.triangle_instances()) > 0)
        )
        if mask == 1:
            if not self.spheres:
                self.spheres = GpuRtPreparedRenderer[.SPHERES](
                    settings, camera, data
                )
            return self.spheres.value().render[integrator](settings, camera)
        if mask == 2:
            if not self.triangles:
                self.triangles = GpuRtPreparedRenderer[.TRIANGLES](
                    settings, camera, data
                )
            return self.triangles.value().render[integrator](settings, camera)
        if mask == 3:
            if not self.spheres_triangles:
                self.spheres_triangles = GpuRtPreparedRenderer[
                    .SPHERES_TRIANGLES
                ](settings, camera, data)
            return self.spheres_triangles.value().render[integrator](
                settings, camera
            )
        if mask == 4:
            if not self.instances:
                self.instances = GpuRtPreparedRenderer[.INSTANCES](
                    settings, camera, data
                )
            return self.instances.value().render[integrator](settings, camera)
        if mask == 5:
            if not self.spheres_instances:
                self.spheres_instances = GpuRtPreparedRenderer[
                    .SPHERES_INSTANCES
                ](settings, camera, data)
            return self.spheres_instances.value().render[integrator](
                settings, camera
            )
        if mask == 6:
            if not self.triangles_instances:
                self.triangles_instances = GpuRtPreparedRenderer[
                    .TRIANGLES_INSTANCES
                ](settings, camera, data)
            return self.triangles_instances.value().render[integrator](
                settings, camera
            )
        if mask == 7:
            if not self.all:
                self.all = GpuRtPreparedRenderer[.ALL](settings, camera, data)
            return self.all.value().render[integrator](settings, camera)
        raise Error("scene has no supported geometry")


struct CompletedFrame:
    var pixels: List[Color]
    var generation: UInt64
    var samples: Int
    var width: Int
    var height: Int
    var render_ms: Float64

    def __init__(
        out self,
        var result: RenderResult,
        generation: UInt64,
        settings: RenderSettings,
    ):
        self.render_ms = Float64(result.timings.render_ns) / 1_000_000
        self.pixels = result.pixels^
        result.pixels = List[Color]()
        self.generation = generation
        self.samples = settings.samples_per_pixel
        self.width = settings.image_width
        self.height = settings.image_height


struct WorkerState:
    var scene: ViewerScene[]
    var gpu: GpuBackend
    var camera: Camera
    var settings: RenderSettings
    var integrator: Int
    var backend: Int
    var generation: UInt64
    var status: UInt32
    var stop: UInt32
    var result: Optional[CompletedFrame]
    var error: String
    var build_method: Int
    var traversal: Int
    var debug_requested: Bool
    var trace_requested: Bool
    var boxes: List[DebugBox]

    def __init__(
        out self, var scene: ViewerScene[], build_method: Int, traversal: Int
    ):
        self.build_method = build_method
        self.traversal = traversal
        self.camera = scene.state.camera
        self.settings = scene.state.settings.copy()
        self.scene = scene^
        self.gpu = GpuBackend()
        self.integrator = 0
        self.backend = 0
        self.generation = 0
        self.status = 0
        self.stop = 0
        self.result = Optional[CompletedFrame]()
        self.error = ""
        self.debug_requested = False
        self.trace_requested = True
        self.boxes = List[DebugBox]()

    def take_boxes(mut self) -> List[DebugBox]:
        var boxes = self.boxes^
        self.boxes = List[DebugBox]()
        return boxes^

    def render_integrator[
        integrator: Integrator
    ](mut self) raises -> RenderResult:
        if self.backend == 1:
            comptime if has_accelerator():
                return self.gpu.render[integrator](
                    self.scene, self.settings, self.camera
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
    ](self) raises -> RenderResult:
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
        ](self.settings, self.camera, self.scene.cpu_scene())

    def prepare_cpu(mut self) raises:
        match self.build_method:
            case 1:
                self.scene.prepare_cpu[.LBVH]()
            case 2:
                self.scene.prepare_cpu[.HPLOC]()
            case 3:
                self.scene.prepare_cpu[.MEDIAN]()
            case _:
                self.scene.prepare_cpu()

    def render(mut self) raises:
        if self.debug_requested:
            self.prepare_cpu()
            self.boxes = cpu_bvh_boxes(self.scene.cpu_scene())
        if not self.trace_requested:
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
        self.result = CompletedFrame(result^, self.generation, self.settings)


@export
def _bajo_viewer_render_thread(address: UInt) abi("C") -> UInt:
    var state = Pointer[WorkerState, MutUntrackedOrigin](
        unsafe_from_address=Int(address)
    )
    while not Atomic.load[ordering=Ordering.ACQUIRE](Pointer(to=state[].stop)):
        if (
            Atomic.load[ordering=Ordering.ACQUIRE](Pointer(to=state[].status))
            == 1
        ):
            try:
                state[].render()
            except error:
                state[].error = String(error)
            Atomic.store[ordering=Ordering.RELEASE](
                Pointer(to=state[].status), UInt32(2)
            )
        else:
            _ = external_call["usleep", c_int](UInt32(1000))
    return 0


struct RenderWorker:
    var state: Pointer[WorkerState, MutUntrackedOrigin]
    var thread: UInt

    def __init__(
        out self,
        var scene: ViewerScene[],
        build_method: Int = 0,
        traversal: Int = 0,
    ) raises:
        var state = unsafe_alloc[WorkerState](1)
        state.unsafe_write(WorkerState(scene^, build_method, traversal))
        var thread = UInt(0)
        var code = external_call["pthread_create", c_int](
            Pointer(to=thread),
            UInt(0),
            _bajo_viewer_render_thread,
            UInt(Int(state)),
        )
        if code:
            state.unsafe_deinit_pointee()
            state.unsafe_free()
            raise Error("could not start native rendering thread")
        self.state = state.unsafe_origin_cast[MutUntrackedOrigin]()
        self.thread = thread

    def idle(self) -> Bool:
        return (
            Atomic.load[ordering=Ordering.ACQUIRE](
                Pointer(to=self.state[].status)
            )
            == 0
        )

    def ready(self) -> Bool:
        return (
            Atomic.load[ordering=Ordering.ACQUIRE](
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
        self.state[].error = ""
        Atomic.store[ordering=Ordering.RELEASE](
            Pointer(to=self.state[].status), UInt32(1)
        )

    def acknowledge(self):
        # Call only after consuming the published result/error/debug snapshot.
        Atomic.store[ordering=Ordering.RELEASE](
            Pointer(to=self.state[].status), UInt32(0)
        )

    def __deinit__(deinit self):
        Atomic.store[ordering=Ordering.RELEASE](
            Pointer(to=self.state[].stop), UInt32(1)
        )
        _ = external_call["pthread_join", c_int](self.thread, UInt(0))
        self.state.unsafe_deinit_pointee()
        self.state.unsafe_free()
