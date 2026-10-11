"""Scene preparation extension: owns every handle it creates."""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.os import abort
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns
from bajo.bvh.cpu import CpuBvhBuildMethod
from bajo.core.utils import ns_to_ms
from bajo.parser.pbrt import read_pbrt, read_pbrt_camera
from bajo.rt.cpu.scene import CpuScene
from bajo.rt.render_types import RenderSettings
from bajo.rt.scene_data import SceneData
from bajo.bvh import Camera
from bajo.rt.gpu.prepared_renderer import GpuRtPreparedRenderer
from bajo.rt.gpu.config import GpuRtSceneKind
from bajo.viewer.common import (
    CpuViewerState,
    _settings,
    _camera,
)
from examples.viewer import (
    _viewer_bvh_stats,
    _make_viewer_world,
    load_viewer_scene_data,
)

comptime VIEWER_BACKEND = get_defined_int["VIEWER_BACKEND", 0]()
comptime VIEWER_BUILD = get_defined_int["VIEWER_BUILD", 0]()
comptime CPU_VIEWER_BUILD_METHOD = (
    CpuBvhBuildMethod.SAH if VIEWER_BUILD
    == 0 else CpuBvhBuildMethod.LBVH if VIEWER_BUILD
    == 1 else CpuBvhBuildMethod.HPLOC if VIEWER_BUILD
    == 2 else CpuBvhBuildMethod.MEDIAN
)


def _allocate_state[
    kind: GpuRtSceneKind,
](
    settings: RenderSettings,
    camera: Camera,
    data: SceneData,
) raises -> Tuple[
    Int, Int
]:
    comptime State = GpuRtPreparedRenderer[kind]
    var ptr = unsafe_alloc[State](1)
    ptr.unsafe_write(State(settings, camera, data))
    return (Int(ptr), Int(kind.bits))


def _create_gpu_state(
    settings: RenderSettings,
    camera: Camera,
    data: SceneData,
) raises -> Tuple[Int, Int]:
    if len(data.triangle_instances()) > 0:
        if len(data.spheres()) > 0:
            if len(data.triangle_vertices()) > 0:
                return _allocate_state[.ALL](settings, camera, data)
            return _allocate_state[.SPHERES_INSTANCES](settings, camera, data)
        if len(data.triangle_vertices()) > 0:
            return _allocate_state[.TRIANGLES_INSTANCES](settings, camera, data)
        return _allocate_state[.INSTANCES](settings, camera, data)
    if len(data.spheres()) > 0:
        if len(data.triangle_vertices()) > 0:
            return _allocate_state[.SPHERES_TRIANGLES](settings, camera, data)
        return _allocate_state[.SPHERES](settings, camera, data)
    return _allocate_state[.TRIANGLES](settings, camera, data)


def create_gpu_state(config: PythonObject) raises -> PythonObject:
    comptime if VIEWER_BACKEND != 1:
        raise Error("persistent GPU state requires the GPU viewer backend")
    var settings = _settings(config)
    var camera = _camera(config)
    var scene = Int(py=config["scene"])
    var scene_path = String(py=config["scene_path"])
    var parse_ms = Float64(0.0)
    var data: SceneData
    var build_t0 = perf_counter_ns()
    if scene == 8 or scene == 9:
        var parse_t0 = perf_counter_ns()
        data = load_viewer_scene_data(scene, scene_path)
        parse_ms = ns_to_ms(Int(perf_counter_ns() - parse_t0))
        build_t0 = perf_counter_ns()
    else:
        data = load_viewer_scene_data(scene, scene_path)
    var bvh_stats = _viewer_bvh_stats[1](data)
    var state = _create_gpu_state(settings, camera, data)
    var build_ms = ns_to_ms(Int(perf_counter_ns() - build_t0))
    return Python.list(state[0], state[1], parse_ms, build_ms, bvh_stats)


def create_cpu_state(config: PythonObject) raises -> PythonObject:
    comptime if VIEWER_BACKEND != 0:
        raise Error("persistent CPU state requires the CPU viewer backend")
    var scene = Int(py=config["scene"])
    var scene_path = String(py=config["scene_path"])
    var parse_ms = Float64(0.0)
    var world: CpuScene[16, 16]
    var build_t0: Int
    if scene == 8 or scene == 9:
        var parse_t0 = perf_counter_ns()
        var parsed = read_pbrt(scene_path)
        parse_ms = ns_to_ms(Int(perf_counter_ns() - parse_t0))
        build_t0 = perf_counter_ns()
        world = CpuScene[16, 16].__init__[CPU_VIEWER_BUILD_METHOD](
            parsed^.take_data()
        )
    else:
        build_t0 = perf_counter_ns()
        world = _make_viewer_world[16, 16, CPU_VIEWER_BUILD_METHOD](scene)
    var bvh_stats = _viewer_bvh_stats[
        0,
        16,
        16,
        CPU_VIEWER_BUILD_METHOD,
    ](world.scene_data(), False)
    var ptr = unsafe_alloc[CpuViewerState](1)
    ptr.unsafe_write(CpuViewerState(world^))
    var build_ms = ns_to_ms(Int(perf_counter_ns() - build_t0))
    return Python.list(Int(ptr), parse_ms, build_ms, bvh_stats)


def destroy_cpu_state(handle: PythonObject) raises:
    var ptr = Pointer[CpuViewerState, MutUntrackedOrigin](
        unsafe_from_address=Int(py=handle)
    )
    ptr.unsafe_deinit_pointee()
    ptr.unsafe_free()


def _destroy_state[
    kind: GpuRtSceneKind,
](handle: Int):
    comptime State = GpuRtPreparedRenderer[kind]
    var ptr = Pointer[State, MutUntrackedOrigin](unsafe_from_address=handle)
    ptr.unsafe_deinit_pointee()
    ptr.unsafe_free()


def destroy_gpu_state(handle: PythonObject, tag: PythonObject) raises:
    var address = Int(py=handle)
    var state_tag = Int(py=tag)
    var kind = GpuRtSceneKind(state_tag)
    __match kind:
        case .SPHERES:
            return _destroy_state[.SPHERES](address)
        case .TRIANGLES:
            return _destroy_state[.TRIANGLES](address)
        case .SPHERES_TRIANGLES:
            return _destroy_state[.SPHERES_TRIANGLES](address)
        case .INSTANCES:
            return _destroy_state[.INSTANCES](address)
        case .SPHERES_INSTANCES:
            return _destroy_state[.SPHERES_INSTANCES](address)
        case .TRIANGLES_INSTANCES:
            return _destroy_state[.TRIANGLES_INSTANCES](address)
        case .ALL:
            return _destroy_state[.ALL](address)


def pbrt_camera(path: PythonObject) raises -> PythonObject:
    var camera = read_pbrt_camera(String(py=path))
    return Python.list(
        camera.origin.x,
        camera.origin.y,
        camera.origin.z,
        camera.forward.x,
        camera.forward.y,
        camera.forward.z,
        camera.fov_scale,
    )


@export
def PyInit_bajo_viewer_prepare() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("bajo_viewer_prepare")
        comptime if VIEWER_BACKEND == 1:
            module.def_function[create_gpu_state]("create_gpu_state")
            module.def_function[destroy_gpu_state]("destroy_gpu_state")
        else:
            module.def_function[create_cpu_state]("create_cpu_state")
            module.def_function[destroy_cpu_state]("destroy_cpu_state")
        module.def_function[pbrt_camera]("pbrt_camera")
        return module.finalize()
    except e:
        abort(String("could not initialize viewer preparation: ", e))
