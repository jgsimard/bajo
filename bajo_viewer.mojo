"""Specialized rendering extension: borrows preparation-owned handles."""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.memory import Pointer
from std.os import abort
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns
from bajo.bvh.cpu import CpuTraversalMode
from bajo.core.utils import ns_to_ms
from bajo.rt.material_types import Integrator
from bajo.rt.cpu.wavefront import render_wavefront_configured
from bajo.rt.cpu.scheduler_mode import CpuSchedulerMode
from bajo.rt.gpu.prepared_renderer import GpuRtPreparedRenderer
from bajo.rt.gpu.config import GpuRtSceneKind
from bajo.viewer.common import (
    CpuViewerState,
    _settings,
    _camera,
    _linear_frame,
)

comptime VIEWER_BACKEND = get_defined_int["VIEWER_BACKEND", 0]()
comptime VIEWER_INTEGRATOR = get_defined_int["VIEWER_INTEGRATOR", 0]()
comptime VIEWER_TRAVERSAL = get_defined_int["VIEWER_TRAVERSAL", 0]()
comptime CPU_VIEWER_TRAVERSAL_MODE = (
    CpuTraversalMode.AUTO_COHERENT if VIEWER_TRAVERSAL
    == 0 else CpuTraversalMode.FIXED_PACKET if VIEWER_TRAVERSAL
    == 1 else CpuTraversalMode.ADAPTIVE
)


def _render_cpu_state[
    integrator: Integrator
](handle: Int, config: PythonObject) raises -> PythonObject:
    var ptr = Pointer[CpuViewerState, MutUntrackedOrigin](
        unsafe_from_address=handle
    )
    var settings = _settings(config)
    var camera = _camera(config)
    var render_t0 = perf_counter_ns()
    var result = render_wavefront_configured[
        CPU_VIEWER_TRAVERSAL_MODE,
        integrator,
        16,
        1024,
        True,
        CpuSchedulerMode.TASK_PARTITIONS,
        16,
        16,
        16,
        8,
        4,
    ](settings, camera, ptr[].world)
    var render_ns = Int(perf_counter_ns() - render_t0)
    var frame = _linear_frame(
        settings.image_width, settings.image_height, result.pixels
    )
    var render_ms = ns_to_ms(render_ns)
    var primary_rays = (
        settings.image_width
        * settings.image_height
        * settings.samples_per_pixel
    )
    var mrays = Float64(0.0)
    if render_ms > 0.0:
        mrays = Float64(primary_rays) / (render_ms * 1000.0)
    return Python.list(render_ms, Float64(0.0), mrays, frame)


def render_cpu_state(
    handle: PythonObject, config: PythonObject
) raises -> PythonObject:
    var address = Int(py=handle)
    comptime if VIEWER_INTEGRATOR == 0:
        return _render_cpu_state[.PATH](address, config)
    elif VIEWER_INTEGRATOR == 1:
        return _render_cpu_state[.NEE](address, config)
    elif VIEWER_INTEGRATOR == 2:
        return _render_cpu_state[.MIS](address, config)
    elif VIEWER_INTEGRATOR == 3:
        return _render_cpu_state[.NORMALS](address, config)
    else:
        return _render_cpu_state[.AO](address, config)


def _render_state[
    integrator: Integrator,
    kind: GpuRtSceneKind,
](handle: Int, config: PythonObject) raises -> PythonObject:
    comptime State = GpuRtPreparedRenderer[kind]
    var ptr = Pointer[State, MutUntrackedOrigin](unsafe_from_address=handle)
    var settings = _settings(config)
    var result = ptr[].render[integrator](settings, _camera(config))
    var frame = _linear_frame(
        settings.image_width, settings.image_height, result.pixels
    )
    var render_ms = ns_to_ms(result.timings.render_ns)
    var init_ms = ns_to_ms(result.timings.init_ns)
    var primary_rays = (
        settings.image_width
        * settings.image_height
        * settings.samples_per_pixel
    )
    var mrays = 0.0
    if render_ms > 0.0:
        mrays = Float64(primary_rays) / (render_ms * 1000.0)
    return Python.list(render_ms, init_ms, mrays, frame)


def _dispatch_render[
    integrator: Integrator
](handle: Int, tag: Int, config: PythonObject) raises -> PythonObject:
    var kind = GpuRtSceneKind(tag)
    match kind:
        case .SPHERES:
            return _render_state[integrator, .SPHERES](handle, config)
        case .TRIANGLES:
            return _render_state[integrator, .TRIANGLES](handle, config)
        case .SPHERES_TRIANGLES:
            return _render_state[integrator, .SPHERES_TRIANGLES](handle, config)
        case .INSTANCES:
            return _render_state[integrator, .INSTANCES](handle, config)
        case .SPHERES_INSTANCES:
            return _render_state[integrator, .SPHERES_INSTANCES](handle, config)
        case .TRIANGLES_INSTANCES:
            return _render_state[integrator, .TRIANGLES_INSTANCES](
                handle, config
            )
        case .ALL:
            return _render_state[integrator, .ALL](handle, config)


def render_gpu_state(
    handle: PythonObject, tag: PythonObject, config: PythonObject
) raises -> PythonObject:
    comptime integrators = (
        Integrator.PATH,
        Integrator.NEE,
        Integrator.MIS,
        Integrator.NORMALS,
        Integrator.AO,
    )
    comptime assert 0 <= VIEWER_INTEGRATOR < len(integrators)
    return _dispatch_render[integrators[VIEWER_INTEGRATOR]](
        Int(py=handle), Int(py=tag), config
    )


@export
def PyInit_bajo_viewer() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("bajo_viewer")
        comptime if VIEWER_BACKEND == 1:
            module.def_function[render_gpu_state]("render_gpu_state")
        else:
            module.def_function[render_cpu_state]("render_cpu_state")
        return module.finalize()
    except e:
        abort(String("could not initialize viewer rendering: ", e))
