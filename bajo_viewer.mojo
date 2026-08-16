from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.time import perf_counter_ns

from bajo.core import Point3W
from bajo.core.utils import ns_to_ms
from bajo.pbrt import read_pbrt
from examples.viewer import render_frame as _render_frame


def render_frame(config: PythonObject) raises -> PythonObject:
    var output = String(py=config["output"])
    var width = Int(py=config["width"])
    var height = Int(py=config["height"])
    var samples = Int(py=config["samples"])
    var max_depth = Int(py=config["max_depth"])
    var scene = Int(py=config["scene"])
    var scene_path = String(py=config["scene_path"])
    var origin = Point3W(
        Float32(py=config["x"]),
        Float32(py=config["y"]),
        Float32(py=config["z"]),
    )
    var total_t0 = perf_counter_ns()
    var render_stats = _render_frame(
        output,
        width,
        height,
        samples,
        max_depth,
        origin,
        Float32(py=config["yaw"]),
        Float32(py=config["pitch"]),
        Float32(py=config["vfov"]),
        scene,
        scene_path,
    )
    var total_ms = ns_to_ms(Int(perf_counter_ns() - total_t0))
    var build_ms = total_ms - render_stats.render_ms
    if build_ms < 0.0:
        build_ms = 0.0
    var primary_rays = width * height * samples
    var mrays = 0.0
    if render_stats.render_ms > 0.0:
        mrays = Float64(primary_rays) / (render_stats.render_ms * 1000.0)
    return Python.list(
        render_stats.render_ms,
        build_ms,
        mrays,
        render_stats.bvh_stats,
    )


def pbrt_camera(path: PythonObject) raises -> PythonObject:
    var scene = read_pbrt(String(py=path))
    var camera = scene.camera
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
def PyInit_bajo_viewer() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("bajo_viewer")
        module.def_function[render_frame]("render_frame")
        module.def_function[pbrt_camera]("pbrt_camera")
        return module.finalize()
    except e:
        abort(String("could not initialize bajo_viewer: ", e))
