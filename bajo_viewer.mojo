from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.os import abort
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns

from bajo.core import Point3W
from bajo.core.random import Sampler
from bajo.core.utils import ns_to_ms
from bajo.parser.pbrt import read_pbrt
from bajo.rt import (
    Camera,
    Integrator,
    RenderSettings,
    SceneData,
)
from bajo.rt.gpu import GpuRtPreparedRenderer
from bajo.rt.gpu.config import (
    GpuRtBvhFormat,
    GpuRtSceneKind,
    GPU_RT_BVH_CWBVH8,
    GPU_RT_BVH_TLAS2,
    GPU_RT_BVH_WIDE4,
)
from bajo.rt.gpu.render import (
    _prefer_cwbvh8_blases,
    _prefer_cwbvh8_triangles,
)
from examples.viewer import (
    _write_linear_frame,
    _viewer_bvh_stats,
    _viewer_camera,
    load_viewer_scene_data,
    render_frame as _render_frame,
)


comptime VIEWER_BACKEND = get_defined_int["VIEWER_BACKEND", 0]()
comptime VIEWER_INTEGRATOR = get_defined_int["VIEWER_INTEGRATOR", 0]()
comptime STATE_KIND_MASK = 7
comptime STATE_COMPRESSED_TRIANGLES = 8
comptime STATE_COMPRESSED_BLASES = 16


def _settings(config: PythonObject) raises -> RenderSettings:
    return RenderSettings(
        Int(py=config["width"]),
        Int(py=config["height"]),
        Int(py=config["samples"]),
        UInt64(1234),
        Int(py=config["max_depth"]),
        Sampler(UInt32(Int(py=config["sampler"]))),
        Int(py=config["sample_offset"]),
        Int(py=config["sample_sequence_length"]),
    )


def _camera(config: PythonObject) raises -> Camera:
    return _viewer_camera(
        Point3W(
            Float32(py=config["x"]),
            Float32(py=config["y"]),
            Float32(py=config["z"]),
        ),
        Float32(py=config["yaw"]),
        Float32(py=config["pitch"]),
        Float32(py=config["vfov"]),
    )


def _state_tag[
    kind: GpuRtSceneKind,
    triangle_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
]() -> Int:
    comptime triangle_bit = (
        STATE_COMPRESSED_TRIANGLES if triangle_format.layout.compressed else 0
    )
    comptime blas_bit = (
        STATE_COMPRESSED_BLASES if blas_format.layout.compressed else 0
    )
    return Int(kind.bits) | triangle_bit | blas_bit


def _allocate_state[
    kind: GpuRtSceneKind,
    triangle_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
](
    settings: RenderSettings,
    camera: Camera,
    data: SceneData,
) raises -> Tuple[
    Int, Int
]:
    comptime State = GpuRtPreparedRenderer[
        kind,
        GPU_RT_BVH_WIDE4,
        triangle_format,
        GPU_RT_BVH_TLAS2,
        blas_format,
    ]
    var ptr = unsafe_alloc[State](1)
    ptr.unsafe_write(State(settings, camera, data))
    return (Int(ptr), _state_tag[kind, triangle_format, blas_format]())


def _allocate_kind[
    kind: GpuRtSceneKind,
](
    settings: RenderSettings,
    camera: Camera,
    data: SceneData,
    compressed_triangles: Bool,
    compressed_blases: Bool,
) raises -> Tuple[Int, Int]:
    comptime if kind.has_triangles():
        if compressed_triangles:
            comptime if kind.has_instances():
                if compressed_blases:
                    return _allocate_state[
                        kind, GPU_RT_BVH_CWBVH8, GPU_RT_BVH_CWBVH8
                    ](settings, camera, data)
                return _allocate_state[
                    kind, GPU_RT_BVH_CWBVH8, GPU_RT_BVH_WIDE4
                ](settings, camera, data)
            return _allocate_state[kind, GPU_RT_BVH_CWBVH8, GPU_RT_BVH_WIDE4](
                settings, camera, data
            )
        comptime if kind.has_instances():
            if compressed_blases:
                return _allocate_state[
                    kind, GPU_RT_BVH_WIDE4, GPU_RT_BVH_CWBVH8
                ](settings, camera, data)
        return _allocate_state[kind, GPU_RT_BVH_WIDE4, GPU_RT_BVH_WIDE4](
            settings, camera, data
        )

    comptime if kind.has_instances():
        if compressed_blases:
            return _allocate_state[kind, GPU_RT_BVH_WIDE4, GPU_RT_BVH_CWBVH8](
                settings, camera, data
            )
    return _allocate_state[kind, GPU_RT_BVH_WIDE4, GPU_RT_BVH_WIDE4](
        settings, camera, data
    )


def _create_gpu_state(
    settings: RenderSettings,
    camera: Camera,
    data: SceneData,
) raises -> Tuple[Int, Int]:
    var compressed_triangles = _prefer_cwbvh8_triangles(data)
    var compressed_blases = _prefer_cwbvh8_blases(data)
    if len(data.triangle_instances()) > 0:
        if len(data.spheres()) > 0:
            if len(data.triangle_vertices()) > 0:
                return _allocate_kind[.ALL](
                    settings,
                    camera,
                    data,
                    compressed_triangles,
                    compressed_blases,
                )
            return _allocate_kind[.SPHERES_INSTANCES](
                settings, camera, data, False, compressed_blases
            )
        if len(data.triangle_vertices()) > 0:
            return _allocate_kind[.TRIANGLES_INSTANCES](
                settings,
                camera,
                data,
                compressed_triangles,
                compressed_blases,
            )
        return _allocate_kind[.INSTANCES](
            settings, camera, data, False, compressed_blases
        )
    if len(data.spheres()) > 0:
        if len(data.triangle_vertices()) > 0:
            return _allocate_kind[.SPHERES_TRIANGLES](
                settings, camera, data, compressed_triangles, False
            )
        return _allocate_kind[.SPHERES](settings, camera, data, False, False)
    return _allocate_kind[.TRIANGLES](
        settings, camera, data, compressed_triangles, False
    )


def create_gpu_state(config: PythonObject) raises -> PythonObject:
    comptime if VIEWER_BACKEND != 1:
        raise Error("persistent GPU state requires the GPU viewer backend")
    var settings = _settings(config)
    var camera = _camera(config)
    var t0 = perf_counter_ns()
    var data = load_viewer_scene_data(
        Int(py=config["scene"]), String(py=config["scene_path"])
    )
    var bvh_stats = _viewer_bvh_stats[1](data)
    var state = _create_gpu_state(settings, camera, data)
    var build_ms = ns_to_ms(Int(perf_counter_ns() - t0))
    return Python.list(state[0], state[1], build_ms, bvh_stats)


def _render_state[
    integrator: Integrator,
    kind: GpuRtSceneKind,
    triangle_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
](handle: Int, config: PythonObject) raises -> PythonObject:
    comptime State = GpuRtPreparedRenderer[
        kind,
        GPU_RT_BVH_WIDE4,
        triangle_format,
        GPU_RT_BVH_TLAS2,
        blas_format,
    ]
    var ptr = Pointer[State, MutUntrackedOrigin](unsafe_from_address=handle)
    var settings = _settings(config)
    var result = ptr[].render[integrator](settings, _camera(config))
    _write_linear_frame(String(py=config["output"]), result.pixels)
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
    return Python.list(render_ms, init_ms, mrays)


def _destroy_state[
    kind: GpuRtSceneKind,
    triangle_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
](handle: Int):
    comptime State = GpuRtPreparedRenderer[
        kind,
        GPU_RT_BVH_WIDE4,
        triangle_format,
        GPU_RT_BVH_TLAS2,
        blas_format,
    ]
    var ptr = Pointer[State, MutUntrackedOrigin](unsafe_from_address=handle)
    ptr.unsafe_deinit_pointee()
    ptr.unsafe_free()


def _render_kind[
    integrator: Integrator,
    kind: GpuRtSceneKind,
](handle: Int, tag: Int, config: PythonObject) raises -> PythonObject:
    var compressed_triangles = Bool(tag & STATE_COMPRESSED_TRIANGLES)
    var compressed_blases = Bool(tag & STATE_COMPRESSED_BLASES)
    comptime if kind.has_triangles():
        if compressed_triangles:
            comptime if kind.has_instances():
                if compressed_blases:
                    return _render_state[
                        integrator, kind, GPU_RT_BVH_CWBVH8, GPU_RT_BVH_CWBVH8
                    ](handle, config)
            return _render_state[
                integrator, kind, GPU_RT_BVH_CWBVH8, GPU_RT_BVH_WIDE4
            ](handle, config)
    comptime if kind.has_instances():
        if compressed_blases:
            return _render_state[
                integrator, kind, GPU_RT_BVH_WIDE4, GPU_RT_BVH_CWBVH8
            ](handle, config)
    return _render_state[integrator, kind, GPU_RT_BVH_WIDE4, GPU_RT_BVH_WIDE4](
        handle, config
    )


def _dispatch_render[
    integrator: Integrator
](handle: Int, tag: Int, config: PythonObject) raises -> PythonObject:
    var kind = tag & STATE_KIND_MASK
    if kind == 1:
        return _render_kind[integrator, .SPHERES](handle, tag, config)
    if kind == 2:
        return _render_kind[integrator, .TRIANGLES](handle, tag, config)
    if kind == 3:
        return _render_kind[integrator, .SPHERES_TRIANGLES](handle, tag, config)
    if kind == 4:
        return _render_kind[integrator, .INSTANCES](handle, tag, config)
    if kind == 5:
        return _render_kind[integrator, .SPHERES_INSTANCES](handle, tag, config)
    if kind == 6:
        return _render_kind[integrator, .TRIANGLES_INSTANCES](
            handle, tag, config
        )
    if kind == 7:
        return _render_kind[integrator, .ALL](handle, tag, config)
    raise Error("unknown persistent GPU viewer state")


def render_gpu_state(
    handle: PythonObject, tag: PythonObject, config: PythonObject
) raises -> PythonObject:
    var address = Int(py=handle)
    var state_tag = Int(py=tag)
    comptime if VIEWER_INTEGRATOR == 0:
        return _dispatch_render[.PATH](address, state_tag, config)
    elif VIEWER_INTEGRATOR == 1:
        return _dispatch_render[.NEE](address, state_tag, config)
    elif VIEWER_INTEGRATOR == 2:
        return _dispatch_render[.MIS](address, state_tag, config)
    elif VIEWER_INTEGRATOR == 3:
        return _dispatch_render[.NORMALS](address, state_tag, config)
    else:
        return _dispatch_render[.AO](address, state_tag, config)


def _destroy_kind[
    kind: GpuRtSceneKind,
](handle: Int, tag: Int):
    var compressed_triangles = Bool(tag & STATE_COMPRESSED_TRIANGLES)
    var compressed_blases = Bool(tag & STATE_COMPRESSED_BLASES)
    comptime if kind.has_triangles():
        if compressed_triangles:
            comptime if kind.has_instances():
                if compressed_blases:
                    return _destroy_state[
                        kind, GPU_RT_BVH_CWBVH8, GPU_RT_BVH_CWBVH8
                    ](handle)
            return _destroy_state[kind, GPU_RT_BVH_CWBVH8, GPU_RT_BVH_WIDE4](
                handle
            )
    comptime if kind.has_instances():
        if compressed_blases:
            return _destroy_state[kind, GPU_RT_BVH_WIDE4, GPU_RT_BVH_CWBVH8](
                handle
            )
    return _destroy_state[kind, GPU_RT_BVH_WIDE4, GPU_RT_BVH_WIDE4](handle)


def destroy_gpu_state(handle: PythonObject, tag: PythonObject) raises:
    var address = Int(py=handle)
    var state_tag = Int(py=tag)
    var kind = state_tag & STATE_KIND_MASK
    if kind == 1:
        return _destroy_kind[.SPHERES](address, state_tag)
    if kind == 2:
        return _destroy_kind[.TRIANGLES](address, state_tag)
    if kind == 3:
        return _destroy_kind[.SPHERES_TRIANGLES](address, state_tag)
    if kind == 4:
        return _destroy_kind[.INSTANCES](address, state_tag)
    if kind == 5:
        return _destroy_kind[.SPHERES_INSTANCES](address, state_tag)
    if kind == 6:
        return _destroy_kind[.TRIANGLES_INSTANCES](address, state_tag)
    if kind == 7:
        return _destroy_kind[.ALL](address, state_tag)
    raise Error("unknown persistent GPU viewer state")


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
        Sampler(UInt32(Int(py=config["sampler"]))),
        Int(py=config["sample_offset"]),
        Int(py=config["sample_sequence_length"]),
        True,
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
        comptime if VIEWER_BACKEND == 1:
            module.def_function[create_gpu_state]("create_gpu_state")
            module.def_function[render_gpu_state]("render_gpu_state")
            module.def_function[destroy_gpu_state]("destroy_gpu_state")
        module.def_function[pbrt_camera]("pbrt_camera")
        return module.finalize()
    except e:
        abort(String("could not initialize bajo_viewer: ", e))
