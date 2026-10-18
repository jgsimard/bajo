from bajo.viewer.common import _viewer_camera, _write_linear_frame
from std.sys.arg import argv
from std.sys import has_accelerator, simd_width_of
from std.sys.defines import get_defined_int

from bajo.core import (
    Point3W,
    Vec3W,
)
from bajo.core.random import Sampler
from bajo.core.utils import ns_to_ms
from bajo.bvh.cpu import CpuBvhBuildMethod, CpuTraversalMode
from bajo.parser.number import parse_f32_at
from bajo.rt import (
    Camera,
    Integrator,
    RenderSettings,
    SceneData,
    CpuScene,
    CpuSchedulerMode,
    render_gpu,
    render_wavefront_configured,
    write_ppm_from_colors,
)
from examples.rtiaw import make_weekend_scene
from examples.cornell_box import make_cornell_scene
from examples.mis_showcase import make_mis_showcase_scene
from examples.lbvh_scene import make_lbvh_scene
from examples.emissive_instances import make_emissive_instance_scene
from examples.stress_scenes import (
    make_indirect_hall_scene,
    make_many_lights_scene,
    make_specular_transport_scene,
)
from bajo.parser.pbrt import read_pbrt


comptime VIEWER_BACKEND = get_defined_int["VIEWER_BACKEND", 0]()
comptime VIEWER_INTEGRATOR = get_defined_int["VIEWER_INTEGRATOR", 0]()
comptime VIEWER_BUILD = get_defined_int["VIEWER_BUILD", 0]()
comptime VIEWER_TRAVERSAL = get_defined_int["VIEWER_TRAVERSAL", 0]()
comptime CPU_VIEWER_BUILD_METHOD = (
    CpuBvhBuildMethod.SAH if VIEWER_BUILD
    == 0 else CpuBvhBuildMethod.LBVH if VIEWER_BUILD
    == 1 else CpuBvhBuildMethod.HPLOC if VIEWER_BUILD
    == 2 else CpuBvhBuildMethod.MEDIAN
)
comptime CPU_VIEWER_TRAVERSAL_MODE = (
    CpuTraversalMode.AUTO_COHERENT if VIEWER_TRAVERSAL
    == 0 else CpuTraversalMode.FIXED_PACKET if VIEWER_TRAVERSAL
    == 1 else CpuTraversalMode.ADAPTIVE
)


@fieldwise_init
struct ViewerRenderStats(Copyable):
    var render_ms: Float64
    var bvh_stats: String


def _write_viewer_frame(
    path: String,
    width: Int,
    height: Int,
    pixels: ImmSpan[Vec3W, _],
    linear_output: Bool,
) raises:
    if linear_output:
        _write_linear_frame(path, pixels)
    else:
        write_ppm_from_colors(path, width, height, pixels)


def _float_arg(text: String) raises -> Float32:
    var span = StringSpan(text)
    var parsed = parse_f32_at(span.as_bytes(), 0)
    if parsed.pos != span.byte_length():
        raise Error("invalid viewer float: " + text)
    return parsed.value


def _int_arg(text: String) raises -> Int:
    var value = _float_arg(text)
    var integer = Int(value)
    if Float32(integer) != value:
        raise Error("invalid viewer integer: " + text)
    return integer


def _viewer_bvh_stats[
    BACKEND: Int,
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
    build_method: CpuBvhBuildMethod = .SAH,
    traversal_mode: CpuTraversalMode = .AUTO_COHERENT,
](data: SceneData, include_traversal: Bool = True) -> String:
    var result: String
    if BACKEND == 0:
        result = "CPU W" + String(Int(world_bvh_width))
        result += "/I" + String(Int(instance_bvh_width)) + " | "
        if len(data.spheres()) > 0:
            result += (
                "sphere"
                + String(Int(world_bvh_width))
                + "/"
                + build_method.name()
            )
        if len(data.triangle_vertices()) > 0:
            if len(data.spheres()) > 0:
                result += " "
            result += (
                "tri" + String(Int(world_bvh_width)) + "/" + build_method.name()
            )
        if len(data.triangle_instances()) > 0:
            if len(data.spheres()) > 0 or len(data.triangle_vertices()) > 0:
                result += " "
            result += (
                "BLAS"
                + String(Int(instance_bvh_width))
                + "/"
                + build_method.name()
                + " TLAS"
                + String(Int(instance_bvh_width))
                + "/1 LBVH"
            )
        if include_traversal:
            comptime if traversal_mode == .ADAPTIVE:
                result += " | traversal adaptive-16-8-4-scalar"
            else:
                result += " | traversal " + traversal_mode.name()
    else:
        result = "GPU | "
        if len(data.spheres()) > 0:
            result += "sphere4/4 LBVH"
        if len(data.triangle_vertices()) > 0:
            if len(data.spheres()) > 0:
                result += " "
            result += "tri8/4 H-PLOC CWBVH8"
        if len(data.triangle_instances()) > 0:
            if len(data.spheres()) > 0 or len(data.triangle_vertices()) > 0:
                result += " "
            result += "BLAS8/4 H-PLOC CWBVH8"
            result += " TLAS2/1 LBVH"
    result += " | S" + String(len(data.spheres()))
    result += " T" + String(len(data.triangle_vertices()) // 3)
    result += " I" + String(len(data.triangle_instances()))
    return result


def load_viewer_scene_data(scene: Int, scene_path: String) raises -> SceneData:
    """Create scene data before either backend prepares acceleration."""
    match scene:
        case 0:
            return make_weekend_scene()
        case 1:
            return make_cornell_scene()
        case 2:
            return make_mis_showcase_scene()
        case 3:
            return make_lbvh_scene()
        case 4:
            return make_emissive_instance_scene()
        case 5:
            return make_many_lights_scene()
        case 6:
            return make_indirect_hall_scene()
        case 7:
            return make_specular_transport_scene()
        case 8 | 9:
            var parsed = read_pbrt(scene_path)
            return parsed^.take_data()
        case _:
            raise Error("unsupported viewer scene")


def _render_gpu_frame[
    integrator: Integrator
](
    output: String,
    width: Int,
    height: Int,
    samples: Int,
    max_depth: Int,
    origin: Point3W,
    yaw_degrees: Float32,
    pitch_degrees: Float32,
    vfov: Float32,
    data: SceneData,
    sampler: Sampler = .INDEPENDENT,
    sample_offset: Int = 0,
    sample_sequence_length: Int = 0,
    linear_output: Bool = False,
) raises -> ViewerRenderStats:
    comptime if not has_accelerator():
        raise Error("GPU backend requested, but no accelerator is available")

    var camera = _viewer_camera(origin, yaw_degrees, pitch_degrees, vfov)
    var settings = RenderSettings(
        width,
        height,
        samples,
        UInt64(1234),
        max_depth,
        sampler,
        sample_offset,
        sample_sequence_length,
    )
    var result = render_gpu[integrator](settings, camera, data)
    _write_viewer_frame(output, width, height, result.pixels, linear_output)
    return ViewerRenderStats(
        ns_to_ms(result.timings.total_ns), _viewer_bvh_stats[1](data)
    )


def _render_cpu_frame[
    integrator: Integrator,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
    build_method: CpuBvhBuildMethod,
    traversal_mode: CpuTraversalMode,
    *adaptive_packet_sizes: SIMDLength,
](
    request: _ViewerRenderRequest, var data: SceneData
) raises -> ViewerRenderStats:
    var world = CpuScene[world_bvh_width, instance_bvh_width].__init__[
        build_method
    ](data^)
    var camera = _viewer_camera(
        request.origin, request.yaw_degrees, request.pitch_degrees, request.vfov
    )
    var settings = RenderSettings(
        request.width,
        request.height,
        request.samples,
        UInt64(1234),
        request.max_depth,
        request.sampler,
        request.sample_offset,
        request.sample_sequence_length,
    )
    var result = render_wavefront_configured[
        traversal_mode,
        integrator,
        16,
        1024,
        True,
        CpuSchedulerMode.TASK_PARTITIONS,
        world_bvh_width,
        instance_bvh_width,
        *adaptive_packet_sizes,
    ](settings, camera, world)
    _write_viewer_frame(
        request.output,
        request.width,
        request.height,
        result.pixels,
        request.linear_output,
    )
    return ViewerRenderStats(
        ns_to_ms(result.timings.total_ns),
        _viewer_bvh_stats[
            0, world_bvh_width, instance_bvh_width, build_method, traversal_mode
        ](world.scene_data()),
    )


@fieldwise_init
struct _ViewerRenderRequest(Copyable):
    var output: String
    var width: Int
    var height: Int
    var samples: Int
    var max_depth: Int
    var origin: Point3W
    var yaw_degrees: Float32
    var pitch_degrees: Float32
    var vfov: Float32
    var scene: Int
    var scene_path: String
    var sampler: Sampler
    var sample_offset: Int
    var sample_sequence_length: Int
    var linear_output: Bool


def _render_scene[
    integrator: Integrator,
    BACKEND: Int,
    build_method: CpuBvhBuildMethod,
    traversal_mode: CpuTraversalMode,
    *adaptive_packet_sizes: SIMDLength,
](request: _ViewerRenderRequest) raises -> ViewerRenderStats:
    var data = load_viewer_scene_data(request.scene, request.scene_path)
    comptime if BACKEND == 1:
        return _render_gpu_frame[integrator](
            request.output,
            request.width,
            request.height,
            request.samples,
            request.max_depth,
            request.origin,
            request.yaw_degrees,
            request.pitch_degrees,
            request.vfov,
            data,
            request.sampler,
            request.sample_offset,
            request.sample_sequence_length,
            request.linear_output,
        )
    else:
        comptime assert BACKEND == 0, "unsupported viewer rendering backend"
        if request.scene == 8 or request.scene == 9:
            return _render_cpu_frame[
                integrator,
                16,
                16,
                build_method,
                traversal_mode,
                *adaptive_packet_sizes,
            ](request, data^)
        # Procedural scenes use the native SIMD width on each host.
        comptime width = simd_width_of[DType.float32]()
        return _render_cpu_frame[
            integrator,
            width,
            width,
            build_method,
            traversal_mode,
            *adaptive_packet_sizes,
        ](request, data^)


def _render_scene_for_policy[
    integrator: Integrator,
](request: _ViewerRenderRequest) raises -> ViewerRenderStats:
    return _render_scene[
        integrator,
        VIEWER_BACKEND,
        CPU_VIEWER_BUILD_METHOD,
        CPU_VIEWER_TRAVERSAL_MODE,
        16,
        8,
        4,
    ](request)


def render_frame(
    output: String,
    width: Int,
    height: Int,
    samples: Int,
    max_depth: Int,
    origin: Point3W,
    yaw_degrees: Float32,
    pitch_degrees: Float32,
    vfov: Float32,
    scene: Int,
    scene_path: String,
    sampler: Sampler = .INDEPENDENT,
    sample_offset: Int = 0,
    sample_sequence_length: Int = 0,
    linear_output: Bool = False,
) raises -> ViewerRenderStats:
    comptime assert VIEWER_BACKEND >= 0 and VIEWER_BACKEND <= 1
    comptime assert VIEWER_INTEGRATOR >= 0 and VIEWER_INTEGRATOR <= 4
    comptime assert VIEWER_BUILD >= 0 and VIEWER_BUILD <= 3
    comptime assert VIEWER_TRAVERSAL >= 0 and VIEWER_TRAVERSAL <= 2
    var request = _ViewerRenderRequest(
        output,
        width,
        height,
        samples,
        max_depth,
        origin,
        yaw_degrees,
        pitch_degrees,
        vfov,
        scene,
        scene_path,
        sampler,
        sample_offset,
        sample_sequence_length,
        linear_output,
    )
    comptime integrators = {
        0: Integrator.PATH,
        1: Integrator.NEE,
        2: Integrator.MIS,
        3: Integrator.NORMALS,
        4: Integrator.AO,
    }
    comptime assert VIEWER_INTEGRATOR in [0, 1, 2, 3, 4]
    comptime integrator = integrators.get(VIEWER_INTEGRATOR).value()
    return _render_scene_for_policy[integrator](request)


def main() raises:
    # output width height spp origin_x origin_y origin_z yaw pitch vfov
    var args = argv()
    if len(args) != 11:
        raise Error(
            "viewer expects: output width height spp x y z yaw pitch vfov"
        )

    var output = String(args[1])
    var width = _int_arg(String(args[2]))
    var height = _int_arg(String(args[3]))
    var samples = _int_arg(String(args[4]))
    var origin = Point3W(
        _float_arg(String(args[5])),
        _float_arg(String(args[6])),
        _float_arg(String(args[7])),
    )
    var render_stats = render_frame(
        output,
        width,
        height,
        samples,
        8,
        origin,
        _float_arg(String(args[8])),
        _float_arg(String(args[9])),
        _float_arg(String(args[10])),
        0,
        "",
    )
    print(t"render_ms={round(render_stats.render_ms, 2)}")
