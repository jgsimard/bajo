from std.math import cos, sin
from std.sys.arg import argv
from std.sys import has_accelerator, simd_width_of
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns

from bajo.core import (
    Point3W,
    Vec3W,
)
from bajo.core.utils import degrees_to_radians, ns_to_ms
from bajo.obj.f32 import parse_f32_at
from bajo.rt import (
    Camera,
    RENDER,
    RenderSettings,
    World,
    render_depth_first,
    render_gpu,
    render_wavefront,
    write_ppm_from_colors,
)
from examples.rtiaw import make_weekend_world
from examples.cornell_box import make_cornell_world
from examples.mis_showcase import make_mis_showcase_world
from examples.lbvh_scene import make_lbvh_world
from bajo.pbrt import read_pbrt


comptime VIEWER_BACKEND = get_defined_int["VIEWER_BACKEND", 0]()
comptime VIEWER_ALGORITHM = get_defined_int["VIEWER_ALGORITHM", 0]()


@fieldwise_init
struct ViewerRenderStats(Copyable):
    var render_ms: Float64
    var bvh_stats: String


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
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](world: World[world_bvh_width, instance_bvh_width]) -> String:
    var result = String()
    if BACKEND == 0:
        result = "CPU W" + String(Int(world_bvh_width))
        result += "/I" + String(Int(instance_bvh_width)) + " | "
        if len(world.spheres) > 0:
            result += "sphere" + String(Int(world_bvh_width)) + "/SAH"
        if len(world.triangle_vertices) > 0:
            if len(world.spheres) > 0:
                result += " "
            result += "tri" + String(Int(world_bvh_width)) + "/SAH"
        if len(world.triangle_instances) > 0:
            if len(world.spheres) > 0 or len(world.triangle_vertices) > 0:
                result += " "
            result += (
                "BLAS"
                + String(Int(instance_bvh_width))
                + "/SAH TLAS"
                + String(Int(instance_bvh_width))
                + "/1 LBVH"
            )
    else:
        result = "GPU host W" + String(Int(world_bvh_width))
        result += "/I" + String(Int(instance_bvh_width)) + " | "
        if len(world.spheres) > 0:
            result += "sphere4/4 LBVH"
        if len(world.triangle_vertices) > 0:
            if len(world.spheres) > 0:
                result += " "
            if len(world.triangle_vertices) / 3 >= 32:
                result += "tri8/4 H-PLOC CWBVH8"
            else:
                result += "tri4/4 LBVH"
        if len(world.triangle_instances) > 0:
            if len(world.spheres) > 0 or len(world.triangle_vertices) > 0:
                result += " "
            var weighted_triangles = 0
            for instance in world.triangle_instances:
                weighted_triangles += (
                    len(world.triangle_meshes[Int(instance.blas_idx)]) / 3
                )
            if weighted_triangles >= len(world.triangle_instances) * 32:
                result += "BLAS8/4 H-PLOC CWBVH8"
            else:
                result += "BLAS4/4 LBVH"
            result += " TLAS2/1 LBVH"
    result += " | S" + String(len(world.spheres))
    result += " T" + String(len(world.triangle_vertices) / 3)
    result += " I" + String(len(world.triangle_instances))
    return result


def _render_frame[
    ALGORITHM: RENDER,
    BACKEND: Int,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
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
    world: World[world_bvh_width, instance_bvh_width],
) raises -> ViewerRenderStats:
    var yaw = degrees_to_radians(yaw_degrees)
    var pitch = degrees_to_radians(pitch_degrees)
    var cos_pitch = cos(pitch)
    var forward = Vec3W(
        sin(yaw) * cos_pitch,
        sin(pitch),
        -cos(yaw) * cos_pitch,
    )
    var camera = Camera.from_vfov(
        origin,
        origin + forward,
        Vec3W(0.0, 1.0, 0.0),
        vfov,
        10.0,
        0.0,
    )

    var settings = RenderSettings(
        width, height, samples, UInt64(1234), max_depth
    )
    var t0 = perf_counter_ns()
    var t1 = 0
    comptime if BACKEND == 0:
        comptime if (
            ALGORITHM == RENDER.PATH
            or ALGORITHM == RENDER.NEE
            or ALGORITHM == RENDER.MIS
        ):
            var result = render_wavefront[ALGORITHM](settings, camera, world)
            t1 = perf_counter_ns()
            write_ppm_from_colors(output, width, height, result.pixels)
        else:
            var result = render_depth_first[ALGORITHM](settings, camera, world)
            t1 = perf_counter_ns()
            write_ppm_from_colors(output, width, height, result.pixels)
    elif BACKEND == 1:
        comptime if has_accelerator():
            var result = render_gpu[ALGORITHM](settings, camera, world)
            t1 = perf_counter_ns()
            write_ppm_from_colors(output, width, height, result.pixels)
        else:
            raise Error(
                "GPU backend requested, but no accelerator is available"
            )
    else:
        raise Error("unsupported viewer rendering backend")
    return ViewerRenderStats(
        ns_to_ms(Int(t1 - t0)),
        _viewer_bvh_stats[BACKEND, world_bvh_width, instance_bvh_width](world),
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


def _render_scene[
    ALGORITHM: RENDER,
    BACKEND: Int,
](request: _ViewerRenderRequest) raises -> ViewerRenderStats:
    if request.scene == 4 or request.scene == 5:
        var parsed = read_pbrt(request.scene_path)
        return _render_frame[ALGORITHM, BACKEND](
            request.output,
            request.width,
            request.height,
            request.samples,
            request.max_depth,
            request.origin,
            request.yaw_degrees,
            request.pitch_degrees,
            request.vfov,
            parsed.world,
        )
    # not everyone has avx-512
    comptime world_bvh_width = simd_width_of[DType.float32]()
    comptime instance_bvh_width = simd_width_of[DType.float32]()

    var world: World[world_bvh_width, instance_bvh_width]
    if request.scene == 0:
        world = make_weekend_world[world_bvh_width, instance_bvh_width]()
    elif request.scene == 1:
        world = make_cornell_world[world_bvh_width, instance_bvh_width]()
    elif request.scene == 2:
        world = make_mis_showcase_world[world_bvh_width, instance_bvh_width]()
    elif request.scene == 3:
        world = make_lbvh_world[world_bvh_width, instance_bvh_width]()
    else:
        raise Error("unsupported viewer scene")

    return _render_frame[ALGORITHM, BACKEND](
        request.output,
        request.width,
        request.height,
        request.samples,
        request.max_depth,
        request.origin,
        request.yaw_degrees,
        request.pitch_degrees,
        request.vfov,
        world,
    )


def _render_frame_for_config(
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
) raises -> ViewerRenderStats:
    comptime assert VIEWER_BACKEND >= 0 and VIEWER_BACKEND <= 1
    comptime assert VIEWER_ALGORITHM >= 0 and VIEWER_ALGORITHM <= 4
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
    )
    comptime if VIEWER_ALGORITHM == 0:
        return _render_scene[RENDER.PATH, VIEWER_BACKEND](request)
    elif VIEWER_ALGORITHM == 1:
        return _render_scene[RENDER.NEE, VIEWER_BACKEND](request)
    elif VIEWER_ALGORITHM == 2:
        return _render_scene[RENDER.MIS, VIEWER_BACKEND](request)
    elif VIEWER_ALGORITHM == 3:
        return _render_scene[RENDER.NORMALS, VIEWER_BACKEND](request)
    else:
        return _render_scene[RENDER.AO, VIEWER_BACKEND](request)


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
) raises -> ViewerRenderStats:
    return _render_frame_for_config(
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
    )


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
