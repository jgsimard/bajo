from std.math import cos, max, sin
from std.sys.arg import argv
from std.sys import has_accelerator, simd_width_of
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns

from bajo.bvh.host_utils import compute_bounds
from bajo.core import (
    AABB,
    Affine3f32,
    Frame,
    Point3W,
    Point3f32,
    Quat,
    Vec3W,
    Vec3f32,
)
from bajo.core.utils import degrees_to_radians, ns_to_ms
from bajo.core.random import Rng
from bajo.obj.pack import pack_obj_triangles
from bajo.obj.f32 import parse_f32_at
from bajo.rt import (
    Camera,
    Color,
    Instance,
    RENDER,
    RenderSettings,
    Sphere,
    SurfaceId,
    SurfaceStore,
    World,
    add_sphere,
    add_triangle,
    add_triangle_instance,
    render_depth_first,
    render_gpu,
    render_wavefront,
    write_ppm_from_colors,
)
from examples.rtiaw import make_weekend_world
from examples.cornell_box import make_cornell_world
from examples.mis_showcase import make_mis_showcase_world
from bajo.pbrt import read_pbrt


comptime VIEWER_BACKEND = get_defined_int["VIEWER_BACKEND", 0]()
comptime VIEWER_ALGORITHM = get_defined_int["VIEWER_ALGORITHM", 0]()
comptime LBVH_OBJ_PATH_0 = "assets/bunny/bunny.obj"
comptime LBVH_OBJ_PATH_1 = "assets/buddha/buddha.obj"
comptime LBVH_OBJ_PATH_2 = "assets/dragon/dragon.obj"
comptime LBVH_GRID_X = 6
comptime LBVH_GRID_Z = 6


@fieldwise_init
struct ViewerRenderStats(Copyable):
    var render_ms: Float64
    var bvh_stats: String


def _lbvh_centered_transform(
    bounds: AABB[Frame.LOCAL],
    rotation: Quat,
    scale: Vec3f32[Frame.LOCAL],
    bottom_center: Vec3f32[Frame.WORLD],
) -> Affine3f32[Frame.LOCAL, Frame.WORLD]:
    var transform = Affine3f32[
        Frame.LOCAL, Frame.WORLD
    ].from_rotation_scale_translation(
        rotation, scale, Vec3f32[Frame.WORLD](0.0)
    )
    var center = bounds.centroid()
    var local_anchor = Vec3f32[Frame.LOCAL](center.x, bounds._min.y, center.z)
    var anchor_delta = transform.vector(local_anchor)
    transform.tx = bottom_center.x - anchor_delta.x
    transform.ty = bottom_center.y - anchor_delta.y
    transform.tz = bottom_center.z - anchor_delta.z
    return transform^


def make_lbvh_world[
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
]() raises -> World[world_bvh_width, instance_bvh_width]:
    var mesh0 = pack_obj_triangles(LBVH_OBJ_PATH_0)
    var mesh1 = pack_obj_triangles(LBVH_OBJ_PATH_1)
    var mesh2 = pack_obj_triangles(LBVH_OBJ_PATH_2)
    var bounds0 = compute_bounds(mesh0)
    var bounds1 = compute_bounds(mesh1)
    var bounds2 = compute_bounds(mesh2)

    var surfaces = SurfaceStore()
    var bunny_surface = surfaces.add_lambertian(Color(0.72, 0.34, 0.12))
    var buddha_surface = surfaces.add_lambertian(Color(0.18, 0.42, 0.78))
    var dragon_surface = surfaces.add_lambertian(Color(0.68, 0.12, 0.10))
    var ground_surface = surfaces.add_lambertian(Color(0.28, 0.30, 0.34))
    var light_surface = surfaces.add_emissive(Color(18.0, 16.0, 13.0))

    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 12.0, 0.0),
        2.0,
        light_surface,
    )

    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    add_triangle(
        triangle_vertices,
        triangle_surfaces,
        Point3f32[Frame.WORLD](-26.0, 0.0, -26.0),
        Point3f32[Frame.WORLD](26.0, 0.0, -26.0),
        Point3f32[Frame.WORLD](26.0, 0.0, 26.0),
        ground_surface,
    )
    add_triangle(
        triangle_vertices,
        triangle_surfaces,
        Point3f32[Frame.WORLD](-26.0, 0.0, -26.0),
        Point3f32[Frame.WORLD](26.0, 0.0, 26.0),
        Point3f32[Frame.WORLD](-26.0, 0.0, 26.0),
        ground_surface,
    )

    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]](capacity=3)
    triangle_meshes.append(mesh0.copy())
    triangle_meshes.append(mesh1.copy())
    triangle_meshes.append(mesh2.copy())
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()

    var rng = Rng(123, 123)
    comptime TARGET_EXTENT = Float32(1.60)
    var mesh_bounds = [bounds0, bounds1, bounds2]
    var mesh_surfaces = [
        bunny_surface.copy(),
        buddha_surface.copy(),
        dragon_surface.copy(),
    ]
    var cell_spacing = TARGET_EXTENT * Float32(5.2)
    var mesh_spacing = TARGET_EXTENT * Float32(2.0)
    for z in range(LBVH_GRID_Z):
        for x in range(LBVH_GRID_X):
            for mesh_idx in range(3):
                ref bounds = mesh_bounds[mesh_idx]
                var extent = bounds.extent()
                var local_extent = max(max(extent.x, extent.y), extent.z)
                if local_extent < Float32(1.0e-6):
                    local_extent = Float32(1.0)
                var variation = Float32(1.0) + Float32(
                    (x + z * LBVH_GRID_X) % 3
                ) * Float32(0.025)
                var scale = Vec3f32[Frame.LOCAL](
                    TARGET_EXTENT / local_extent * variation * 1.5
                )
                var angle = rng.f32(-1.0, 1.0)
                var rotation = Quat.from_axis_angle(
                    Vec3f32[Frame.LOCAL](0.0, 1.0, 0.0), angle
                )
                var cell_x = (
                    Float32(x) - Float32(LBVH_GRID_X - 1) * 0.5
                ) * cell_spacing
                var cell_z = (
                    Float32(z) - Float32(LBVH_GRID_Z - 1) * 0.5
                ) * cell_spacing
                var local_x = (
                    Float32(mesh_idx) - Float32(2 - 1) * 0.5
                ) * mesh_spacing
                var transform = _lbvh_centered_transform(
                    bounds,
                    rotation,
                    scale,
                    Vec3f32[Frame.WORLD](cell_x + local_x, 0.0, cell_z),
                )
                add_triangle_instance(
                    triangle_instances,
                    triangle_instance_surfaces,
                    UInt32(mesh_idx),
                    transform,
                    bounds,
                    mesh_surfaces[mesh_idx],
                )

    return World[world_bvh_width, instance_bvh_width](
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )


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
                + "/LBVH"
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
            result += " TLAS2/2 LBVH"
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
