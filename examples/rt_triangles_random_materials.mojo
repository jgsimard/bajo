from std.math import max, round
from std.time import perf_counter_ns

from bajo.bvh import Camera, Instance
from bajo.bvh.host_utils import compute_bounds
from bajo.core import AABB, Affine3f32, Point3f32, Quat, Vec3f32
from bajo.core.random import Rng
from bajo.core.utils import ns_to_ms
from bajo.parser.obj.pack import pack_obj_triangles
from bajo.rt import (
    Color,
    Integrator,
    RenderSettings,
    SceneBuilder,
    SurfaceId,
    CpuScene,
    render_depth_first,
    write_ppm_from_colors,
)


comptime OBJ_PATH_0 = "./assets/bunny/bunny.obj"
comptime OBJ_PATH_1 = "./assets/buddha/buddha.obj"
comptime OBJ_PATH_2 = "./assets/dragon/dragon.obj"
comptime OBJ_PATH_3 = "./assets/armadillo/armadillo.obj"
comptime OBJ_PATH_4 = "./assets/lucy/lucy.obj"
comptime OBJ_PATH_5 = "./assets/nefertiti/nefertiti.obj"
comptime OBJ_PATH_6 = "./assets/igea/igea.obj"
comptime OUTPUT_PATH = "rt_triangles_random_materials.ppm"
comptime IMAGE_WIDTH = 960
comptime IMAGE_HEIGHT = 540
comptime SAMPLES_PER_PIXEL = 50
comptime MAX_DEPTH = 8
comptime RNG_SEED = UInt64(2026)
comptime INTEGRATOR = Integrator.PATH
comptime MESH_COUNT = 7
comptime COPY_COUNT = 3
comptime TRIANGLE_SAMPLE_STRIDE = 1
comptime _PointL = Point3f32[.LOCAL]
comptime _PointW = Point3f32[.WORLD]
comptime _VecL = Vec3f32[.LOCAL]
comptime _VecW = Vec3f32[.WORLD]
comptime _BoundsL = AABB[.LOCAL]
comptime _BoundsW = AABB[.WORLD]
comptime _Transform = Affine3f32[.LOCAL, .WORLD]


def _normalized_instance_scale(
    bounds: _BoundsL,
    target_extent: Float32,
    variation: Float32,
) -> _VecL:
    var extent = bounds.extent()
    var local_extent = max(max(extent.x, extent.y), extent.z)
    if local_extent < Float32(1.0e-6):
        local_extent = Float32(1.0)

    return _VecL(target_extent / local_extent * variation)


def _make_centered_transform(
    bounds: _BoundsL,
    rotation: Quat,
    scale: _VecL,
    bottom_center: _VecW,
) -> _Transform:
    var transform = _Transform.from_rotation_scale_translation(
        rotation, scale, _VecW(0.0)
    )
    var c = bounds.centroid()
    var local_anchor = _VecL(c.x, bounds._min.y, c.z)
    var anchor_delta = transform.vector(local_anchor)
    transform.tx = bottom_center.x - anchor_delta.x
    transform.ty = bottom_center.y - anchor_delta.y
    transform.tz = bottom_center.z - anchor_delta.z
    return transform^


def _random_surface(mut builder: SceneBuilder, mut rng: Rng) -> SurfaceId[1]:
    var choose = rng.f32()
    if choose < 0.58:
        var albedo = rng.vec3f32[.WORLD](0.15, 0.95) * rng.vec3f32[.WORLD](
            0.35, 1.0
        )
        return builder.add_lambertian(albedo)

    if choose < 0.88:
        var albedo = rng.vec3f32[.WORLD](0.45, 1.0)
        var fuzz = rng.f32(0.0, 0.22)
        return builder.add_metal(albedo, fuzz)

    var ior = rng.f32(1.25, 1.7)
    return builder.add_dielectric(ior)


def _random_ground_surface(mut builder: SceneBuilder) -> SurfaceId[1]:
    return builder.add_lambertian(Color(0.42, 0.45, 0.40))


def _mesh_x(instance_idx: Int) -> Float32:
    if instance_idx == 0:
        return -2.85
    elif instance_idx == 1:
        return -0.95
    elif instance_idx == 2:
        return 0.95
    elif instance_idx == 3:
        return 2.85
    elif instance_idx == 4:
        return -1.85
    elif instance_idx == 5:
        return 0.0
    return 1.85


def _mesh_z(instance_idx: Int) -> Float32:
    if instance_idx == 0:
        return -0.25
    elif instance_idx == 1:
        return -0.20
    elif instance_idx == 2:
        return -0.20
    elif instance_idx == 3:
        return -0.25
    elif instance_idx == 4:
        return 1.25
    elif instance_idx == 5:
        return 1.35
    return 1.25


def _append_mesh_instance(
    mut builder: SceneBuilder,
    mesh_idx: UInt32,
    bounds: _BoundsL,
    instance_idx: Int,
    copy_idx: Int,
    mut scene_rng: Rng,
    mut material_rng: Rng,
):
    comptime TARGET_WORLD_EXTENT = Float32(0.82)
    var x = _mesh_x(instance_idx)
    var z = _mesh_z(instance_idx) + (Float32(copy_idx) - 1.0) * 1.38
    var angle = scene_rng.f32(-0.45, 0.45)
    var rotation = Quat.from_axis_angle(_VecL(0.0, 1.0, 0.0), angle)
    var variation = Float32(0.95) + Float32(
        (instance_idx + copy_idx) % 4
    ) * Float32(0.035)
    var scale = _normalized_instance_scale(
        bounds, TARGET_WORLD_EXTENT, variation
    )
    var transform = _make_centered_transform(
        bounds, rotation, scale, _VecW(x, 0.0, z)
    )
    var surface = _random_surface(builder, material_rng)
    builder.add_triangle_instance(mesh_idx, transform, bounds, surface)


def _append_hero_glass_triangle(
    mut builder: SceneBuilder,
    bounds: _BoundsW,
):
    var c = bounds.centroid()
    var extent = bounds.extent()
    var glass = builder.add_dielectric(1.45)
    var y0 = bounds._min.y + extent.y * 0.12
    var y1 = bounds._min.y + extent.y * 0.82
    var z = bounds._min.z - max(extent.x, extent.z) * 0.035
    var w = max(extent.x, extent.z) * 0.055
    builder.add_triangle(
        _PointW(c.x - w, y0, z),
        _PointW(c.x + w, y0, z),
        _PointW(c.x, y1, z),
        glass,
    )


def _append_ground(
    mut builder: SceneBuilder,
    bounds: _BoundsW,
):
    var extent = bounds.extent()
    var pad = max(extent.x, extent.z) * 0.12
    if pad < 2.0:
        pad = 2.0

    var y = bounds._min.y - 0.02
    var x0 = bounds._min.x - pad
    var x1 = bounds._max.x + pad
    var z0 = bounds._min.z - pad
    var z1 = bounds._max.z + pad
    var ground = _random_ground_surface(builder)

    builder.add_quad(
        _PointW(x0, y, z0),
        _PointW(x1, y, z0),
        _PointW(x1, y, z1),
        _PointW(x0, y, z1),
        ground,
    )


def _make_camera(bounds: _BoundsW) -> Camera:
    var center = bounds.centroid()
    var extent = bounds.extent()
    var scene_w = max(extent.x, extent.z)
    if scene_w < 1.0:
        scene_w = 1.0

    var eye = _PointW(
        center.x,
        bounds._min.y + extent.y * 0.70,
        center.z - scene_w * 1.12,
    )
    var target = _PointW(center.x, bounds._min.y + extent.y * 0.30, center.z)
    return Camera.from_vfov(
        eye,
        target,
        _VecW(0.0, 1.0, 0.0),
        42.0,
    )


def _instance_bounds(instances: List[Instance]) -> _BoundsW:
    var bounds = _BoundsW.invalid()
    for inst in instances:
        bounds.grow(inst.bounds)
    return bounds


def _triangle_count(world: CpuScene[16, 16]) -> Int:
    var total = len(world.scene_data().triangle_vertices()) / 3
    for inst in world.scene_data().triangle_instances():
        total += (
            len(world.scene_data().triangle_meshes()[Int(inst.blas_idx)]) / 3
        )
    return total


def _count_primary_hits(
    settings: RenderSettings, camera: Camera, world: CpuScene[16, 16]
) -> Int:
    var hits = 0
    for py in range(settings.image_height):
        for px in range(settings.image_width):
            var ray = camera.make_ray(
                px,
                py,
                settings.image_width,
                settings.image_height,
            )
            if world.trace(ray):
                hits += 1

    return hits


def make_triangle_world() raises -> CpuScene[16, 16]:
    var scene_rng = Rng(seed=99, id=17)
    var material_rng = Rng(seed=RNG_SEED, id=0)

    var mesh0 = pack_obj_triangles(OBJ_PATH_0)
    var mesh1 = pack_obj_triangles(OBJ_PATH_1)
    var mesh2 = pack_obj_triangles(OBJ_PATH_2)
    var mesh3 = pack_obj_triangles(OBJ_PATH_3)
    var mesh4 = pack_obj_triangles(OBJ_PATH_4)
    var mesh5 = pack_obj_triangles(OBJ_PATH_5)
    var mesh6 = pack_obj_triangles(OBJ_PATH_6)

    var bounds0 = compute_bounds(mesh0)
    var bounds1 = compute_bounds(mesh1)
    var bounds2 = compute_bounds(mesh2)
    var bounds3 = compute_bounds(mesh3)
    var bounds4 = compute_bounds(mesh4)
    var bounds5 = compute_bounds(mesh5)
    var bounds6 = compute_bounds(mesh6)

    var builder = SceneBuilder()
    builder.triangle_meshes.append(mesh0.copy())
    builder.triangle_meshes.append(mesh3.copy())
    builder.triangle_meshes.append(mesh1.copy())
    builder.triangle_meshes.append(mesh5.copy())
    builder.triangle_meshes.append(mesh4.copy())
    builder.triangle_meshes.append(mesh2.copy())
    builder.triangle_meshes.append(mesh6.copy())

    for copy_idx in range(COPY_COUNT):
        _append_mesh_instance(
            builder,
            UInt32(0),
            bounds0,
            0,
            copy_idx,
            scene_rng,
            material_rng,
        )
        _append_mesh_instance(
            builder,
            UInt32(1),
            bounds3,
            1,
            copy_idx,
            scene_rng,
            material_rng,
        )
        _append_mesh_instance(
            builder,
            UInt32(2),
            bounds1,
            2,
            copy_idx,
            scene_rng,
            material_rng,
        )
        _append_mesh_instance(
            builder,
            UInt32(3),
            bounds5,
            3,
            copy_idx,
            scene_rng,
            material_rng,
        )
        _append_mesh_instance(
            builder,
            UInt32(4),
            bounds4,
            4,
            copy_idx,
            scene_rng,
            material_rng,
        )
        _append_mesh_instance(
            builder,
            UInt32(5),
            bounds2,
            5,
            copy_idx,
            scene_rng,
            material_rng,
        )
        _append_mesh_instance(
            builder,
            UInt32(6),
            bounds6,
            6,
            copy_idx,
            scene_rng,
            material_rng,
        )

    var scene_bounds = _instance_bounds(builder.triangle_instances)
    _append_hero_glass_triangle(builder, scene_bounds)
    _append_ground(builder, scene_bounds)

    var scene = builder^.finish()
    return CpuScene[16, 16](scene^)


def main() raises:
    print("Triangle mesh RT scene, random materials")
    print(t"OBJ 0: {OBJ_PATH_0}")
    print(t"OBJ 1: {OBJ_PATH_1}")
    print(t"OBJ 2: {OBJ_PATH_2}")
    print(t"OBJ 3: {OBJ_PATH_3}")
    print(t"OBJ 4: {OBJ_PATH_4}")
    print(t"OBJ 5: {OBJ_PATH_5}")
    print(t"OBJ 6: {OBJ_PATH_6}")
    print(t"image: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(t"samples: {SAMPLES_PER_PIXEL} | depth: {MAX_DEPTH}")
    print(t"triangle sample stride: {TRIANGLE_SAMPLE_STRIDE}")

    var settings = RenderSettings(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        SAMPLES_PER_PIXEL,
        RNG_SEED,
    )

    var build_t0 = perf_counter_ns()
    var world = make_triangle_world()
    var build_t1 = perf_counter_ns()
    var bounds = _instance_bounds(world.scene_data().triangle_instances())
    var camera = _make_camera(bounds)

    print(t"meshes: {len(world.scene_data().triangle_meshes())}")
    print(t"instances: {len(world.scene_data().triangle_instances())}")
    print(t"triangles traced: {_triangle_count(world)}")
    print(
        t"surfaces:"
        t" {len(world.scene_data().surfaces().lambertians) + len(world.scene_data().surfaces().metals) + len(world.scene_data().surfaces().dielectrics)}"
    )
    print(t"build ms: {round(ns_to_ms(Int(build_t1 - build_t0)), 3)}")
    print(t"primary hits: {_count_primary_hits(settings, camera, world)}")

    var render_t0 = perf_counter_ns()
    settings.max_depth = MAX_DEPTH
    var result = render_depth_first[INTEGRATOR](settings, camera, world)
    var render_t1 = perf_counter_ns()

    write_ppm_from_colors(
        OUTPUT_PATH,
        settings.image_width,
        settings.image_height,
        result.pixels,
    )
    print(t"render ms: {round(ns_to_ms(Int(render_t1 - render_t0)), 3)}")
    print(t"  total  : {round(ns_to_ms(result.timings.total_ns), 3)} ms")
    print(t"  init   : {round(ns_to_ms(result.timings.init_ns), 3)} ms")
    print(t"  kernel : {round(ns_to_ms(result.timings.render_ns), 3)} ms")
    print(t"wrote {OUTPUT_PATH}")
