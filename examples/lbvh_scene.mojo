"""Shared heavy instancing scene used by the viewer and GPU RT benchmarks."""

from std.math import cos, max, sin

from bajo.bvh.host_utils import compute_bounds
from bajo.core import (
    AABB,
    Affine3f32,
    Frame,
    Point3f32,
    Quat,
    Vec3f32,
)
from bajo.core.random import Rng
from bajo.core.utils import degrees_to_radians
from bajo.parser.obj.pack import pack_obj_triangles
from bajo.rt import (
    Camera,
    Color,
    Instance,
    Sphere,
    SurfaceId,
    SurfaceStore,
    World,
    add_sphere,
    add_triangle,
    add_triangle_instance,
)


comptime LBVH_OBJ_PATH_0 = "assets/bunny/bunny.obj"
comptime LBVH_OBJ_PATH_1 = "assets/buddha/buddha.obj"
comptime LBVH_OBJ_PATH_2 = "assets/dragon/dragon.obj"
comptime LBVH_GRID_X = 6
comptime LBVH_GRID_Z = 6


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


def make_lbvh_camera() -> Camera:
    """Return the viewer's default LBVH camera without involving Python."""
    comptime yaw_degrees = Float32(180.0)
    comptime pitch_degrees = Float32(-8.0)
    var yaw = degrees_to_radians(yaw_degrees)
    var pitch = degrees_to_radians(pitch_degrees)
    var cos_pitch = cos(pitch)
    var origin = Point3f32[Frame.WORLD](0.0, 6.0, -28.0)
    var forward = Vec3f32[Frame.WORLD](
        sin(yaw) * cos_pitch,
        sin(pitch),
        -cos(yaw) * cos_pitch,
    )
    return Camera.from_vfov(
        origin,
        origin + forward,
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
        35.0,
        10.0,
        0.0,
    )


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
