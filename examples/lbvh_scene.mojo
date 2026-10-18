"""Shared heavy instancing scene used by the viewer and GPU RT benchmarks."""

from std.math import cos, sin

from bajo.bvh.cpu import CpuBvhBuildMethod
from bajo.bvh.host_utils import compute_bounds
from bajo.core import (
    AABB,
    Point3f32,
    Quat,
    Vec3f32,
)
from bajo.core.random import Rng
from examples.mesh_helpers import (
    normalized_instance_scale,
    bottom_centered_transform,
)
from bajo.core.utils import degrees_to_radians
from bajo.parser.obj.pack import pack_obj_triangles
from bajo.rt import (
    Camera,
    Color,
    SceneBuilder,
    CpuScene,
)


comptime LBVH_OBJ_PATHS = (
    "assets/bunny/bunny.obj",
    "assets/buddha/buddha.obj",
    "assets/dragon/dragon.obj",
)
comptime LBVH_GRID_X = 6
comptime LBVH_GRID_Z = 6


def make_lbvh_camera() -> Camera:
    """Return the viewer's default LBVH camera without involving Python."""
    comptime yaw_degrees = Float32(180.0)
    comptime pitch_degrees = Float32(-8.0)
    var yaw = degrees_to_radians(yaw_degrees)
    var pitch = degrees_to_radians(pitch_degrees)
    var cos_pitch = cos(pitch)
    var origin = Point3f32[.WORLD](0.0, 6.0, -28.0)
    var forward = Vec3f32[.WORLD](
        sin(yaw) * cos_pitch,
        sin(pitch),
        -cos(yaw) * cos_pitch,
    )
    return Camera.from_vfov(
        origin,
        origin + forward,
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        35.0,
        10.0,
        0.0,
    )


def make_lbvh_world[
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
    build_method: CpuBvhBuildMethod = .SAH,
]() raises -> CpuScene[world_bvh_width, instance_bvh_width]:
    var builder = SceneBuilder()
    var bunny_surface = builder.add_lambertian(Color(0.72, 0.34, 0.12))
    var buddha_surface = builder.add_lambertian(Color(0.18, 0.42, 0.78))
    var dragon_surface = builder.add_lambertian(Color(0.68, 0.12, 0.10))
    var ground_surface = builder.add_lambertian(Color(0.28, 0.30, 0.34))
    var light_surface = builder.add_emissive(Color(18.0, 16.0, 13.0))

    builder.add_sphere(
        Point3f32[.WORLD](0.0, 12.0, 0.0),
        2.0,
        light_surface,
    )

    builder.add_triangle(
        Point3f32[.WORLD](-26.0, 0.0, -26.0),
        Point3f32[.WORLD](26.0, 0.0, -26.0),
        Point3f32[.WORLD](26.0, 0.0, 26.0),
        ground_surface,
    )
    builder.add_triangle(
        Point3f32[.WORLD](-26.0, 0.0, -26.0),
        Point3f32[.WORLD](26.0, 0.0, 26.0),
        Point3f32[.WORLD](-26.0, 0.0, 26.0),
        ground_surface,
    )

    var mesh_bounds = List[AABB[.LOCAL]](capacity=len(LBVH_OBJ_PATHS))
    try:
        comptime for mesh in range(len(LBVH_OBJ_PATHS)):
            var vertices = pack_obj_triangles(LBVH_OBJ_PATHS[mesh])
            mesh_bounds.append(compute_bounds(vertices))
            _ = builder.add_owned_triangle_mesh(
                vertices^, List[Float32](), List[Float32]()
            )
    except e:
        builder^.abort()
        raise e

    var rng = Rng(123, 123)
    comptime TARGET_EXTENT = Float32(1.60)
    var mesh_surfaces = [
        bunny_surface.copy(),
        buddha_surface.copy(),
        dragon_surface.copy(),
    ]
    var cell_spacing = TARGET_EXTENT * Float32(5.2)
    var mesh_spacing = TARGET_EXTENT * Float32(2.0)
    for z in range(LBVH_GRID_Z):
        for x in range(LBVH_GRID_X):
            for mesh_idx in range(len(LBVH_OBJ_PATHS)):
                ref bounds = mesh_bounds[mesh_idx]
                var variation = Float32(1.0) + Float32(
                    (x + z * LBVH_GRID_X) % 3
                ) * Float32(0.025)
                var scale = (
                    normalized_instance_scale(bounds, TARGET_EXTENT, variation)
                    * 1.5
                )
                var angle = rng.f32(-1.0, 1.0)
                var rotation = Quat.from_axis_angle(
                    Vec3f32[.LOCAL](0.0, 1.0, 0.0), angle
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
                var transform = bottom_centered_transform(
                    bounds,
                    rotation,
                    scale,
                    Vec3f32[.WORLD](cell_x + local_x, 0.0, cell_z),
                )
                builder.add_triangle_instance(
                    UInt32(mesh_idx),
                    transform,
                    bounds,
                    mesh_surfaces[mesh_idx],
                )

    var scene = builder^.finish()
    return CpuScene[world_bvh_width, instance_bvh_width].__init__[build_method](
        scene^
    )
