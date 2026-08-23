"""Viewer scene lit by a transformed emissive triangle-mesh instance."""

from std.math import max

from bajo.bvh.host_utils import compute_bounds
from bajo.core import Affine3f32, Frame, Point3f32, Vec3f32
from bajo.parser.obj.pack import pack_obj_triangles
from bajo.rt import Color, CpuScene, SceneBuilder


comptime EMISSIVE_BUNNY_PATH = "assets/bunny/bunny.obj"


def make_emissive_instance_world[
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
]() raises -> CpuScene[world_bvh_width, instance_bvh_width]:
    var bunny = pack_obj_triangles(EMISSIVE_BUNNY_PATH)
    var bunny_bounds = compute_bounds(bunny)
    var builder = SceneBuilder()
    var white = builder.add_lambertian(Color(0.72, 0.72, 0.72))
    var warm = builder.add_lambertian(Color(0.72, 0.24, 0.10))
    var cool = builder.add_lambertian(Color(0.12, 0.30, 0.70))
    var metal = builder.add_metal(Color(0.82, 0.78, 0.68), 0.12)
    var light = builder.add_emissive(Color(12.0, 9.0, 6.0))

    builder.add_quad(
        Point3f32[Frame.WORLD](-4.0, 0.0, -5.0),
        Point3f32[Frame.WORLD](-4.0, 0.0, 2.0),
        Point3f32[Frame.WORLD](4.0, 0.0, 2.0),
        Point3f32[Frame.WORLD](4.0, 0.0, -5.0),
        white,
    )
    builder.add_quad(
        Point3f32[Frame.WORLD](-4.0, 0.0, -5.0),
        Point3f32[Frame.WORLD](4.0, 0.0, -5.0),
        Point3f32[Frame.WORLD](4.0, 4.0, -5.0),
        Point3f32[Frame.WORLD](-4.0, 4.0, -5.0),
        white,
    )
    builder.add_quad(
        Point3f32[Frame.WORLD](-4.0, 0.0, 2.0),
        Point3f32[Frame.WORLD](-4.0, 0.0, -5.0),
        Point3f32[Frame.WORLD](-4.0, 4.0, -5.0),
        Point3f32[Frame.WORLD](-4.0, 4.0, 2.0),
        warm,
    )
    builder.add_quad(
        Point3f32[Frame.WORLD](4.0, 0.0, -5.0),
        Point3f32[Frame.WORLD](4.0, 0.0, 2.0),
        Point3f32[Frame.WORLD](4.0, 4.0, 2.0),
        Point3f32[Frame.WORLD](4.0, 4.0, -5.0),
        cool,
    )
    builder.add_quad(
        Point3f32[Frame.WORLD](-4.0, 4.0, -5.0),
        Point3f32[Frame.WORLD](4.0, 4.0, -5.0),
        Point3f32[Frame.WORLD](4.0, 4.0, 2.0),
        Point3f32[Frame.WORLD](-4.0, 4.0, 2.0),
        white,
    )

    builder.add_sphere(Point3f32[Frame.WORLD](-1.25, 0.75, -1.5), 0.75, metal)
    builder.add_sphere(Point3f32[Frame.WORLD](1.05, 0.85, -2.2), 0.85, white)
    builder.add_sphere(Point3f32[Frame.WORLD](0.35, 0.55, 0.0), 0.55, warm)

    var extent = bunny_bounds.extent()
    var local_extent = max(max(extent.x, extent.y), extent.z)
    var scale = 1.55 / local_extent
    var center = bunny_bounds.centroid()
    var bunny_transform = Affine3f32[Frame.LOCAL, Frame.WORLD].from_scale(
        Vec3f32[Frame.LOCAL](scale)
    )
    bunny_transform.tx = -scale * center.x
    bunny_transform.ty = 2.15 - scale * bunny_bounds._min.y
    bunny_transform.tz = -2.35 - scale * center.z
    _ = builder.add_triangle_mesh_instance(
        bunny,
        bunny_transform,
        bunny_bounds,
        light,
    )

    var scene = builder^.finish()
    return CpuScene[world_bvh_width, instance_bvh_width](scene^)
