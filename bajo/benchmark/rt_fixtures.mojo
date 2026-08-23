"""Shared cameras and scenes used by RT benchmarks."""

from std.math import max

from bajo.bvh.host_utils import compute_bounds
from bajo.core import (
    AABB,
    Affine3f32,
    Frame,
    Point3f32,
    Point3W,
    Rayf32,
    Vec3f32,
    Vec3W,
)
from bajo.rt import (
    Camera,
    Color,
    SceneBuilder,
    CpuScene,
)
from .bvh_fixtures import make_grid_triangles, make_hit_and_miss_rays


comptime TRIANGLE_GRID = 64


def weekend_camera(aperture: Float32 = 0.6) -> Camera:
    return Camera.from_vfov(
        Point3W(13.0, 2.0, 3.0),
        Point3W(0.0, 0.0, 0.0),
        Vec3W(0.0, 1.0, 0.0),
        20.0,
        10.0,
        aperture,
    )


def make_mixed_triangle_mesh() -> List[Point3f32[.LOCAL]]:
    var vertices = List[Point3f32[.LOCAL]](
        capacity=TRIANGLE_GRID * TRIANGLE_GRID * 6
    )
    var inv_grid = 1.0 / Float32(TRIANGLE_GRID)
    for z in range(TRIANGLE_GRID):
        for x in range(TRIANGLE_GRID):
            var x0 = -0.9 + 1.8 * Float32(x) * inv_grid
            var x1 = -0.9 + 1.8 * Float32(x + 1) * inv_grid
            var z0 = -0.9 + 1.8 * Float32(z) * inv_grid
            var z1 = -0.9 + 1.8 * Float32(z + 1) * inv_grid
            var y00 = Float32(0.08) if (x + z) % 7 == 0 else Float32(0.0)
            var y10 = Float32(0.08) if (x + 1 + z) % 7 == 0 else Float32(0.0)
            var y01 = Float32(0.08) if (x + z + 1) % 7 == 0 else Float32(0.0)
            var y11 = Float32(0.08) if (x + z + 2) % 7 == 0 else Float32(0.0)
            var p00 = Point3f32[.LOCAL](x0, y00, z0)
            var p10 = Point3f32[.LOCAL](x1, y10, z0)
            var p01 = Point3f32[.LOCAL](x0, y01, z1)
            var p11 = Point3f32[.LOCAL](x1, y11, z1)
            vertices.append(p00)
            vertices.append(p11)
            vertices.append(p10)
            vertices.append(p00)
            vertices.append(p01)
            vertices.append(p11)
    return vertices^


def make_mixed_triangle_world() raises -> CpuScene[]:
    var builder = SceneBuilder()
    var diffuse = builder.add_lambertian(Color(0.55, 0.32, 0.18))
    var ground = builder.add_lambertian(Color(0.35, 0.38, 0.32))
    var metal = builder.add_metal(Color(0.75, 0.78, 0.82), 0.12)
    var glass = builder.add_dielectric(1.45)

    builder.add_triangle(
        Point3W(-7.0, -0.2, -7.0),
        Point3W(7.0, -0.2, 7.0),
        Point3W(7.0, -0.2, -7.0),
        ground,
    )
    builder.add_triangle(
        Point3W(-7.0, -0.2, -7.0),
        Point3W(-7.0, -0.2, 7.0),
        Point3W(7.0, -0.2, 7.0),
        ground,
    )

    var mesh = make_mixed_triangle_mesh()
    var mesh_bounds = compute_bounds(mesh)
    var first_transform = Affine3f32[.LOCAL, .WORLD].from_translation(
        Vec3f32[.WORLD](-4.0, 0.0, -4.0)
    )
    var mesh_idx = builder.add_triangle_mesh_instance(
        mesh,
        first_transform,
        mesh_bounds,
        diffuse,
    )
    for iz in range(5):
        for ix in range(5):
            if ix == 0 and iz == 0:
                continue
            var transform = Affine3f32[
                .LOCAL, .WORLD
            ].from_translation(
                Vec3f32[.WORLD](
                    Float32(ix) * 2.0 - 4.0,
                    0.0,
                    Float32(iz) * 2.0 - 4.0,
                )
            )
            var selector = (ix + 2 * iz) % 5
            var surface = diffuse.copy()
            if selector == 1:
                surface = metal.copy()
            elif selector == 2:
                surface = glass.copy()
            builder.add_triangle_instance(
                mesh_idx,
                transform,
                mesh_bounds,
                surface,
            )

    var scene = builder^.finish()
    return CpuScene[](scene^)


def mixed_triangle_camera(world: CpuScene[]) -> Camera:
    var bounds = AABB[.WORLD].invalid()
    for inst in world.scene_data().triangle_instances():
        bounds.grow(inst.bounds)
    var center = bounds.centroid()
    var extent = bounds.extent()
    var scene_w = max(extent.x, extent.z)
    if scene_w < 1.0:
        scene_w = 1.0
    var eye = Point3W(center.x, center.y + scene_w * 0.78, center.z + scene_w)
    return Camera.from_vfov(
        eye, Point3W(center.x, center.y, center.z), Vec3W(0.0, 1.0, 0.0), 44.0
    )


def make_grid_triangle_world() raises -> CpuScene[]:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    var vertices = make_grid_triangles()
    builder.add_triangle_mesh(vertices, matte)
    var scene = builder^.finish()
    return CpuScene[](scene^)


def make_bounded_grid_rays() -> List[Rayf32[.WORLD]]:
    var source = make_hit_and_miss_rays()
    var rays = List[Rayf32[.WORLD]](capacity=len(source))
    for ray in source:
        rays.append(Rayf32[.WORLD](ray.o, ray.d, ray.t_min, Float32(3.0)))
    return rays^
