from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
    assert_false,
)

from bajo.bvh.camera import Camera
from bajo.bvh.host_utils import compute_bounds
from bajo.core import (
    Affine3f32,
    Frame,
    Vec3f32,
    assert_vec_equal,
    dot,
    length,
    Point3f32,
    Rayf32,
)
from bajo.core.random import Rng
from bajo.rt import (
    Color,
    Dielectric,
    Instance,
    Lambertian,
    Metal,
    PrimitiveId,
    RENDER_AO,
    RENDER_NORMALS,
    RENDER_PATH,
    RenderSettings,
    Sphere,
    SurfaceId,
    SurfaceStore,
    World,
    add_sphere,
    add_triangle,
    add_triangle_instance,
    add_triangle_mesh,
    add_triangle_mesh_instance,
    render,
    render_wavefront,
)
from bajo.rt.cpu import (
    reflect,
    reflectance,
    scatter,
    scatter_dielectric,
    scatter_lambertian,
    scatter_metal,
)
from bajo.rt.types import HitRecord
from bajo.rt.types import MAT_LAMBERTIAN
from bajo.rt.types import PRIM_SPHERE, PRIM_TRIANGLE, PRIM_TRIANGLE_INSTANCE


def _front_hit() -> HitRecord:
    return HitRecord(
        PrimitiveId(PRIM_SPHERE, UInt32(0)),
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        Vec3f32[Frame.WORLD](0.0, 0.0, 1.0),
        SurfaceId(MAT_LAMBERTIAN, UInt32(0)),
        1.0,
        True,
    )


def test_reflect_and_reflectance() raises:
    var reflected = reflect(
        Vec3f32[Frame.WORLD](1.0, -1.0, 0.0),
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
    )
    assert_vec_equal(reflected, Vec3f32[Frame.WORLD](1.0, 1.0, 0.0))

    assert_almost_equal(reflectance(1.0, 1.5), 0.04, atol=1e-5)


def test_surface_id_is_packed() raises:
    var surface = SurfaceId(UInt32(2), UInt32(123))
    assert_equal(surface.kind(), UInt32(2))
    assert_equal(surface.index(), UInt32(123))
    assert_equal(surface.value, (UInt32(2) << UInt32(28)) | UInt32(123))


def test_lambertian_scatter_is_explicit() raises:
    var rng = Rng(seed=1, id=0)
    var material = Lambertian(Color(0.2, 0.4, 0.8))
    var surfaces = SurfaceStore()
    var surface = surfaces.add_lambertian(material.albedo)
    var hit = _front_hit()
    var incoming = Rayf32[Frame.WORLD](
        Point3f32[Frame.WORLD](0.0), Vec3f32[Frame.WORLD](0.0, 0.0, -1.0)
    )

    var scattered = scatter_lambertian(material, hit, rng)
    assert_true(scattered.ok)
    assert_vec_equal(scattered.attenuation, material.albedo)
    assert_true(length(scattered.ray.d) > 0.0)

    var dispatched = scatter(surface, surfaces, incoming, hit, rng)
    assert_true(dispatched.ok)
    assert_vec_equal(dispatched.attenuation, material.albedo)


def test_metal_scatter_can_absorb() raises:
    var rng = Rng(seed=2, id=0)
    var material = Metal(Color(0.8, 0.7, 0.6), 0.0)
    var hit = _front_hit()
    var incoming = Rayf32[Frame.WORLD](
        Point3f32[Frame.WORLD](0.0), Vec3f32[Frame.WORLD](0.0, 0.0, -1.0)
    )

    var scattered = scatter_metal(material, incoming, hit, rng)
    assert_true(scattered.ok)
    assert_vec_equal(scattered.attenuation, material.albedo)
    assert_true(dot(scattered.ray.d, hit.normal) > 0.0)

    var back_face_hit = HitRecord(
        hit.primitive.copy(),
        hit.p,
        -hit.normal,
        hit.surface.copy(),
        hit.t,
        False,
    )
    var absorbed = scatter_metal(material, incoming, back_face_hit, rng)
    assert_false(absorbed.ok)


def test_dielectric_scatter_is_explicit() raises:
    var rng = Rng(seed=3, id=0)
    var material = Dielectric(1.5)
    var surfaces = SurfaceStore()
    var surface = surfaces.add_dielectric(material.refraction_index)
    var hit = _front_hit()
    var incoming = Rayf32[Frame.WORLD](
        Point3f32[Frame.WORLD](0.0), Vec3f32[Frame.WORLD](0.0, 0.0, -1.0)
    )

    var scattered = scatter_dielectric(material, incoming, hit, rng)
    assert_true(scattered.ok)
    assert_vec_equal(scattered.attenuation, Color(1.0))
    assert_true(length(scattered.ray.d) > 0.0)

    var dispatched = scatter(surface, surfaces, incoming, hit, rng)
    assert_true(dispatched.ok)
    assert_vec_equal(dispatched.attenuation, Color(1.0))


def test_world_hit_maps_material_and_normal() raises:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId]()
    var matte = surfaces.add_lambertian(Color(0.5))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    var world = World(
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )

    var hit = (
        world.hit(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit.primitive.kind(), PRIM_SPHERE)
    assert_equal(hit.primitive.index(), UInt32(0))
    assert_equal(hit.surface.kind(), matte.kind())
    assert_equal(hit.surface.index(), matte.index())
    assert_true(hit.front_face)
    assert_vec_equal(hit.normal, Vec3f32[Frame.WORLD](0.0, 0.0, 1.0))
    assert_almost_equal(hit.t, 0.5)


def test_world_preserves_signed_radius_normals() raises:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId]()
    var glass = surfaces.add_dielectric(1.5)
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        -0.5,
        glass,
    )
    var world = World(
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )

    var hit = (
        world.hit(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_false(hit.front_face)
    assert_vec_equal(hit.normal, Vec3f32[Frame.WORLD](0.0, 0.0, 1.0))
    assert_almost_equal(hit.t, 0.5)


def test_world_hits_triangle() raises:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId]()
    var matte = surfaces.add_lambertian(Color(0.25, 0.5, 0.75))
    add_triangle(
        triangle_vertices,
        triangle_surfaces,
        Point3f32[Frame.WORLD](-1.0, -1.0, -2.0),
        Point3f32[Frame.WORLD](1.0, -1.0, -2.0),
        Point3f32[Frame.WORLD](0.0, 1.0, -2.0),
        matte,
    )
    var world = World(
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )

    var hit = (
        world.hit(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit.primitive.kind(), PRIM_TRIANGLE)
    assert_equal(hit.primitive.index(), UInt32(0))
    assert_equal(hit.surface.value, matte.value)
    assert_true(hit.front_face)
    assert_vec_equal(hit.normal, Vec3f32[Frame.WORLD](0.0, 0.0, 1.0))
    assert_almost_equal(hit.t, 2.0)


def test_world_picks_closest_sphere_or_triangle() raises:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId]()
    var sphere_surface = surfaces.add_lambertian(Color(0.5))
    var tri_surface = surfaces.add_metal(Color(0.9), 0.0)

    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.25,
        sphere_surface,
    )
    add_triangle(
        triangle_vertices,
        triangle_surfaces,
        Point3f32[Frame.WORLD](-1.0, -1.0, -2.0),
        Point3f32[Frame.WORLD](1.0, -1.0, -2.0),
        Point3f32[Frame.WORLD](0.0, 1.0, -2.0),
        tri_surface,
    )
    var world = World(
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )

    var hit = (
        world.hit(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit.primitive.kind(), PRIM_SPHERE)
    assert_equal(hit.surface.value, sphere_surface.value)
    assert_almost_equal(hit.t, 0.75)


def test_add_triangle_mesh_assigns_surface_per_triangle() raises:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId]()
    var matte = surfaces.add_lambertian(Color(0.3, 0.4, 0.5))
    var mesh = List[Point3f32[Frame.WORLD]]()
    mesh.append(Point3f32[Frame.WORLD](-1.0, -1.0, -2.0))
    mesh.append(Point3f32[Frame.WORLD](1.0, -1.0, -2.0))
    mesh.append(Point3f32[Frame.WORLD](0.0, 1.0, -2.0))
    mesh.append(Point3f32[Frame.WORLD](-1.0, -1.0, -3.0))
    mesh.append(Point3f32[Frame.WORLD](1.0, -1.0, -3.0))
    mesh.append(Point3f32[Frame.WORLD](0.0, 1.0, -3.0))

    add_triangle_mesh(triangle_vertices, triangle_surfaces, mesh, matte)
    assert_equal(len(triangle_vertices), 6)
    assert_equal(len(triangle_surfaces), 2)

    var world = World(
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )

    var hit = (
        world.hit(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit.primitive.kind(), PRIM_TRIANGLE)
    assert_equal(hit.surface.value, matte.value)
    assert_almost_equal(hit.t, 2.0)


def test_triangle_mesh_instances_use_instance_surfaces() raises:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId]()
    var matte = surfaces.add_lambertian(Color(0.2, 0.4, 0.8))
    var metal = surfaces.add_metal(Color(0.9, 0.8, 0.7), 0.0)

    var mesh = List[Point3f32[Frame.LOCAL]]()
    mesh.append(Point3f32[Frame.LOCAL](-0.5, -0.5, -2.0))
    mesh.append(Point3f32[Frame.LOCAL](0.5, -0.5, -2.0))
    mesh.append(Point3f32[Frame.LOCAL](0.0, 0.5, -2.0))
    var mesh_bounds = compute_bounds(mesh)

    var transform = Affine3f32[Frame.LOCAL, Frame.WORLD].identity()
    var inv_transform = Affine3f32[Frame.WORLD, Frame.LOCAL].identity()
    var mesh_idx = add_triangle_mesh_instance(
        triangle_meshes,
        triangle_instances,
        triangle_instance_surfaces,
        mesh,
        transform,
        inv_transform,
        mesh_bounds,
        matte,
    )

    var t = Affine3f32[Frame.LOCAL, Frame.WORLD].from_translation(
        Vec3f32[Frame.WORLD](1.5, 0.0, 0.0)
    )
    var inv_t = Affine3f32[Frame.WORLD, Frame.LOCAL].from_translation(
        Vec3f32[Frame.LOCAL](-1.5, 0.0, 0.0)
    )
    add_triangle_instance(
        triangle_instances,
        triangle_instance_surfaces,
        mesh_idx,
        t,
        inv_t,
        mesh_bounds,
        metal,
    )

    var world = World(
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )

    var hit0 = (
        world.hit(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit0.primitive.kind(), PRIM_TRIANGLE_INSTANCE)
    assert_equal(hit0.primitive.index(), UInt32(0))
    assert_equal(hit0.surface.value, matte.value)
    assert_almost_equal(hit0.t, 2.0)

    var hit1 = (
        world.hit(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](1.5, 0.0, 0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit1.primitive.kind(), PRIM_TRIANGLE_INSTANCE)
    assert_equal(hit1.primitive.index(), UInt32(1))
    assert_equal(hit1.surface.value, metal.value)
    assert_almost_equal(hit1.t, 2.0)


def test_render_settings_and_tiny_render() raises:
    var settings = RenderSettings(4, 2, 2, UInt64(9))
    assert_equal(settings.image_width, 4)
    assert_equal(settings.image_height, 2)
    assert_equal(settings.rng_seed, 9)

    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId]()
    var matte = surfaces.add_lambertian(Color(0.5))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    var world = World(
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )
    var camera = Camera.from_vfov(
        Point3f32[Frame.WORLD](0.0, 0.0, 0.0),
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
        90.0,
    )

    var result = render[RENDER_PATH, 2](settings, camera, world)
    assert_equal(len(result.pixels), 8)
    assert_equal(result.timings.pixel_count, 8)
    assert_equal(result.timings.sample_count, 16)
    assert_equal(result.timings.max_depth, 2)
    assert_true(result.timings.total_ns >= result.timings.render_ns)
    for p in result.pixels:
        assert_true(p.x >= 0.0 and p.y >= 0.0 and p.z >= 0.0)


def test_render_can_select_normal_algorithm_at_compile_time() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(11))
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId]()
    var matte = surfaces.add_lambertian(Color(0.5))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    var world = World(
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )
    var camera = Camera.from_vfov(
        Point3f32[Frame.WORLD](0.0, 0.0, 0.0),
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
        90.0,
    )

    var result = render[RENDER_NORMALS, 1](settings, camera, world)
    assert_equal(len(result.pixels), 1)
    assert_true(result.pixels[0].z >= result.pixels[0].x)


def test_render_can_select_ao_algorithm_at_compile_time() raises:
    var settings = RenderSettings(1, 1, 2, UInt64(12))
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId]()
    var matte = surfaces.add_lambertian(Color(0.5))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    var world = World(
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )
    var camera = Camera.from_vfov(
        Point3f32[Frame.WORLD](0.0, 0.0, 0.0),
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
        90.0,
    )

    var result = render[RENDER_AO, 1](settings, camera, world)
    assert_equal(len(result.pixels), 1)
    assert_true(
        result.pixels[0].x >= 0.0
        and result.pixels[0].y >= 0.0
        and result.pixels[0].z >= 0.0
    )


def test_wavefront_tiny_render() raises:
    var settings = RenderSettings(4, 2, 2, UInt64(9))
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId]()
    var matte = surfaces.add_lambertian(Color(0.5))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    var world = World(
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )
    var camera = Camera.from_vfov(
        Point3f32[Frame.WORLD](0.0, 0.0, 0.0),
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
        90.0,
    )

    var result = render_wavefront[RENDER_PATH, 2](settings, camera, world)
    assert_equal(len(result.pixels), 8)
    assert_equal(result.timings.sample_count, 16)
    for p in result.pixels:
        assert_true(p.x >= 0.0 and p.y >= 0.0 and p.z >= 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
