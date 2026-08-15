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
    RENDER,
    RenderSettings,
    ShadingPoint,
    Sphere,
    SurfaceId,
    SurfaceHit,
    SurfaceStore,
    World,
    add_sphere,
    add_triangle,
    add_triangle_instance,
    add_triangle_mesh,
    add_triangle_mesh_instance,
    evaluate_bsdf,
    render_depth_first,
    render_wavefront,
    sample_bsdf,
    wavefront_rng_roulette_stage,
)
from examples.cornell_box import make_cornell_world
from bajo.rt.cpu import reflect, reflectance
from bajo.rt.cpu.common import _path_stage_rng, _russian_roulette
from bajo.rt.types import MAT, PRIM


def _front_point() -> ShadingPoint[1]:
    return ShadingPoint(
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        Vec3f32[Frame.WORLD](0.0, 0.0, 1.0),
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
    var surface = SurfaceId(MAT(UInt32(2)), UInt32(123))
    assert_equal(surface.kind().v, UInt32(2))
    assert_equal(surface.index(), UInt32(123))
    assert_equal(surface.value, (UInt32(2) << UInt32(28)) | UInt32(123))


def test_surface_hit_is_width_generic() raises:
    var hits = SurfaceHit[4](SIMD[DType.float32, 4](100.0))
    hits.normal.x[2] = 1.0
    hits.normal.y[2] = 2.0
    hits.normal.z[2] = 3.0
    hits.surface.value[2] = SurfaceId(MAT.METAL, UInt32(7)).value
    hits.t[2] = 4.5
    hits.front_face[2] = False
    hits.hit[2] = True

    var lane = hits.get(2)
    assert_vec_equal(lane.normal, Vec3f32[Frame.WORLD](1.0, 2.0, 3.0))
    assert_equal(lane.surface.kind(), MAT.METAL)
    assert_equal(lane.surface.index(), UInt32(7))
    assert_almost_equal(lane.t, 4.5)
    assert_false(lane.front_face)
    assert_true(lane.hit)


def test_wavefront_philox_streams_are_deterministic_and_separate() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(91))
    var first = _path_stage_rng(settings, UInt32(7), UInt32(3))
    var replay = _path_stage_rng(settings, UInt32(7), UInt32(3))
    var next_stage = _path_stage_rng(settings, UInt32(7), UInt32(4))
    var roulette = _path_stage_rng(
        settings,
        UInt32(7),
        wavefront_rng_roulette_stage(UInt32(2)),
    )

    assert_equal(first.f32(), replay.f32())
    assert_true(first.f32() != next_stage.f32())
    assert_true(replay.f32() != roulette.f32())


def test_russian_roulette_is_deterministic_and_unbiased() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(8128))
    var throughput = Color(0.25, 0.2, 0.1)
    var early = _russian_roulette(settings, UInt32(17), UInt32(4), throughput)
    assert_true(early.survived)
    assert_vec_equal(early.throughput, throughput)

    var first = _russian_roulette(settings, UInt32(17), UInt32(5), throughput)
    var replay = _russian_roulette(settings, UInt32(17), UInt32(5), throughput)
    assert_equal(first.survived, replay.survived)
    assert_vec_equal(first.throughput, replay.throughput)

    comptime TRIALS = 20000
    var weighted_sum = Float64(0.0)
    for path_idx in range(TRIALS):
        var result = _russian_roulette(
            settings, UInt32(path_idx), UInt32(5), throughput
        )
        if result.survived:
            weighted_sum += Float64(result.throughput.x)
    assert_almost_equal(
        weighted_sum / Float64(TRIALS),
        Float64(throughput.x),
        atol=0.01,
    )


def test_lambertian_scatter_is_explicit() raises:
    var rng = Rng(seed=1, id=0)
    var material = Lambertian(Color(0.2, 0.4, 0.8))
    var surfaces = SurfaceStore()
    var surface = surfaces.add_lambertian(material.albedo)
    var point = _front_point()
    var incoming = Rayf32[Frame.WORLD](
        Point3f32[Frame.WORLD](0.0), Vec3f32[Frame.WORLD](0.0, 0.0, -1.0)
    )

    var scattered = sample_bsdf(surface, surfaces, incoming, point, rng)
    assert_true(scattered.ok)
    assert_vec_equal(scattered.weight, material.albedo)
    assert_true(length(scattered.direction) > 0.0)

    var sampled = sample_bsdf(surface, surfaces, incoming, point, rng)
    var evaluated = evaluate_bsdf(
        surface, surfaces, incoming, point, sampled.direction
    )
    assert_false(sampled.delta)
    assert_false(evaluated.delta)
    assert_true(sampled.pdf > 0.0)
    assert_almost_equal(sampled.pdf, evaluated.pdf)
    assert_almost_equal(evaluated.value.x, material.albedo.x / 3.14159265)


def test_metal_scatter_can_absorb() raises:
    var rng = Rng(seed=2, id=0)
    var material = Metal(Color(0.8, 0.7, 0.6), 0.0)
    var surfaces = SurfaceStore()
    var surface = surfaces.add_metal(material.albedo, material.fuzz)
    var point = _front_point()
    var incoming = Rayf32[Frame.WORLD](
        Point3f32[Frame.WORLD](0.0), Vec3f32[Frame.WORLD](0.0, 0.0, -1.0)
    )

    var scattered = sample_bsdf(surface, surfaces, incoming, point, rng)
    assert_true(scattered.ok)
    assert_vec_equal(scattered.weight, material.albedo)
    assert_true(dot(scattered.direction, point.normal) > 0.0)
    assert_true(scattered.delta)
    assert_equal(scattered.pdf, 1.0)

    var back_face_point = ShadingPoint(point.p, -point.normal, False)
    var absorbed = sample_bsdf(
        surface, surfaces, incoming, back_face_point, rng
    )
    assert_false(absorbed.ok)


def test_dielectric_scatter_is_explicit() raises:
    var rng = Rng(seed=3, id=0)
    var material = Dielectric(1.5)
    var surfaces = SurfaceStore()
    var surface = surfaces.add_dielectric(material.refraction_index)
    var point = _front_point()
    var incoming = Rayf32[Frame.WORLD](
        Point3f32[Frame.WORLD](0.0), Vec3f32[Frame.WORLD](0.0, 0.0, -1.0)
    )

    var scattered = sample_bsdf(surface, surfaces, incoming, point, rng)
    assert_true(scattered.ok)
    assert_vec_equal(scattered.weight, Color(1.0))
    assert_true(length(scattered.direction) > 0.0)
    assert_true(scattered.delta)
    assert_true(scattered.pdf > 0.0 and scattered.pdf <= 1.0)

    var dispatched = sample_bsdf(surface, surfaces, incoming, point, rng)
    assert_true(dispatched.ok)
    assert_vec_equal(dispatched.weight, Color(1.0))


def test_world_hit_maps_material_and_normal() raises:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var matte = surfaces.add_lambertian(Color(0.5))
    var light = surfaces.add_emissive(Color(4.0))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](10.0, 0.0, -1.0),
        0.25,
        light,
    )
    var world = World[4, 8](
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )
    assert_equal(len(world.lights.records), 1)
    assert_equal(world.lights.records[0].primitive.kind(), PRIM.SPHERE)
    assert_equal(world.lights.records[0].surface.value, light.value)
    assert_true(world.lights.total_weight > 0.0)

    var hit = (
        world.trace(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit.primitive.kind(), PRIM.SPHERE)
    assert_equal(hit.primitive.index(), UInt32(0))
    assert_equal(hit.surface.kind(), matte.kind())
    assert_equal(hit.surface.index(), matte.index())
    assert_true(hit.front_face)
    assert_vec_equal(hit.normal, Vec3f32[Frame.WORLD](0.0, 0.0, 1.0))
    assert_almost_equal(hit.t, 0.5)
    var compact = world.trace_surface(
        Rayf32[Frame.WORLD](
            Point3f32[Frame.WORLD](0.0),
            Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
        )
    )
    assert_true(compact.hit)
    assert_equal(compact.surface.value, hit.surface.value)
    assert_vec_equal(compact.normal, hit.normal)
    assert_almost_equal(compact.t, hit.t)


def test_world_preserves_signed_radius_normals() raises:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var glass = surfaces.add_dielectric(1.5)
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        -0.5,
        glass,
    )
    var world = World[](
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
        world.trace(
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
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var matte = surfaces.add_lambertian(Color(0.25, 0.5, 0.75))
    add_triangle(
        triangle_vertices,
        triangle_surfaces,
        Point3f32[Frame.WORLD](-1.0, -1.0, -2.0),
        Point3f32[Frame.WORLD](1.0, -1.0, -2.0),
        Point3f32[Frame.WORLD](0.0, 1.0, -2.0),
        matte,
    )
    var world = World[](
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
        world.trace(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit.primitive.kind(), PRIM.TRIANGLE)
    assert_equal(hit.primitive.index(), UInt32(0))
    assert_equal(hit.surface.value, matte.value)
    assert_true(hit.front_face)
    assert_vec_equal(hit.normal, Vec3f32[Frame.WORLD](0.0, 0.0, 1.0))
    assert_almost_equal(hit.t, 2.0)

    var back_hit = (
        world.trace(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](0.0, 0.0, -4.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, 1.0),
            )
        )
        .value()
        .copy()
    )
    assert_false(back_hit.front_face)
    assert_vec_equal(back_hit.normal, Vec3f32[Frame.WORLD](0.0, 0.0, -1.0))
    assert_almost_equal(back_hit.t, 2.0)
    var compact = world.trace_surface(
        Rayf32[Frame.WORLD](
            Point3f32[Frame.WORLD](0.0),
            Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
        )
    )
    assert_true(compact.hit)
    assert_equal(compact.surface.value, hit.surface.value)
    assert_vec_equal(compact.normal, hit.normal)
    assert_almost_equal(compact.t, hit.t)


def test_world_picks_closest_sphere_or_triangle() raises:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
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
    var world = World[](
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
        world.trace(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit.primitive.kind(), PRIM.SPHERE)
    assert_equal(hit.surface.value, sphere_surface.value)
    assert_almost_equal(hit.t, 0.75)


def test_add_triangle_mesh_assigns_surface_per_triangle() raises:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
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

    var world = World[](
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
        world.trace(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit.primitive.kind(), PRIM.TRIANGLE)
    assert_equal(hit.surface.value, matte.value)
    assert_almost_equal(hit.t, 2.0)


def test_triangle_mesh_instances_use_instance_surfaces() raises:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var matte = surfaces.add_lambertian(Color(0.2, 0.4, 0.8))
    var metal = surfaces.add_metal(Color(0.9, 0.8, 0.7), 0.0)

    var mesh = List[Point3f32[Frame.LOCAL]]()
    mesh.append(Point3f32[Frame.LOCAL](-0.5, -0.5, -2.0))
    mesh.append(Point3f32[Frame.LOCAL](0.5, -0.5, -2.0))
    mesh.append(Point3f32[Frame.LOCAL](0.0, 0.5, -2.0))
    var mesh_bounds = compute_bounds(mesh)

    var transform = Affine3f32[Frame.LOCAL, Frame.WORLD].identity()
    var mesh_idx = add_triangle_mesh_instance(
        triangle_meshes,
        triangle_instances,
        triangle_instance_surfaces,
        mesh,
        transform,
        mesh_bounds,
        matte,
    )

    var t = Affine3f32[Frame.LOCAL, Frame.WORLD].from_translation(
        Vec3f32[Frame.WORLD](1.5, 0.0, 0.0)
    )
    add_triangle_instance(
        triangle_instances,
        triangle_instance_surfaces,
        mesh_idx,
        t,
        mesh_bounds,
        metal,
    )

    var world = World[4, 8](
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
        world.trace(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit0.primitive.kind(), PRIM.TRIANGLE_INSTANCE)
    assert_equal(hit0.primitive.index(), UInt32(0))
    assert_equal(hit0.surface.value, matte.value)
    assert_almost_equal(hit0.t, 2.0)

    var hit1 = (
        world.trace(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](1.5, 0.0, 0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit1.primitive.kind(), PRIM.TRIANGLE_INSTANCE)
    assert_equal(hit1.primitive.index(), UInt32(1))
    assert_equal(hit1.surface.value, metal.value)
    assert_almost_equal(hit1.t, 2.0)
    var compact = world.trace_surface(
        Rayf32[Frame.WORLD](
            Point3f32[Frame.WORLD](1.5, 0.0, 0.0),
            Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
        )
    )
    assert_true(compact.hit)
    assert_equal(compact.surface.value, hit1.surface.value)
    assert_vec_equal(compact.normal, hit1.normal)
    assert_almost_equal(compact.t, hit1.t)


def test_world_occluded_covers_all_geometry_and_ray_interval() raises:
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var matte = surfaces.add_lambertian(Color(0.5))

    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](-2.0, 0.0, -2.0),
        0.5,
        matte,
    )
    add_triangle(
        triangle_vertices,
        triangle_surfaces,
        Point3f32[Frame.WORLD](-0.75, -0.75, -3.0),
        Point3f32[Frame.WORLD](0.75, -0.75, -3.0),
        Point3f32[Frame.WORLD](0.0, 0.75, -3.0),
        matte,
    )

    var mesh = List[Point3f32[Frame.LOCAL]]()
    mesh.append(Point3f32[Frame.LOCAL](-0.5, -0.5, -4.0))
    mesh.append(Point3f32[Frame.LOCAL](0.5, -0.5, -4.0))
    mesh.append(Point3f32[Frame.LOCAL](0.0, 0.5, -4.0))
    var mesh_bounds = compute_bounds(mesh)
    var transform = Affine3f32[Frame.LOCAL, Frame.WORLD].from_translation(
        Vec3f32[Frame.WORLD](2.0, 0.0, 0.0)
    )
    _ = add_triangle_mesh_instance(
        triangle_meshes,
        triangle_instances,
        triangle_instance_surfaces,
        mesh,
        transform,
        mesh_bounds,
        matte,
    )

    var world = World[](
        spheres^,
        sphere_surfaces^,
        triangle_vertices^,
        triangle_surfaces^,
        triangle_meshes^,
        triangle_instances^,
        triangle_instance_surfaces^,
        surfaces^,
    )

    var sphere_ray = Rayf32[Frame.WORLD](
        Point3f32[Frame.WORLD](-2.0, 0.0, 0.0),
        Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.001,
        3.0,
    )
    assert_true(world.occluded(sphere_ray))
    assert_false(
        world.occluded(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](-2.0, 0.0, 0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
                0.001,
                1.0,
            )
        )
    )

    var triangle_ray = Rayf32[Frame.WORLD](
        Point3f32[Frame.WORLD](0.0),
        Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.001,
        4.0,
    )
    assert_true(world.occluded(triangle_ray))
    assert_false(
        world.occluded(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
                0.001,
                2.0,
            )
        )
    )

    var instance_ray = Rayf32[Frame.WORLD](
        Point3f32[Frame.WORLD](2.0, 0.0, 0.0),
        Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.001,
        5.0,
    )
    assert_true(world.occluded(instance_ray))
    assert_false(
        world.occluded(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](2.0, 0.0, 0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
                0.001,
                3.0,
            )
        )
    )

    assert_false(
        world.occluded(
            Rayf32[Frame.WORLD](
                Point3f32[Frame.WORLD](5.0, 0.0, 0.0),
                Vec3f32[Frame.WORLD](0.0, 0.0, -1.0),
            )
        )
    )


def test_render_settings_and_tiny_render() raises:
    var settings = RenderSettings(4, 2, 2, UInt64(9))
    assert_equal(settings.image_width, 4)
    assert_equal(settings.image_height, 2)
    assert_equal(settings.rng_seed, 9)

    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var matte = surfaces.add_lambertian(Color(0.5))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    var world = World[4, 8](
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

    var result = render_depth_first[RENDER.PATH, 2](settings, camera, world)
    assert_equal(len(result.pixels), 8)
    assert_equal(result.timings.pixel_count, 8)
    assert_equal(result.timings.sample_count, 16)
    assert_equal(result.timings.max_depth, 2)
    assert_true(result.timings.total_ns >= result.timings.render_ns)
    for p in result.pixels:
        assert_true(p.x >= 0.0 and p.y >= 0.0 and p.z >= 0.0)

    var one_pixel_tiles = render_depth_first[RENDER.PATH, 2, 1, 1](
        settings, camera, world
    )
    for i in range(len(result.pixels)):
        assert_vec_equal(one_pixel_tiles.pixels[i], result.pixels[i])

    # Renderer packet length and both acceleration widths are independent.
    var packet_result = render_wavefront[RENDER.PATH, 2, 4, 16, False](
        settings, camera, world
    )
    assert_equal(len(packet_result.pixels), len(result.pixels))


def test_render_can_select_normal_algorithm_at_compile_time() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(11))
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var matte = surfaces.add_lambertian(Color(0.5))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    var world = World[](
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

    var result = render_depth_first[RENDER.NORMALS, 1](settings, camera, world)
    assert_equal(len(result.pixels), 1)
    assert_true(result.pixels[0].z >= result.pixels[0].x)


def test_render_can_select_ao_algorithm_at_compile_time() raises:
    var settings = RenderSettings(1, 1, 2, UInt64(12))
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var matte = surfaces.add_lambertian(Color(0.5))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    var world = World[](
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

    var result = render_depth_first[RENDER.AO, 1](settings, camera, world)
    assert_equal(len(result.pixels), 1)
    assert_true(
        result.pixels[0].x >= 0.0
        and result.pixels[0].y >= 0.0
        and result.pixels[0].z >= 0.0
    )


def test_wavefront_tiny_render() raises:
    var settings = RenderSettings(3, 2, 2, UInt64(9))
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()
    var matte = surfaces.add_lambertian(Color(0.5))
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    var world = World[](
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

    var result = render_wavefront[RENDER.PATH, 2](settings, camera, world)
    assert_equal(len(result.pixels), 6)
    assert_equal(result.timings.sample_count, 12)
    for p in result.pixels:
        assert_true(p.x >= 0.0 and p.y >= 0.0 and p.z >= 0.0)

    # Whole-pixel chunks preserve Philox stream ownership and accumulation
    # order. The small chunk also exercises a target not divisible by samples
    # per pixel.
    var chunked = render_wavefront[RENDER.PATH, 2, 1, 3, False](
        settings, camera, world
    )
    var parallel = render_wavefront[RENDER.PATH, 2, 1, 3, True](
        settings, camera, world
    )
    # Instantiate multiple packet widths and a six-path chunk so every width
    # exercises partial packets as well as the generic queue indexing.
    var width1 = render_wavefront[RENDER.PATH, 2, 1, 7, False](
        settings, camera, world
    )
    var packet4 = render_wavefront[RENDER.PATH, 2, 4, 7, False](
        settings, camera, world
    )
    var packet8 = render_wavefront[RENDER.PATH, 2, 8, 7, False](
        settings, camera, world
    )
    var packet16 = render_wavefront[RENDER.PATH, 2, 16, 7, False](
        settings, camera, world
    )
    assert_equal(len(chunked.pixels), len(result.pixels))
    for i in range(len(result.pixels)):
        assert_equal(chunked.pixels[i].x, result.pixels[i].x)
        assert_equal(chunked.pixels[i].y, result.pixels[i].y)
        assert_equal(chunked.pixels[i].z, result.pixels[i].z)
        assert_equal(parallel.pixels[i].x, result.pixels[i].x)
        assert_equal(parallel.pixels[i].y, result.pixels[i].y)
        assert_equal(parallel.pixels[i].z, result.pixels[i].z)
        assert_equal(width1.pixels[i].x, packet4.pixels[i].x)
        assert_equal(width1.pixels[i].y, packet4.pixels[i].y)
        assert_equal(width1.pixels[i].z, packet4.pixels[i].z)
        assert_equal(packet4.pixels[i].x, packet8.pixels[i].x)
        assert_equal(packet4.pixels[i].y, packet8.pixels[i].y)
        assert_equal(packet4.pixels[i].z, packet8.pixels[i].z)
        assert_equal(packet8.pixels[i].x, packet16.pixels[i].x)
        assert_equal(packet8.pixels[i].y, packet16.pixels[i].y)
        assert_equal(packet8.pixels[i].z, packet16.pixels[i].z)


def test_packet_widths_match_width1_for_mixed_bsdfs() raises:
    var settings = RenderSettings(9, 3, 3, UInt64(314159))
    var surfaces = SurfaceStore()
    var spheres = List[Sphere[Frame.WORLD]]()
    var sphere_surfaces = List[SurfaceId[1]]()
    var triangle_vertices = List[Point3f32[Frame.WORLD]]()
    var triangle_surfaces = List[SurfaceId[1]]()
    var triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
    var triangle_instances = List[Instance]()
    var triangle_instance_surfaces = List[SurfaceId[1]]()

    var ground = surfaces.add_lambertian(Color(0.5))
    var diffuse = surfaces.add_lambertian(Color(0.7, 0.2, 0.1))
    var rough_metal = surfaces.add_metal(Color(0.8, 0.75, 0.65), 0.35)
    var glass = surfaces.add_dielectric(1.5)
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, -100.6, -1.5),
        100.0,
        ground,
    )
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](-1.1, 0.0, -1.5),
        0.55,
        diffuse,
    )
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](0.0, 0.0, -1.5),
        0.55,
        rough_metal,
    )
    add_sphere(
        spheres,
        sphere_surfaces,
        Point3f32[Frame.WORLD](1.1, 0.0, -1.5),
        0.55,
        glass,
    )
    var world = World[](
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
        Point3f32[Frame.WORLD](0.0, 0.35, 2.5),
        Point3f32[Frame.WORLD](0.0, 0.0, -1.5),
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
        52.0,
    )

    var packet1 = render_wavefront[RENDER.PATH, 4, 1, 10, False](
        settings, camera, world
    )
    var packet4 = render_wavefront[RENDER.PATH, 4, 4, 10, False](
        settings, camera, world
    )
    var packet8 = render_wavefront[RENDER.PATH, 4, 8, 10, False](
        settings, camera, world
    )
    var packet16 = render_wavefront[RENDER.PATH, 4, 16, 10, False](
        settings, camera, world
    )
    for pixel_idx in range(len(packet1.pixels)):
        assert_vec_equal(packet4.pixels[pixel_idx], packet1.pixels[pixel_idx])
        assert_vec_equal(packet8.pixels[pixel_idx], packet1.pixels[pixel_idx])
        assert_vec_equal(packet16.pixels[pixel_idx], packet1.pixels[pixel_idx])


def test_direct_light_algorithms_render_cornell() raises:
    var settings = RenderSettings(8, 8, 2, UInt64(2026))
    var world = make_cornell_world()
    var camera = Camera.from_vfov(
        Point3f32[Frame.WORLD](0.0, 1.0, 3.2),
        Point3f32[Frame.WORLD](0.0, 1.0, -1.0),
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
        28.0,
        4.2,
    )
    var result = render_wavefront[RENDER.NEE, 4](settings, camera, world)
    assert_true(len(world.lights.records) > 0)
    assert_true(world.lights.total_weight > 0.0)
    for light in world.lights.records:
        assert_equal(light.surface.kind(), MAT.EMISSIVE)
        assert_true(light.weight > 0.0)
    var depth_first = render_depth_first[RENDER.NEE, 4](settings, camera, world)
    var mis = render_wavefront[RENDER.MIS, 4](settings, camera, world)
    var depth_first_mis = render_depth_first[RENDER.MIS, 4](
        settings, camera, world
    )
    var packet_nee1 = render_wavefront[RENDER.NEE, 4, 1, 14, False](
        settings, camera, world
    )
    var packet_nee4 = render_wavefront[RENDER.NEE, 4, 4, 14, False](
        settings, camera, world
    )
    var packet_nee8 = render_wavefront[RENDER.NEE, 4, 8, 14, False](
        settings, camera, world
    )
    var packet_nee16 = render_wavefront[RENDER.NEE, 4, 16, 14, False](
        settings, camera, world
    )
    var packet_mis1 = render_wavefront[RENDER.MIS, 4, 1, 14, False](
        settings, camera, world
    )
    var packet_mis4 = render_wavefront[RENDER.MIS, 4, 4, 14, False](
        settings, camera, world
    )
    var packet_mis8 = render_wavefront[RENDER.MIS, 4, 8, 14, False](
        settings, camera, world
    )
    var packet_mis16 = render_wavefront[RENDER.MIS, 4, 16, 14, False](
        settings, camera, world
    )
    var total = Float32(0.0)
    var depth_first_total = Float32(0.0)
    var mis_total = Float32(0.0)
    var depth_first_mis_total = Float32(0.0)
    for pixel_idx in range(len(result.pixels)):
        var pixel = result.pixels[pixel_idx]
        assert_true(pixel.x >= 0.0 and pixel.y >= 0.0 and pixel.z >= 0.0)
        total += pixel.x + pixel.y + pixel.z
        assert_vec_equal(packet_nee1.pixels[pixel_idx], pixel)
        assert_vec_equal(
            packet_nee4.pixels[pixel_idx], packet_nee1.pixels[pixel_idx]
        )
        assert_vec_equal(
            packet_nee8.pixels[pixel_idx], packet_nee1.pixels[pixel_idx]
        )
        assert_vec_equal(
            packet_nee16.pixels[pixel_idx], packet_nee1.pixels[pixel_idx]
        )
    for pixel in depth_first.pixels:
        assert_true(pixel.x >= 0.0 and pixel.y >= 0.0 and pixel.z >= 0.0)
        depth_first_total += pixel.x + pixel.y + pixel.z
    for pixel_idx in range(len(mis.pixels)):
        var pixel = mis.pixels[pixel_idx]
        assert_true(pixel.x >= 0.0 and pixel.y >= 0.0 and pixel.z >= 0.0)
        mis_total += pixel.x + pixel.y + pixel.z
        assert_vec_equal(packet_mis1.pixels[pixel_idx], pixel)
        assert_vec_equal(
            packet_mis4.pixels[pixel_idx], packet_mis1.pixels[pixel_idx]
        )
        assert_vec_equal(
            packet_mis8.pixels[pixel_idx], packet_mis1.pixels[pixel_idx]
        )
        assert_vec_equal(
            packet_mis16.pixels[pixel_idx], packet_mis1.pixels[pixel_idx]
        )
    for pixel in depth_first_mis.pixels:
        assert_true(pixel.x >= 0.0 and pixel.y >= 0.0 and pixel.z >= 0.0)
        depth_first_mis_total += pixel.x + pixel.y + pixel.z
    assert_true(total > 0.0)
    assert_true(depth_first_total > 0.0)
    assert_true(mis_total > 0.0)
    assert_true(depth_first_mis_total > 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
