from test.helpers import assert_vec_equal
from test.rt.helpers import assert_pixels_close

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
    assert_false,
)
from std.sys import num_logical_cores, simd_width_of
from std.math import ceildiv
from std.memory import bitcast
from std.math import pi, sqrt

from bajo.bvh.camera import Camera
from bajo.bvh.host_utils import compute_bounds
from bajo.core import (
    AABB,
    Affine3f32,
    Vec3,
    Vec3f32,
    dot,
    length,
    Point3f32,
    Point3,
    Ray,
    Rayf32,
)
from bajo.core.random import Rng, Sampler, _sobol_bits, _sz_bits
from bajo.rt import (
    Color,
    CpuSchedulerMode,
    Dielectric,
    Emissive,
    Environment,
    ImageTexture,
    Integrator,
    Instance,
    Lambertian,
    LightRecord,
    LightStore,
    Metal,
    PrimitiveId,
    RenderSettings,
    SceneBuilder,
    ShadingPoint,
    Sphere,
    SurfaceId,
    SurfaceHit,
    SurfaceStore,
    CpuScene,
    evaluate_bsdf,
    render_wavefront,
    sample_bsdf,
)
from examples.cornell_box import make_cornell_world
from bajo.rt.cpu import reflect, reflectance
from bajo.rt.common import (
    environment_light_pdf,
    environment_pdf_from_radiance,
    environment_radiance,
    equal_area_square_to_sphere,
    equal_area_sphere_to_square,
    path_stage_rng,
    path_stage_seed,
    russian_roulette,
)
from bajo.rt.lighting import (
    _draw_alias_column,
    _emissive_hit_light_pdf,
    _emissive_hit_weight_from_pdf,
    _resolve_alias_draw,
    _solid_angle_light_pdf,
)
from bajo.rt.geometry import (
    orient_surface_normal,
    triangle_area,
    triangle_is_valid,
)
from bajo.bvh.constants import PrimitiveKind
from bajo.rt.render_types import SamplingConfig
from bajo.rt.material_types import MaterialKind
from bajo.rt.wavefront_contract import wavefront_rng_roulette_stage
from bajo.rt.wavefront_queue import PathPacket, ShadePacket
from bajo.rt.material_storage import pack_coated_materials
from bajo.rt.film import FilmResponse
from bajo.rt.cpu.wavefront.packet import (
    _DirectLightBatch,
    _accumulate_direct_light_packet,
    _accumulate_environment_packet,
    _sample_bsdf_values,
)


def _front_point() -> ShadingPoint[1]:
    return ShadingPoint(
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        Vec3f32[.WORLD](0.0, 0.0, 1.0),
        True,
    )


def test_equal_area_sphere_to_square_is_width_generic() raises:
    var directions = Vec3[.float32, .WORLD, 4](
        SIMD[.float32, 4](1.0, 0.0, 0.0, -1.0),
        SIMD[.float32, 4](0.0, 1.0, 0.0, -1.0),
        SIMD[.float32, 4](0.0, 0.0, 1.0, -1.0),
    )
    var packet = equal_area_sphere_to_square(directions)
    for lane in range(4):
        var scalar = equal_area_sphere_to_square(
            Vec3f32[.WORLD](
                directions.x[lane], directions.y[lane], directions.z[lane]
            )
        )
        assert_almost_equal(packet.u[lane], scalar.u[0])
        assert_almost_equal(packet.v[lane], scalar.v[0])


def test_environment_distribution_tracks_texel_importance() raises:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    builder.add_sphere(Point3f32[.WORLD](0.0, 0.0, -1.0), 0.5, matte)
    var pixels: List[Float32] = [1.0, 0.0, 0.0, 0.0, 0.0, 3.0]
    var texture_index = builder.surfaces.add_image_texture(
        ImageTexture(2, 1, pixels^)
    )
    builder.set_environment(
        Environment.image(
            texture_index,
            Color(1.0),
            Affine3f32[.WORLD, .LOCAL].identity(),
            Affine3f32[.LOCAL, .WORLD].identity(),
        )
    )
    var scene = builder^.finish()
    assert_equal(len(scene.environment_alias_probabilities()), 2)
    assert_almost_equal(scene.environment_alias_probabilities()[0], 0.5)
    assert_equal(scene.environment_alias_indices()[0], UInt32(1))
    assert_almost_equal(scene.environment_weight(), 8.0 * pi / 3.0)
    assert_almost_equal(scene.total_light_weight(), scene.environment_weight())

    var red_direction = equal_area_square_to_sphere[.WORLD, 1](0.25, 0.5)
    var blue_direction = equal_area_square_to_sphere[.WORLD, 1](0.75, 0.5)
    var red_pdf = environment_light_pdf(
        scene.environment(),
        scene.surfaces(),
        scene.environment_weight(),
        scene.total_light_weight(),
        red_direction,
    )[0]
    var blue_pdf = environment_light_pdf(
        scene.environment(),
        scene.surfaces(),
        scene.environment_weight(),
        scene.total_light_weight(),
        blue_direction,
    )[0]
    assert_almost_equal(red_pdf, 1.0 / (8.0 * pi), atol=1.0e-6)
    assert_almost_equal(blue_pdf, 3.0 / (8.0 * pi), atol=1.0e-6)


def test_orient_surface_normal_is_width_generic() raises:
    var scalar = orient_surface_normal(
        Vec3f32[.WORLD](0.0, 0.0, -1.0),
        Vec3f32[.WORLD](0.0, 0.0, 1.0),
    )
    assert_true(scalar.front_face)
    assert_vec_equal(scalar.normal, Vec3f32[.WORLD](0.0, 0.0, 1.0))

    var directions = Vec3[.float32, .WORLD, 2](
        SIMD[.float32, 2](0.0),
        SIMD[.float32, 2](0.0),
        SIMD[.float32, 2](-1.0, 1.0),
    )
    var outward = Vec3[.float32, .WORLD, 2](0.0, 0.0, 1.0)
    var packet = orient_surface_normal(directions, outward)
    assert_true(packet.front_face[0])
    assert_false(packet.front_face[1])
    assert_almost_equal(packet.normal.z[0], 1.0)
    assert_almost_equal(packet.normal.z[1], -1.0)


def test_signed_sphere_acceleration_policy() raises:
    var sphere = Sphere[.WORLD](Point3f32[.WORLD](1.0, 2.0, 3.0), -2.5)
    var acceleration = sphere.for_acceleration()
    assert_almost_equal(sphere.physical_radius(), 2.5)
    assert_almost_equal(acceleration.radius, 2.5)
    assert_vec_equal(acceleration.center, sphere.center)
    assert_almost_equal(sphere.radius, -2.5)
    assert_true(sphere.bounds().is_valid()[0])


def test_triangle_geometry_policy() raises:
    var v0 = Point3f32[.WORLD](0.0, 0.0, 0.0)
    var v1 = Point3f32[.WORLD](1.0, 0.0, 0.0)
    var v2 = Point3f32[.WORLD](0.0, 1.0, 0.0)
    assert_true(triangle_is_valid(v0, v1, v2))
    assert_almost_equal(triangle_area(v0, v1, v2), 0.5)
    assert_false(triangle_is_valid(v0, v1, Point3f32[.WORLD](2.0, 0.0, 0.0)))


def test_geometry_structs_own_validity_and_orientation_queries() raises:
    var identity = Affine3f32[.LOCAL, .WORLD].identity()
    assert_true(identity.is_finite()[0])
    assert_false(identity.reverses_orientation()[0])

    var mirrored = Affine3f32[.LOCAL, .WORLD].from_scale(
        Vec3f32[.LOCAL](-1.0, 1.0, 1.0)
    )
    assert_true(mirrored.reverses_orientation()[0])

    var finite_bounds = AABB[.WORLD].point(Point3f32[.WORLD](1.0, 2.0, 3.0))
    assert_true(finite_bounds.is_finite()[0])
    assert_true(finite_bounds.is_valid()[0])
    assert_false(AABB[.WORLD].invalid().is_valid()[0])

    var nonfinite = identity.copy()
    nonfinite.m00 = bitcast[.float32](UInt32(0x7FC00000))
    assert_false(nonfinite.is_finite()[0])

    var points = Vec3[.float32, .WORLD, 2](
        SIMD[.float32, 2](1.0, nonfinite.m00[0]),
        SIMD[.float32, 2](2.0),
        SIMD[.float32, 2](3.0),
    )
    var finite_lanes = points.is_finite()
    assert_true(finite_lanes[0])
    assert_false(finite_lanes[1])


def test_materials_own_domain_validation() raises:
    Lambertian(Color(0.5)).validate()
    Metal(Color(0.5), 0.25).validate()
    Dielectric(1.5).validate()
    Emissive(Color(2.0)).validate()

    with assert_raises():
        Lambertian(Color(-0.1, 0.5, 0.5)).validate()
    with assert_raises():
        Metal(Color(0.5), 1.1).validate()
    with assert_raises():
        Dielectric(0.0).validate()
    with assert_raises():
        Emissive(Color(1.0, -0.1, 1.0)).validate()

    var surfaces = SurfaceStore()
    var light = surfaces.add_emissive(Color(2.0, 3.0, 4.0))
    assert_true(surfaces.contains(light))
    assert_vec_equal(
        surfaces.emitted_radiance(light, True), Color(2.0, 3.0, 4.0)
    )
    assert_vec_equal(surfaces.emitted_radiance(light, False), Color(0.0))


def test_reflect_and_reflectance() raises:
    var reflected = reflect(
        Vec3f32[.WORLD](1.0, -1.0, 0.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
    )
    assert_vec_equal(reflected, Vec3f32[.WORLD](1.0, 1.0, 0.0))

    assert_almost_equal(reflectance(1.0, 1.5), 0.04, atol=1e-5)


def test_surface_id_is_packed() raises:
    var surface = SurfaceId(MaterialKind(UInt32(2)), UInt32(123))
    assert_equal(surface.kind().value, UInt32(2))
    assert_equal(surface.index(), UInt32(123))
    assert_equal(surface.value, (UInt32(2) << UInt32(28)) | UInt32(123))
    var decoded = SurfaceId.from_raw(surface.value[0])
    assert_equal(decoded.kind(), surface.kind())
    assert_equal(decoded.index(), surface.index())
    assert_equal(
        SurfaceId.pack_raw(surface.kind(), surface.index()), surface.value[0]
    )


def test_surface_hit_is_width_generic() raises:
    var hits = SurfaceHit[4](SIMD[.float32, 4](100.0))
    hits.normal.x[2] = 1.0
    hits.normal.y[2] = 2.0
    hits.normal.z[2] = 3.0
    hits.surface.value[2] = SurfaceId(.METAL, UInt32(7)).value
    hits.t[2] = 4.5
    hits.front_face[2] = False
    hits.hit[2] = True

    var lane = hits.get(2)
    assert_vec_equal(lane.normal, Vec3f32[.WORLD](1.0, 2.0, 3.0))
    assert_equal(lane.surface.kind(), .METAL)
    assert_equal(lane.surface.index(), UInt32(7))
    assert_almost_equal(lane.t, 4.5)
    assert_false(lane.front_face)
    assert_true(lane.hit)

    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](0.0, 0.0, 1.0),
        Vec3f32[.WORLD](0.0, 0.0, 2.0),
    )
    var shading = ShadingPoint.from_hit(ray, lane)
    assert_vec_equal(shading.p, Point3f32[.WORLD](0.0, 0.0, 10.0))
    assert_vec_equal(shading.normal, lane.normal)
    assert_false(shading.front_face)


def test_light_alias_table_matches_power_distribution() raises:
    var lights = LightStore()
    var surface = SurfaceId(.EMISSIVE, UInt32(0))
    lights.append(
        LightRecord.sphere(
            PrimitiveId(PrimitiveKind.SPHERE, UInt32(0)),
            surface.copy(),
            1.0,
            Point3f32[.WORLD](0.0),
            1.0,
        )
    )
    lights.append(
        LightRecord.sphere(
            PrimitiveId(PrimitiveKind.SPHERE, UInt32(1)),
            surface.copy(),
            3.0,
            Point3f32[.WORLD](0.0),
            1.0,
        )
    )
    lights.append(
        LightRecord.sphere(
            PrimitiveId(PrimitiveKind.SPHERE, UInt32(2)),
            surface.copy(),
            6.0,
            Point3f32[.WORLD](0.0),
            1.0,
        )
    )
    lights.build_alias_table()

    assert_equal(len(lights.alias_probabilities), 3)
    assert_equal(len(lights.alias_indices), 3)
    var reconstructed = List[Float32](length=3, fill=0.0)
    for column in range(3):
        var probability = lights.alias_probabilities[column]
        var alias_idx = Int(lights.alias_indices[column])
        assert_true(probability >= 0.0 and probability <= 1.0)
        assert_true(alias_idx >= 0 and alias_idx < 3)
        reconstructed[column] += probability / 3.0
        reconstructed[alias_idx] += (1.0 - probability) / 3.0

    assert_almost_equal(reconstructed[0], 0.1, atol=1.0e-6)
    assert_almost_equal(reconstructed[1], 0.3, atol=1.0e-6)
    assert_almost_equal(reconstructed[2], 0.6, atol=1.0e-6)


def test_shared_light_selection_and_pdf_contract() raises:
    var draw = _draw_alias_column(0.45, 3)
    assert_equal(draw.column, 1)
    assert_almost_equal(draw.fraction, 0.35, atol=1.0e-6)
    assert_equal(_resolve_alias_draw(draw, 0.25, UInt32(2)), 2)
    assert_equal(_resolve_alias_draw(draw, 0.50, UInt32(2)), 1)
    assert_almost_equal(_solid_angle_light_pdf(4.0, 0.5, Color(3.0), 6.0), 4.0)


def test_scene_builder_finalizes_derived_lights() raises:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    var light = builder.add_emissive(Color(6.0, 4.0, 2.0))
    builder.add_sphere(Point3f32[.WORLD](0.0, 0.0, -1.0), 0.5, matte)
    builder.add_sphere(Point3f32[.WORLD](0.0, 2.0, -1.0), 0.25, light)
    builder.add_quad(
        Point3f32[.WORLD](-1.0, -1.0, -2.0),
        Point3f32[.WORLD](1.0, -1.0, -2.0),
        Point3f32[.WORLD](1.0, 1.0, -2.0),
        Point3f32[.WORLD](-1.0, 1.0, -2.0),
        matte,
    )

    var scene = builder^.finish()
    assert_equal(len(scene.triangle_vertices()), 6)
    assert_equal(len(scene.triangle_surfaces()), 2)
    assert_equal(len(scene.lights().records), 1)
    assert_equal(
        scene.lights().records[0].primitive.kind(), PrimitiveKind.SPHERE
    )
    assert_true(scene.lights().total_weight > 0.0)


def test_scene_builder_finalizes_emissive_triangle_instance_lights() raises:
    var builder = SceneBuilder()
    var light = builder.add_emissive(Color(4.0))
    var mesh = List[Point3f32[.LOCAL]]()
    mesh.append(Point3f32[.LOCAL](-1.0, -1.0, 0.0))
    mesh.append(Point3f32[.LOCAL](1.0, -1.0, 0.0))
    mesh.append(Point3f32[.LOCAL](0.0, 1.0, 0.0))
    var bounds = compute_bounds(mesh)
    _ = builder.add_triangle_mesh_instance(
        mesh,
        Affine3f32[.LOCAL, .WORLD].from_scale(Vec3f32[.LOCAL](-1.0, 1.0, 1.0)),
        bounds,
        light,
    )

    var scene = builder^.finish()
    assert_equal(len(scene.lights().records), 1)
    ref record = scene.lights().records[0]
    assert_equal(record.primitive.kind(), PrimitiveKind.TRIANGLE_INSTANCE)
    assert_equal(record.primitive.index(), UInt32(0))
    assert_almost_equal(record.weight, 8.0)
    assert_vec_equal(record.p0, Point3f32[.WORLD](1.0, -1.0, 0.0))
    assert_vec_equal(record.p1, Point3f32[.WORLD](0.0, 1.0, 0.0))
    assert_vec_equal(record.p2, Point3f32[.WORLD](-1.0, -1.0, 0.0))


def test_scene_builder_rejects_invalid_material_domains() raises:
    var invalid_albedo = SceneBuilder()
    var albedo_surface = invalid_albedo.add_lambertian(Color(1.01, 0.5, 0.5))
    invalid_albedo.add_sphere(Point3f32[.WORLD](0.0), 1.0, albedo_surface)
    with assert_raises():
        _ = invalid_albedo^.finish()

    var invalid_fuzz = SceneBuilder()
    var metal_surface = invalid_fuzz.add_metal(Color(0.5), 1.01)
    invalid_fuzz.add_sphere(Point3f32[.WORLD](0.0), 1.0, metal_surface)
    with assert_raises():
        _ = invalid_fuzz^.finish()

    var invalid_ior = SceneBuilder()
    var glass_surface = invalid_ior.add_dielectric(0.0)
    invalid_ior.add_sphere(Point3f32[.WORLD](0.0), 1.0, glass_surface)
    with assert_raises():
        _ = invalid_ior^.finish()

    var invalid_emission = SceneBuilder()
    var light_surface = invalid_emission.add_emissive(Color(1.0, -0.1, 1.0))
    invalid_emission.add_sphere(Point3f32[.WORLD](0.0), 1.0, light_surface)
    with assert_raises():
        _ = invalid_emission^.finish()

    var nonfinite_material = SceneBuilder()
    var nan = bitcast[.float32](UInt32(0x7FC00000))
    var nan_surface = nonfinite_material.add_lambertian(Color(nan, 0.5, 0.5))
    nonfinite_material.add_sphere(Point3f32[.WORLD](0.0), 1.0, nan_surface)
    with assert_raises():
        _ = nonfinite_material^.finish()

    var overflowed_light = SceneBuilder()
    var f32_max = bitcast[.float32](UInt32(0x7F7FFFFF))
    var overflowed_surface = overflowed_light.add_emissive(Color(f32_max))
    overflowed_light.add_sphere(Point3f32[.WORLD](0.0), 1.0, overflowed_surface)
    with assert_raises():
        _ = overflowed_light^.finish()


def test_scene_builder_rejects_invalid_geometry() raises:
    var zero_radius = SceneBuilder()
    var zero_radius_surface = zero_radius.add_lambertian(Color(0.5))
    zero_radius.add_sphere(Point3f32[.WORLD](0.0), 0.0, zero_radius_surface)
    with assert_raises():
        _ = zero_radius^.finish()

    var nonfinite_sphere = SceneBuilder()
    var nan = bitcast[.float32](UInt32(0x7FC00000))
    var sphere_surface = nonfinite_sphere.add_lambertian(Color(0.5))
    nonfinite_sphere.add_sphere(
        Point3f32[.WORLD](nan, 0.0, 0.0), 1.0, sphere_surface
    )
    with assert_raises():
        _ = nonfinite_sphere^.finish()

    var degenerate_triangle = SceneBuilder()
    var triangle_surface = degenerate_triangle.add_lambertian(Color(0.5))
    degenerate_triangle.add_triangle(
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Point3f32[.WORLD](1.0, 0.0, 0.0),
        Point3f32[.WORLD](2.0, 0.0, 0.0),
        triangle_surface,
    )
    with assert_raises():
        _ = degenerate_triangle^.finish()

    var degenerate_mesh = SceneBuilder()
    var mesh_surface = degenerate_mesh.add_lambertian(Color(0.5))
    var vertices = List[Point3f32[.LOCAL]]()
    vertices.append(Point3f32[.LOCAL](0.0, 0.0, 0.0))
    vertices.append(Point3f32[.LOCAL](1.0, 0.0, 0.0))
    vertices.append(Point3f32[.LOCAL](2.0, 0.0, 0.0))
    _ = degenerate_mesh.add_triangle_mesh_instance(
        vertices,
        Affine3f32[.LOCAL, .WORLD].identity(),
        compute_bounds(vertices),
        mesh_surface,
    )
    with assert_raises():
        _ = degenerate_mesh^.finish()


def test_scene_builder_validates_and_rebuilds_instance_derivatives() raises:
    var vertices = List[Point3f32[.LOCAL]]()
    vertices.append(Point3f32[.LOCAL](-1.0, -2.0, 0.0))
    vertices.append(Point3f32[.LOCAL](3.0, -2.0, 0.0))
    vertices.append(Point3f32[.LOCAL](0.0, 4.0, 0.0))

    var stale_bounds = SceneBuilder()
    var stale_surface = stale_bounds.add_lambertian(Color(0.5))
    _ = stale_bounds.add_triangle_mesh_instance(
        vertices,
        Affine3f32[.LOCAL, .WORLD].identity(),
        AABB[.LOCAL].invalid(),
        stale_surface,
    )
    var scene = stale_bounds^.finish()
    ref bounds = scene.triangle_instances()[0].bounds
    assert_almost_equal(bounds._min.x[0], -1.0)
    assert_almost_equal(bounds._min.y[0], -2.0)
    assert_almost_equal(bounds._min.z[0], 0.0)
    assert_almost_equal(bounds._max.x[0], 3.0)
    assert_almost_equal(bounds._max.y[0], 4.0)
    assert_almost_equal(bounds._max.z[0], 0.0)

    var singular_transform = SceneBuilder()
    var singular_surface = singular_transform.add_lambertian(Color(0.5))
    var singular = Affine3f32[.LOCAL, .WORLD].from_scale(
        Vec3f32[.LOCAL](1.0, 0.0, 1.0)
    )
    _ = singular_transform.add_triangle_mesh_instance(
        vertices, singular, compute_bounds(vertices), singular_surface
    )
    with assert_raises():
        _ = singular_transform^.finish()

    var overflowed_bounds = SceneBuilder()
    var overflow_surface = overflowed_bounds.add_lambertian(Color(0.5))
    var large_vertices = List[Point3f32[.LOCAL]]()
    large_vertices.append(Point3f32[.LOCAL](1.0, 0.0, 0.0))
    large_vertices.append(Point3f32[.LOCAL](2.0, 0.0, 0.0))
    large_vertices.append(Point3f32[.LOCAL](1.0, 1.0, 0.0))
    var f32_max = bitcast[.float32](UInt32(0x7F7FFFFF))
    var enormous = Affine3f32[.LOCAL, .WORLD].from_scale(
        Vec3f32[.LOCAL](f32_max, 1.0, 1.0)
    )
    _ = overflowed_bounds.add_triangle_mesh_instance(
        large_vertices,
        enormous,
        compute_bounds(large_vertices),
        overflow_surface,
    )
    with assert_raises():
        _ = overflowed_bounds^.finish()


def test_wavefront_philox_streams_are_deterministic_and_separate() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(91))
    var first = path_stage_rng(settings.rng_seed, UInt32(7), UInt32(3))
    var replay = path_stage_rng(settings.rng_seed, UInt32(7), UInt32(3))
    var next_stage = path_stage_rng(settings.rng_seed, UInt32(7), UInt32(4))
    var roulette = path_stage_rng(
        settings.rng_seed,
        UInt32(7),
        wavefront_rng_roulette_stage(UInt32(2)),
    )

    assert_equal(first.f32(), replay.f32())
    assert_true(first.f32() != next_stage.f32())
    assert_true(replay.f32() != roulette.f32())


def test_sample_sequences_are_deterministic_and_batch_invariant() raises:
    for sampler_idx in range(6):
        var sampler = Sampler(UInt32(sampler_idx))
        var full = RenderSettings(3, 1, 4, UInt64(123), 2, sampler, 0, 4)
        var first = RenderSettings(3, 1, 2, UInt64(123), 2, sampler, 0, 4)
        var second = RenderSettings(3, 1, 2, UInt64(123), 2, sampler, 2, 4)
        for pixel in range(3):
            for sample in range(4):
                var full_path = UInt32(pixel * 4 + sample)
                var batch_path = UInt32(pixel * 2 + sample % 2)
                var batch_config = SamplingConfig.from_settings(first)
                if sample >= 2:
                    batch_config = SamplingConfig.from_settings(second)
                var expected = path_stage_rng(
                    SamplingConfig.from_settings(full), full_path, UInt32(7)
                )
                var actual = path_stage_rng(batch_config, batch_path, UInt32(7))
                assert_equal(
                    path_stage_seed(batch_config, batch_path, UInt32(7)),
                    path_stage_seed(
                        SamplingConfig.from_settings(full), full_path, UInt32(7)
                    ),
                )
                for _ in range(4):
                    assert_equal(actual.f32(), expected.f32())


def test_sobol_direction_numbers_match_known_prefix() raises:
    assert_equal(_sobol_bits(UInt32(0), 0), UInt32(0x00000000))
    assert_equal(_sobol_bits(UInt32(1), 0), UInt32(0x80000000))
    assert_equal(_sobol_bits(UInt32(2), 0), UInt32(0x40000000))
    assert_equal(_sobol_bits(UInt32(3), 0), UInt32(0xC0000000))
    assert_equal(_sobol_bits(UInt32(1), 1), UInt32(0x80000000))
    assert_equal(_sobol_bits(UInt32(2), 1), UInt32(0xC0000000))


def test_sz_prefix_is_a_four_dimensional_base4_net() raises:
    # The first 4^4 points contain every combination of the leading base-4
    # digit in the four dimensions exactly once.
    for first in range(256):
        var first_cell = UInt32(0)
        for dimension in range(4):
            first_cell |= (
                _sz_bits(UInt32(first), dimension) >> UInt32(30)
            ) << UInt32(2 * dimension)
        for second in range(first + 1, 256):
            var second_cell = UInt32(0)
            for dimension in range(4):
                second_cell |= (
                    _sz_bits(UInt32(second), dimension) >> UInt32(30)
                ) << UInt32(2 * dimension)
            assert_true(first_cell != second_cell)


def test_russian_roulette_is_deterministic_and_unbiased() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(8128))
    var throughput = Color(0.25, 0.2, 0.1)
    var early = russian_roulette(
        settings.rng_seed, UInt32(17), UInt32(4), throughput
    )
    assert_true(early.survived)
    assert_vec_equal(early.throughput, throughput)

    var first = russian_roulette(
        settings.rng_seed, UInt32(17), UInt32(5), throughput
    )
    var replay = russian_roulette(
        settings.rng_seed, UInt32(17), UInt32(5), throughput
    )
    assert_equal(first.survived, replay.survived)
    assert_vec_equal(first.throughput, replay.throughput)

    comptime TRIALS = 20000
    var weighted_sum = Float64(0.0)
    for path_idx in range(TRIALS):
        var result = russian_roulette(
            settings.rng_seed, UInt32(path_idx), UInt32(5), throughput
        )
        if result.survived:
            weighted_sum += Float64(result.throughput.x)
    assert_almost_equal(
        weighted_sum / Float64(TRIALS),
        Float64(throughput.x),
        atol=0.01,
    )


def test_emissive_hit_pdf_and_integrator_weight_are_shared() raises:
    var light_pdf = _emissive_hit_light_pdf(
        Vec3f32[.WORLD](0.0, 0.0, -2.0),
        3.0,
        Vec3f32[.WORLD](0.0, 0.0, 1.0),
        Color(2.0),
        4.0,
    )
    assert_almost_equal(light_pdf, 18.0)
    assert_almost_equal(
        _emissive_hit_weight_from_pdf[.PATH](UInt32(1), False, 6.0, light_pdf),
        1.0,
    )
    assert_almost_equal(
        _emissive_hit_weight_from_pdf[.NEE](UInt32(1), False, 6.0, light_pdf),
        0.0,
    )
    assert_almost_equal(
        _emissive_hit_weight_from_pdf[.MIS](UInt32(1), False, 6.0, 8.0),
        0.36,
        atol=1.0e-6,
    )
    assert_almost_equal(
        _emissive_hit_weight_from_pdf[.MIS](UInt32(0), False, 6.0, 8.0),
        1.0,
    )
    assert_almost_equal(
        _emissive_hit_weight_from_pdf[.MIS](UInt32(1), True, 6.0, 8.0),
        1.0,
    )


def test_lambertian_scatter_is_explicit() raises:
    var rng = Rng(seed=1, id=0)
    var material = Lambertian(Color(0.2, 0.4, 0.8))
    var surfaces = SurfaceStore()
    var surface = surfaces.add_lambertian(material.albedo)
    var point = _front_point()
    var incoming = Rayf32[.WORLD](
        Point3f32[.WORLD](0.0), Vec3f32[.WORLD](0.0, 0.0, -1.0)
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
    assert_almost_equal(evaluated.value.x, material.albedo.x / pi)


def test_metal_scatter_can_absorb() raises:
    var rng = Rng(seed=2, id=0)
    var material = Metal(Color(0.8, 0.7, 0.6), 0.0)
    var surfaces = SurfaceStore()
    var surface = surfaces.add_metal(material.albedo, material.fuzz)
    var point = _front_point()
    var incoming = Rayf32[.WORLD](
        Point3f32[.WORLD](0.0), Vec3f32[.WORLD](0.0, 0.0, -1.0)
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
    var incoming = Rayf32[.WORLD](
        Point3f32[.WORLD](0.0), Vec3f32[.WORLD](0.0, 0.0, -1.0)
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
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    var light = builder.add_emissive(Color(4.0))
    builder.add_sphere(
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    builder.add_sphere(
        Point3f32[.WORLD](10.0, 0.0, -1.0),
        0.25,
        light,
    )
    var scene = builder^.finish()
    var world = CpuScene[4, 8](scene^)
    assert_equal(len(world.scene_data().lights().records), 1)
    assert_equal(
        world.scene_data().lights().records[0].primitive.kind(),
        PrimitiveKind.SPHERE,
    )
    assert_equal(
        world.scene_data().lights().records[0].surface.value, light.value
    )
    assert_true(world.scene_data().lights().total_weight > 0.0)

    var trace_result = world.trace(
        Rayf32[.WORLD](
            Point3f32[.WORLD](0.0),
            Vec3f32[.WORLD](0.0, 0.0, -1.0),
        )
    )
    match trace_result:
        case .Some(ref hit):
            assert_equal(hit.primitive.kind(), PrimitiveKind.SPHERE)
            assert_equal(hit.primitive.index(), UInt32(0))
            assert_equal(hit.surface.kind(), matte.kind())
            assert_equal(hit.surface.index(), matte.index())
            assert_true(hit.front_face)
            assert_vec_equal(hit.normal, Vec3f32[.WORLD](0.0, 0.0, 1.0))
            assert_almost_equal(hit.t, 0.5)
            var compact = world.trace_surface(
                Rayf32[.WORLD](
                    Point3f32[.WORLD](0.0),
                    Vec3f32[.WORLD](0.0, 0.0, -1.0),
                )
            )
            assert_true(compact.hit)
            assert_equal(compact.surface.value, hit.surface.value)
            assert_vec_equal(compact.normal, hit.normal)
            assert_almost_equal(compact.t, hit.t)
        case .None:
            raise Error("expected ray hit")

    var miss_result = world.trace(
        Rayf32[.WORLD](
            Point3f32[.WORLD](0.0),
            Vec3f32[.WORLD](0.0, 1.0, 0.0),
        )
    )
    match miss_result:
        case .Some(_):
            raise Error("expected ray miss")
        case .None:
            pass


def test_world_preserves_signed_radius_normals() raises:
    var builder = SceneBuilder()
    var glass = builder.add_dielectric(1.5)
    builder.add_sphere(
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        -0.5,
        glass,
    )
    var scene = builder^.finish()
    var world = CpuScene[16, 16](scene^)

    var hit = (
        world.trace(
            Rayf32[.WORLD](
                Point3f32[.WORLD](0.0),
                Vec3f32[.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_false(hit.front_face)
    assert_vec_equal(hit.normal, Vec3f32[.WORLD](0.0, 0.0, 1.0))
    assert_almost_equal(hit.t, 0.5)


def test_world_hits_triangle() raises:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.25, 0.5, 0.75))
    builder.add_triangle(
        Point3f32[.WORLD](-1.0, -1.0, -2.0),
        Point3f32[.WORLD](1.0, -1.0, -2.0),
        Point3f32[.WORLD](0.0, 1.0, -2.0),
        matte,
    )
    var scene = builder^.finish()
    var world = CpuScene[16, 16](scene^)

    var hit = (
        world.trace(
            Rayf32[.WORLD](
                Point3f32[.WORLD](0.0),
                Vec3f32[.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit.primitive.kind(), PrimitiveKind.TRIANGLE)
    assert_equal(hit.primitive.index(), UInt32(0))
    assert_equal(hit.surface.value, matte.value)
    assert_true(hit.front_face)
    assert_vec_equal(hit.normal, Vec3f32[.WORLD](0.0, 0.0, 1.0))
    assert_almost_equal(hit.t, 2.0)

    var back_hit = (
        world.trace(
            Rayf32[.WORLD](
                Point3f32[.WORLD](0.0, 0.0, -4.0),
                Vec3f32[.WORLD](0.0, 0.0, 1.0),
            )
        )
        .value()
        .copy()
    )
    assert_false(back_hit.front_face)
    assert_vec_equal(back_hit.normal, Vec3f32[.WORLD](0.0, 0.0, -1.0))
    assert_almost_equal(back_hit.t, 2.0)
    var compact = world.trace_surface(
        Rayf32[.WORLD](
            Point3f32[.WORLD](0.0),
            Vec3f32[.WORLD](0.0, 0.0, -1.0),
        )
    )
    assert_true(compact.hit)
    assert_equal(compact.surface.value, hit.surface.value)
    assert_vec_equal(compact.normal, hit.normal)
    assert_almost_equal(compact.t, hit.t)


def test_world_picks_closest_sphere_or_triangle() raises:
    var builder = SceneBuilder()
    var sphere_surface = builder.add_lambertian(Color(0.5))
    var tri_surface = builder.add_metal(Color(0.9), 0.0)

    builder.add_sphere(
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        0.25,
        sphere_surface,
    )
    builder.add_triangle(
        Point3f32[.WORLD](-1.0, -1.0, -2.0),
        Point3f32[.WORLD](1.0, -1.0, -2.0),
        Point3f32[.WORLD](0.0, 1.0, -2.0),
        tri_surface,
    )
    var scene = builder^.finish()
    var world = CpuScene[16, 16](scene^)

    var hit = (
        world.trace(
            Rayf32[.WORLD](
                Point3f32[.WORLD](0.0),
                Vec3f32[.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit.primitive.kind(), PrimitiveKind.SPHERE)
    assert_equal(hit.surface.value, sphere_surface.value)
    assert_almost_equal(hit.t, 0.75)


def test_add_triangle_mesh_assigns_surface_per_triangle() raises:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.3, 0.4, 0.5))
    var mesh = List[Point3f32[.WORLD]]()
    mesh.append(Point3f32[.WORLD](-1.0, -1.0, -2.0))
    mesh.append(Point3f32[.WORLD](1.0, -1.0, -2.0))
    mesh.append(Point3f32[.WORLD](0.0, 1.0, -2.0))
    mesh.append(Point3f32[.WORLD](-1.0, -1.0, -3.0))
    mesh.append(Point3f32[.WORLD](1.0, -1.0, -3.0))
    mesh.append(Point3f32[.WORLD](0.0, 1.0, -3.0))

    builder.add_triangle_mesh(mesh, matte)
    var scene = builder^.finish()
    var world = CpuScene[16, 16](scene^)
    assert_equal(len(world.scene_data().triangle_vertices()), 6)
    assert_equal(len(world.scene_data().triangle_surfaces()), 2)

    var hit = (
        world.trace(
            Rayf32[.WORLD](
                Point3f32[.WORLD](0.0),
                Vec3f32[.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit.primitive.kind(), PrimitiveKind.TRIANGLE)
    assert_equal(hit.surface.value, matte.value)
    assert_almost_equal(hit.t, 2.0)


def test_triangle_mesh_instances_use_instance_surfaces() raises:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.2, 0.4, 0.8))
    var metal = builder.add_metal(Color(0.9, 0.8, 0.7), 0.0)

    var mesh = List[Point3f32[.LOCAL]]()
    mesh.append(Point3f32[.LOCAL](-0.5, -0.5, -2.0))
    mesh.append(Point3f32[.LOCAL](0.5, -0.5, -2.0))
    mesh.append(Point3f32[.LOCAL](0.0, 0.5, -2.0))
    var mesh_bounds = compute_bounds(mesh)

    var transform = Affine3f32[.LOCAL, .WORLD].identity()
    var mesh_idx = builder.add_triangle_mesh_instance(
        mesh,
        transform,
        mesh_bounds,
        matte,
    )

    var t = Affine3f32[.LOCAL, .WORLD].from_translation(
        Vec3f32[.WORLD](1.5, 0.0, 0.0)
    )
    builder.add_triangle_instance(
        mesh_idx,
        t,
        mesh_bounds,
        metal,
    )

    var scene = builder^.finish()
    var world = CpuScene[4, 8](scene^)

    var hit0 = (
        world.trace(
            Rayf32[.WORLD](
                Point3f32[.WORLD](0.0),
                Vec3f32[.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit0.primitive.kind(), PrimitiveKind.TRIANGLE_INSTANCE)
    assert_equal(hit0.primitive.index(), UInt32(0))
    assert_equal(hit0.surface.value, matte.value)
    assert_almost_equal(hit0.t, 2.0)

    var hit1 = (
        world.trace(
            Rayf32[.WORLD](
                Point3f32[.WORLD](1.5, 0.0, 0.0),
                Vec3f32[.WORLD](0.0, 0.0, -1.0),
            )
        )
        .value()
        .copy()
    )
    assert_equal(hit1.primitive.kind(), PrimitiveKind.TRIANGLE_INSTANCE)
    assert_equal(hit1.primitive.index(), UInt32(1))
    assert_equal(hit1.surface.value, metal.value)
    assert_almost_equal(hit1.t, 2.0)
    var compact = world.trace_surface(
        Rayf32[.WORLD](
            Point3f32[.WORLD](1.5, 0.0, 0.0),
            Vec3f32[.WORLD](0.0, 0.0, -1.0),
        )
    )
    assert_true(compact.hit)
    assert_equal(compact.surface.value, hit1.surface.value)
    assert_vec_equal(compact.normal, hit1.normal)
    assert_almost_equal(compact.t, hit1.t)


def test_world_occluded_covers_all_geometry_and_ray_interval() raises:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))

    builder.add_sphere(
        Point3f32[.WORLD](-2.0, 0.0, -2.0),
        0.5,
        matte,
    )
    builder.add_triangle(
        Point3f32[.WORLD](-0.75, -0.75, -3.0),
        Point3f32[.WORLD](0.75, -0.75, -3.0),
        Point3f32[.WORLD](0.0, 0.75, -3.0),
        matte,
    )

    var mesh = List[Point3f32[.LOCAL]]()
    mesh.append(Point3f32[.LOCAL](-0.5, -0.5, -4.0))
    mesh.append(Point3f32[.LOCAL](0.5, -0.5, -4.0))
    mesh.append(Point3f32[.LOCAL](0.0, 0.5, -4.0))
    var mesh_bounds = compute_bounds(mesh)
    var transform = Affine3f32[.LOCAL, .WORLD].from_translation(
        Vec3f32[.WORLD](2.0, 0.0, 0.0)
    )
    _ = builder.add_triangle_mesh_instance(
        mesh,
        transform,
        mesh_bounds,
        matte,
    )

    var scene = builder^.finish()
    var world = CpuScene[16, 16](scene^)

    var sphere_ray = Rayf32[.WORLD](
        Point3f32[.WORLD](-2.0, 0.0, 0.0),
        Vec3f32[.WORLD](0.0, 0.0, -1.0),
        0.001,
        3.0,
    )
    assert_true(world.occluded(sphere_ray))
    assert_false(
        world.occluded(
            Rayf32[.WORLD](
                Point3f32[.WORLD](-2.0, 0.0, 0.0),
                Vec3f32[.WORLD](0.0, 0.0, -1.0),
                0.001,
                1.0,
            )
        )
    )

    var triangle_ray = Rayf32[.WORLD](
        Point3f32[.WORLD](0.0),
        Vec3f32[.WORLD](0.0, 0.0, -1.0),
        0.001,
        4.0,
    )
    assert_true(world.occluded(triangle_ray))
    assert_false(
        world.occluded(
            Rayf32[.WORLD](
                Point3f32[.WORLD](0.0),
                Vec3f32[.WORLD](0.0, 0.0, -1.0),
                0.001,
                2.0,
            )
        )
    )

    var instance_ray = Rayf32[.WORLD](
        Point3f32[.WORLD](2.0, 0.0, 0.0),
        Vec3f32[.WORLD](0.0, 0.0, -1.0),
        0.001,
        5.0,
    )
    assert_true(world.occluded(instance_ray))
    assert_false(
        world.occluded(
            Rayf32[.WORLD](
                Point3f32[.WORLD](2.0, 0.0, 0.0),
                Vec3f32[.WORLD](0.0, 0.0, -1.0),
                0.001,
                3.0,
            )
        )
    )

    assert_false(
        world.occluded(
            Rayf32[.WORLD](
                Point3f32[.WORLD](5.0, 0.0, 0.0),
                Vec3f32[.WORLD](0.0, 0.0, -1.0),
            )
        )
    )


def test_render_settings_validation() raises:
    var settings = RenderSettings(4, 2, 2, UInt64(9), 2)
    settings.validate()
    assert_equal(settings.image_width, 4)
    assert_equal(settings.image_height, 2)
    assert_equal(settings.rng_seed, 9)

    var invalid_settings = settings.copy()
    invalid_settings.sample_offset = 2
    invalid_settings.sample_sequence_length = 3
    with assert_raises():
        invalid_settings.validate()


def test_render_can_select_normal_integrator_at_compile_time() raises:
    var settings = RenderSettings(1, 1, 1, UInt64(11))
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    builder.add_sphere(
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    var scene = builder^.finish()
    var world = CpuScene[16, 16](scene^)
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        90.0,
    )

    var result = render_wavefront[.NORMALS](settings, camera, world)
    assert_equal(len(result.pixels), 1)
    assert_true(result.pixels[0].z >= result.pixels[0].x)


def test_render_can_select_ao_integrator_at_compile_time() raises:
    var settings = RenderSettings(1, 1, 2, UInt64(12))
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    builder.add_sphere(
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    var scene = builder^.finish()
    var world = CpuScene[16, 16](scene^)
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        90.0,
    )

    var result = render_wavefront[.AO](settings, camera, world)
    assert_equal(len(result.pixels), 1)
    assert_true(
        result.pixels[0].x >= 0.0
        and result.pixels[0].y >= 0.0
        and result.pixels[0].z >= 0.0
    )


def test_wavefront_tiny_render() raises:
    var settings = RenderSettings(3, 2, 2, UInt64(9), 2)
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    builder.add_sphere(
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    var scene = builder^.finish()
    var world = CpuScene[4, 8](scene^)
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        90.0,
    )

    var result = render_wavefront[.PATH](settings, camera, world)
    assert_equal(len(result.pixels), 6)
    assert_equal(result.timings.pixel_count, 6)
    assert_equal(result.timings.sample_count, 12)
    assert_equal(result.timings.max_depth, 2)
    assert_true(result.timings.total_ns >= result.timings.render_ns)
    for p in result.pixels:
        assert_true(p.x >= 0.0 and p.y >= 0.0 and p.z >= 0.0)

    # Whole-pixel chunks preserve Philox stream ownership and accumulation
    # order. The small chunk also exercises a target not divisible by samples
    # per pixel.
    var chunked = render_wavefront[.PATH, 1, 3, False](settings, camera, world)
    var parallel = render_wavefront[.PATH, 1, 3, True](settings, camera, world)
    assert_pixels_close(chunked.pixels, result.pixels)
    assert_pixels_close(parallel.pixels, result.pixels)


def test_packet_widths_match_width1_for_mixed_bsdfs() raises:
    var settings = RenderSettings(9, 3, 3, UInt64(314159))
    var builder = SceneBuilder()

    var ground = builder.add_lambertian(Color(0.5))
    var diffuse = builder.add_lambertian(Color(0.7, 0.2, 0.1))
    var rough_metal = builder.add_metal(Color(0.8, 0.75, 0.65), 0.35)
    var glass = builder.add_dielectric(1.5)
    builder.add_sphere(
        Point3f32[.WORLD](0.0, -100.6, -1.5),
        100.0,
        ground,
    )
    builder.add_sphere(
        Point3f32[.WORLD](-1.1, 0.0, -1.5),
        0.55,
        diffuse,
    )
    builder.add_sphere(
        Point3f32[.WORLD](0.0, 0.0, -1.5),
        0.55,
        rough_metal,
    )
    builder.add_sphere(
        Point3f32[.WORLD](1.1, 0.0, -1.5),
        0.55,
        glass,
    )
    var scene = builder^.finish()
    var world = CpuScene[16, 16](scene^)
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.35, 2.5),
        Point3f32[.WORLD](0.0, 0.0, -1.5),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        52.0,
    )

    var packet1 = render_wavefront[.PATH, 1, 10, False](settings, camera, world)
    comptime for width in [4, 16]:
        var packet = render_wavefront[.PATH, width, 10, False](
            settings, camera, world
        )
        assert_pixels_close(packet.pixels, packet1.pixels)


def test_direct_light_integrators_render_cornell() raises:
    var settings = RenderSettings(8, 8, 2, UInt64(2026))
    var world = make_cornell_world()
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0.0, 1.0, 3.2),
        Point3f32[.WORLD](0.0, 1.0, -1.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        28.0,
        4.2,
    )
    assert_true(len(world.scene_data().lights().records) > 0)
    assert_true(world.scene_data().lights().total_weight > 0.0)
    for light in world.scene_data().lights().records:
        assert_equal(light.surface.kind(), .EMISSIVE)
        assert_true(light.weight > 0.0)
    comptime for integrator in [Integrator.NEE, Integrator.MIS]:
        var result = render_wavefront[integrator](settings, camera, world)
        var packet1 = render_wavefront[integrator, 1, 14, False](
            settings, camera, world
        )
        var total = Float32(0.0)
        for i, pixel in enumerate(result.pixels):
            assert_true(pixel.x >= 0.0 and pixel.y >= 0.0 and pixel.z >= 0.0)
            total += pixel.x + pixel.y + pixel.z
            assert_vec_equal(packet1.pixels[i], pixel)
        assert_true(total > 0.0)
        comptime for width in [4, 16]:
            var packet = render_wavefront[integrator, width, 14, False](
                settings, camera, world
            )
            assert_pixels_close(packet.pixels, packet1.pixels)


def test_environment_pdf_reuses_rotated_packet_radiance() raises:
    var surfaces = SurfaceStore()
    var pixels: List[Float32] = [1.0, 0.0, 0.0, 0.0, 0.0, 3.0]
    var texture = surfaces.add_image_texture(ImageTexture(2, 1, pixels^))
    var rotation = Affine3f32[.WORLD, .LOCAL].identity()
    rotation.m00 = -1.0
    rotation.m22 = -1.0
    var inverse = Affine3f32[.LOCAL, .WORLD].identity()
    inverse.m00 = -1.0
    inverse.m22 = -1.0
    var environment = Environment.image(
        texture, Color(2.0, 1.0, 0.5), rotation, inverse
    )
    var directions = Vec3[.float32, .WORLD, 4](
        SIMD[.float32, 4](1.0, -1.0, 0.0, 0.0),
        SIMD[.float32, 4](0.0, 0.0, 1.0, -1.0),
        SIMD[.float32, 4](0.0),
    )
    var emission = environment_radiance(environment, surfaces, directions)
    var pdf = environment_pdf_from_radiance(environment, 2.0, 5.0, emission)
    var original = environment_light_pdf(
        environment, surfaces, 2.0, 5.0, directions
    )
    for lane in range(4):
        assert_equal(pdf[lane], original[lane])
        assert_almost_equal(
            pdf[lane],
            max(
                (emission.x[lane] + emission.y[lane] + emission.z[lane]) / 3.0,
                0.0,
            )
            / 5.0,
        )
    assert_equal(
        environment_pdf_from_radiance(environment, 0.0, 5.0, emission),
        SIMD[.float32, 4](0.0),
    )
    assert_equal(
        environment_pdf_from_radiance(environment, 2.0, 0.0, emission),
        SIMD[.float32, 4](0.0),
    )
    var uniform = environment_pdf_from_radiance(
        Environment.uniform(Color(2.0)), 2.0, 5.0, emission
    )
    for lane in range(4):
        assert_almost_equal(uniform[lane], 2.0 / (20.0 * pi))


def test_instance_packet_matches_scalar_normals_and_uvs() raises:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    var vertices: List[Point3f32[.LOCAL]] = [
        Point3f32[.LOCAL](0.0, 0.0, 0.0),
        Point3f32[.LOCAL](1.0, 0.0, 0.0),
        Point3f32[.LOCAL](0.0, 1.0, 0.0),
    ]
    var normals: List[Float32] = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    var texcoords: List[Float32] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    var transform = Affine3f32[.LOCAL, .WORLD].identity()
    transform.m00 = 2.0
    var bounds = compute_bounds(vertices)
    _ = builder.add_owned_triangle_mesh_instance(
        vertices^, normals^, texcoords^, transform, bounds, matte
    )
    var world = CpuScene[4, 4](builder^.finish())
    var rays = Ray[.float32, .WORLD, 4](
        Point3[.float32, .WORLD, 4](
            SIMD[.float32, 4](0.5, 3.0, 0.5, 0.5),
            SIMD[.float32, 4](0.25),
            SIMD[.float32, 4](1.0, 1.0, -1.0, 1.0),
        ),
        Vec3[.float32, .WORLD, 4](
            SIMD[.float32, 4](0.0),
            SIMD[.float32, 4](0.0),
            SIMD[.float32, 4](-1.0, -1.0, 1.0, -1.0),
        ),
        SIMD[.float32, 4](0.001),
        SIMD[.float32, 4](10.0),
    )
    var valid = SIMD[.bool, 4](True, True, True, False)
    var packet = world.trace_surface(rays, valid)
    assert_true(packet.hit[0])
    assert_false(packet.hit[1])
    assert_true(packet.hit[2])
    assert_false(packet.hit[3])
    for lane in range(3):
        var scalar_ray = Rayf32[.WORLD](
            Point3f32[.WORLD](rays.o.x[lane], rays.o.y[lane], rays.o.z[lane]),
            Vec3f32[.WORLD](rays.d.x[lane], rays.d.y[lane], rays.d.z[lane]),
            rays.t_min[lane],
            rays.t_max[lane],
        )
        var scalar = world.trace_surface(scalar_ray)
        var hit = packet.get(lane)
        assert_equal(hit.hit, scalar.hit)
        if not hit.hit:
            continue
        assert_vec_equal(hit.normal, scalar.normal)
        assert_equal(hit.surface.value, scalar.surface.value)
        assert_equal(hit.front_face, scalar.front_face)
        assert_equal(hit.t, scalar.t)
        assert_almost_equal(hit.uv_u[0], 0.25)
        assert_almost_equal(hit.uv_v[0], 0.25)
    assert_almost_equal(packet.normal.x[0], 1.0 / sqrt(Float32(69.0)))
    assert_almost_equal(packet.normal.y[0], 2.0 / sqrt(Float32(69.0)))
    assert_almost_equal(packet.normal.z[0], 8.0 / sqrt(Float32(69.0)))
    assert_true(packet.front_face[0])
    assert_false(packet.front_face[2])


def test_direct_light_packet_matches_scalar_for_mixed_materials() raises:
    var surfaces = SurfaceStore()
    var texture_pixels: List[Float32] = [0.25, 0.5, 0.75, 0.8, 0.6, 0.4]
    var texture = surfaces.add_image_texture(
        ImageTexture(2, 1, texture_pixels^)
    )
    var ids = List[SurfaceId[1]]()
    ids.append(surfaces.add_lambertian(Color(0.7, 0.5, 0.25), texture))
    ids.append(surfaces.add_metal(Color(0.8), 0.25, texture))
    ids.append(surfaces.add_coated_diffuse(Color(0.6), 0.2, 1.5, texture))
    ids.append(ids[2].copy())
    var paths = PathPacket[4]()
    paths.path_ids = SIMD[.uint32, 4](0, 1, 2, 3)
    paths.dz = -1.0
    paths.tx = 1.0
    paths.ty = 1.0
    paths.tz = 1.0
    var direct = _DirectLightBatch[4]()
    direct.point.uv_u = 0.2
    direct.point.uv_v = 0.4
    direct.point.normal.z = 1.0
    direct.sample.direction.z = 1.0
    direct.sample.emission = Vec3[.float32, .WORLD, 4](2.0)
    direct.sample.light_pdf = 0.4
    direct.sample.surface_cosine = 1.0
    direct.sample.valid = SIMD[.bool, 4](True, True, True, False)
    for lane in range(4):
        direct.surface_kinds[lane] = ids[lane].kind().value
        direct.surface_indices[lane] = ids[lane].index()
    var pixels = List[Color](length=4, fill=Color(0.0))
    var sampling = SamplingConfig.from_settings(
        RenderSettings(4, 1, 1, UInt64(2026))
    )
    _accumulate_direct_light_packet[.MIS, 4](
        pixels,
        paths,
        direct,
        4,
        surfaces,
        pack_coated_materials[True](surfaces),
        1,
        sampling,
        UInt32(1),
    )
    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](0.0), Vec3f32[.WORLD](0.0, 0.0, -1.0), 0.001, 10.0
    )
    var point = ShadingPoint(
        Point3f32[.WORLD](0.0), Vec3f32[.WORLD](0.0, 0.0, 1.0), True, 0.2, 0.4
    )
    for lane in range(3):
        var evaluated = evaluate_bsdf(
            ids[lane],
            surfaces,
            ray,
            point,
            Vec3f32[.WORLD](0.0, 0.0, 1.0),
            estimator_seed=path_stage_seed(
                sampling, paths.path_ids[lane], UInt32(1)
            ),
        )
        var light_pdf = Float32(0.4)
        var scale = light_pdf / (
            light_pdf * light_pdf + evaluated.pdf[0] * evaluated.pdf[0]
        )
        assert_vec_equal(pixels[lane], evaluated.value * (2.0 * scale))
        assert_true(pixels[lane].x > 0.0)
    assert_vec_equal(pixels[3], Color(0.0))


def test_wavefront_workers_reuse_chunks_without_changing_pixels() raises:
    # Several chunks per logical core force reuse, including a final partial
    # chunk and three samples per pixel (target chunk size is not divisible).
    var settings = RenderSettings(
        17, max(2, ceildiv(num_logical_cores() * 6 + 5, 17)), 3, UInt64(2026)
    )
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    builder.add_sphere(Point3f32[.WORLD](0.0, 0.0, -1.0), 0.5, matte)
    var world = CpuScene[16, 16](builder^.finish())
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0.0),
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        90.0,
    )
    var serial = render_wavefront[.PATH, 4, 7, False](settings, camera, world)
    comptime for mode_idx in range(3):
        comptime mode = CpuSchedulerMode(mode_idx)
        var parallel = render_wavefront[.PATH, 4, 7, True, mode](
            settings, camera, world
        )
        for lane in range(len(serial.pixels)):
            assert_equal(parallel.pixels[lane].x, serial.pixels[lane].x)
            assert_equal(parallel.pixels[lane].y, serial.pixels[lane].y)
            assert_equal(parallel.pixels[lane].z, serial.pixels[lane].z)


def test_environment_packet_accumulates_only_misses() raises:
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    builder.add_sphere(Point3f32[.WORLD](0.0, 0.0, -2.0), 0.5, matte)
    var texels: List[Float32] = [1.0, 0.25, 0.5, 0.1, 0.8, 0.2]
    var texture = builder.surfaces.add_image_texture(
        ImageTexture(2, 1, texels^)
    )
    builder.environment = Environment.image(
        texture,
        Color(1.0),
        Affine3f32[.WORLD, .LOCAL].identity(),
        Affine3f32[.LOCAL, .WORLD].identity(),
    )
    var world = CpuScene[16, 16](builder^.finish())
    ref scene = world.scene_data()
    var paths = PathPacket[4]()
    paths.path_ids = SIMD[.uint32, 4](0, 1, 2, 3)
    paths.dz = 1.0
    paths.tx = 0.5
    paths.ty = 0.75
    paths.tz = 1.0
    var pixels = List[Color](length=4, fill=Color(0.25))
    _accumulate_environment_packet[4, .PATH, 16, 16](
        pixels,
        paths,
        4,
        SIMD[.bool, 4](fill=False),
        1,
        scene.environment(),
        scene.surfaces(),
        world,
        0,
    )
    for pixel in pixels:
        assert_vec_equal(pixel, Color(0.25))
    _accumulate_environment_packet[4, .PATH, 16, 16](
        pixels,
        paths,
        3,
        SIMD[.bool, 4](True, False, True, False),
        1,
        scene.environment(),
        scene.surfaces(),
        world,
        0,
    )
    var radiance = environment_radiance(
        scene.environment(), scene.surfaces(), Vec3f32[.WORLD](0.0, 0.0, 1.0)
    )
    var expected = Color(0.25) + Color(0.5, 0.75, 1.0) * radiance
    assert_vec_equal(pixels[0], expected)
    assert_vec_equal(pixels[1], Color(0.25))
    assert_vec_equal(pixels[2], expected)
    assert_vec_equal(pixels[3], Color(0.25))


def test_textured_coated_packet_scattering_matches_scalar() raises:
    var surfaces = SurfaceStore()
    var texels: List[Float32] = [0.25, 0.5, 0.75, 0.8, 0.6, 0.4]
    var texture = surfaces.add_image_texture(ImageTexture(2, 1, texels^))
    var first = surfaces.add_coated_diffuse(Color(0.7), 0.2, 1.3, texture)
    var second = surfaces.add_coated_diffuse(
        Color(0.6),
        0.4,
        1.6,
        texture,
        thickness=0.2,
        layer_albedo=Color(0.1, 0.3, 0.5),
        g=0.2,
        max_depth=4,
        n_samples=2,
    )
    var settings = RenderSettings(2, 2, 1, UInt64(2026))
    var sampling = SamplingConfig.from_settings(settings)
    var batch = ShadePacket[4]()
    batch.surface_indices = SIMD[.uint32, 4](
        first.index(), second.index(), 12345, 12345
    )
    batch.path_ids = SIMD[.uint32, 4](1, 2, 3, 4)
    batch.dz = -1.0
    batch.nz = 1.0
    batch.front_faces = SIMD[.bool, 4](fill=True)
    batch.uv_u = SIMD[.float32, 4](0.2, 0.7, 0.0, 0.0)
    batch.uv_v = 0.4
    batch.tx = 1.0
    batch.ty = 1.0
    batch.tz = 1.0
    var packet = _sample_bsdf_values[.COATED_DIFFUSE, 4](
        batch,
        2,
        surfaces,
        pack_coated_materials[True](surfaces),
        sampling,
        UInt32(1),
    )
    for lane in range(2):
        var rng = path_stage_rng(sampling, batch.path_ids[lane], UInt32(1))
        var scalar = sample_bsdf(
            SurfaceId(.COATED_DIFFUSE, batch.surface_indices[lane]),
            surfaces,
            Rayf32[.WORLD](
                Point3f32[.WORLD](0.0), Vec3f32[.WORLD](0.0, 0.0, -1.0)
            ),
            ShadingPoint(
                Point3f32[.WORLD](0.0),
                Vec3f32[.WORLD](0.0, 0.0, 1.0),
                True,
                batch.uv_u[lane],
                batch.uv_v[lane],
            ),
            rng,
            estimator_seed=path_stage_seed(
                sampling, batch.path_ids[lane], UInt32(1)
            ),
        )
        assert_equal(packet.ok[lane], scalar.ok[0])
        assert_almost_equal(packet.direction.x[lane], scalar.direction.x[0])
        assert_almost_equal(packet.direction.y[lane], scalar.direction.y[0])
        assert_almost_equal(packet.direction.z[lane], scalar.direction.z[0])
        assert_almost_equal(packet.weight.x[lane], scalar.weight.x[0])
        assert_almost_equal(packet.weight.y[lane], scalar.weight.y[0])
        assert_almost_equal(packet.weight.z[lane], scalar.weight.z[0])
        assert_almost_equal(packet.pdf[lane], scalar.pdf[0])
        assert_equal(packet.delta[lane], scalar.delta[0])
    assert_false(packet.ok[2])
    assert_false(packet.ok[3])


def test_film_response_simd_rgb_and_tails() raises:
    var film = FilmResponse(
        0.81,
        Color(0.7, 0.2, 0.1),
        Color(-0.1, 1.1, 0.05),
        Color(0.15, -0.2, 0.9),
    )
    comptime for width in [1, 4, simd_width_of[Float32]()]:
        for count in [0, 1, width - 1, width, width + 1, 2 * width + 3]:
            var pixels = List[Color](length=count + 2, fill=Color(-17.0))
            for i in range(count):
                pixels[i + 1] = Color(
                    Float32(i % 7) / 3.0,
                    -Float32(i % 11) / 5.0,
                    Float32(i % 13) / 7.0,
                )
            var expected = pixels.copy()
            for i in range(1, count + 1):
                var color = expected[i] * 0.25
                expected[i] = (
                    Color(
                        dot(film.red, color),
                        dot(film.green, color),
                        dot(film.blue, color),
                    )
                    * film.exposure
                )
            # Offset views exercise unaligned loads; sentinels catch tail writes.
            film.apply_to[width](pixels[1 : count + 1], 0.25)
            for i in range(count + 2):
                comptime for channel in range(3):
                    assert_equal(
                        bitcast[.uint32](pixels[i][channel]),
                        bitcast[.uint32](expected[i][channel]),
                    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
