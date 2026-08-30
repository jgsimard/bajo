from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
    assert_false,
)
from std.memory import bitcast

from bajo.bvh.camera import Camera
from bajo.bvh.host_utils import compute_bounds
from bajo.core import (
    AABB,
    Affine3f32,
    Vec3,
    Vec3f32,
    assert_vec_equal,
    dot,
    length,
    Point3f32,
    Rayf32,
)
from bajo.core.random import Rng, Sampler, _sobol_bits, _sz_bits
from bajo.rt import (
    Color,
    CpuSchedulerMode,
    Dielectric,
    Emissive,
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
    render_depth_first,
    render_wavefront,
    sample_bsdf,
)
from examples.cornell_box import make_cornell_world
from bajo.rt.cpu import reflect, reflectance
from bajo.rt.common import path_stage_rng, russian_roulette
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
from bajo.rt.types import MaterialKind, PrimitiveKind, SamplingConfig
from bajo.rt.wavefront_contract import wavefront_rng_roulette_stage


def _front_point() -> ShadingPoint[1]:
    return ShadingPoint(
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        Vec3f32[.WORLD](0.0, 0.0, 1.0),
        True,
    )


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
    assert_almost_equal(evaluated.value.x, material.albedo.x / 3.14159265)


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


def test_world_preserves_signed_radius_normals() raises:
    var builder = SceneBuilder()
    var glass = builder.add_dielectric(1.5)
    builder.add_sphere(
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        -0.5,
        glass,
    )
    var scene = builder^.finish()
    var world = CpuScene[](scene^)

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
    var world = CpuScene[](scene^)

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
    var world = CpuScene[](scene^)

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
    var world = CpuScene[](scene^)
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
    var world = CpuScene[](scene^)

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


def test_compile_time_semantic_classifications() raises:
    comptime assert Integrator.is_path_tracing[Integrator.PATH]
    comptime assert Integrator.is_path_tracing[Integrator.NEE]
    comptime assert Integrator.is_path_tracing[Integrator.MIS]
    comptime assert not Integrator.is_path_tracing[Integrator.NORMALS]
    comptime assert not Integrator.is_path_tracing[Integrator.AO]
    comptime assert Integrator.uses_direct_lighting[Integrator.NEE]
    comptime assert Integrator.uses_direct_lighting[Integrator.MIS]
    comptime assert not Integrator.uses_direct_lighting[Integrator.PATH]
    comptime assert not Integrator.uses_direct_lighting[Integrator.NORMALS]
    comptime assert not Integrator.uses_direct_lighting[Integrator.AO]
    comptime assert Integrator.uses_visibility[Integrator.AO]
    comptime assert Integrator.uses_visibility[Integrator.NEE]
    comptime assert Integrator.uses_visibility[Integrator.MIS]
    comptime assert not Integrator.uses_visibility[Integrator.PATH]
    comptime assert not Integrator.uses_visibility[Integrator.NORMALS]
    comptime assert MaterialKind.has_bsdf[MaterialKind.LAMBERTIAN]
    comptime assert MaterialKind.has_bsdf[MaterialKind.METAL]
    comptime assert MaterialKind.has_bsdf[MaterialKind.DIELECTRIC]
    comptime assert not MaterialKind.has_bsdf[MaterialKind.EMISSIVE]
    comptime assert CpuSchedulerMode.is_valid[CpuSchedulerMode.RUNTIME_DEFAULT]
    comptime assert CpuSchedulerMode.is_valid[CpuSchedulerMode.LOGICAL_CORES]
    comptime assert CpuSchedulerMode.is_valid[CpuSchedulerMode.TASK_PARTITIONS]
    comptime assert not CpuSchedulerMode.is_valid[CpuSchedulerMode(3)]


def test_render_settings_and_tiny_render() raises:
    var settings = RenderSettings(4, 2, 2, UInt64(9), 2)
    assert_equal(settings.image_width, 4)
    assert_equal(settings.image_height, 2)
    assert_equal(settings.rng_seed, 9)

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

    var result = render_depth_first[.PATH](settings, camera, world)
    assert_equal(len(result.pixels), 8)
    assert_equal(result.timings.pixel_count, 8)
    assert_equal(result.timings.sample_count, 16)
    assert_equal(result.timings.max_depth, 2)
    assert_true(result.timings.total_ns >= result.timings.render_ns)
    for p in result.pixels:
        assert_true(p.x >= 0.0 and p.y >= 0.0 and p.z >= 0.0)

    var one_pixel_tiles = render_depth_first[.PATH, 1, 1](
        settings, camera, world
    )
    for i, pixel in enumerate(result.pixels):
        assert_vec_equal(one_pixel_tiles.pixels[i], pixel)

    # Renderer packet length and both acceleration widths are independent.
    var packet_result = render_wavefront[.PATH, 4, 16, False](
        settings, camera, world
    )
    assert_equal(len(packet_result.pixels), len(result.pixels))


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
    var world = CpuScene[](scene^)
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        90.0,
    )

    var result = render_depth_first[.NORMALS](settings, camera, world)
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
    var world = CpuScene[](scene^)
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        90.0,
    )

    var result = render_depth_first[.AO](settings, camera, world)
    assert_equal(len(result.pixels), 1)
    assert_true(
        result.pixels[0].x >= 0.0
        and result.pixels[0].y >= 0.0
        and result.pixels[0].z >= 0.0
    )


def test_wavefront_tiny_render() raises:
    var settings = RenderSettings(3, 2, 2, UInt64(9))
    var builder = SceneBuilder()
    var matte = builder.add_lambertian(Color(0.5))
    builder.add_sphere(
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        0.5,
        matte,
    )
    var scene = builder^.finish()
    var world = CpuScene[](scene^)
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Point3f32[.WORLD](0.0, 0.0, -1.0),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        90.0,
    )

    var result = render_wavefront[.PATH](settings, camera, world)
    assert_equal(len(result.pixels), 6)
    assert_equal(result.timings.sample_count, 12)
    for p in result.pixels:
        assert_true(p.x >= 0.0 and p.y >= 0.0 and p.z >= 0.0)

    # Whole-pixel chunks preserve Philox stream ownership and accumulation
    # order. The small chunk also exercises a target not divisible by samples
    # per pixel.
    var chunked = render_wavefront[.PATH, 1, 3, False](settings, camera, world)
    var parallel = render_wavefront[.PATH, 1, 3, True](settings, camera, world)
    # Instantiate multiple packet widths and a six-path chunk so every width
    # exercises partial packets as well as the generic queue indexing.
    var width1 = render_wavefront[.PATH, 1, 7, False](settings, camera, world)
    var packet4 = render_wavefront[.PATH, 4, 7, False](settings, camera, world)
    var packet8 = render_wavefront[.PATH, 8, 7, False](settings, camera, world)
    var packet16 = render_wavefront[.PATH, 16, 7, False](
        settings, camera, world
    )
    assert_equal(len(chunked.pixels), len(result.pixels))
    for i, pixel in enumerate(result.pixels):
        assert_equal(chunked.pixels[i].x, pixel.x)
        assert_equal(chunked.pixels[i].y, pixel.y)
        assert_equal(chunked.pixels[i].z, pixel.z)
        assert_equal(parallel.pixels[i].x, pixel.x)
        assert_equal(parallel.pixels[i].y, pixel.y)
        assert_equal(parallel.pixels[i].z, pixel.z)
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
    var world = CpuScene[](scene^)
    var camera = Camera.from_vfov(
        Point3f32[.WORLD](0.0, 0.35, 2.5),
        Point3f32[.WORLD](0.0, 0.0, -1.5),
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        52.0,
    )

    var packet1 = render_wavefront[.PATH, 1, 10, False](settings, camera, world)
    var packet4 = render_wavefront[.PATH, 4, 10, False](settings, camera, world)
    var packet8 = render_wavefront[.PATH, 8, 10, False](settings, camera, world)
    var packet16 = render_wavefront[.PATH, 16, 10, False](
        settings, camera, world
    )
    for pixel_idx, pixel in enumerate(packet1.pixels):
        assert_vec_equal(packet4.pixels[pixel_idx], pixel)
        assert_vec_equal(packet8.pixels[pixel_idx], pixel)
        assert_vec_equal(packet16.pixels[pixel_idx], pixel)


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
    var result = render_wavefront[.NEE](settings, camera, world)
    assert_true(len(world.scene_data().lights().records) > 0)
    assert_true(world.scene_data().lights().total_weight > 0.0)
    for light in world.scene_data().lights().records:
        assert_equal(light.surface.kind(), .EMISSIVE)
        assert_true(light.weight > 0.0)
    var depth_first = render_depth_first[.NEE](settings, camera, world)
    var mis = render_wavefront[.MIS](settings, camera, world)
    var depth_first_mis = render_depth_first[.MIS](settings, camera, world)
    var packet_nee1 = render_wavefront[.NEE, 1, 14, False](
        settings, camera, world
    )
    var packet_nee4 = render_wavefront[.NEE, 4, 14, False](
        settings, camera, world
    )
    var packet_nee8 = render_wavefront[.NEE, 8, 14, False](
        settings, camera, world
    )
    var packet_nee16 = render_wavefront[.NEE, 16, 14, False](
        settings, camera, world
    )
    var packet_mis1 = render_wavefront[.MIS, 1, 14, False](
        settings, camera, world
    )
    var packet_mis4 = render_wavefront[.MIS, 4, 14, False](
        settings, camera, world
    )
    var packet_mis8 = render_wavefront[.MIS, 8, 14, False](
        settings, camera, world
    )
    var packet_mis16 = render_wavefront[.MIS, 16, 14, False](
        settings, camera, world
    )
    var total = Float32(0.0)
    var depth_first_total = Float32(0.0)
    var mis_total = Float32(0.0)
    var depth_first_mis_total = Float32(0.0)
    for pixel_idx, pixel in enumerate(result.pixels):
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
    for pixel_idx, pixel in enumerate(mis.pixels):
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
