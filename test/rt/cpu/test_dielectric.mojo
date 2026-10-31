"""Dielectric transport: exact Fresnel, radiance scaling and rough lobes."""

from std.math import abs, sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from bajo.core import Vec3, Vec3f32, normalize
from bajo.core.optics import fresnel_dielectric
from bajo.rt.material_types import MaterialKind, SurfaceStore, ImageTexture
from bajo.rt.texture_math import mapped_normal
from bajo.rt.shading import _sample_material, _evaluate_material
from bajo.rt.cpu.bsdf import _MaterialParameters


def test_smooth_glass_exact_fresnel_and_radiance_weights() raises:
    var direction = Vec3f32[.WORLD](sqrt(Float32(0.75)), 0.0, -0.5)
    var normal = Vec3f32[.WORLD](0.0, 0.0, 1.0)
    # Exact unpolarized Fresnel reflectance for air/glass at 60 degrees.
    var expected = Float32(0.08918671)
    assert_almost_equal(fresnel_dielectric(0.5, 1.5)[0], expected, atol=1e-6)
    var reflected = _sample_material[.DIELECTRIC, 1](
        direction, normal, Vec3f32[.WORLD](0.0), 1.5, True, 0.08, 0.0
    )
    assert_true(reflected.ok[0])
    assert_true(reflected.delta[0])
    assert_true(reflected.direction.z[0] > 0.0)
    assert_almost_equal(reflected.pdf[0], expected, atol=1e-6)
    assert_almost_equal(reflected.weight.x[0], 1.0, atol=1e-6)
    var transmitted = _sample_material[.DIELECTRIC, 1](
        direction, normal, Vec3f32[.WORLD](0.0), 1.5, True, 0.9, 0.0
    )
    assert_true(transmitted.ok[0])
    assert_true(transmitted.direction.z[0] < 0.0)
    assert_almost_equal(transmitted.pdf[0], 1.0 - expected, atol=1e-6)
    assert_almost_equal(transmitted.weight.x[0], 1.0 / 2.25, atol=1e-6)
    var leaving = _sample_material[.DIELECTRIC, 1](
        Vec3f32[.WORLD](0.0, 0.0, -1.0),
        normal,
        Vec3f32[.WORLD](0.0),
        1.5,
        False,
        0.9,
        0.0,
    )
    assert_almost_equal(leaving.weight.x[0], 2.25, atol=1e-6)
    assert_almost_equal(
        leaving.weight.x[0] * transmitted.weight.x[0], 1.0, atol=1e-6
    )


def test_total_internal_reflection_and_matching_indices() raises:
    var normal = Vec3f32[.WORLD](0.0, 0.0, 1.0)
    var reflected = _sample_material[.DIELECTRIC, 1](
        Vec3f32[.WORLD](sqrt(Float32(0.75)), 0.0, -0.5),
        normal,
        Vec3f32[.WORLD](0.0),
        1.5,
        False,
        0.999,
        0.0,
    )
    assert_true(reflected.ok[0])
    assert_true(reflected.direction.z[0] > 0.0)
    assert_almost_equal(reflected.pdf[0], 1.0)
    assert_almost_equal(reflected.weight.x[0], 1.0)
    var incoming = normalize(Vec3f32[.WORLD](0.3, 0.1, -1.0))
    var transparent = _sample_material[.DIELECTRIC, 1](
        incoming, normal, Vec3f32[.WORLD](0.0), 1.0, True, 0.5, 0.2, 0.5, 0.8
    )
    assert_true(transparent.ok[0])
    assert_true(transparent.delta[0])
    assert_almost_equal(transparent.weight.x[0], 1.0)
    for axis in range(3):
        assert_almost_equal(
            transparent.direction[axis][0], incoming[axis][0], atol=1e-6
        )


def test_rough_glass_sampling_evaluation_and_pdf_agree() raises:
    var normal = Vec3f32[.WORLD](0.0, 0.0, 1.0)
    var incoming = Vec3f32[.WORLD](0.6, 0.0, -0.8)
    var reflected_count = 0
    var transmitted_count = 0
    for front in [True, False]:
        for uc in [Float32(0.0), Float32(0.5), Float32(0.99)]:
            for u in [Float32(0.01), Float32(0.25), Float32(0.75)]:
                var sampled = _sample_material[.DIELECTRIC, 1](
                    incoming,
                    normal,
                    Vec3f32[.WORLD](0.0),
                    1.5,
                    front,
                    uc,
                    u,
                    0.5,
                    0.37,
                )
                if not sampled.ok[0]:
                    continue
                assert_equal(sampled.delta[0], False)
                assert_true(sampled.direction.is_finite()[0])
                var evaluation = _evaluate_material[.DIELECTRIC, 1](
                    incoming,
                    normal,
                    Vec3f32[.WORLD](0.0),
                    1.5,
                    sampled.direction,
                    0.5,
                    front,
                )
                assert_true(evaluation.pdf[0] > 0.0)
                assert_almost_equal(
                    sampled.pdf[0], evaluation.pdf[0], rtol=1e-5, atol=1e-6
                )
                var weight = (
                    evaluation.value.x[0]
                    * abs(sampled.direction.z[0])
                    / evaluation.pdf[0]
                )
                assert_almost_equal(
                    sampled.weight.x[0], weight, rtol=1e-5, atol=1e-6
                )
                if sampled.direction.z[0] > 0.0:
                    reflected_count += 1
                else:
                    transmitted_count += 1
    assert_true(reflected_count > 0)
    assert_true(transmitted_count > 0)


def test_packet_materials_gather_distinct_glass_roughness() raises:
    var surfaces = SurfaceStore()
    for value in [Float32(0.0), Float32(0.1), Float32(0.5)]:
        var surface = surfaces.add_dielectric(1.5)
        surfaces.dielectrics[Int(surface.index())].roughness = value
    var material = _MaterialParameters[4]()
    material.gather[.DIELECTRIC](
        surfaces,
        SIMD[.uint32, 4](0, 1, UInt32.MAX, 2),
        SIMD[.bool, 4](True, True, False, True),
    )
    assert_almost_equal(material.parameter[0], 1.5)
    assert_almost_equal(material.roughness[0], 0.0)
    assert_almost_equal(material.roughness[1], 0.1)
    assert_almost_equal(material.roughness[2], 0.0)
    assert_almost_equal(material.roughness[3], 0.5)


def test_normal_map_uses_uv_frame_and_backface_orientation() raises:
    var surfaces = SurfaceStore()
    var material = surfaces.add_dielectric(1.5)
    var image = surfaces.add_image_texture(ImageTexture(1, 1, [0.75, 0.5, 1.0]))
    surfaces.normal_maps[material.value[0]] = image
    var normal = Vec3f32[.WORLD](0.0, 0.0, 1.0)
    var tangent = Vec3f32[.WORLD](0.0, 3.0, 0.0)
    var mapped = surfaces.shading_normal(
        material, normal, 0.1, 0.9, dpdu=tangent
    )
    assert_almost_equal(mapped.x[0], 0.0, atol=1e-6)
    assert_almost_equal(mapped.y[0], 0.4472136, atol=1e-6)
    assert_almost_equal(mapped.z[0], 0.8944272, atol=1e-6)
    var back = surfaces.shading_normal(
        material, normal, 0.1, 0.9, dpdu=tangent, front_face=False
    )
    assert_almost_equal(back.y[0], -0.4472136, atol=1e-6)
    assert_almost_equal(back.z[0], 0.8944272, atol=1e-6)
    var neutral = mapped_normal(normal, tangent, Vec3f32[.WORLD](0.5, 0.5, 1.0))
    assert_almost_equal(neutral.z[0], 1.0, atol=1e-6)


def test_bilinear_normal_texture_centers_edges_and_wrapping() raises:
    var image = ImageTexture(2, 2, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    assert_almost_equal(image.sample_bilinear(0.25, 0.75).x[0], 0.0)
    assert_almost_equal(image.sample_bilinear(0.75, 0.25).x[0], 3.0)
    assert_almost_equal(image.sample_bilinear(0.5, 0.5).x[0], 1.5)
    assert_almost_equal(image.sample_bilinear(0.0, 0.0).x[0], 1.5)
    assert_almost_equal(image.sample_bilinear(-0.5, 2.5).x[0], 1.5)
    var us = SIMD[.float32, 4](0.25, 0.75, 0.5, -0.5)
    var vs = SIMD[.float32, 4](0.75, 0.25, 0.5, 2.5)
    var packet = image.sample_bilinear(us, vs)
    comptime for lane in range(4):
        assert_almost_equal(
            packet.x[lane], image.sample_bilinear(us[lane], vs[lane]).x[0]
        )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
