from bajo.rt.material_parameters import CoatedParameters
from std.math import pi
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_false,
    assert_true,
)

from bajo.core import dot, Vec3f32
from bajo.rt.coated_diffuse import (
    _dielectric_f,
    _evaluate_coated_diffuse,
    _fresnel_dielectric,
    _sample_coated_diffuse,
    _sample_dielectric,
)


def test_exact_dielectric_fresnel() raises:
    assert_almost_equal(_fresnel_dielectric(1.0, 1.5), 0.04, atol=1.0e-6)
    # A ray leaving glass beyond the critical angle must totally reflect.
    assert_almost_equal(_fresnel_dielectric(-0.5, 1.5), 1.0)


def test_ggx_dielectric_normal_incidence_value() raises:
    var wo = Vec3f32[.WORLD](0.0, 0.0, 1.0)
    var wi = Vec3f32[.WORLD](0.0, 0.0, 1.0)
    var alpha = Float32(0.2)
    var expected = Float32(0.04) / (Float32(4.0 * pi) * alpha * alpha)
    assert_almost_equal(
        _dielectric_f(wo, wi, alpha, 1.5, True),
        expected,
        rtol=1.0e-5,
    )


def test_ggx_dielectric_samples_reflection_and_transmission() raises:
    var wo = Vec3f32[.WORLD](0.0, 0.0, 1.0)
    var reflection = _sample_dielectric(wo, 0.2, 1.5, 0.0, 0.3, 0.7, True)
    var transmission = _sample_dielectric(wo, 0.2, 1.5, 0.99, 0.3, 0.7, True)
    assert_true(reflection.ok)
    assert_true(reflection.reflection)
    assert_true(reflection.direction.z[0] > 0.0)
    assert_true(transmission.ok)
    assert_false(transmission.reflection)
    assert_true(transmission.direction.z[0] < 0.0)


def test_layered_samples_are_finite_and_leave_the_coating() raises:
    var ray = Vec3f32[.WORLD](0.0, 0.0, -1.0)
    var normal = Vec3f32[.WORLD](0.0, 0.0, 1.0)
    var albedo = Vec3f32[.WORLD](0.7, 0.5, 0.25)
    var successful = 0
    for i in range(256):
        var u = Float32(i) / Float32(256)
        var v = Float32((i * 73) % 256) / Float32(256)
        var sampled = _sample_coated_diffuse(
            ray,
            normal,
            albedo,
            CoatedParameters[1](
                SIMD[.float32, 1](0.2),
                SIMD[.float32, 1](1.5),
                SIMD[.float32, 1](0.01),
                Vec3f32[.WORLD](0.0),
                SIMD[.float32, 1](0.0),
                SIMD[.float32, 1](10.0),
                SIMD[.float32, 1](1.0),
                Vec3f32[.WORLD](0.0),
                Vec3f32[.WORLD](0.0),
                0.0,
            ),
            SIMD[.float32, 1](u),
            SIMD[.float32, 1](v),
        )
        if not sampled.ok[0]:
            continue
        successful += 1
        assert_true(sampled.direction.is_finite()[0])
        assert_true(sampled.weight.is_finite()[0])
        assert_true(sampled.pdf[0] > 0.0)
        assert_true(dot(sampled.direction, normal)[0] > 0.0)
        assert_true(sampled.weight.x[0] >= 0.0)
        assert_true(sampled.weight.y[0] >= 0.0)
        assert_true(sampled.weight.z[0] >= 0.0)
    # Some substrate paths legitimately fail to leave within maxDepth.
    assert_true(successful > 150)


def test_coated_sampling_and_evaluation_share_estimator_seed() raises:
    var incoming = Vec3f32[.WORLD](0.1, -0.2, -1.0)
    var normal = Vec3f32[.WORLD](0.0, 0.0, 1.0)
    var eta = Vec3f32[.WORLD](1.2, 0.95, 0.65)
    var k = Vec3f32[.WORLD](7.3, 6.7, 5.6)
    var successes = 0
    for i in range(32):
        var seed = UInt32(i * 73 + 19)
        var sampled = _sample_coated_diffuse(
            incoming,
            normal,
            Vec3f32[.WORLD](0.7),
            CoatedParameters[1](
                0.2,
                1.5,
                0.01,
                Vec3f32[.WORLD](0.0),
                0.0,
                16.0,
                4.0,
                eta,
                k,
                0.2,
            ),
            Float32(i) / 32.0,
            Float32((i * 13) % 32) / 32.0,
            estimator_seed=seed,
        )
        if not sampled.ok[0] or sampled.delta[0]:
            continue
        successes += 1
        var evaluated = _evaluate_coated_diffuse(
            incoming,
            normal,
            Vec3f32[.WORLD](0.7),
            CoatedParameters[1](
                0.2,
                1.5,
                0.01,
                Vec3f32[.WORLD](0.0),
                0.0,
                16.0,
                4.0,
                eta,
                k,
                0.2,
            ),
            sampled.direction,
            estimator_seed=seed,
        )
        assert_almost_equal(
            sampled.pdf[0], evaluated.pdf[0], atol=1.0e-5, rtol=1.0e-5
        )
    assert_true(successes > 16)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
