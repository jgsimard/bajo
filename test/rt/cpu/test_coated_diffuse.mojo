from std.math import pi
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_false,
    assert_true,
)

from bajo.core import dot, Vec3, Vec3f32
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
            SIMD[.float32, 1](0.2),
            SIMD[.float32, 1](1.5),
            SIMD[.float32, 1](0.01),
            Vec3f32[.WORLD](0.0),
            SIMD[.float32, 1](0.0),
            SIMD[.float32, 1](10.0),
            SIMD[.float32, 1](1.0),
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


def test_coated_packet_evaluation_skips_inactive_lanes() raises:
    var rays = Vec3[.float32, .WORLD, 4](0.0, 0.0, -1.0)
    var normals = Vec3[.float32, .WORLD, 4](0.0, 0.0, 1.0)
    var albedo = Vec3[.float32, .WORLD, 4](0.7, 0.5, 0.25)
    var out_direction = Vec3[.float32, .WORLD, 4](0.0, 0.0, 1.0)
    var active = SIMD[.bool, 4](True, False, True, False)
    var original = _evaluate_coated_diffuse(
        rays,
        normals,
        albedo,
        SIMD[.float32, 4](0.2),
        SIMD[.float32, 4](1.5),
        SIMD[.float32, 4](0.01),
        Vec3[.float32, .WORLD, 4](0.0),
        SIMD[.float32, 4](0.0),
        SIMD[.float32, 4](10.0),
        SIMD[.float32, 4](1.0),
        out_direction,
    )
    var masked = _evaluate_coated_diffuse(
        rays,
        normals,
        albedo,
        SIMD[.float32, 4](0.2),
        SIMD[.float32, 4](1.5),
        SIMD[.float32, 4](0.01),
        Vec3[.float32, .WORLD, 4](0.0),
        SIMD[.float32, 4](0.0),
        SIMD[.float32, 4](10.0),
        SIMD[.float32, 4](1.0),
        out_direction,
        active,
    )
    for lane in range(4):
        if active[lane]:
            assert_almost_equal(masked.pdf[lane], original.pdf[lane])
            assert_almost_equal(masked.value.x[lane], original.value.x[lane])
            assert_almost_equal(masked.value.y[lane], original.value.y[lane])
            assert_almost_equal(masked.value.z[lane], original.value.z[lane])
            assert_true(masked.pdf[lane] > 0.0)
        else:
            assert_almost_equal(masked.pdf[lane], 0.0)
            assert_almost_equal(masked.value.x[lane], 0.0)
            assert_almost_equal(masked.value.y[lane], 0.0)
            assert_almost_equal(masked.value.z[lane], 0.0)


def test_coated_packet_sampling_skips_inactive_lanes() raises:
    var rays = Vec3[.float32, .WORLD, 4](0.0, 0.0, -1.0)
    var normals = Vec3[.float32, .WORLD, 4](0.0, 0.0, 1.0)
    var albedo = Vec3[.float32, .WORLD, 4](0.7, 0.5, 0.25)
    var active = SIMD[.bool, 4](True, False, True, False)
    var original = _sample_coated_diffuse(
        rays,
        normals,
        albedo,
        SIMD[.float32, 4](0.2),
        SIMD[.float32, 4](1.5),
        SIMD[.float32, 4](0.01),
        Vec3[.float32, .WORLD, 4](0.0),
        SIMD[.float32, 4](0.0),
        SIMD[.float32, 4](10.0),
        SIMD[.float32, 4](1.0),
        SIMD[.float32, 4](0.0),
        SIMD[.float32, 4](0.3),
    )
    var masked = _sample_coated_diffuse(
        rays,
        normals,
        albedo,
        SIMD[.float32, 4](0.2),
        SIMD[.float32, 4](1.5),
        SIMD[.float32, 4](0.01),
        Vec3[.float32, .WORLD, 4](0.0),
        SIMD[.float32, 4](0.0),
        SIMD[.float32, 4](10.0),
        SIMD[.float32, 4](1.0),
        SIMD[.float32, 4](0.0),
        SIMD[.float32, 4](0.3),
        active,
    )
    for lane in range(4):
        if active[lane]:
            assert_true(masked.ok[lane] == original.ok[lane])
            assert_almost_equal(masked.pdf[lane], original.pdf[lane])
            assert_almost_equal(
                masked.direction.z[lane], original.direction.z[lane]
            )
            assert_almost_equal(masked.weight.x[lane], original.weight.x[lane])
            assert_almost_equal(masked.weight.y[lane], original.weight.y[lane])
            assert_almost_equal(masked.weight.z[lane], original.weight.z[lane])
        else:
            assert_false(masked.ok[lane])
            assert_almost_equal(masked.pdf[lane], 0.0)
            assert_almost_equal(masked.weight.x[lane], 0.0)
            assert_almost_equal(masked.weight.y[lane], 0.0)
            assert_almost_equal(masked.weight.z[lane], 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
