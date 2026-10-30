"""Shared layered estimators: cross-width checks and original scalar fixtures."""

from std.memory import bitcast
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)
from bajo.core import Vec3, Vec3f32, normalize, orthonormal_basis
from bajo.rt.material_parameters import CoatedParameters
from bajo.rt.coated_diffuse import (
    _from_local,
    _random,
    _evaluate_coated_diffuse,
    _sample_coated_diffuse,
)


def _parameters(index: Int) -> CoatedParameters[1]:
    var p = CoatedParameters[1]()
    p.roughness = 0.2
    p.max_depth = Float32(1 + index % 19)
    p.n_samples = Float32(1 + index % 4)
    if index % 23 == 0:
        # Adapters truncate these register fields to scalar integer limits.
        p.max_depth += 0.75
        p.n_samples += 0.25
    var kind = index % 8
    if kind == 1:
        p.roughness = 0.0
        p.thickness = 0.0
    elif kind == 2 or kind == 6:
        p.layer_albedo = Vec3f32[.WORLD](0.7, 0.4, 0.2)
        p.thickness = 0.4 if kind == 2 else 3.5
        p.g = 0.5 if kind == 2 else -0.99
        p.roughness = 0.35 if kind == 2 else 0.001
    elif kind == 3 or kind == 4 or kind == 5:
        p.conductor_eta = Vec3f32[.WORLD](1.2, 0.95, 0.65)
        p.conductor_k = Vec3f32[.WORLD](7.3, 6.7, 5.6)
        p.conductor_roughness = 0.2
        if kind == 3 and index % 16 == 11:
            p.layer_albedo = Vec3f32[.WORLD](0.15, 0.05, 0.02)
            p.thickness = 0.2
            p.g = -0.4
        if kind == 4 or kind == 5:
            p.eta = 1.0
            p.thickness = 0.0
            if kind == 5:
                p.conductor_roughness = 0.0
    elif kind == 7:
        p.eta = 0.75
        p.roughness = 0.0009
    return p


def _check[width: SIMDLength]() raises:
    var successes = 0
    var deltas = 0
    for batch in range(16):
        var rays = Vec3[.float32, .WORLD, width](0.0)
        var normals = Vec3[.float32, .WORLD, width](0.0)
        var directions = Vec3[.float32, .WORLD, width](0.0)
        var albedo = Vec3[.float32, .WORLD, width](0.7, 0.5, 0.25)
        var p = CoatedParameters[width]()
        var active = SIMD[.bool, width](fill=True)
        var u = SIMD[.float32, width](0.0)
        var v = u
        var seeds = SIMD[.uint32, width](0)
        for lane in range(Int(width)):
            var index = batch * Int(width) + lane
            var normal = Vec3f32[.WORLD](0.0, 0.0, 1.0)
            if index % 3 == 1:
                normal = Vec3f32[.WORLD](0.0, 1.0, 0.0)
            elif index % 3 == 2:
                normal = normalize(Vec3f32[.WORLD](1.0, 2.0, 3.0))
            var tangent, bitangent = orthonormal_basis(normal)
            var local = normalize(
                Vec3f32[.WORLD](
                    Float32(index % 7) * 0.2,
                    -0.2,
                    Float32(0.1 if index % 5 == 0 else 1.0),
                )
            )
            var ray = -_from_local(local, tangent, bitangent, normal)
            var out_direction = _from_local(
                normalize(
                    Vec3f32[.WORLD](
                        -0.3, 0.1, Float32(-1.0 if index % 11 == 0 else 1.0)
                    )
                ),
                tangent,
                bitangent,
                normal,
            )
            comptime for axis in range(3):
                rays[axis][lane] = ray[axis]
                normals[axis][lane] = normal[axis]
                directions[axis][lane] = out_direction[axis]
            p.set_lane(lane, _parameters(index))
            seeds[lane] = UInt32(index) * 0x9E3779B9 + 73
            u[lane] = _random(seeds[lane], 3)
            v[lane] = _random(seeds[lane], 7)
            active[lane] = index % 7 != 0 and batch != 3
            if not active[lane]:
                # Poison even iteration counts: inactive lanes cannot affect a
                # reduction, integer conversion, another lane or its result.
                var nan = bitcast[.float32](UInt32(0x7FC00000))
                p.roughness[lane] = nan
                p.eta[lane] = nan
                p.max_depth[lane] = nan
                p.n_samples[lane] = nan
                rays.x[lane] = nan
                directions.z[lane] = nan
        var sampled = _sample_coated_diffuse(
            rays, normals, albedo, p, u, v, active, seeds
        )
        var evaluated = _evaluate_coated_diffuse(
            rays,
            normals,
            albedo,
            p,
            directions,
            active,
            seeds,
        )
        for lane in range(Int(width)):
            if not active[lane]:
                assert_false(sampled.ok[lane])
                assert_false(sampled.delta[lane])
                assert_equal(sampled.pdf[lane], Float32(0.0))
                assert_equal(evaluated.pdf[lane], Float32(0.0))
                comptime for axis in range(3):
                    assert_equal(sampled.weight[axis][lane], Float32(0.0))
                    assert_equal(evaluated.value[axis][lane], Float32(0.0))
                    assert_equal(
                        sampled.direction[axis][lane], normals[axis][lane]
                    )
                continue
            var index = batch * Int(width) + lane
            var ray = Vec3f32[.WORLD](rays.x[lane], rays.y[lane], rays.z[lane])
            var normal = Vec3f32[.WORLD](
                normals.x[lane], normals.y[lane], normals.z[lane]
            )
            var out_direction = Vec3f32[.WORLD](
                directions.x[lane], directions.y[lane], directions.z[lane]
            )
            var scalar = _sample_coated_diffuse(
                ray,
                normal,
                Vec3f32[.WORLD](0.7, 0.5, 0.25),
                _parameters(index),
                u[lane],
                v[lane],
                estimator_seed=seeds[lane],
            )
            var scalar_eval = _evaluate_coated_diffuse(
                ray,
                normal,
                Vec3f32[.WORLD](0.7, 0.5, 0.25),
                _parameters(index),
                out_direction,
                estimator_seed=seeds[lane],
            )
            assert_equal(sampled.ok[lane], scalar.ok[0])
            successes += Int(sampled.ok[lane])
            deltas += Int(sampled.ok[lane] and sampled.delta[lane])
            assert_equal(sampled.delta[lane], scalar.delta[0])
            assert_equal(evaluated.delta[lane], scalar_eval.delta[0])
            assert_almost_equal(
                sampled.pdf[lane], scalar.pdf[0], atol=1e-6, rtol=2e-5
            )
            assert_almost_equal(
                evaluated.pdf[lane], scalar_eval.pdf[0], atol=1e-6, rtol=2e-5
            )
            comptime for axis in range(3):
                assert_almost_equal(
                    sampled.direction[axis][lane],
                    scalar.direction[axis][0],
                    atol=2e-6,
                    rtol=2e-5,
                )
                assert_almost_equal(
                    sampled.weight[axis][lane],
                    scalar.weight[axis][0],
                    atol=2e-6,
                    rtol=2e-5,
                )
                assert_almost_equal(
                    evaluated.value[axis][lane],
                    scalar_eval.value[axis][0],
                    atol=2e-6,
                    rtol=2e-5,
                )

    assert_true(successes > Int(width) * 4)
    assert_true(deltas > Int(width))


def test_simd_layered_matches_scalar_across_widths_and_masks() raises:
    _check[4]()
    _check[16]()


def test_shared_coating_matches_original_scalar_reference() raises:
    # Captured from the original scalar implementation before SIMD/refactoring.
    # Per case: direction xyz, weight rgb, PDF, delta, ok, value rgb, PDF, delta.
    var expected: List[UInt32] = [
        # Coated diffuse.
        0x00000000,
        0x00000000,
        0x3F800000,
        0x3F333333,
        0x3F000000,
        0x3E800000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x3DA03DD9,
        0x3D762FC4,
        0x3D1950DA,
        0x3F1C8498,
        0x00000000,
        # Coated conductor.
        0x3EFD5772,
        0xBEEC72E7,
        0x3F3C7344,
        0x3F636635,
        0x3F648628,
        0x3F64F5C8,
        0x3F155D27,
        0x00000000,
        0x3F800000,
        0x3EE5AB01,
        0x3EE6C7EC,
        0x3EE73647,
        0x3F4343C7,
        0x00000000,
        # Delta conductor.
        0x3DC7DD46,
        0xBE47DD46,
        0x3F79D497,
        0x3F6ADB1E,
        0x3F6C043F,
        0x3F6C7756,
        0x3F800000,
        0x3F800000,
        0x3F800000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x3F800000,
        # Scattering coat.
        0x00000000,
        0x00000000,
        0x3F800000,
        0x3F333333,
        0x3F000000,
        0x3E800000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x3DF8E1FE,
        0x3D4DE819,
        0x3CCC49F2,
        0x3F00C208,
        0x00000000,
    ]
    for case_index, i in enumerate([0, 3, 5, 10]):
        var seed = UInt32(i) * 0x9E3779B9 + 73
        var ray = Vec3f32[.WORLD](0.1, -0.2, -1.0)
        var normal = Vec3f32[.WORLD](0.0, 0.0, 1.0)
        var albedo = Vec3f32[.WORLD](0.7, 0.5, 0.25)
        var p = _parameters(i)
        var sampled = _sample_coated_diffuse(
            ray,
            normal,
            albedo,
            p,
            _random(seed, 3),
            _random(seed, 7),
            estimator_seed=seed,
        )
        var evaluated = _evaluate_coated_diffuse(
            ray,
            normal,
            albedo,
            p,
            Vec3f32[.WORLD](-0.3, 0.1, 1.0),
            estimator_seed=seed,
        )
        var actual: List[Float32] = [
            sampled.direction.x[0],
            sampled.direction.y[0],
            sampled.direction.z[0],
            sampled.weight.x[0],
            sampled.weight.y[0],
            sampled.weight.z[0],
            sampled.pdf[0],
            Float32(sampled.delta[0]),
            Float32(sampled.ok[0]),
            evaluated.value.x[0],
            evaluated.value.y[0],
            evaluated.value.z[0],
            evaluated.pdf[0],
            Float32(evaluated.delta[0]),
        ]
        for field in range(14):
            var reference = bitcast[.float32](expected[case_index * 14 + field])
            if field == 7 or field == 8 or field == 13:
                assert_equal(actual[field], reference)
            else:
                assert_almost_equal(
                    actual[field], reference, atol=2e-6, rtol=2e-5
                )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
