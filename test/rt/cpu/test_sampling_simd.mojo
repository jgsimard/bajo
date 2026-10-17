"""Packet sampling preserves streams, distributions, UV addressing and masks."""

from std.math import floor
from std.memory import bitcast
from std.random import Random
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
)
from bajo.core import Affine3f32, Point3, Point3f32, Vec3
from bajo.core.random import Rng
from bajo.rt.common import path_stage_rng
from bajo.rt.render_types import Color, SamplingConfig
from bajo.rt.material_types import (
    Environment,
    ImageTexture,
    NO_TEXTURE,
    SurfaceId,
    SurfaceStore,
)
from bajo.rt.scene_data import SceneBuilder
from bajo.rt.cpu import CpuScene
from bajo.rt.cpu.lighting import _sample_direct_light_candidate
from bajo.rt.shading_types import ShadingPoint
from bajo.rt.lighting import _draw_alias_column, _resolve_alias_draw
from bajo.rt.lighting_types import _build_alias_distribution


def _streams[length: SIMDLength]() raises:
    var ids = SIMD[.uint64, length](0)
    for lane in range(Int(length)):
        ids[lane] = UInt64(lane) * 0x1234567887654321
    for seed in [UInt64(0), UInt64(2026), UInt64.MAX]:
        var rng = Rng[length](seed, ids)
        var expected = List[Float32]()
        for lane in range(Int(length)):
            var reference = Random[10](seed=seed, subsequence=ids[lane])
            for _ in range(5):
                var values = reference.step_uniform()
                for dim in range(4):
                    expected.append(values[dim])
        for dim in range(20):
            var values = rng.f32()
            for lane in range(Int(length)):
                assert_equal(values[lane], expected[lane * 20 + dim])
    for sampler in range(6):
        for stage in [
            UInt32(0),
            UInt32(9),
            UInt32(0x40000009),
            UInt32(0x80000009),
            UInt32(0xC0000009),
        ]:
            var config = SamplingConfig(
                2026, UInt32(sampler), 7, 100, 1024, 320
            )
            var paths = ids.cast[.uint32]()
            var expected = List[Float32]()
            for lane in range(Int(length)):
                var scalar = path_stage_rng(config, paths[lane], stage)
                for _ in range(20):
                    expected.append(scalar.f32())
            var wide = path_stage_rng(config, paths, stage)
            for dim in range(20):
                var values = wide.f32()
                for lane in range(Int(length)):
                    assert_equal(values[lane], expected[lane * 20 + dim])


def test_packet_random_streams_match_philox_and_all_samplers() raises:
    _streams[1]()
    _streams[4]()
    _streams[8]()
    _streams[16]()


def _textures[length: SIMDLength]() raises:
    var surfaces = SurfaceStore()
    var image = surfaces.add_image_texture(
        ImageTexture(2, 2, [0, 0, 0, 2, 1, 0.5, 0.2, 0.4, 0.6, 1, 1, 1])
    )
    var constant = surfaces.add_image_texture(
        ImageTexture(1, 1, [0.1, 0.3, 0.7])
    )
    var matte = surfaces.add_lambertian(Color(0.75), image)
    var metal = surfaces.add_metal(Color(0.4), 0.2, constant)
    var coat = surfaces.add_coated_diffuse(
        Color(0.6), 0.2, 1.5, image, image, 0.25, 2.0, 0.5
    )
    surfaces.coated_diffuses[0].roughness_texture_index = image
    surfaces.coated_diffuses[0].roughness_scale = 0.3
    var glass = surfaces.add_dielectric(1.5)
    surfaces.dielectrics[0].displacement_texture_index = constant
    surfaces.dielectrics[0].displacement_scale = 0.2
    var bare = surfaces.add_lambertian(Color(0.8))
    var choices = [
        matte.value[0],
        metal.value[0],
        coat.value[0],
        glass.value[0],
        bare.value[0],
    ]
    for offset in range(9):
        var ids = SIMD[.uint32, length](0)
        var u = SIMD[.float32, length](0)
        var v = u
        var active = SIMD[.bool, length](fill=True)
        var normal = Vec3[.float32, .WORLD, length](0.0, 0.0, 1.0)
        for lane in range(Int(length)):
            ids[lane] = choices[(lane + offset) % 5]
            u[lane] = Float32(lane - offset) * 0.25
            v[lane] = Float32(lane + offset) * -0.13
            active[lane] = (lane + offset) % 4 != 0
            if not active[lane]:
                ids[lane] = UInt32.MAX
                u[lane] = bitcast[.float32](UInt32(0x7FC00000))
                v[lane] = bitcast[.float32](UInt32(0x7F800000))
        var albedo = surfaces.sample_albedo(
            SurfaceId[length](ids), u, v, active
        )
        var bumped = surfaces.shading_normal(
            SurfaceId[length](ids), normal, u, v, active
        )
        var coated = active & ids.eq(coat.value[0])
        var roughness = surfaces.sample_conductor_roughness(
            SurfaceId[length](ids), u, v, coated
        )
        for lane in range(Int(length)):
            if not active[lane]:
                assert_equal(albedo.x[lane], Float32(0))
                assert_equal(bumped.z[lane], Float32(1))
                continue
            var surface = SurfaceId.from_raw(ids[lane])
            var scalar = surfaces.sample_albedo(surface, u[lane], v[lane])
            assert_equal(albedo.x[lane], scalar.x[0])
            assert_equal(albedo.y[lane], scalar.y[0])
            assert_equal(albedo.z[lane], scalar.z[0])
            var expected = surfaces.shading_normal(
                surface, Vec3[.float32, .WORLD, 1](0, 0, 1), u[lane], v[lane]
            )
            assert_almost_equal(bumped.x[lane], expected.x[0], atol=1e-6)
            assert_almost_equal(bumped.y[lane], expected.y[0], atol=1e-6)
            assert_almost_equal(bumped.z[lane], expected.z[0], atol=1e-6)
            if coated[lane]:
                assert_equal(
                    roughness[lane],
                    surfaces.sample_conductor_roughness(
                        surface, u[lane], v[lane]
                    ),
                )
    # Independent reference for repeat addressing at UV seams and negatives.
    for n in range(-64, 65):
        var u = Float32(n) / 16.0
        var v = Float32(n * 3) / 17.0
        var x = min(Int((u - floor(u)) * 2), 1)
        var y = min(Int((1 - (v - floor(v))) * 2), 1)
        var sampled = surfaces.image_textures[0].sample(u, v)
        assert_equal(
            sampled.x[0], surfaces.image_textures[0].pixels[3 * (2 * y + x)]
        )


def test_packet_texture_and_bump_masks_and_mixed_images() raises:
    _textures[1]()
    _textures[4]()
    _textures[8]()
    _textures[16]()


def test_alias_distribution_preserves_sparse_hdr_weights_and_zero_boundary() raises:
    var weights: List[Float32] = [0, 1e-10, 1, 10000, 0, 3]
    var total = Float64(0)
    for weight in weights:
        total += Float64(weight)
    var probabilities = List[Float32]()
    var indices = List[UInt32]()
    _build_alias_distribution(Span(weights), total, probabilities, indices)
    var reconstructed = List[Float64](length=len(weights), fill=0)
    for column in range(len(weights)):
        assert_true(probabilities[column] >= 0 and probabilities[column] <= 1)
        assert_true(Int(indices[column]) < len(weights))
        reconstructed[column] += Float64(probabilities[column]) / Float64(
            len(weights)
        )
        reconstructed[Int(indices[column])] += (
            1 - Float64(probabilities[column])
        ) / Float64(len(weights))
    for i in range(len(weights)):
        assert_almost_equal(
            reconstructed[i], Float64(weights[i]) / total, atol=1e-8
        )
    var draw = _draw_alias_column(Float32(0), len(weights))
    assert_true(_resolve_alias_draw(draw, probabilities[0], indices[0]) != 0)
    draw = _draw_alias_column(Float32(1), len(weights))
    assert_equal(Int(draw.column), len(weights) - 1)
    var zeros = List[Float32](length=3, fill=0)
    _build_alias_distribution(Span(zeros), 0, probabilities, indices)
    assert_equal(len(probabilities), 0)
    assert_equal(len(indices), 0)


def _direct_lights[length: SIMDLength](image_environment: Bool) raises:
    var builder = SceneBuilder()
    var emitter = builder.add_emissive(Color(3, 2, 1))
    builder.add_sphere(Point3f32[.WORLD](0, 3, 0), 1, emitter)
    builder.add_triangle(
        Point3f32[.WORLD](-3, 4, -3),
        Point3f32[.WORLD](3, 4, -3),
        Point3f32[.WORLD](0, 4, 3),
        emitter,
    )
    if image_environment:
        var texture = builder.surfaces.add_image_texture(
            ImageTexture(2, 1, [0, 0, 0, 3, 2, 1])
        )
        builder.set_environment(
            Environment.image(
                texture,
                Color(1),
                Affine3f32[.WORLD, .LOCAL].identity(),
                Affine3f32[.LOCAL, .WORLD].identity(),
            )
        )
    else:
        builder.set_environment(Environment.uniform(Color(0.25)))
    var world = CpuScene[4, 4](builder^.finish())
    var point = ShadingPoint[length](
        Point3[.float32, .WORLD, length](0),
        Vec3[.float32, .WORLD, length](0, 1, 0),
        SIMD[.bool, length](fill=True),
    )
    for sampler in range(6):
        var config = SamplingConfig(2026, UInt32(sampler), 7, 100, 1024, 320)
        for packet in range(20):
            var ids = SIMD[.uint32, length](0)
            var active = SIMD[.bool, length](fill=True)
            for lane in range(Int(length)):
                ids[lane] = UInt32(packet * Int(length) + lane)
                active[lane] = (packet + lane) % 5 != 0
            var rng = path_stage_rng(config, ids, UInt32(0x80000001))
            var sampled = _sample_direct_light_candidate(
                world, point, rng, active
            )
            for lane in range(Int(length)):
                var scalar_rng = path_stage_rng(
                    config, ids[lane], UInt32(0x80000001)
                )
                var scalar_point = ShadingPoint[1](
                    Point3f32[.WORLD](0),
                    Vec3[.float32, .WORLD, 1](0, 1, 0),
                    True,
                )
                var expected = _sample_direct_light_candidate(
                    world, scalar_point, scalar_rng
                )
                assert_equal(
                    sampled.valid[lane], expected.valid[0] and active[lane]
                )
                if sampled.valid[lane]:
                    assert_almost_equal(
                        sampled.direction.x[lane],
                        expected.direction.x[0],
                        atol=2e-6,
                    )
                    assert_almost_equal(
                        sampled.direction.y[lane],
                        expected.direction.y[0],
                        atol=2e-6,
                    )
                    assert_almost_equal(
                        sampled.direction.z[lane],
                        expected.direction.z[0],
                        atol=2e-6,
                    )
                    assert_almost_equal(
                        sampled.light_pdf[lane],
                        expected.light_pdf[0],
                        atol=2e-5,
                        rtol=1e-5,
                    )
                    assert_equal(
                        sampled.emission.x[lane], expected.emission.x[0]
                    )


def test_packet_direct_lights_match_scalar_with_mixed_emitter_kinds() raises:
    _direct_lights[1](False)
    _direct_lights[4](False)
    _direct_lights[8](True)
    _direct_lights[16](True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
