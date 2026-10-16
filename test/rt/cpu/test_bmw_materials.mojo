"""Conductor optics, layered sampling, and mixed-material rendering."""

from bajo.rt.material_parameters import CoatedParameters


from std.math import pi
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)
from bajo.core import Vec3f32, dot
from bajo.core import Point3f32, Rayf32
from bajo.core.random import Rng
from bajo.parser.pbrt import parse_pbrt
from bajo.rt import (
    CpuScene,
    SurfaceStore,
    Color,
    render_depth_first,
    render_wavefront,
)
from bajo.rt.cpu.bsdf import evaluate_bsdf, sample_bsdf
from bajo.rt.shading_types import ShadingPoint
from bajo.rt.coated_diffuse import (
    _fresnel_conductor,
    _fresnel_dielectric,
    _sample_substrate,
    _substrate_f,
    _substrate_pdf,
    _sample_coated_diffuse,
)


def test_complex_fresnel_normal_grazing_and_dielectric_limit() raises:
    var eta = Vec3f32[.WORLD](1.2, 0.95, 0.65)
    var k = Vec3f32[.WORLD](7.3, 6.7, 5.6)
    var normal = _fresnel_conductor(1.0, eta, k)
    for lane in range(3):
        var e = eta.x[0] if lane == 0 else eta.y[0] if lane == 1 else eta.z[0]
        var extinction = (
            k.x[0] if lane == 0 else k.y[0] if lane == 1 else k.z[0]
        )
        var expected = ((e - 1.0) * (e - 1.0) + extinction * extinction) / (
            (e + 1.0) * (e + 1.0) + extinction * extinction
        )
        var actual = (
            normal.x[0] if lane
            == 0 else normal.y[0] if lane
            == 1 else normal.z[0]
        )
        assert_almost_equal(actual, expected, atol=1.0e-6)
    var grazing = _fresnel_conductor(0.0, eta, k)
    assert_almost_equal(grazing.x[0], 1.0, atol=1.0e-6)
    for c in [0.1, 0.3, 0.7, 1.0]:
        var value = _fresnel_conductor(
            Float32(c), Vec3f32[.WORLD](1.5), Vec3f32[.WORLD](0.0)
        )
        assert_almost_equal(
            value.x[0], _fresnel_dielectric(Float32(c), 1.5), atol=1.0e-5
        )


def test_ggx_conductor_samples_match_bsdf_and_pdf() raises:
    var wo = Vec3f32[.WORLD](0.0, 0.0, 1.0)
    var eta = Vec3f32[.WORLD](1.2, 0.95, 0.65)
    var k = Vec3f32[.WORLD](7.3, 6.7, 5.6)
    var p = CoatedParameters[1]()
    p.conductor_eta = eta
    p.conductor_k = k
    p.conductor_roughness = 0.2
    var successes = 0
    for i in range(256):
        var sampled = _sample_substrate(
            wo,
            Color(1.0),
            p,
            Float32(i) / 256.0,
            Float32((i * 73) % 256) / 256.0,
        )
        if not sampled.ok[0]:
            continue
        successes += 1
        assert_false(sampled.delta[0])
        var value = _substrate_f(wo, sampled.direction, Color(1.0), p)
        var pdf = _substrate_pdf(wo, sampled.direction, p)
        assert_almost_equal(sampled.pdf[0], pdf, atol=1.0e-6)
        assert_almost_equal(
            sampled.weight.x[0],
            value.x[0] * sampled.direction.z[0] / pdf,
            atol=1.0e-6,
        )
        assert_true(sampled.weight.is_finite()[0])
        assert_true(sampled.direction.z[0] > 0.0)
    assert_true(successes > 200)
    var sampled = _sample_substrate(wo, Color(1.0), p, 0.0, 0.0)
    assert_almost_equal(
        sampled.pdf[0], Float32(1.0 / (4.0 * pi * 0.2 * 0.2)), atol=1.0e-5
    )
    p.conductor_roughness = 0.0
    var smooth = _sample_substrate(wo, Color(1.0), p, 0.2, 0.4)
    assert_true(smooth.delta[0])
    assert_equal(smooth.pdf[0], Float32(1.0))
    assert_almost_equal(
        smooth.weight.x[0], _fresnel_conductor(1.0, eta, k).x[0], atol=1.0e-6
    )


def test_coated_conductor_contains_both_interface_and_substrate_events() raises:
    var delta_count = 0
    var successes = 0
    var energy = Float32(0.0)
    for i in range(256):
        var sampled = _sample_coated_diffuse(
            Vec3f32[.WORLD](0.0, 0.0, -1.0),
            Vec3f32[.WORLD](0.0, 0.0, 1.0),
            Color(1.0),
            CoatedParameters[1](
                SIMD[.float32, 1](0.0),
                SIMD[.float32, 1](1.5),
                SIMD[.float32, 1](0.01),
                Color(0.0),
                SIMD[.float32, 1](0.0),
                SIMD[.float32, 1](10.0),
                SIMD[.float32, 1](2.0),
                Color(1.2, 0.95, 0.65) / 1.5,
                Color(7.3, 6.7, 5.6) / 1.5,
                SIMD[.float32, 1](0.2236068),
            ),
            SIMD[.float32, 1](Float32(i) / 256.0),
            SIMD[.float32, 1](Float32((i * 73) % 256) / 256.0),
        )
        if not sampled.ok[0]:
            continue
        successes += 1
        delta_count += Int(sampled.delta[0])
        assert_true(sampled.weight.is_finite()[0])
        assert_true(sampled.pdf[0] > 0.0)
        assert_true(sampled.direction.z[0] > 0.0)
        energy += sampled.weight.x[0]
    assert_true(successes > 200)
    assert_equal(delta_count, 11)
    assert_true(energy / 256.0 > 0.5 and energy / 256.0 <= 1.01)


def test_nested_mix_probabilities_and_endpoints() raises:
    var surfaces = SurfaceStore()
    var red = surfaces.add_lambertian(Color(1.0, 0.0, 0.0))
    var green = surfaces.add_lambertian(Color(0.0, 1.0, 0.0))
    var blue = surfaces.add_lambertian(Color(0.0, 0.0, 1.0))
    var inner = surfaces.add_mix(red, green, 0.5)
    var outer = surfaces.add_mix(inner, blue, 0.2)
    var counts = List[Int](length=3, fill=0)
    for i in range(10000):
        var selected = surfaces.resolve_mix(outer, (Float32(i) + 0.5) / 10000.0)
        counts[Int(selected.index())] += 1
    assert_equal(counts[0], 4000)
    assert_equal(counts[1], 4000)
    assert_equal(counts[2], 2000)
    var first_only = surfaces.add_mix(red, blue, 0.0)
    var second_only = surfaces.add_mix(red, blue, 1.0)
    for random in [0.0, 0.5, 0.99999]:
        assert_equal(
            surfaces.resolve_mix(first_only, Float32(random)).value[0],
            red.value[0],
        )
        assert_equal(
            surfaces.resolve_mix(second_only, Float32(random)).value[0],
            blue.value[0],
        )


def test_public_mix_bsdf_value_pdf_and_sample_weight() raises:
    var surfaces = SurfaceStore()
    var red = surfaces.add_lambertian(Color(1.0, 0.0, 0.0))
    var green = surfaces.add_lambertian(Color(0.0, 1.0, 0.0))
    var mixed = surfaces.add_mix(red, green, 0.25)
    var point = ShadingPoint[1](
        Point3f32[.WORLD](0.0), Vec3f32[.WORLD](0.0, 0.0, 1.0), True
    )
    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](0.0, 0.0, 1.0), Vec3f32[.WORLD](0.0, 0.0, -1.0)
    )
    var evaluated = evaluate_bsdf(
        mixed, surfaces, ray, point, Vec3f32[.WORLD](0.0, 0.0, 1.0)
    )
    assert_almost_equal(evaluated.value.x[0], Float32(0.75 / pi), atol=1.0e-6)
    assert_almost_equal(evaluated.value.y[0], Float32(0.25 / pi), atol=1.0e-6)
    assert_almost_equal(evaluated.pdf[0], Float32(1.0 / pi), atol=1.0e-6)
    var rng = Rng(UInt64(73), UInt64(0))
    for _ in range(64):
        var sampled = sample_bsdf(mixed, surfaces, ray, point, rng)
        assert_true(sampled.ok[0])
        assert_almost_equal(sampled.weight.x[0], 0.75, atol=1.0e-6)
        assert_almost_equal(sampled.weight.y[0], 0.25, atol=1.0e-6)
        assert_almost_equal(
            sampled.pdf[0], sampled.direction.z[0] / Float32(pi), atol=1.0e-6
        )


def test_mixed_and_coated_materials_agree_across_cpu_schedulers() raises:
    var scene = parse_pbrt(
        """Film "rgb" "integer xresolution" 12 "integer yresolution" 8
Sampler "independent" "integer pixelsamples" 8
LookAt 0 0 5 0 0 0 0 1 0
Camera "perspective" "float fov" 35
WorldBegin
LightSource "infinite" "rgb L" [1 1 1]
MakeNamedMaterial "red" "string type" "diffuse" "rgb reflectance" [.7 .05 .02]
MakeNamedMaterial "rim" "string type" "coatedconductor" "spectrum conductor.eta" "metal-Al-eta" "spectrum conductor.k" "metal-Al-k" "float conductor.roughness" .05
Material "mix" "string materials" ["red" "rim"] "float amount" .3
Shape "sphere"
"""
    )
    var settings = scene.settings.copy()
    var camera = scene.camera
    var world = CpuScene[4, 4](scene^.take_data())
    var depth = render_depth_first[
        .MIS, world_bvh_width=4, instance_bvh_width=4
    ](settings, camera, world)
    var packet = render_wavefront[.MIS, 4](settings, camera, world)
    var energy = Float32(0.0)
    for i in range(len(depth.pixels)):
        assert_true(packet.pixels[i].is_finite()[0])
        assert_almost_equal(
            packet.pixels[i].x[0], depth.pixels[i].x[0], atol=1.0e-5
        )
        assert_almost_equal(
            packet.pixels[i].y[0], depth.pixels[i].y[0], atol=1.0e-5
        )
        assert_almost_equal(
            packet.pixels[i].z[0], depth.pixels[i].z[0], atol=1.0e-5
        )
        energy += packet.pixels[i].x[0]
    assert_true(energy > 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
