"""Geometry emission preserves scattering, scoped bindings, and light PDFs."""

from std.math import pi
from std.memory import bitcast
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
    assert_raises,
)
from bajo.core import Point3f32, Vec3f32, Rayf32, Point3, Vec3, Ray
from bajo.core.color import Color
from bajo.core.random import Rng
from bajo.parser.pbrt import parse_pbrt
from bajo.rt import CpuScene, SceneBuilder, render_wavefront
from bajo.rt.scene_description import SceneDescription
from bajo.rt.emission import AreaEmitter
from bajo.rt.material_types import PrimitiveId, Integrator
from bajo.rt.shading_types import ShadingPoint
from bajo.rt.cpu.lighting import (
    _sample_direct_light_candidate,
    light_pdf_for_emissive_hit,
)
from bajo.rt.lighting import _emissive_hit_light_pdf
from bajo.rt.rays import RT_SHADOW_END_OFFSET


def make_emissive_plane(
    two_sided: Bool = False, back: Bool = False, model: String = "diffuse"
) raises -> SceneDescription:
    var source = """Film "rgb" "integer xresolution" 4 "integer yresolution" 3
Sampler "halton" "integer pixelsamples" 32
Integrator "path" "integer maxdepth" 3
"""
    source += (
        "LookAt 0 0 -1 0 0 0 0 1 0\n" if back else "LookAt 0 0 1 0 0 0 0 1 0\n"
    )
    source += """Camera "perspective" "float fov" 10
WorldBegin
LightSource "infinite" "rgb L" [.4 .2 .1]
"""
    if model == "dielectric":
        source += (
            'Material "dielectric" "float eta" 1.5 "float roughness" .15\n'
        )
    else:
        source += 'Material "' + model + '" "rgb reflectance" [.25 .5 .75]\n'
    source += """
AreaLightSource "diffuse" "rgb L" [.1 .2 .3] "bool twosided" """
    source += '"true"\n' if two_sided else '"false"\n'
    source += """Shape "trianglemesh" "point3 P" [-2 -2 0 2 -2 0 0 2 0]
 "point2 uv" [0 0 1 0 .5 1]
"""
    return parse_pbrt(source)


def test_emissive_diffuse_retains_bsdf_and_reflects_the_environment() raises:
    for two_sided in [False, True]:
        for back in [False, True]:
            var parsed = make_emissive_plane(two_sided, back)
            assert_equal(len(parsed.data.surfaces().emissives), 0)
            assert_equal(len(parsed.data.emission().records), 1)
            assert_equal(
                parsed.data.triangle_instance_surfaces()[0].kind().value,
                UInt32(0),
            )
            var settings = parsed.settings.copy()
            var camera = parsed.camera
            var world = CpuScene[4, 4](parsed^.take_data())
            var image = render_wavefront[.PATH, 4](settings, camera, world)
            var expected = Color(0.1, 0.1, 0.075)
            if two_sided or not back:
                expected += Color(0.1, 0.2, 0.3)
            for pixel in image.pixels:
                comptime for channel in range(3):
                    assert_almost_equal(
                        pixel[channel], expected[channel], atol=1e-6
                    )


def test_area_light_scope_preserves_shared_material_and_sidedness() raises:
    var parsed = parse_pbrt(
        """WorldBegin
Material "diffuse" "rgb reflectance" [.25 .5 .75]
Shape "sphere"
AttributeBegin
 AreaLightSource "diffuse" "rgb L" [2 3 4] "bool twosided" "true"
 Translate 3 0 0
 Shape "sphere"
AttributeEnd
Translate -3 0 0
Shape "sphere"
"""
    )
    ref data = parsed.data
    assert_equal(len(data.emission().records), 1)
    assert_equal(
        data.sphere_surfaces()[0].value, data.sphere_surfaces()[1].value
    )
    assert_equal(
        data.sphere_surfaces()[0].value, data.sphere_surfaces()[2].value
    )
    ref emitter = data.emission().records[PrimitiveId(.SPHERE, 1).value]
    assert_equal(emitter.two_sided, True)
    assert_almost_equal(emitter.radiance.y, 3.0)
    assert_equal(len(data.lights().records), 1)
    assert_almost_equal(data.lights().records[0].radiance.y, 3.0)
    assert_almost_equal(
        data.lights().records[0].weight, 8.0 * pi * 3.0, atol=1e-4
    )


def test_two_sided_direct_light_and_hit_pdfs_agree_from_the_back() raises:
    for two_sided in [False, True]:
        var source = (
            'WorldBegin\nMaterial "diffuse" "rgb reflectance" [0 0'
            ' 0]\nAreaLightSource "diffuse" "rgb L" [4 4 4] "bool twosided" '
        )
        source += '"true"\n' if two_sided else '"false"\n'
        source += 'Shape "trianglemesh" "point3 P" [-1 -1 2 1 -1 2 0 1 2]\n'
        var parsed = parse_pbrt(source)
        var world = CpuScene[4, 4](parsed^.take_data())
        var point = ShadingPoint[1](
            Point3f32[.WORLD](0), Vec3f32[.WORLD](0, 0, 1), True
        )
        var rng = Rng[1](17, UInt64(0))
        var sample = _sample_direct_light_candidate(world, point, rng)
        assert_equal(sample.valid[0], two_sided)
        if two_sided:
            var distance = sample.shadow_t_max + RT_SHADOW_END_OFFSET
            var pdf = _emissive_hit_light_pdf(
                sample.direction,
                distance,
                Vec3f32[.WORLD](0, 0, 1),
                Color(4.0),
                world.scene_data().total_light_weight(),
                True,
            )
            assert_almost_equal(sample.light_pdf, pdf, atol=1e-6)
            var ray = Rayf32[.WORLD](point.p, sample.direction)
            var hit = world.trace_surface(ray)
            assert_equal(hit.emission_two_sided[0], True)
            assert_almost_equal(
                light_pdf_for_emissive_hit(world, ray, hit), pdf, atol=1e-5
            )


def test_invalid_geometry_emission_is_rejected() raises:
    for invalid in [0, 1, 2]:
        var builder = SceneBuilder()
        var matte = builder.add_lambertian(Color(0.5))
        builder.add_sphere(Point3f32[.WORLD](0), 1.0, matte)
        var key = PrimitiveId(.SPHERE, UInt32(10 if invalid == 0 else 0)).value
        var radiance = (
            Color(-1.0) if invalid
            == 1 else Color(bitcast[.float32](UInt32(0x7F800000))) if invalid
            == 2 else Color(1.0)
        )
        builder.emission.records[key] = AreaEmitter(radiance)
        with assert_raises():
            _ = builder^.finish()


def test_emission_packet_merging_and_scalar_alpha_path_keep_geometry_identity() raises:
    for alpha_path in [False, True]:
        var source = """WorldBegin
Material "diffuse" "rgb reflectance" [.5 .5 .5]
AttributeBegin
 Translate -3 0 0
 AreaLightSource "diffuse" "rgb L" [1 0 0]
 Shape "sphere" "float radius" .5
AttributeEnd
AttributeBegin
 AreaLightSource "diffuse" "rgb L" [0 2 0]
 Shape "trianglemesh" "point3 P" [-1 -1 0 1 -1 0 0 1 0]
AttributeEnd
AttributeBegin
 Translate 3 0 0
 AreaLightSource "diffuse" "rgb L" [0 0 3]
 Shape "trianglemesh" "point3 P" [-1 -1 0 1 -1 0 0 1 0]
  "point2 uv" [0 0 1 0 .5 1]
AttributeEnd
"""
        if alpha_path:
            source += (
                'Translate 100 0 0\nShape "trianglemesh" "float alpha" 0'
                ' "point3 P" [-1 -1 0 1 -1 0 0 1 0]\n'
            )
        var parsed = parse_pbrt(source)
        assert_equal(len(parsed.data.emission().records), 3)
        var world = CpuScene[4, 4](parsed^.take_data())
        var rays = Ray[.float32, .WORLD, 4](
            Point3[.float32, .WORLD, 4](
                SIMD[.float32, 4](-3, 0, 3, 9999), 0.1, 2
            ),
            Vec3[.float32, .WORLD, 4](0, 0, -1),
        )
        var hits = world.trace_surface(
            rays, SIMD[.bool, 4](True, True, True, False)
        )
        for lane in range(4):
            assert_equal(hits.hit[lane], lane < 3)
            var scalar = hits.get(lane)
            comptime for channel in range(3):
                var expected = Float32(
                    lane + 1
                ) if lane == channel else Float32(0.0)
                assert_equal(hits.emission[channel][lane], expected)
                assert_equal(scalar.emission[channel], expected)


def make_depth_scene(
    diffuse: Bool = False, miss: Bool = False
) raises -> SceneDescription:
    var source = """Film "rgb" "integer xresolution" 3 "integer yresolution" 2
Sampler "halton" "integer pixelsamples" 8
LookAt 0 0 3 0 0 0 0 1 0
Camera "perspective" "float fov" 5
WorldBegin
LightSource "infinite" "rgb L" [.7 .8 .9]
"""
    if miss:
        source += "Translate 100 0 0\n"
    source += (
        'Material "diffuse" "rgb reflectance" [.25 .5 .75]\n' if diffuse else 'Material "dielectric" "float eta" 1 "float roughness" 0\n'
    )
    source += """AreaLightSource "diffuse" "rgb L" [.1 .2 .3]
Shape "trianglemesh" "point3 P" [-20 -20 2 20 -20 2 0 20 2]
 "point2 uv" [0 0 1 0 .5 1]
"""
    if not diffuse:
        source += """AreaLightSource "diffuse" "rgb L" [.4 .5 .6]
Shape "trianglemesh" "point3 P" [-20 -20 1 20 -20 1 0 20 1]
 "point2 uv" [0 0 1 0 .5 1]
"""
    return parse_pbrt(source)


def depth_scene_expected(depth: Int, diffuse: Bool, miss: Bool) -> Color:
    if miss:
        return Color(0.7, 0.8, 0.9)
    var expected = Color(0.1, 0.2, 0.3)
    if depth > 0 and not diffuse:
        expected += Color(0.4, 0.5, 0.6)
        if depth > 1:
            expected += Color(0.7, 0.8, 0.9)
    return expected


def test_depth_limits_preserve_terminal_emission_without_scattering() raises:
    for scenario in range(3):
        var diffuse = scenario == 1
        var miss = scenario == 2
        var parsed = make_depth_scene(diffuse, miss)
        var settings = parsed.settings.copy()
        var camera = parsed.camera
        var world = CpuScene[4, 4](parsed^.take_data())
        for depth in range(3 if not diffuse else 1):
            settings.max_depth = depth
            var expected = depth_scene_expected(depth, diffuse, miss)
            comptime for integrator in [
                Integrator.PATH,
                Integrator.NEE,
                Integrator.MIS,
            ]:
                comptime for width in [1, 4]:
                    var result = render_wavefront[integrator, width](
                        settings, camera, world
                    )
                    for pixel in result.pixels:
                        comptime for channel in range(3):
                            assert_almost_equal(
                                pixel[channel], expected[channel], atol=1e-6
                            )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
