"""Coated-conductor reflectance parameterization and optical defaults."""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_raises,
    assert_true,
)
from bajo.core import Vec3, Vec3f32
from bajo.core.optics import conductor_k_from_reflectance, fresnel_conductor
from bajo.core.color import Color
from bajo.parser import MemoryAssetLoader
from bajo.parser.pbrt import parse_pbrt, read_pbrt
from bajo.parser.pbrt.spectrum import _named_optical_constant
from bajo.rt.material_storage import pack_coated_materials
from bajo.rt.material_types import SurfaceStore, ImageTexture, NO_TEXTURE
from bajo.rt.cpu.bsdf import _MaterialParameters
from bajo.rt.material_parameters import CoatedParameters
from bajo.rt.coated_diffuse import _layered_pdf, _evaluate_coated_diffuse
from bajo.rt.scene_description import SceneDescription
from test.parser.png.test_png import _encode


def test_reflectance_conductor_normal_incidence_and_white_limit() raises:
    var r = Color(0.0, 0.25, 0.75)
    var k = conductor_k_from_reflectance(r)
    var recovered = fresnel_conductor(Float32(1.0), Color(1.0), k)
    comptime for channel in range(3):
        assert_almost_equal(recovered[channel], r[channel], atol=1e-6)
    k = conductor_k_from_reflectance(Color(1.0))
    assert_true(k.is_finite()[0])
    recovered = fresnel_conductor(Float32(1.0), Color(1.0), k)
    assert_almost_equal(recovered.x, 0.9999, atol=1e-6)


def test_coated_reflectance_and_optical_constants_are_exclusive() raises:
    for name in ["rgb conductor.eta", "rgb conductor.k"]:
        with assert_raises():
            _ = parse_pbrt(
                'WorldBegin\nMaterial "coatedconductor" "rgb reflectance" [.2'
                ' .4 .6] "'
                + name
                + '" [1 1 1]\nShape "sphere"'
            )


def test_coated_conductor_defaults_retain_original_copper_spectra() raises:
    var parsed = parse_pbrt(
        'WorldBegin\nMaterial "coatedconductor"\nShape "sphere"'
    )
    ref surfaces = parsed.data.surfaces()
    ref material = surfaces.coated_diffuses[0]
    assert_equal(material.conductor_reflectance, False)
    assert_equal(len(surfaces.optical_spectra), 2)
    assert_true(material.conductor_eta_spectrum != NO_TEXTURE)
    assert_true(material.conductor_k_spectrum != NO_TEXTURE)
    var eta = _named_optical_constant("metal-Cu-eta") / 1.5
    var k = _named_optical_constant("metal-Cu-k") / 1.5
    comptime for channel in range(3):
        assert_almost_equal(
            material.conductor_eta[channel], eta[channel], atol=1e-6
        )
        assert_almost_equal(
            material.conductor_k[channel], k[channel], atol=1e-6
        )
    assert_almost_equal(
        surfaces.optical_spectra[Int(material.conductor_eta_spectrum)].evaluate(
            Float32(550.0)
        ),
        _named_optical_constant("metal-Cu-eta").y,
        atol=1e-6,
    )


def _check_material_packet[length: SIMDLength]() raises:
    var surfaces = SurfaceStore()
    var texture = surfaces.add_image_texture(
        ImageTexture(2, 1, [0.1, 0.4, 0.8, 0.6, 0.3, 0.2])
    )
    var from_r = surfaces.add_coated_conductor_reflectance(
        Color(0.8), 0.2, 0.1, 1.5, texture_index=texture
    )
    var explicit = surfaces.add_coated_conductor(
        Color(0.2, 0.3, 0.4), Color(2.0, 3.0, 4.0), 0.3, 0.2, 1.4
    )
    var ids = SIMD[.uint32, length](0)
    var u = SIMD[.float32, length](0.25)
    var active = SIMD[.bool, length](fill=True)
    for lane in range(Int(length)):
        ids[lane] = from_r.index() if lane % 3 != 1 else explicit.index()
        u[lane] = 0.25 if lane % 2 else 0.75
        if lane % 4 == 3:
            active[lane] = False
            ids[lane] = UInt32.MAX
    var columns = pack_coated_materials[True](surfaces)
    for prepared in [False, True]:
        var parameters = _MaterialParameters[length]()
        if prepared:
            parameters.load[.COATED_DIFFUSE](
                surfaces, ids, u, SIMD[.float32, length](0.5), active, columns
            )
        else:
            parameters.load[.COATED_DIFFUSE](
                surfaces, ids, u, SIMD[.float32, length](0.5), active
            )
        for lane in range(Int(length)):
            if not active[lane]:
                continue
            var expected_eta: Color
            var expected_k: Color
            if ids[lane] == from_r.index():
                var r = Color(0.8) * surfaces.image_textures[0].sample(
                    u[lane], 0.5
                )
                expected_eta = Color(1.0 / 1.5)
                expected_k = conductor_k_from_reflectance(r) / 1.5
            else:
                expected_eta = Color(0.2, 0.3, 0.4) / 1.4
                expected_k = Color(2.0, 3.0, 4.0) / 1.4
            comptime for channel in range(3):
                assert_almost_equal(
                    parameters.coated.conductor_eta[channel][lane],
                    expected_eta[channel],
                    atol=1e-6,
                )
                assert_almost_equal(
                    parameters.coated.conductor_k[channel][lane],
                    expected_k[channel],
                    atol=1e-6,
                )


def test_textured_reflectance_scalar_and_masked_packet_optics() raises:
    comptime for width in [1, 4, 8, 16]:
        _check_material_packet[width]()


def make_reflectance_scene(
    interface_eta: String = "1.5",
) raises -> SceneDescription:
    var loader = MemoryAssetLoader()
    loader.add_file(
        "main.pbrt",
        """Film "rgb" "integer xresolution" 4 "integer yresolution" 3
Sampler "halton" "integer pixelsamples" 64
Integrator "path" "integer maxdepth" 5
LookAt 0 0 3 0 0 0 0 1 0
Camera "perspective" "float fov" 10
WorldBegin
LightSource "infinite" "rgb L" [1 .8 .6]
Texture "r" "spectrum" "imagemap" "string filename" "r.png"
Material "coatedconductor" "texture reflectance" "r"
 "float conductor.roughness" .15 "float interface.roughness" .1
 "float interface.eta" """
        + interface_eta
        + """
Shape "trianglemesh" "point3 P" [-5 -5 0 5 -5 0 0 5 0]
 "point2 uv" [0 0 1 0 .5 1]
""",
    )
    loader.add_image_file(
        "r.png", _encode([UInt8(0), 64, 128, 192, 192, 128, 64], 2, 1, color=2)
    )
    return read_pbrt("main.pbrt", loader)


def test_parser_retains_textured_reflectance_binding() raises:
    var parsed = make_reflectance_scene()
    ref material = parsed.data.surfaces().coated_diffuses[0]
    assert_equal(material.conductor_reflectance, True)
    assert_equal(material.texture_index, UInt32(0))
    assert_equal(material.conductor_eta_spectrum, NO_TEXTURE)
    assert_equal(material.conductor_k_spectrum, NO_TEXTURE)


def test_matching_interface_ignores_roughness_in_layered_value_and_pdf() raises:
    var p = CoatedParameters[1]()
    p.eta = 1.0
    p.thickness = 0.01
    p.conductor_eta = Color(1.0)
    p.conductor_k = conductor_k_from_reflectance(Color(0.2, 0.5, 0.8))
    p.conductor_roughness = 0.2
    p.roughness = 0.0
    var smooth = p
    p.roughness = 0.5
    var wo = Vec3f32[.WORLD](0, 0, 1)
    var wi = Vec3f32[.WORLD](0.6, 0, 0.8)
    for seed in range(64):
        var reference_pdf = _layered_pdf(wo, wi, smooth, UInt32(seed), True)
        var actual_pdf = _layered_pdf(wo, wi, p, UInt32(seed), True)
        assert_almost_equal(actual_pdf, reference_pdf, atol=1e-6)
        var reference = _evaluate_coated_diffuse(
            -wo, wo, Color(1.0), smooth, wi, estimator_seed=UInt32(seed)
        )
        var actual = _evaluate_coated_diffuse(
            -wo, wo, Color(1.0), p, wi, estimator_seed=UInt32(seed)
        )
        comptime for channel in range(3):
            assert_almost_equal(
                actual.value[channel], reference.value[channel], atol=1e-6
            )
        assert_almost_equal(actual.pdf, reference.pdf, atol=1e-6)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
