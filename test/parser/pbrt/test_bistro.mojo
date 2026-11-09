"""Bistro input features, checked without downloading scene assets."""

from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

from bajo.core import Point3f32, Rayf32, Vec3f32
from bajo.parser.pbrt import parse_pbrt, read_pbrt
from bajo.parser import MemoryAssetLoader, AssetLoader
from bajo.parser.ply import PlyMesh
from bajo.rt.material_types import ImageTexture
from bajo.rt import CpuScene
from bajo.core.random import Sampler
from bajo.core.spectrum import PiecewiseLinearSpectrum
from bajo.parser.pbrt.optical_data import named_optical_spectrum
from test.parser.png.test_png import _encode


struct _NormalMapLoader(AssetLoader):
    var source: String

    def __init__(out self, source: String):
        self.source = source

    def read_text(self, path: String) raises -> String:
        return self.source

    def read_ply_mesh(self, path: String) raises -> PlyMesh:
        raise Error("unexpected normal-map fixture mesh read")

    def read_image_texture(self, path: String) raises -> ImageTexture:
        raise Error("normal-map fixture must use linear data decoding")

    def read_data_texture(self, path: String) raises -> ImageTexture:
        if path != "scene/normal.png":
            raise Error("incorrect normal-map path: " + path)
        return ImageTexture(1, 1, [0.75, 0.5, 1.0])


def test_zsobol_sampler_retains_sequence_length_and_seed() raises:
    var source = (
        'Film "rgb" "integer xresolution" 9 "integer yresolution" 73\n'
        'Sampler "zsobol" "integer pixelsamples" 256\n'
        'WorldBegin\nShape "sphere"\n'
    )
    var scene = parse_pbrt(source)
    assert_equal(scene.settings.sampler, Sampler.Z_SOBOL)
    assert_equal(scene.settings.samples_per_pixel, 256)
    assert_equal(scene.settings.sample_sequence_length, 256)
    assert_equal(scene.settings.rng_seed, UInt64(0))
    scene.settings.validate()
    for seed in [0, 31, -1]:
        var seeded = parse_pbrt(
            'Sampler "zsobol" "integer pixelsamples" 3 "integer seed" '
            + String(seed)
            + ' "string randomization" "fastowen"\nWorldBegin\nShape "sphere"'
        )
        assert_equal(seeded.settings.sampler, Sampler.Z_SOBOL)
        assert_equal(seeded.settings.samples_per_pixel, 3)
        assert_equal(seeded.settings.rng_seed, UInt64(UInt32(seed)))
    with assert_raises():
        _ = parse_pbrt(
            'Sampler "zsobol" "string randomization" "owen"\n'
            'WorldBegin\nShape "sphere"'
        )


def test_inline_mesh_interpolates_uvs_and_inverse_transpose_normals() raises:
    var scene = parse_pbrt(
        """WorldBegin
Scale 2 3 1
Shape "trianglemesh"
 "integer indices" [0 1 2]
 "point3 P" [0 0 0 1 0 0 0 1 0]
 "normal N" [0 0 1 .6 0 .8 0 .6 .8]
 "point2 uv" [0 0 1 0 0 1]
"""
    )
    assert_equal(len(scene.data.triangle_meshes()), 1)
    assert_equal(len(scene.data.triangle_instances()), 1)
    assert_equal(len(scene.data.triangle_mesh_normals()[0]), 9)
    assert_equal(len(scene.data.triangle_mesh_texcoords()[0]), 6)
    var world = CpuScene[4, 4](scene^.take_data())
    var hit = world.trace_surface(
        Rayf32[.WORLD](
            Point3f32[.WORLD](0.4, 0.9, 2.0), Vec3f32[.WORLD](0.0, 0.0, -1.0)
        )
    )
    assert_equal(hit.hit[0], True)
    assert_almost_equal(hit.uv_u[0], 0.2, atol=1e-6)
    assert_almost_equal(hit.uv_v[0], 0.3, atol=1e-6)
    var magnitude = sqrt(Float32(0.06 * 0.06 + 0.06 * 0.06 + 0.9 * 0.9))
    assert_almost_equal(hit.normal.x[0], 0.06 / magnitude, atol=1e-6)
    assert_almost_equal(hit.normal.y[0], 0.06 / magnitude, atol=1e-6)
    assert_almost_equal(hit.normal.z[0], 0.9 / magnitude, atol=1e-6)


def test_inline_mesh_reversal_keeps_corner_attributes_attached() raises:
    var scene = parse_pbrt(
        """WorldBegin
ReverseOrientation
Shape "trianglemesh"
 "integer indices" [0 1 2]
 "point3 P" [0 0 0 1 0 0 0 1 0]
 "normal N" [0 0 1 .6 0 .8 0 .6 .8]
 "point2 uv" [0 0 1 0 0 1]
"""
    )
    ref positions = scene.data.triangle_meshes()[0]
    ref normals = scene.data.triangle_mesh_normals()[0]
    ref uvs = scene.data.triangle_mesh_texcoords()[0]
    assert_equal(positions[1].y[0], Float32(1.0))
    assert_equal(positions[2].x[0], Float32(1.0))
    assert_almost_equal(normals[4], 0.6)
    assert_almost_equal(normals[6], 0.6)
    assert_equal(uvs[2], Float32(0.0))
    assert_equal(uvs[3], Float32(1.0))
    assert_equal(uvs[4], Float32(1.0))
    assert_equal(uvs[5], Float32(0.0))
    var world = CpuScene[4, 4](scene^.take_data())
    var hit = world.trace_surface(
        Rayf32[.WORLD](
            Point3f32[.WORLD](0.2, 0.3, -2.0), Vec3f32[.WORLD](0.0, 0.0, 1.0)
        )
    )
    assert_equal(hit.hit[0], True)
    assert_equal(hit.front_face[0], True)
    assert_almost_equal(hit.uv_u[0], 0.2, atol=1e-6)
    assert_almost_equal(hit.uv_v[0], 0.3, atol=1e-6)


def test_inline_unindexed_mesh_uv_aliases_and_optional_normals() raises:
    for declaration in ['"point2 uv"', '"float uv"', '"float st"']:
        var scene = parse_pbrt(
            'WorldBegin\nShape "trianglemesh" "point3 P" [0 0 0 1 0 0 0 1 0] '
            + declaration
            + " [.1 .2 .3 .4 .5 .6]"
        )
        assert_equal(len(scene.data.triangle_meshes()[0]), 3)
        assert_equal(len(scene.data.triangle_mesh_normals()[0]), 0)
        assert_almost_equal(scene.data.triangle_mesh_texcoords()[0][5], 0.6)
    var normal_scene = parse_pbrt(
        'WorldBegin\nShape "trianglemesh" '
        '"point3 P" [0 0 0 1 0 0 0 1 0] '
        '"normal3 N" [0 0 1 0 0 1 0 0 1]'
    )
    assert_equal(len(normal_scene.data.triangle_mesh_normals()[0]), 9)
    assert_equal(len(normal_scene.data.triangle_mesh_texcoords()[0]), 0)


def test_inline_mesh_rejects_mismatched_attributes_and_bad_indices() raises:
    for parameters in [
        '"normal N" [0 0 1]',
        '"point2 uv" [0 0 1 0]',
        '"normal N" [0 0 1 0 0 1 0 0 1] "integer indices" [0 1 3]',
        '"point2 uv" [0 0 1 0 0 1] "integer indices" [-1 1 2]',
        '"point2 uv" [0 0 1 bad 0 1]',
    ]:
        with assert_raises():
            _ = parse_pbrt(
                'WorldBegin\nShape "trianglemesh" '
                '"point3 P" [0 0 0 1 0 0 0 1 0] '
                + parameters
            )


def test_brass_original_spectra_and_wavelength_interpolation() raises:
    var eta = named_optical_spectrum("metal-CuZn-eta")
    var k = named_optical_spectrum("metal-CuZn-k")
    assert_equal(len(eta.samples), 122)
    assert_equal(len(k.samples), 122)
    assert_almost_equal(eta.evaluate(290.0)[0], 1.358)
    assert_almost_equal(eta.evaluate(550.0)[0], 0.527)
    assert_almost_equal(eta.evaluate(555.0)[0], 0.516)
    assert_almost_equal(k.evaluate(550.0)[0], 2.765)
    assert_almost_equal(k.evaluate(555.0)[0], 2.8125)
    assert_almost_equal(k.evaluate(890.0)[0], 5.575)
    assert_equal(eta.evaluate(289.0)[0], Float32(0.0))
    assert_equal(eta.evaluate(891.0)[0], Float32(0.0))
    var wavelengths = SIMD[.float32, 4](450.0, 555.0, 610.0, 890.0)
    var packet = eta.evaluate(wavelengths)
    comptime for lane in range(4):
        assert_almost_equal(packet[lane], eta.evaluate(wavelengths[lane])[0])


def test_named_spectra_retained_and_shared_by_materials() raises:
    var scene = parse_pbrt(
        """WorldBegin
Material "coatedconductor"
 "spectrum conductor.eta" "metal-CuZn-eta"
 "spectrum conductor.k" "metal-CuZn-k"
Shape "sphere"
Translate 3 0 0
Material "conductor"
 "spectrum eta" "metal-CuZn-eta" "spectrum k" "metal-CuZn-k"
Shape "sphere"
"""
    )
    ref surfaces = scene.data.surfaces()
    assert_equal(len(surfaces.optical_spectra), 2)
    for index in range(2):
        ref material = surfaces.coated_diffuses[index]
        ref eta = surfaces.optical_spectra[Int(material.conductor_eta_spectrum)]
        ref k = surfaces.optical_spectra[Int(material.conductor_k_spectrum)]
        assert_almost_equal(eta.evaluate(555.0)[0], 0.516)
        assert_almost_equal(k.evaluate(555.0)[0], 2.8125)


def test_dielectric_wavelength_curve_is_retained() raises:
    var scene = parse_pbrt(
        'WorldBegin\nMaterial "dielectric" "spectrum eta" [400 1.6 700 1.5]'
        '\nShape "sphere"'
    )
    ref surfaces = scene.data.surfaces()
    var index = surfaces.dielectrics[0].eta_spectrum
    assert_equal(len(surfaces.optical_spectra), 1)
    assert_almost_equal(
        surfaces.optical_spectra[Int(index)].evaluate(475.0)[0], 1.575
    )


def test_spectra_reject_invalid_tables() raises:
    var tables: List[List[Float32]] = [
        [],
        [400.0, 1.0, 500.0],
        [500.0, 1.0, 400.0, 1.0],
        [400.0, 1.0, 400.0, 2.0],
        [0.0, 1.0],
        [400.0, -1.0],
    ]
    for values in tables:
        with assert_raises():
            _ = PiecewiseLinearSpectrum(values.copy())


def test_pbrt_matrix_columns_and_concatenation() raises:
    var scene = parse_pbrt(
        """WorldBegin
Transform [1 2 3 0 4 5 6 0 7 8 10 0 11 12 13 1]
ConcatTransform [1 0 0 0 0 1 0 0 0 0 1 0 2 3 4 1]
Shape "trianglemesh" "point3 P" [0 0 0 1 0 0 0 1 0]
"""
    )
    ref vertices = scene.data.triangle_vertices()
    assert_almost_equal(vertices[0].x[0], 53.0)
    assert_almost_equal(vertices[0].y[0], 63.0)
    assert_almost_equal(vertices[0].z[0], 77.0)
    assert_almost_equal(vertices[1].x[0], 54.0)
    assert_almost_equal(vertices[1].y[0], 65.0)
    assert_almost_equal(vertices[1].z[0], 80.0)
    assert_almost_equal(vertices[2].x[0], 57.0)
    assert_almost_equal(vertices[2].y[0], 68.0)
    assert_almost_equal(vertices[2].z[0], 83.0)
    for matrix in [
        "1 0 0 .1 0 1 0 0 0 0 1 0 0 0 0 1",
        "1 0 0 0 0 1 0 .1 0 0 1 0 0 0 0 1",
        "1 0 0 0 0 1 0 0 0 0 1 .1 0 0 0 1",
    ]:
        with assert_raises():
            _ = parse_pbrt(
                "WorldBegin\nTransform [" + matrix + ']\nShape "sphere"'
            )


def test_normal_map_file_uses_linear_encoding_and_include_directory() raises:
    var loader = MemoryAssetLoader()
    loader.add_file(
        "root/main.pbrt", 'WorldBegin\nInclude "nested/material.pbrt"'
    )
    loader.add_file(
        "root/nested/material.pbrt",
        """
Texture "base" "spectrum" "imagemap" "string filename" "map.png"
Material "coateddiffuse" "texture reflectance" "base" "string normalmap" "map.png"
Shape "trianglemesh" "point3 P" [0 0 0 1 0 0 0 1 0] "point2 uv" [0 0 1 0 0 1]
""",
    )
    loader.add_image_file(
        "root/nested/map.png", _encode([UInt8(0), 128, 128, 255], 1, 1)
    )
    var scene = read_pbrt("root/main.pbrt", loader)
    ref surfaces = scene.data.surfaces()
    assert_equal(len(surfaces.image_textures), 2)
    var surface = scene.data.triangle_instance_surfaces()[0].value[0]
    var index = surfaces.normal_maps[surface]
    assert_almost_equal(
        surfaces.image_textures[Int(index)].pixels[0],
        Float32(128.0 / 255.0),
        atol=1e-7,
    )
    var base_index = surfaces.coated_diffuses[0].texture_index
    assert_almost_equal(
        surfaces.image_textures[Int(base_index)].pixels[0], 0.2158605, atol=1e-6
    )
    assert_equal(index == base_index, False)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
