"""BMW material declarations and robust PLY ingestion without asset fixtures."""

from std.memory import bitcast
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)
from bajo.parser import MemoryAssetLoader
from bajo.parser.pbrt import parse_pbrt, read_pbrt
from bajo.rt.material_types import MaterialKind
from bajo.core.random import Sampler
from test.parser.pbrt.test_pbrt import (
    _append_ply_text,
    _append_ply_vertex,
    _append_ply_u32,
)


def test_mix_named_materials_and_probability() raises:
    var scene = parse_pbrt(
        """WorldBegin
MakeNamedMaterial "black" "string type" "diffuse" "rgb reflectance" [.01 .01 .01]
MakeNamedMaterial "white" "string type" "coateddiffuse" "rgb reflectance" [.8 .8 .8]
Material "mix" "string materials" ["black" "white"] "float amount" .2
Shape "sphere"
"""
    )
    ref surfaces = scene.data.surfaces()
    var mixed = scene.data.sphere_surfaces()[0].copy()
    assert_true(mixed.kind() == MaterialKind.MIX)
    assert_equal(len(surfaces.mix_amounts), 1)
    assert_almost_equal(surfaces.mix_amounts[0], 0.2)
    assert_true(
        surfaces.resolve_mix(mixed, 0.199).kind() == MaterialKind.COATED_DIFFUSE
    )
    assert_true(
        surfaces.resolve_mix(mixed, 0.2).kind() == MaterialKind.LAMBERTIAN
    )


def test_bmw_halton_sampler_is_preserved() raises:
    var scene = parse_pbrt(
        'Sampler "halton" "integer pixelsamples" 64\nWorldBegin\nShape "sphere"'
    )
    assert_true(scene.settings.sampler == Sampler.HALTON)
    assert_equal(scene.settings.samples_per_pixel, 64)


def test_conductor_optics_and_independent_coating_roughness() raises:
    var scene = parse_pbrt(
        """WorldBegin
Material "coatedconductor"
 "spectrum conductor.eta" "metal-Al-eta" "spectrum conductor.k" "metal-Al-k"
 "float conductor.roughness" .05 "float interface.roughness" 0
Shape "sphere"
AttributeBegin
Translate 3 0 0
Material "conductor" "spectrum eta" "metal-Al-eta" "spectrum k" "metal-Al-k"
 "float roughness" .01
Shape "sphere"
AttributeEnd
"""
    )
    ref coatings = scene.data.surfaces().coated_diffuses
    assert_equal(len(coatings), 2)
    assert_equal(coatings[0].roughness, Float32(0.0))
    assert_almost_equal(coatings[0].conductor_roughness, 0.2236068, atol=1.0e-6)
    assert_almost_equal(
        coatings[0].conductor_eta.x[0] * 1.5, 1.24750867, atol=1.0e-6
    )
    assert_almost_equal(
        coatings[0].conductor_k.z[0] * 1.5, 5.59014846, atol=1.0e-6
    )
    assert_equal(coatings[1].eta, Float32(1.0))
    assert_equal(coatings[1].thickness, Float32(0.0))
    assert_almost_equal(coatings[1].conductor_roughness, 0.1, atol=1.0e-6)
    assert_almost_equal(coatings[1].conductor_eta.y[0], 0.95856902, atol=1.0e-6)


def test_new_materials_reject_invalid_definitions() raises:
    for declaration in [
        'Material "mix" "string materials" ["missing" "missing"]',
        'Material "mix" "string materials" ["a"]',
        'Material "mix" "string materials" ["a" "a"] "float amount" 1.01',
        'Material "mix" "string materials" ["a" "a"] "float amount" -.1',
        'Material "conductor" "spectrum eta" "metal-Unknown-eta"',
        'Material "conductor" "rgb eta" [-1 1 1] "rgb k" [1 1 1]',
        'Material "coatedconductor" "float interface.eta" 0',
    ]:
        with assert_raises():
            _ = parse_pbrt(
                'WorldBegin\nMakeNamedMaterial "a" "string type" "diffuse"\n'
                + declaration
                + '\nShape "sphere"'
            )


def _ply_with_degenerate_faces(nonfinite: Bool = False) -> List[UInt8]:
    var bytes = List[UInt8]()
    _append_ply_text(
        bytes,
        (
            "ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty"
            " float x\nproperty float y\nproperty float z\nproperty float"
            " nx\nproperty float ny\nproperty float nz\nproperty float"
            " u\nproperty float v\nelement face 4\nproperty list uchar int"
            " vertex_indices\nend_header\n"
        ),
    )
    _append_ply_vertex(bytes, 0.0, 0.0, 0.0, 0.0)
    _append_ply_vertex(
        bytes,
        bitcast[.float32](UInt32(0x7F800000)) if nonfinite else Float32(1.0),
        0.0,
        1.0,
        0.0,
    )
    _append_ply_vertex(bytes, 0.0, 1.0, 0.0, 1.0)
    _append_ply_vertex(bytes, 2.0, 0.0, 2.0, 0.0)
    for face in range(4):
        bytes.append(UInt8(3))
        var indices: List[Int] = [0, 1, 2, 0, 0, 2, 0, 1, 3, 2, 1, 0]
        for lane in range(3):
            _append_ply_u32(bytes, UInt32(indices[3 * face + lane]))
    return bytes^


def test_ply_degenerate_faces_are_removed_with_attributes_and_orientation() raises:
    var loader = MemoryAssetLoader()
    loader.add_file(
        "scene/main.pbrt",
        (
            'WorldBegin\nReverseOrientation\nShape "plymesh" "string filename"'
            ' "mesh.ply"'
        ),
    )
    loader.add_ply_file("scene/mesh.ply", _ply_with_degenerate_faces())
    var scene = read_pbrt("scene/main.pbrt", loader)
    ref mesh = scene.data.triangle_meshes()[0]
    ref uv = scene.data.triangle_mesh_texcoords()[0]
    assert_equal(len(mesh), 6)
    assert_equal(len(uv), 12)
    assert_equal(mesh[1].x[0], Float32(0.0))
    assert_equal(mesh[1].y[0], Float32(1.0))
    assert_equal(uv[2], Float32(0.0))
    assert_equal(uv[3], Float32(1.0))
    assert_equal(mesh[4].x[0], Float32(0.0))
    assert_equal(mesh[4].y[0], Float32(0.0))
    assert_equal(uv[8], Float32(0.0))
    assert_equal(uv[9], Float32(0.0))
    assert_equal(len(scene.data.triangle_mesh_normals()[0]), 18)


def test_ply_nonfinite_geometry_remains_an_error() raises:
    var loader = MemoryAssetLoader()
    loader.add_file(
        "scene/main.pbrt",
        'WorldBegin\nShape "plymesh" "string filename" "mesh.ply"',
    )
    loader.add_ply_file("scene/mesh.ply", _ply_with_degenerate_faces(True))
    with assert_raises():
        _ = read_pbrt("scene/main.pbrt", loader)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
