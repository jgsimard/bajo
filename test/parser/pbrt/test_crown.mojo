"""Crown optical inputs, media scoping, textured parameters and camera."""

from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
    assert_raises,
)
from bajo.parser import AssetLoader
from bajo.parser.pbrt import parse_pbrt, read_pbrt, read_pbrt_camera
from bajo.parser.pbrt.spectrum import _blackbody_rgb
from bajo.parser.ply import PlyMesh
from bajo.rt.material_types import ImageTexture, NO_TEXTURE, MaterialKind
from bajo.rt.media import SceneMedia
from bajo.rt.render_types import Color
from bajo.core import Vec3f32, dot


struct _TextureLoader(AssetLoader):
    var source: String

    def __init__(out self, source: String):
        self.source = source

    def read_text(self, path: String) raises -> String:
        return self.source

    def read_ply_mesh(self, path: String) raises -> PlyMesh:
        raise Error("unexpected mesh read")

    def read_image_texture(self, path: String) raises -> ImageTexture:
        return ImageTexture(2, 1, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])


def test_gold_and_curve_ior() raises:
    var scene = parse_pbrt(
        """WorldBegin
Material "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k"
Shape "sphere"
Material "dielectric" "spectrum eta" [200 3.5 900 3.3]
Translate 3 0 0
Shape "sphere"
"""
    )
    ref surfaces = scene.data.surfaces()
    assert_almost_equal(
        surfaces.coated_diffuses[0].conductor_eta.x[0], 0.20605660
    )
    assert_almost_equal(
        surfaces.coated_diffuses[0].conductor_k.z[0], 1.84310538
    )
    assert_almost_equal(surfaces.dielectrics[0].refraction_index, 3.4)
    for curve in ["[550 1.5 450 1.6]", "[400 -1 700 1.5]", "[400 1 500 1 600]"]:
        with assert_raises():
            _ = parse_pbrt(
                'WorldBegin\nMaterial "dielectric" "spectrum eta" '
                + curve
                + '\nShape "sphere"'
            )


def test_blackbody_lighting_and_film() raises:
    var scene = parse_pbrt(
        """Film "rgb" "float iso" 150 "string sensor" "canon_eos_5d_mkiv"
WorldBegin
AreaLightSource "diffuse" "blackbody L" 5500 "float scale" 10
Shape "sphere"
"""
    )
    var expected = _blackbody_rgb(5500)
    var actual = scene.data.surfaces().emissives[0].radiance
    assert_almost_equal(actual.x[0], expected.x[0] * 10)
    assert_true(expected.x > expected.z)
    assert_true(expected.is_finite()[0])
    assert_almost_equal(scene.settings.film.exposure, 1.5)
    assert_true(scene.settings.film.red.x > 0.99)
    with assert_raises():
        _ = _blackbody_rgb(0)
    with assert_raises():
        _ = parse_pbrt(
            'Film "rgb" "string sensor" "missing"\nWorldBegin\nShape "sphere"'
        )


def test_medium_boundary_scoping() raises:
    var scene = parse_pbrt(
        """WorldBegin
MakeNamedMedium "ruby" "string type" "homogeneous" "rgb sigma_a" [.01 .2 .5] "rgb sigma_s" [0 0 0]
Material "dielectric" "float eta" 1
AttributeBegin
MediumInterface "ruby" ""
Shape "sphere"
AttributeEnd
Translate 3 0 0
Shape "sphere"
AttributeBegin
MediumInterface "" "ruby"
Translate 3 0 0
Shape "sphere"
AttributeEnd
"""
    )
    ref media = scene.data.media()
    assert_equal(len(media.absorption), 2)
    assert_almost_equal(media.absorption[1].z[0], 0.5)
    assert_equal(SceneMedia.boundary(media.spheres, 0), UInt32(1))
    assert_equal(SceneMedia.boundary(media.spheres, 1), UInt32(0))
    assert_equal(SceneMedia.boundary(media.spheres, 2), UInt32(1 << 16))
    # The material remains shared while geometry owns distinct interfaces.
    assert_equal(
        scene.data.sphere_surfaces()[0].value[0],
        scene.data.sphere_surfaces()[1].value[0],
    )
    for source in [
        'MediumInterface "missing" ""',
        'MakeNamedMedium "m" "string type" "homogeneous" "rgb sigma_s" [1 0 0]',
    ]:
        with assert_raises():
            _ = parse_pbrt("WorldBegin\n" + source + '\nShape "sphere"')


def test_textured_masks_roughness_and_bump() raises:
    var loader = _TextureLoader(
        """WorldBegin
Texture "mask" "float" "imagemap" "string filename" "mask.png"
Texture "rough" "float" "scale" "texture tex" "mask" "float scale" .1
Texture "bump" "float" "scale" "texture tex" "mask" "float scale" .25
MakeNamedMaterial "white" "string type" "diffuse" "rgb reflectance" [.8 .8 .8]
MakeNamedMaterial "gold" "string type" "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k" "texture roughness" "rough" "texture displacement" "bump"
Material "mix" "string materials" ["white" "gold"] "texture amount" "mask"
Shape "sphere"
Material "dielectric" "float eta" 3 "texture displacement" "bump"
Translate 3 0 0
Shape "sphere"
"""
    )
    var scene = read_pbrt("scene.pbrt", loader)
    ref surfaces = scene.data.surfaces()
    var mixed = scene.data.sphere_surfaces()[0].copy()
    var first = surfaces.resolve_mix(mixed, 0.5, 0.1, 0.5)
    var second = surfaces.resolve_mix(mixed, 0.5, 0.6, 0.5)
    assert_true(first.kind() == MaterialKind.LAMBERTIAN)
    assert_true(second.kind() == MaterialKind.COATED_DIFFUSE)
    assert_almost_equal(
        surfaces.sample_conductor_roughness(second, 0.1, 0.5), 0.0
    )
    assert_almost_equal(
        surfaces.sample_conductor_roughness(second, 0.6, 0.5),
        sqrt(Float32(0.1)),
    )
    assert_equal(len(surfaces.image_textures), 1)
    assert_almost_equal(surfaces.coated_diffuses[0].displacement_scale, 0.25)
    assert_almost_equal(surfaces.dielectrics[0].displacement_scale, 0.25)
    var normal = Vec3f32[.WORLD](0, 0, 1)
    var bumped = surfaces.shading_normal(
        scene.data.sphere_surfaces()[1], normal, 0.25, 0.5
    )
    assert_true(dot(normal, bumped) < 0.999)


def test_portrait_camera_transform_lens_and_camera_only() raises:
    var loader = _TextureLoader(
        """Scale -1 1 1
Film "rgb" "integer xresolution" 1000 "integer yresolution" 1400
LookAt 0 0 5 0 0 0 0 1 0
Camera "perspective" "float fov" 47 "float lensradius" .1 "float focaldistance" 17
WorldBegin
Shape "sphere"
"""
    )
    var scene = read_pbrt("scene.pbrt", loader)
    var camera = read_pbrt_camera("scene.pbrt", loader)
    assert_almost_equal(camera.origin.z[0], 5.0)
    assert_almost_equal(camera.right.x[0], 1.0)
    assert_almost_equal(camera.forward.z[0], -1.0)
    assert_almost_equal(camera.focus_dist, 17.0)
    assert_almost_equal(camera.defocus_disk_u.x[0], 0.1)
    assert_almost_equal(camera.fov_scale, scene.camera.fov_scale)
    assert_true(camera.fov_scale > 0.6)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
