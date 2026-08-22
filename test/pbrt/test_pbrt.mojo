from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_raises,
    assert_true,
)

from bajo.parser.pbrt import MemoryPbrtTextLoader, parse_pbrt, read_pbrt
from bajo.rt import CpuScene, MAT, RENDER, render_wavefront


def test_parse_checked_in_scene_with_include() raises:
    var scene = read_pbrt("examples/scenes/pbrt_showcase.pbrt")
    assert_true(scene.settings.image_width == 640)
    assert_true(scene.settings.image_height == 480)
    assert_true(scene.settings.samples_per_pixel == 64)
    assert_true(scene.settings.max_depth == 10)
    assert_true(scene.integrator == RENDER.PATH)
    assert_true(len(scene.data.spheres()) == 3)
    assert_true(len(scene.data.triangle_vertices()) / 3 == 10)
    assert_true(len(scene.data.lights().records) == 2)
    assert_true(len(scene.data.surfaces().lambertians) == 4)
    assert_true(len(scene.data.surfaces().metals) == 1)
    assert_true(len(scene.data.surfaces().dielectrics) == 1)
    assert_true(len(scene.data.surfaces().emissives) == 1)


def test_transform_and_named_material() raises:
    comptime source = """Film "rgb" "integer xresolution" [4] "integer yresolution" [3]
ColorSpace "srgb"
PixelFilter "gaussian" "float xradius" [1.5]
Accelerator "bvh" "integer maxnodeprims" [4]
WorldBegin
MakeNamedMaterial "blue" "string type" "diffuse" "rgb reflectance" [0.1 0.2 0.8]
AttributeBegin
NamedMaterial "blue"
Translate 1 2 3
Scale 2 2 2
Shape "sphere" "float radius" [0.5]
AttributeEnd
WorldEnd
"""
    var scene = parse_pbrt(source)
    assert_true(scene.settings.image_width == 4)
    assert_true(scene.settings.image_height == 3)
    assert_true(len(scene.data.spheres()) == 1)
    assert_almost_equal(scene.data.spheres()[0].center.x, 1.0)
    assert_almost_equal(scene.data.spheres()[0].center.y, 2.0)
    assert_almost_equal(scene.data.spheres()[0].center.z, 3.0)
    assert_almost_equal(scene.data.spheres()[0].radius, 1.0)
    assert_true(scene.data.sphere_surfaces()[0].kind() == MAT.LAMBERTIAN)


def test_coateddiffuse_uses_diffuse_substrate() raises:
    comptime source = """WorldBegin
Material "coateddiffuse" "rgb reflectance" [0.4 0.2 0.1] "float roughness" [0.025]
Shape "sphere"
"""
    var scene = parse_pbrt(source)
    var surface = scene.data.sphere_surfaces()[0].copy()
    assert_true(surface.kind() == MAT.LAMBERTIAN)
    var albedo = scene.data.surfaces().lambertians[Int(surface.index())].albedo
    assert_almost_equal(albedo.x, 0.4)
    assert_almost_equal(albedo.y, 0.2)
    assert_almost_equal(albedo.z, 0.1)


def test_loopsubdiv_loads_control_cage() raises:
    comptime source = """WorldBegin
Shape "loopsubdiv" "integer levels" [1]
    "point3 P" [0 0 0  1 0 0  0 1 0]
    "integer indices" [0 1 2]
"""
    var scene = parse_pbrt(source)
    assert_true(len(scene.data.triangle_vertices()) == 3)
    assert_true(len(scene.data.triangle_surfaces()) == 1)


def test_memory_include_and_mis_render() raises:
    var loader = MemoryPbrtTextLoader()
    loader.add_file(
        "scene/main.pbrt",
        """LookAt 0 0 4  0 0 0  0 1 0
Camera "perspective" "float fov" 40
Film "rgb" "integer xresolution" 4 "integer yresolution" 4
Sampler "independent" "integer pixelsamples" 1
WorldBegin
Include "geometry.pbrt"
WorldEnd
""",
    )
    loader.add_file(
        "scene/geometry.pbrt",
        """AttributeBegin
AreaLightSource "diffuse" "rgb L" [8 8 8]
Translate 0 2 0
Shape "sphere" "float radius" 0.5
AttributeEnd
Material "diffuse" "rgb reflectance" [0.7 0.7 0.7]
Shape "sphere" "float radius" 1
""",
    )
    var scene = read_pbrt("scene/main.pbrt", loader)
    var settings = scene.settings.copy()
    var camera = scene.camera
    var world = CpuScene[](scene^.take_data())
    var result = render_wavefront[RENDER.MIS, 1, 64, False](
        settings, camera, world
    )
    assert_true(len(result.pixels) == 16)
    var energy: Float32 = 0.0
    for pixel in result.pixels:
        energy += pixel.x + pixel.y + pixel.z
    assert_true(energy > 0.0)


def test_rejects_unsupported_and_invalid_geometry() raises:
    with assert_raises():
        _ = parse_pbrt('Integrator "bdpt"\nWorldBegin\nShape "sphere"')
    with assert_raises():
        _ = parse_pbrt('WorldBegin\nShape "plymesh" "string filename" "x.ply"')
    with assert_raises():
        _ = parse_pbrt('WorldBegin\nScale 1 2 1\nShape "sphere"')
    with assert_raises():
        _ = parse_pbrt(
            'WorldBegin\nShape "trianglemesh" "point3 P" [0 0 0 1 0 0 0 1 0]'
            ' "integer indices" [0 1 3]'
        )
    with assert_raises():
        _ = parse_pbrt('WorldBegin\nShape "sphere" "float radius" [0]')


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
