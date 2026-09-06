from std.memory import bitcast
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_raises,
    assert_true,
)

from bajo.parser.pbrt import (
    MemoryTextLoader,
    parse_pbrt,
    read_pbrt,
    read_pbrt_camera,
)
from bajo.core import Point3f32, Rayf32, Vec3f32
from bajo.rt import CpuScene, render_wavefront


def _append_ply_text(mut bytes: List[UInt8], text: String):
    for value in StringSpan(text).as_bytes():
        bytes.append(value)


def _append_ply_u32(mut bytes: List[UInt8], value: UInt32):
    bytes.append(UInt8(value & UInt32(0xFF)))
    bytes.append(UInt8((value >> UInt32(8)) & UInt32(0xFF)))
    bytes.append(UInt8((value >> UInt32(16)) & UInt32(0xFF)))
    bytes.append(UInt8(value >> UInt32(24)))


def _append_ply_f32(mut bytes: List[UInt8], value: Float32):
    _append_ply_u32(bytes, bitcast[.uint32](value))


def _append_ply_vertex(
    mut bytes: List[UInt8], x: Float32, y: Float32, u: Float32, v: Float32
):
    _append_ply_f32(bytes, x)
    _append_ply_f32(bytes, y)
    _append_ply_f32(bytes, 0.0)
    _append_ply_f32(bytes, 0.0)
    _append_ply_f32(bytes, 0.0)
    _append_ply_f32(bytes, 1.0)
    _append_ply_f32(bytes, u)
    _append_ply_f32(bytes, v)


def _triangle_ply() -> List[UInt8]:
    var bytes = List[UInt8]()
    _append_ply_text(
        bytes,
        (
            "ply\nformat binary_little_endian 1.0\n"
            "element vertex 3\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "property float u\nproperty float v\n"
            "element face 1\n"
            "property list uchar int vertex_indices\nend_header\n"
        ),
    )
    _append_ply_vertex(bytes, 0.0, 0.0, 0.0, 0.0)
    _append_ply_vertex(bytes, 1.0, 0.0, 1.0, 0.0)
    _append_ply_vertex(bytes, 0.0, 1.0, 0.0, 1.0)
    bytes.append(UInt8(3))
    _append_ply_u32(bytes, UInt32(0))
    _append_ply_u32(bytes, UInt32(1))
    _append_ply_u32(bytes, UInt32(2))
    return bytes^


def test_parse_checked_in_scene_with_include() raises:
    var scene = read_pbrt("examples/scenes/pbrt_showcase.pbrt")
    assert_true(scene.settings.image_width == 640)
    assert_true(scene.settings.image_height == 480)
    assert_true(scene.settings.samples_per_pixel == 64)
    assert_true(scene.settings.max_depth == 10)
    assert_true(scene.integrator == .PATH)
    assert_true(len(scene.data.spheres()) == 3)
    assert_true(len(scene.data.triangle_vertices()) / 3 == 10)
    assert_true(len(scene.data.lights().records) == 2)
    assert_true(len(scene.data.surfaces().lambertians) == 4)
    assert_true(len(scene.data.surfaces().metals) == 1)
    assert_true(len(scene.data.surfaces().dielectrics) == 1)
    assert_true(len(scene.data.surfaces().emissives) == 1)


def test_camera_read_stops_before_scene_assets() raises:
    var loader = MemoryTextLoader()
    loader.add_file(
        "camera-only.pbrt",
        """LookAt 1 2 3  1 2 2  0 1 0
Camera "perspective" "float fov" [31]
WorldBegin
Texture "missing" "spectrum" "imagemap" "string filename" "missing.png"
Shape "plymesh" "string filename" "missing.ply"
""",
    )
    var camera = read_pbrt_camera("camera-only.pbrt", loader)
    assert_almost_equal(camera.origin.x, 1.0)
    assert_almost_equal(camera.origin.y, 2.0)
    assert_almost_equal(camera.origin.z, 3.0)


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
    assert_true(scene.data.sphere_surfaces()[0].kind() == .LAMBERTIAN)


def test_coateddiffuse_material() raises:
    comptime source = """WorldBegin
Material "coateddiffuse" "rgb reflectance" [0.4 0.2 0.1] "float roughness" [0.025]
Shape "sphere"
"""
    var scene = parse_pbrt(source)
    var surface = scene.data.sphere_surfaces()[0].copy()
    assert_true(surface.kind() == .COATED_DIFFUSE)
    ref material = scene.data.surfaces().coated_diffuses[Int(surface.index())]
    var albedo = material.albedo
    assert_almost_equal(albedo.x, 0.4)
    assert_almost_equal(albedo.y, 0.2)
    assert_almost_equal(albedo.z, 0.1)
    assert_true(material.roughness > 0.025)
    assert_true(material.roughness < 1.0)
    assert_almost_equal(material.eta, 1.5)


def test_texture_graph_loads_imagemap() raises:
    comptime source = """WorldBegin
Texture "base" "spectrum" "constant" "rgb value" [0.8 0.4 0.2]
Texture "tint" "spectrum" "constant" "rgb value" [0.5 0.25 0.5]
Texture "scaled" "spectrum" "scale"
    "texture tex" "base" "texture scale" "tint"
Texture "cover" "spectrum" "imagemap" "string filename" "cover.png"
Texture "bump_raw" "float" "imagemap" "string filename" "bump.png"
Texture "bump_amount" "float" "constant" "float value" 0.1
Texture "bump" "float" "scale"
    "texture tex" "bump_raw" "texture scale" "bump_amount"
AttributeBegin
Material "diffuse" "texture reflectance" "scaled"
Shape "sphere"
AttributeEnd
AttributeBegin
Material "coateddiffuse"
    "texture reflectance" "cover" "texture displacement" "bump"
Translate 3 0 0
Shape "sphere"
AttributeEnd
"""
    var loader = MemoryTextLoader()
    loader.add_file("scene.pbrt", source)
    var png = List[UInt8]()
    for value in [
        137,
        80,
        78,
        71,
        13,
        10,
        26,
        10,
        0,
        0,
        0,
        13,
        73,
        72,
        68,
        82,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        8,
        2,
        0,
        0,
        0,
        144,
        119,
        83,
        222,
        0,
        0,
        0,
        12,
        73,
        68,
        65,
        84,
        120,
        156,
        99,
        104,
        112,
        80,
        0,
        0,
        2,
        36,
        0,
        225,
        171,
        89,
        98,
        39,
        0,
        0,
        0,
        0,
        73,
        69,
        78,
        68,
        174,
        66,
        96,
        130,
    ]:
        png.append(UInt8(value))
    loader.add_image_file("cover.png", png.copy())
    loader.add_image_file("bump.png", png^)
    var scene = read_pbrt("scene.pbrt", loader)
    assert_true(len(scene.data.spheres()) == 2)
    var first = scene.data.sphere_surfaces()[0].copy()
    var first_albedo = (
        scene.data.surfaces().lambertians[Int(first.index())].albedo
    )
    assert_almost_equal(first_albedo.x, 0.4)
    assert_almost_equal(first_albedo.y, 0.1)
    assert_almost_equal(first_albedo.z, 0.1)

    var second = scene.data.sphere_surfaces()[1].copy()
    assert_true(second.kind() == .COATED_DIFFUSE)
    ref second_material = scene.data.surfaces().coated_diffuses[
        Int(second.index())
    ]
    var second_albedo = second_material.albedo
    assert_almost_equal(second_albedo.x, 1.0)
    assert_almost_equal(second_albedo.y, 1.0)
    assert_almost_equal(second_albedo.z, 1.0)
    var sampled = scene.data.surfaces().sample_albedo(second, 0.0, 0.0)
    assert_almost_equal(sampled.x, 0.21586, atol=0.0001)
    assert_almost_equal(sampled.y, 0.05127, atol=0.0001)
    assert_almost_equal(sampled.z, 0.01444, atol=0.0001)
    assert_true(second_material.displacement_texture_index != UInt32.MAX)
    assert_almost_equal(second_material.displacement_scale, 0.1)


def test_loopsubdiv_loads_control_cage() raises:
    comptime source = """WorldBegin
Shape "loopsubdiv" "integer levels" [1]
    "point3 P" [0 0 0  1 0 0  0 1 0]
    "integer indices" [0 1 2]
"""
    var scene = parse_pbrt(source)
    assert_true(len(scene.data.triangle_vertices()) == 3)
    assert_true(len(scene.data.triangle_surfaces()) == 1)


def test_plymesh_resource_transform_orientation_and_area_light() raises:
    var loader = MemoryTextLoader()
    loader.add_file(
        "scene/main.pbrt",
        """WorldBegin
Include "parts/geometry.pbrt"
WorldEnd
""",
    )
    loader.add_file(
        "scene/parts/geometry.pbrt",
        """AttributeBegin
Material "diffuse" "rgb reflectance" [0.2 0.4 0.6]
Translate 2 3 4
ReverseOrientation
Shape "plymesh" "string filename" ["meshes/triangle.ply"]
AttributeEnd
AttributeBegin
AreaLightSource "diffuse" "rgb L" [4 5 6]
Translate 0 0 2
Shape "plymesh" "string filename" "meshes/triangle.ply"
AttributeEnd
""",
    )
    loader.add_ply_file("scene/parts/meshes/triangle.ply", _triangle_ply())

    var scene = read_pbrt("scene/main.pbrt", loader)
    assert_true(len(scene.data.triangle_meshes()) == 2)
    assert_true(len(scene.data.triangle_instances()) == 2)
    assert_true(len(scene.data.triangle_instance_surfaces()) == 2)
    assert_true(len(scene.data.triangle_meshes()[0]) == 3)
    assert_true(len(scene.data.triangle_mesh_normals()[0]) == 9)
    assert_true(len(scene.data.triangle_mesh_texcoords()[0]) == 6)

    # ReverseOrientation swaps the final two vertices in the triangle soup.
    assert_almost_equal(scene.data.triangle_meshes()[0][1].x, 0.0)
    assert_almost_equal(scene.data.triangle_meshes()[0][1].y, 1.0)
    assert_almost_equal(scene.data.triangle_meshes()[0][2].x, 1.0)
    assert_almost_equal(scene.data.triangle_meshes()[0][2].y, 0.0)
    assert_almost_equal(scene.data.triangle_mesh_normals()[0][2], 1.0)
    assert_almost_equal(scene.data.triangle_mesh_texcoords()[0][2], 0.0)
    assert_almost_equal(scene.data.triangle_mesh_texcoords()[0][3], 1.0)
    assert_almost_equal(scene.data.triangle_mesh_texcoords()[0][4], 1.0)
    assert_almost_equal(scene.data.triangle_mesh_texcoords()[0][5], 0.0)

    ref bounds = scene.data.triangle_instances()[0].bounds
    assert_almost_equal(bounds._min.x[0], 2.0)
    assert_almost_equal(bounds._min.y[0], 3.0)
    assert_almost_equal(bounds._min.z[0], 4.0)
    assert_almost_equal(bounds._max.x[0], 3.0)
    assert_almost_equal(bounds._max.y[0], 4.0)
    assert_almost_equal(bounds._max.z[0], 4.0)

    assert_true(
        scene.data.triangle_instance_surfaces()[0].kind() == .LAMBERTIAN
    )
    assert_true(scene.data.triangle_instance_surfaces()[1].kind() == .EMISSIVE)
    assert_true(len(scene.data.lights().records) == 1)

    var world = CpuScene[16, 16](scene^.take_data())
    var hit = world.trace_surface(
        Rayf32[.WORLD](
            Point3f32[.WORLD](2.25, 3.25, 5.0),
            Vec3f32[.WORLD](0.0, 0.0, -1.0),
        )
    )
    assert_true(hit.hit[0])
    assert_almost_equal(hit.uv_u[0], 0.25)
    assert_almost_equal(hit.uv_v[0], 0.25)
    assert_almost_equal(hit.normal.z[0], 1.0)


def test_memory_include_and_mis_render() raises:
    var loader = MemoryTextLoader()
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
    var world = CpuScene[16, 16](scene^.take_data())
    var result = render_wavefront[.MIS, 1, 64, False](settings, camera, world)
    assert_true(len(result.pixels) == 16)
    var energy: Float32 = 0.0
    for pixel in result.pixels:
        energy += pixel.x + pixel.y + pixel.z
    assert_true(energy > 0.0)


def test_rejects_unsupported_and_invalid_geometry() raises:
    with assert_raises():
        _ = parse_pbrt('Integrator "bdpt"\nWorldBegin\nShape "sphere"')
    with assert_raises():
        _ = parse_pbrt('WorldBegin\nShape "plymesh"')
    with assert_raises():
        _ = parse_pbrt(
            'WorldBegin\nShape "plymesh" "string filename" "missing.ply"'
        )
    with assert_raises():
        _ = parse_pbrt(
            'WorldBegin\nMaterial "diffuse" "texture reflectance" "missing"'
            '\nShape "sphere"'
        )
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
