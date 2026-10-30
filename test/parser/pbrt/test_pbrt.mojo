from std.memory import bitcast
from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_raises,
    assert_true,
    assert_equal,
)

from bajo.core.camera import Camera
from bajo.parser import MemoryAssetLoader
from bajo.parser.pbrt import parse_pbrt, read_pbrt, read_pbrt_camera
from bajo.parser.number import parse_f32_at, parse_int_f32
from bajo.parser.pbrt.parser import _Lexer
from std.sys import simd_width_of
from bajo.parser.asset_loader import AssetLoader
from bajo.parser.ply import PlyMesh

from bajo.core import Point3f32, Rayf32, Vec3f32
from bajo.rt import CpuScene, EnvironmentKind, ImageTexture, render_wavefront


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


struct _CountingAssetLoader[origin: MutOrigin](AssetLoader):
    var source: MemoryAssetLoader
    var calls: MutSpan[Int, Self.origin]

    def __init__(
        out self,
        var source: MemoryAssetLoader,
        calls: MutSpan[Int, Self.origin],
    ):
        self.source = source^
        self.calls = calls

    def read_text(self, path: String) raises -> String:
        return self.source.read_text(path)

    def read_ply_mesh(self, path: String) raises -> PlyMesh:
        self.calls[0 if path == "scene/assets/shared.ply" else 1] += 1
        var mesh = self.source.read_ply_mesh(path)
        # Distinct attributes expose mistakes when reversing cached geometry.
        mesh.normals[3] = 0.25
        mesh.normals[6] = 0.5
        return mesh^

    def read_image_texture(self, path: String) raises -> ImageTexture:
        if path == "scene/assets/shared.png":
            self.calls[2] += 1
            return ImageTexture(
                1, 1, [Float32(0.2), Float32(0.4), Float32(0.6)]
            )
        if path == "scene/other/shared.png":
            self.calls[3] += 1
            return ImageTexture(
                1, 1, [Float32(0.8), Float32(0.6), Float32(0.4)]
            )
        raise Error("unexpected test image: " + path)


def test_images_shared_across_include_color_scalar_and_environment() raises:
    var memory = MemoryAssetLoader()
    memory.add_file(
        "scene/main.pbrt",
        """WorldBegin
Include "assets/textures.pbrt"
Texture "same" "spectrum" "imagemap" "string filename" "assets/shared.png"
Texture "other" "spectrum" "imagemap" "string filename" "other/shared.png"
LightSource "infinite" "string filename" "assets/shared.png" "rgb L" [2 3 4]
AttributeBegin
Material "coateddiffuse" "texture reflectance" "same" "texture displacement" "height"
Shape "sphere"
AttributeEnd
AttributeBegin
Translate 3 0 0
Material "diffuse" "texture reflectance" "other"
Shape "sphere"
AttributeEnd
""",
    )
    memory.add_file(
        "scene/assets/textures.pbrt",
        """Texture "included" "spectrum" "imagemap" "string filename" "shared.png"
Texture "height" "float" "imagemap" "string filename" "shared.png"
    "float scale" [0.25] "float uscale" [2] "float vscale" [3]
""",
    )
    var calls = List[Int](length=4, fill=0)
    var loader = _CountingAssetLoader(memory^, Span(calls))
    # Caches are local to a parse: a new scene must load fresh assets.
    for run in range(2):
        var scene = read_pbrt("scene/main.pbrt", loader)
        assert_equal(calls[2], run + 1)
        assert_equal(calls[3], run + 1)
        assert_equal(len(scene.data.surfaces().image_textures), 2)
        ref material = scene.data.surfaces().coated_diffuses[
            Int(scene.data.sphere_surfaces()[0].index())
        ]
        assert_equal(material.texture_index, UInt32(0))
        assert_equal(material.displacement_texture_index, UInt32(0))
        assert_almost_equal(material.displacement_scale, 0.25)
        assert_almost_equal(material.displacement_u_scale, 2.0)
        assert_almost_equal(material.displacement_v_scale, 3.0)
        assert_equal(scene.data.environment().texture_index, UInt32(0))
        assert_almost_equal(scene.data.environment().scale.x[0], 2.0)
        var color = scene.data.surfaces().sample_albedo(
            scene.data.sphere_surfaces()[0], 0.0, 0.0
        )
        assert_almost_equal(color.x[0], 0.2)
        color = scene.data.surfaces().sample_albedo(
            scene.data.sphere_surfaces()[1], 0.0, 0.0
        )
        assert_almost_equal(color.x[0], 0.8)


def test_ply_assets_share_geometry_and_preserve_both_orientations() raises:
    for first_reversed in [False, True]:
        var memory = MemoryAssetLoader()
        var source = String("WorldBegin\n")
        if first_reversed:
            source += "ReverseOrientation\n"
        source += """Include "assets/first.pbrt"
Translate 2 0 0
Material "diffuse" "rgb reflectance" [0.2 0.4 0.6]
Shape "plymesh" "string filename" "assets/shared.ply"
ReverseOrientation
Translate 2 0 0
AreaLightSource "diffuse" "rgb L" [3 4 5]
Shape "plymesh" "string filename" "assets/shared.ply"
Translate 2 0 0
Shape "plymesh" "string filename" "assets/shared.ply"
"""
        memory.add_file("scene/main.pbrt", source)
        memory.add_file(
            "scene/assets/first.pbrt",
            'Shape "plymesh" "string filename" "shared.ply"',
        )
        memory.add_ply_file("scene/assets/shared.ply", _triangle_ply())
        var calls = List[Int](length=4, fill=0)
        var loader = _CountingAssetLoader(memory^, Span(calls))
        var scene = read_pbrt("scene/main.pbrt", loader)
        assert_equal(calls[0], 1)
        assert_equal(len(scene.data.triangle_meshes()), 2)
        assert_equal(len(scene.data.triangle_instances()), 4)
        assert_equal(len(scene.data.lights().records), 2)
        for instance in range(4):
            assert_equal(
                scene.data.triangle_instances()[instance].blas_idx,
                UInt32(instance // 2),
            )
            assert_almost_equal(
                scene.data.triangle_instances()[instance].transform.tx,
                Float32(2 * instance),
            )
            assert_almost_equal(
                scene.data.triangle_instances()[instance].bounds._min.x[0],
                Float32(2 * instance),
            )
        for mesh_index in range(2):
            var reversed = first_reversed != (mesh_index == 1)
            ref vertices = scene.data.triangle_meshes()[mesh_index]
            ref normals = scene.data.triangle_mesh_normals()[mesh_index]
            ref uv = scene.data.triangle_mesh_texcoords()[mesh_index]
            assert_almost_equal(
                vertices[1].x, Float32(0.0 if reversed else 1.0)
            )
            assert_almost_equal(
                vertices[1].y, Float32(1.0 if reversed else 0.0)
            )
            assert_almost_equal(normals[3], Float32(0.5 if reversed else 0.25))
            assert_almost_equal(normals[6], Float32(0.25 if reversed else 0.5))
            assert_almost_equal(uv[2], Float32(0.0 if reversed else 1.0))
            assert_almost_equal(uv[3], Float32(1.0 if reversed else 0.0))
        assert_true(
            scene.data.triangle_instance_surfaces()[0].kind() == .LAMBERTIAN
        )
        assert_true(
            scene.data.triangle_instance_surfaces()[2].kind() == .EMISSIVE
        )
        var albedo = (
            scene.data.surfaces()
            .lambertians[
                Int(scene.data.triangle_instance_surfaces()[1].index())
            ]
            .albedo
        )
        assert_almost_equal(albedo.x[0], 0.2)


def test_distinct_ply_paths_with_same_filename_are_not_shared() raises:
    var memory = MemoryAssetLoader()
    memory.add_file(
        "scene/main.pbrt",
        """WorldBegin
Shape "plymesh" "string filename" "assets/shared.ply"
Shape "plymesh" "string filename" "other/shared.ply"
""",
    )
    memory.add_ply_file("scene/assets/shared.ply", _triangle_ply())
    memory.add_ply_file("scene/other/shared.ply", _triangle_ply())
    var calls = List[Int](length=4, fill=0)
    var loader = _CountingAssetLoader(memory^, Span(calls))
    var scene = read_pbrt("scene/main.pbrt", loader)
    assert_equal(calls[0], 1)
    assert_equal(calls[1], 1)
    assert_equal(len(scene.data.triangle_meshes()), 2)
    assert_equal(scene.data.triangle_instances()[0].blas_idx, UInt32(0))
    assert_equal(scene.data.triangle_instances()[1].blas_idx, UInt32(1))


def test_camera_read_stops_before_scene_assets() raises:
    var loader = MemoryAssetLoader()
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


def _assert_same_camera(actual: Camera, expected: Camera) raises:
    for y in range(3):
        for x in range(3):
            var actual_ray = actual.make_ray(x, y, 3, 3)
            var expected_ray = expected.make_ray(x, y, 3, 3)
            assert_almost_equal(actual_ray.o.x, expected_ray.o.x)
            assert_almost_equal(actual_ray.o.y, expected_ray.o.y)
            assert_almost_equal(actual_ray.o.z, expected_ray.o.z)
            assert_almost_equal(actual_ray.d.x, expected_ray.d.x)
            assert_almost_equal(actual_ray.d.y, expected_ray.d.y)
            assert_almost_equal(actual_ray.d.z, expected_ray.d.z)


def test_camera_only_matches_full_scene_camera() raises:
    for options in [
        "",
        'LookAt 1 2 3  -1 0 0  0 0 1\nCamera "perspective" "float fov" [31]\n',
        'Camera "perspective" "float fov" [31]\nCamera "perspective"\n',
        (
            'Film "rgb" "integer xresolution" [8]\n'
            'Sampler "independent" "integer pixelsamples" [1]\n'
            'Integrator "path" "integer maxdepth" [2]\n'
            'PixelFilter "box" "float xradius" [0.5]\n'
            'Accelerator "bvh" "integer maxnodeprims" [4]\n'
        ),
    ]:
        var loader = MemoryAssetLoader()
        loader.add_file("camera.pbrt", options + 'WorldBegin\nShape "sphere"')
        var scene = read_pbrt("camera.pbrt", loader)
        var camera = read_pbrt_camera("camera.pbrt", loader)
        _assert_same_camera(camera, scene.camera)


def test_camera_include_stops_parent_at_world_begin() raises:
    var loader = MemoryAssetLoader()
    loader.add_file(
        "scene/main.pbrt",
        (
            'Include "parts/header.pbrt"\nCamera "orthographic"\nInclude'
            ' "missing.pbrt"'
        ),
    )
    loader.add_file(
        "scene/parts/header.pbrt",
        'Include "camera.pbrt"\nWorldBegin\nInclude "missing.pbrt"',
    )
    loader.add_file(
        "scene/parts/camera.pbrt",
        'LookAt 1 2 3  -1 0 0  0 0 1\nCamera "perspective" "float fov" [31]',
    )
    var camera = read_pbrt_camera("scene/main.pbrt", loader)
    var expected = Camera.from_vfov(
        Point3f32[.WORLD](1.0, 2.0, 3.0),
        Point3f32[.WORLD](-1.0, 0.0, 0.0),
        Vec3f32[.WORLD](0.0, 0.0, 1.0),
        31.0,
    )
    expected.right = -expected.right
    _assert_same_camera(camera, expected)


def test_camera_only_preserves_options_validation() raises:
    var loader = MemoryAssetLoader()
    loader.add_file(
        "camera.pbrt",
        (
            'Film "rgb" "integer xresolution" [unused]\n'
            'Sampler "independent" "integer pixelsamples" [unused]\n'
            'Integrator "bdpt" "integer maxdepth" [unused]\n'
            'ColorSpace "aces"\nWorldBegin'
        ),
    )
    var camera = read_pbrt_camera("camera.pbrt", loader)
    assert_almost_equal(camera.origin.z, 5.0)
    with assert_raises():
        _ = read_pbrt("camera.pbrt", loader)

    for source in [
        'Camera "orthographic"',
        'Camera "perspective" "float fov" [invalid]',
        "LookAt 0 0 invalid  0 0 0  0 1 0",
        'Film "rgb" "integer xresolution" [1',
        'Include "missing.pbrt"',
        'Include "camera.pbrt"',
    ]:
        loader.add_file("camera.pbrt", source)
        with assert_raises():
            _ = read_pbrt_camera("camera.pbrt", loader)


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
    "float thickness" [0.03] "rgb albedo" [0.1 0.2 0.3] "float g" [0.4]
    "integer maxdepth" [7] "integer nsamples" [3]
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
    assert_almost_equal(material.roughness, sqrt(Float32(0.025)))
    assert_almost_equal(material.eta, 1.5)
    assert_almost_equal(material.thickness, 0.03)
    assert_almost_equal(material.layer_albedo.x, 0.1)
    assert_almost_equal(material.layer_albedo.y, 0.2)
    assert_almost_equal(material.layer_albedo.z, 0.3)
    assert_almost_equal(material.g, 0.4)
    assert_true(material.max_depth == 7)
    assert_true(material.n_samples == 3)


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
    var loader = MemoryAssetLoader()
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
    var loader = MemoryAssetLoader()
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
    assert_true(scene.data.environment().kind == EnvironmentKind.BLACK)
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
    var loader = MemoryAssetLoader()
    loader.add_file(
        "scene/main.pbrt",
        """LookAt 0 0 4  0 0 0  0 1 0
Camera "perspective" "float fov" 40
Film "rgb" "integer xresolution" 4 "integer yresolution" 4
Sampler "independent" "integer pixelsamples" 1
WorldBegin
LightSource "infinite" "rgb L" [1 1 1]
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
    assert_true(scene.data.environment().kind == EnvironmentKind.UNIFORM)
    assert_almost_equal(scene.data.environment().scale.x[0], 1.0)
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


def test_indexed_mesh_reuses_vertices_and_preserves_unused_values() raises:
    var scene = parse_pbrt(
        "WorldBegin\nTranslate 1 2 3\nReverseOrientation\n"
        'Shape "trianglemesh" "point P" [0 0 0 1 0 0 0 1 0 bad bad bad]'
        ' "integer indices" [0 1 2 0 1 2]'
    )
    ref vertices = scene.data.triangle_vertices()
    assert_true(len(vertices) == 6)
    for base in [0, 3]:
        assert_almost_equal(vertices[base].x, 1.0)
        assert_almost_equal(vertices[base].y, 2.0)
        assert_almost_equal(vertices[base].z, 3.0)
        assert_almost_equal(vertices[base + 1].x, 1.0)
        assert_almost_equal(vertices[base + 1].y, 3.0)
        assert_almost_equal(vertices[base + 2].x, 2.0)
        assert_almost_equal(vertices[base + 2].y, 2.0)
    with assert_raises():
        _ = parse_pbrt(
            'WorldBegin\nShape "trianglemesh" "point3 P" [0 0 0 bad 0 0 0 1 0]'
            ' "integer indices" [0 1 2]'
        )
    with assert_raises():
        _ = parse_pbrt('WorldBegin\nAttributeBegin\nShape "sphere"')
    with assert_raises():
        _ = parse_pbrt(
            'WorldBegin\nTexture "bad" "spectrum" "imagemap" '
            '"string filename" "missing.png"\nShape "sphere"'
        )


def test_numeric_token_views() raises:
    var scene = parse_pbrt(
        'Film "rgb" "integer xresolution" +4 "integer yresolution" [3e0]\n'
        'Sampler "independent" "integer pixelsamples" [2]\n'
        'Integrator "path" "integer maxdepth" 5\n'
        'WorldBegin\nMaterial "diffuse" "rgb reflectance" [.25 +.5 7.5e-1]\n'
        "Transform[1 0 0 +1e0 0 1 0 -2. 0 0 1 .5 0 0 0 1]\n"
        "ConcatTransform[1 0 0 0 0 1 0 0 0 0 1 +.5 0 0 0 1]\n"
        "Translate +.5 0 0\n"
        'Shape "trianglemesh" "point3 P" [0 0 0# adjacent comment\n'
        '1.25E+0 0 0\t0 2.5e-1 0] "integer indices" [+0 1e0 2]\n'
        'Shape "sphere" "float radius" "2.5e-1"'
    )
    assert_true(scene.settings.image_width == 4)
    assert_true(scene.settings.image_height == 3)
    assert_true(scene.settings.samples_per_pixel == 2)
    assert_true(scene.settings.max_depth == 5)
    ref vertices = scene.data.triangle_vertices()
    assert_true(len(vertices) == 3)
    assert_almost_equal(vertices[0].x, 1.5)
    assert_almost_equal(vertices[0].y, -2.0)
    assert_almost_equal(vertices[0].z, 1.0)
    assert_almost_equal(vertices[1].x, 2.75)
    assert_almost_equal(vertices[2].y, -1.75)
    assert_almost_equal(scene.data.spheres()[0].radius, 0.25)
    ref surface = scene.data.sphere_surfaces()[0]
    var albedo = scene.data.surfaces().lambertians[Int(surface.index())].albedo
    assert_almost_equal(albedo.x[0], 0.25)
    assert_almost_equal(albedo.y[0], 0.5)
    assert_almost_equal(albedo.z[0], 0.75)


def test_include_definitions_survive_source_text() raises:
    var loader = MemoryAssetLoader()
    loader.add_file(
        "scene/main.pbrt",
        (
            'WorldBegin\nInclude "definitions.pbrt"\n'
            'NamedMaterial "included"\nShape "sphere"'
        ),
    )
    loader.add_file(
        "scene/definitions.pbrt",
        (
            'Texture "tint" "spectrum" "constant" "rgb value" [.2 .4 .6]\n'
            'MakeNamedMaterial "included" "string type" "diffuse" '
            '"texture reflectance" "tint"'
        ),
    )
    var scene = read_pbrt("scene/main.pbrt", loader)
    ref surface = scene.data.sphere_surfaces()[0]
    var albedo = scene.data.surfaces().lambertians[Int(surface.index())].albedo
    assert_almost_equal(albedo.x[0], 0.2)
    assert_almost_equal(albedo.y[0], 0.4)
    assert_almost_equal(albedo.z[0], 0.6)


def test_numeric_token_errors() raises:
    for source in [
        'WorldBegin\nTranslate 1x 0 0\nShape "sphere"',
        'WorldBegin\nShape "sphere" "float radius" [1e]',
        'WorldBegin\nShape "sphere" "float radius" [1x]',
        'Film "rgb" "integer xresolution" [1.5]\nWorldBegin\nShape "sphere"',
        'WorldBegin\nShape "sphere" "float radius"',
        'WorldBegin\nShape "sphere" "float radius" [1',
        'WorldBegin\nTransform [1 0 0 0]\nShape "sphere"',
        (
            "WorldBegin\nTransform [1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 bad]\n"
            'Shape "sphere"'
        ),
    ]:
        with assert_raises():
            _ = parse_pbrt(source)


def test_integer_fast_path_preserves_float32_fallback_semantics() raises:
    for spelling in [
        "0",
        "-0",
        "+00012",
        "16777215",
        "16777216",
        "16777217",
        "-16777217",
        "0000000000000000000000000000001",
        "1e0",
        "2.0",
        "3.00000001",
        "",
        "+",
        "-",
    ]:
        var text = StringSpan(spelling)
        var expected = Int(parse_f32_at(text.as_bytes(), 0).value)
        assert_equal(parse_int_f32(text), expected)
        assert_equal(parse_int_f32(text, "PBRT"), expected)
    for spelling in ["1.5", "1x", "1e", "1e309", "++1", "-+1"]:
        with assert_raises():
            _ = parse_int_f32(StringSpan(spelling), "PBRT")


def test_simd_lexer_preserves_tokens_quotes_comments_and_line_numbers() raises:
    comptime width = min(64, simd_width_of[UInt8]())
    for length in range(3 * width + 4):
        var prefix = String()
        for _ in range(length):
            prefix += "x"
        var text = (
            prefix
            + '\t # ignored "[\n[\n"rgb\n L"\t-1.25# ignored'
            ' ]"\n]\r\nword"inner"# tail\n'
        )
        var lexer = _Lexer(StringSpan(text).as_bytes())
        if length:
            var token = lexer.next()
            assert_equal(token.value, StringSpan(prefix))
            assert_equal(token.line, 1)
            assert_true(not token.quoted)
        var values = ["[", "rgb\n L", "-1.25", "]", 'word"inner"']
        var lines = [2, 3, 4, 5, 6]
        for index in range(len(values)):
            assert_true(lexer.has_next())
            assert_equal(lexer.next_is_quoted(), index == 1)
            assert_true(lexer.has_next())
            var token = lexer.next()
            assert_equal(token.value, StringSpan(values[index]))
            assert_equal(token.line, lines[index])
            assert_equal(token.quoted, index == 1)
        assert_true(not lexer.has_next())
        assert_equal(lexer.line, 7)


def test_simd_lexer_handles_partial_blocks_and_unterminated_multiline_quotes() raises:
    comptime width = min(64, simd_width_of[UInt8]())
    for length in range(3 * width + 4):
        var prefix = String()
        var start_line = 1
        var body_lines = 0
        var body = String()
        for index in range(length):
            if index % 7 == 0:
                prefix += "\n"
                start_line += 1
            else:
                prefix += "\t "
            if index % 11 == 0:
                body += "\n"
                body_lines += 1
            else:
                body += "β#["
        var text = prefix + '"' + body + '"'
        var lexer = _Lexer(StringSpan(text).as_bytes())
        var token = lexer.next()
        assert_true(token.quoted)
        assert_equal(token.value, StringSpan(body))
        assert_equal(token.line, start_line)
        assert_true(not lexer.has_next())
        assert_equal(lexer.line, start_line + body_lines)
        var incomplete = prefix + '"' + body
        var bad = _Lexer(StringSpan(incomplete).as_bytes())
        with assert_raises():
            _ = bad.next()
        assert_equal(bad.line, start_line + body_lines)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
