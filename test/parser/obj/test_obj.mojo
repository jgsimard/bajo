from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_raises,
    assert_true,
    assert_false,
)

from bajo.parser.asset_loader import MemoryAssetLoader
from bajo.core import Frame
from bajo.parser.obj import (
    ObjMesh,
    read_obj,
    parse_obj,
    triangulated_indices,
)
from bajo.parser.obj.pack import (
    pack_obj_triangles,
    _pack_obj_triangles_parallel,
)
from bajo.parser.obj.cursor import ObjIndexLimit
from bajo.parser.obj.types import ObjIndex
from std.sys import size_of
from std.memory import bitcast
from bajo.parser.number import F32ParseResult, parse_f32_at
from bajo.parser.number import parse_f32_fast_at
from bajo.parser.obj.parallel import _parse_obj_bulk
from bajo.parser.obj.serial import _parse_obj_serial
from bajo.parser.obj.triangulate import _triangulated_indices_parallel


def _assert_obj_parse_error(text: String) raises:
    with assert_raises():
        _ = parse_obj(text, "invalid.obj")


def test_basic_triangle() raises:
    comptime obj = """v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1"""

    var mesh = parse_obj(obj, "triangle.obj")
    assert_true(mesh.position_count(include_dummy=False) == 3)
    assert_true(mesh.texcoord_count(include_dummy=False) == 3)
    assert_true(mesh.normal_count(include_dummy=False) == 1)
    assert_true(mesh.face_count() == 1)
    assert_true(mesh.index_count() == 3)
    assert_true(mesh.indices[0].p == 1)
    assert_true(mesh.indices[0].t == 1)
    assert_true(mesh.indices[0].n == 1)
    assert_true(mesh.indices[2].p == 3)
    assert_true(mesh.indices[2].t == 3)
    assert_true(mesh.indices[2].n == 1)


def test_negative_indices_and_triangulation() raises:
    comptime obj = """v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f -4 -3 -2 -1"""

    var mesh = parse_obj(obj, "quad.obj")
    var tris = triangulated_indices(mesh)
    assert_true(mesh.face_vertices[0] == 4)
    assert_true(mesh.indices[0].p == 1)
    assert_true(mesh.indices[3].p == 4)
    assert_true(len(tris) == 6)
    assert_true(tris[0].p == 1)
    assert_true(tris[1].p == 2)
    assert_true(tris[2].p == 3)
    assert_true(tris[3].p == 1)
    assert_true(tris[4].p == 3)
    assert_true(tris[5].p == 4)


def test_direct_triangle_packing_preserves_fans_and_skips_lines() raises:
    var mesh = parse_obj(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv -1 0 0\n"
        "l 1 2 3\nf -5 -4 -3 -2 -1\nf 1 3 5\nl 4 5"
    )
    var indices = triangulated_indices(mesh)
    var packed = pack_obj_triangles(mesh)
    assert_true(len(packed) == 12)
    assert_true(len(packed) == len(indices))
    for corner in range(len(indices)):
        var base = 3 * Int(indices[corner].p)
        assert_almost_equal(packed[corner].x, mesh.positions[base])
        assert_almost_equal(packed[corner].y, mesh.positions[base + 1])
        assert_almost_equal(packed[corner].z, mesh.positions[base + 2])

    var empty = parse_obj("v 0 0 0\nv 1 0 0\nl 1 2")
    assert_true(len(pack_obj_triangles(empty)) == 0)


def test_parallel_packing_preserves_fans_lines_and_chunk_offsets() raises:
    var text = String("v -0.0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv -1 0 0\n")
    for _ in range(35):
        text += "l 1 2 3\nf 1 3 5\nf 1 2 3 4 5\nl 4 5\nf 1 2 3 4\n"
    var mesh = parse_obj(text)
    var indices = triangulated_indices(mesh)
    for workers in [1, 2, 8, 16]:
        var packed = _pack_obj_triangles_parallel(mesh, workers, 1)
        assert_true(len(packed) == len(indices))
        for corner in range(len(indices)):
            var base = 3 * Int(indices[corner].p)
            assert_true(
                bitcast[.uint32](packed[corner].x)
                == bitcast[.uint32](mesh.positions[base])
            )
            assert_true(
                bitcast[.uint32](packed[corner].y)
                == bitcast[.uint32](mesh.positions[base + 1])
            )
            assert_true(
                bitcast[.uint32](packed[corner].z)
                == bitcast[.uint32](mesh.positions[base + 2])
            )
    var empty = parse_obj("v 0 0 0\nv 1 0 0\nl 1 2\nl 2 1\nl 1 2")
    assert_true(len(_pack_obj_triangles_parallel(empty, 8, 1)) == 0)
    var no_faces = ObjMesh()
    assert_true(len(_pack_obj_triangles_parallel(no_faces, 8, 1)) == 0)
    var world = _pack_obj_triangles_parallel[Frame.WORLD](mesh, 8, 1)
    assert_true(len(world) == len(indices))
    for corner in range(len(indices)):
        var base = 3 * Int(indices[corner].p)
        assert_true(
            bitcast[.uint32](world[corner].x)
            == bitcast[.uint32](mesh.positions[base])
        )


def test_parallel_triangulation_preserves_all_corner_indices_and_order() raises:
    var text = String(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv -1 0 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvt 0.5 0.5\n"
        "vn 1 0 0\nvn 0 1 0\nvn 0 0 1\nvn -1 0 0\nvn 0 -1 0\n"
    )
    for _ in range(35):
        text += "l 1/1 2/2 3/3\nf 1/5/2 3/3/4 5/1/1\n"
        text += "f -5/-1/-4 -4/-2/-3 -3/-3/-2 -2/-4/-1 -1/-5/-5\n"
        text += "l 4 5\nf 1 2 3 4\n"
    var mesh = parse_obj(text)
    # Expected fan order, expressed as source corner offsets in each block.
    var offsets = [3, 4, 5, 6, 7, 8, 6, 8, 9, 6, 9, 10, 13, 14, 15, 13, 15, 16]
    for workers in [1, 2, 8, 16]:
        var actual = _triangulated_indices_parallel(mesh, workers, 1)
        assert_true(len(actual) == 35 * len(offsets))
        for block in range(35):
            for corner in range(len(offsets)):
                var expected = mesh.indices[17 * block + offsets[corner]]
                var index = actual[block * len(offsets) + corner]
                assert_true(index.p == expected.p)
                assert_true(index.t == expected.t)
                assert_true(index.n == expected.n)
    var empty = parse_obj("v 0 0 0\nv 1 0 0\nl 1 2\nl 2 1\nl 1 2")
    assert_true(len(_triangulated_indices_parallel(empty, 8, 1)) == 0)
    var no_faces = ObjMesh()
    assert_true(len(_triangulated_indices_parallel(no_faces, 8, 1)) == 0)


def test_face_indices_are_range_validated() raises:
    _assert_obj_parse_error("v 0 0 0\nf 2 1 1")
    _assert_obj_parse_error("v 0 0 0\nf -2 -1 -1")
    _assert_obj_parse_error("v 0 0 0\nvt 0 0\nf 1/2 1/1 1/1")
    _assert_obj_parse_error("v 0 0 0\nvn 0 0 1\nf 1//2 1//1 1//1")
    _assert_obj_parse_error("v 0 0 0\nf 999999999999999999999 1 1")
    _assert_obj_parse_error("v 0 0 0\nvt 0 0\nvn 0 0 1\nf 1/1/1 1/1 1/1/1")


def test_compact_indices_check_the_storage_limit() raises:
    assert_true(size_of[ObjIndex]() == 12)
    var limit = ObjIndexLimit(0x80000000)
    assert_true(limit.max_magnitude == 0x7FFFFFFF)
    var index = ObjIndex(0x7FFFFFFF, 0, 0)
    assert_true(Int(index.p) == 0x7FFFFFFF)
    with assert_raises():
        _ = ObjIndexLimit(0x80000001)


def test_decimal_exponent_work_is_bounded() raises:
    var mesh = parse_obj("v 1e3 1e-3 1e+2", "exponents.obj")
    assert_almost_equal(mesh.positions[3], 1000.0)
    assert_almost_equal(mesh.positions[4], 0.001)
    assert_almost_equal(mesh.positions[5], 100.0)

    _assert_obj_parse_error("v 1e309 0 0")
    _assert_obj_parse_error("v 1e-309 0 0")
    _assert_obj_parse_error("v 1e999999999999999999999 0 0")


def _assert_fast_decimal_matches_reference(text: String) raises:
    var bytes = StringSpan(text).as_bytes()
    var expected = parse_f32_at(bytes, 0)
    var actual = parse_f32_fast_at(bytes, 0)
    assert_true(actual.pos == expected.pos)
    assert_true(
        bitcast[.uint32](actual.value) == bitcast[.uint32](expected.value)
    )


def test_decimal_fast_path_matches_reference_rounding_and_consumption() raises:
    for text in [
        "",
        "+",
        "-",
        ".",
        "-0",
        "-0.000000",
        "1.23456789",
        ".1234567890123456789",
        "123456789012345.1234567890123456789",
        "1234567890123456.5",
        "1e-3",
        "+1.5E+20",
        "0.000000000000000000000000000000000000001",
        "1.23456789012345678901234567#comment",
        "-20.1537690000000002 0",
        "0.5000000298023223876953125",
        "1.000000059604644775390625",
        "0.9999999701976776123046875",
        "3.4028234663852886e38",
        "1.401298464324817e-45",
    ]:
        _assert_fast_decimal_matches_reference(text)
    var state = UInt64(817)
    for _ in range(20000):
        state = state * UInt64(6364136223846793005) + 1
        var whole = Int(state % 1000000)
        state = state * UInt64(6364136223846793005) + 1
        var fraction = Int(state % 1000000000000000000)
        _assert_fast_decimal_matches_reference(String(t"-{whole}.{fraction} "))


def test_vertex_colors_lazy_fill() raises:
    comptime obj = """v 0 0 0
v 1 0 0 0.25 0.5 0.75
v 2 0 0"""

    var mesh = parse_obj(obj, "colors.obj")
    assert_true(mesh.color_count() == 4)
    assert_almost_equal(mesh.colors[0], 1.0)
    assert_almost_equal(mesh.colors[1], 1.0)
    assert_almost_equal(mesh.colors[2], 1.0)
    assert_almost_equal(mesh.colors[3], 1.0)
    assert_almost_equal(mesh.colors[4], 1.0)
    assert_almost_equal(mesh.colors[5], 1.0)
    assert_almost_equal(mesh.colors[6], 0.25)
    assert_almost_equal(mesh.colors[7], 0.5)
    assert_almost_equal(mesh.colors[8], 0.75)
    assert_almost_equal(mesh.colors[9], 1.0)
    assert_almost_equal(mesh.colors[10], 1.0)
    assert_almost_equal(mesh.colors[11], 1.0)


def test_groups_objects_and_lines() raises:
    comptime obj = """o Cube
g Front
v 0 0 0
v 1 0 0
v 0 1 0
l 1 2
f 1 2 3"""

    var mesh = parse_obj(obj, "groups.obj")
    var tris = triangulated_indices(mesh)
    assert_true(mesh.face_count() == 2)
    assert_true(len(mesh.face_lines) == 2)
    assert_true(mesh.face_lines[0] == 1)
    assert_true(mesh.face_lines[1] == 0)
    assert_true(len(tris) == 3)
    assert_true(mesh.object_count() == 1)
    assert_true(mesh.objects[0].name == "Cube")
    assert_true(mesh.objects[0].face_count == 2)

    assert_true(mesh.group_count() == 1)
    assert_true(mesh.groups[0].name == "Front")
    assert_true(mesh.groups[0].face_count == 2)


def test_material_fallback_replacement_and_textures() raises:
    var loader = MemoryAssetLoader()

    comptime obj = """usemtl matA
mtllib matA.mtl
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3"""

    comptime mtl = """newmtl matA
Ka 0.01 0.02 0.03
Kd 0.1 0.2 0.3
Ks 0.4 0.5 0.6
Ke 0.7 0.8 0.9
Ns 42
Ni 1.45
d 0.5
illum 2
map_Kd diffuse.png
bump normal.png"""

    loader.add_file("assets/material_test.obj", obj)
    loader.add_file("assets/matA.mtl", mtl)

    var mesh = read_obj("assets/material_test.obj", loader)
    var mat = mesh.materials[mesh.face_materials[0]].copy()

    assert_true(mesh.material_count() == 1 and not mat.fallback)
    assert_almost_equal(mat.Kd[0], 0.1)
    assert_almost_equal(mat.Kd[1], 0.2)
    assert_almost_equal(mat.Kd[2], 0.3)
    assert_almost_equal(mat.Ns, 42.0)
    assert_almost_equal(mat.Ni, 1.45)
    assert_almost_equal(mat.d, 0.5)
    assert_true(mat.illum == 2)
    assert_true(mesh.texture_count() == 3)
    assert_true(mesh.texture_count(include_dummy=False) == 2)
    assert_true(mat.map_Kd == 1)
    assert_true(mesh.textures[1].name == "diffuse.png")
    assert_true(mesh.textures[1].path == "assets/diffuse.png")
    assert_true(mat.map_bump == 2)
    assert_true(mesh.textures[2].name == "normal.png")
    assert_true(mesh.textures[2].path == "assets/normal.png")


def test_memory_loader_mtllib_with_spaces_and_texture_dedup() raises:
    var loader = MemoryAssetLoader()

    comptime obj = """mtllib material library.mtl
usemtl matB
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""

    var mtl = String("")
    mtl += "newmtl matB\n"
    mtl += "map_Ka shared.png\n"
    mtl += "map_Kd shared.png\n"
    mtl += "map_bump -bm 0.5 shared.png\n"

    loader.add_file("models/model.obj", obj)
    loader.add_file("models/material library.mtl", mtl)

    var mesh = read_obj("models/model.obj", loader)
    var mat_idx = mesh.face_materials[0]
    var mat = mesh.materials[mat_idx].copy()

    assert_true(mesh.face_count() == 1)
    assert_true(mesh.position_count(include_dummy=False) == 3)
    assert_true(mesh.material_count() == 1 and mat.name == "matB")
    assert_true(mesh.texture_count() == 2)
    assert_true(mesh.texture_count(include_dummy=False) == 1)
    assert_true(mat.map_Ka == 1 and mat.map_Kd == 1 and mat.map_bump == 1)
    assert_true(mesh.textures[1].path == "models/shared.png")


def test_parse_obj_from_string() raises:
    comptime obj = """v 0 0 0 
v 1 0 0
v 0 1 0
f 1 2 3
"""

    var mesh = parse_obj(obj)
    assert_true(mesh.face_count() == 1)
    assert_true(mesh.index_count() == 3)


def test_comments_blank_lines_and_crlf() raises:
    comptime obj = """# comment\r
\r
v 0 0 0 # inline comment\r
v 1 0 0\r
v 0 1 0\r
f 1 2 3\r"""

    var mesh = parse_obj(obj, "comments.obj")
    assert_true(mesh.position_count(include_dummy=False) == 3)
    assert_true(mesh.face_count() == 1)
    assert_true(mesh.index_count() == 3)


def test_missing_texcoord_face_form() raises:
    comptime obj = """v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
f 1//1 2//1 3//1"""

    var mesh = parse_obj(obj, "missing_texcoord.obj")
    assert_true(mesh.face_count() == 1)
    assert_true(mesh.indices[0].p == 1)
    assert_true(mesh.indices[0].t == 0)
    assert_true(mesh.indices[0].n == 1)
    assert_true(mesh.indices[2].p == 3)
    assert_true(mesh.indices[2].t == 0)
    assert_true(mesh.indices[2].n == 1)


def test_multiple_groups_flush_offsets() raises:
    comptime obj = """g A
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
f 1 2 3
g B
f 2 4 3"""

    var mesh = parse_obj(obj, "groups2.obj")
    assert_true(mesh.group_count() == 2)
    assert_true(mesh.groups[0].name == "A")
    assert_true(mesh.groups[0].face_count == 1)
    assert_true(mesh.groups[0].face_offset == 0)
    assert_true(mesh.groups[0].index_offset == 0)
    assert_true(mesh.groups[1].name == "B")
    assert_true(mesh.groups[1].face_count == 1)
    assert_true(mesh.groups[1].face_offset == 1)
    assert_true(mesh.groups[1].index_offset == 3)


def test_tr_only_transparency() raises:
    var loader = MemoryAssetLoader()

    comptime obj = """mtllib glass.mtl
usemtl glass
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3"""

    comptime mtl = """newmtl glass
Tr 0.25"""

    loader.add_file("materials/glass.obj", obj)
    loader.add_file("materials/glass.mtl", mtl)

    var mesh = read_obj("materials/glass.obj", loader)
    var mat = mesh.materials[mesh.face_materials[0]].copy()
    assert_almost_equal(mat.d, 0.75)


def test_d_overrides_tr_order() raises:
    var loader = MemoryAssetLoader()

    comptime obj = """mtllib mat.mtl
usemtl mat
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3"""

    comptime mtl = """newmtl mat
d 0.4
Tr 0.9"""

    loader.add_file("materials/mat.obj", obj)
    loader.add_file("materials/mat.mtl", mtl)

    var mesh = read_obj("materials/mat.obj", loader)
    var mat = mesh.materials[mesh.face_materials[0]].copy()
    assert_almost_equal(mat.d, 0.4)


def _assert_mesh_equal(actual: ObjMesh, expected: ObjMesh) raises:
    assert_true(len(actual.positions) == len(expected.positions))
    assert_true(len(actual.texcoords) == len(expected.texcoords))
    assert_true(len(actual.normals) == len(expected.normals))
    assert_true(len(actual.colors) == len(expected.colors))
    for i in range(len(expected.positions)):
        assert_true(
            bitcast[.uint32](actual.positions[i])
            == bitcast[.uint32](expected.positions[i])
        )
    for i in range(len(expected.texcoords)):
        assert_true(
            bitcast[.uint32](actual.texcoords[i])
            == bitcast[.uint32](expected.texcoords[i])
        )
    for i in range(len(expected.normals)):
        assert_true(
            bitcast[.uint32](actual.normals[i])
            == bitcast[.uint32](expected.normals[i])
        )
    for i in range(len(expected.colors)):
        assert_true(
            bitcast[.uint32](actual.colors[i])
            == bitcast[.uint32](expected.colors[i])
        )
    assert_true(actual.face_count() == expected.face_count())
    assert_true(actual.index_count() == expected.index_count())
    for i in range(expected.face_count()):
        assert_true(actual.face_vertices[i] == expected.face_vertices[i])
        assert_true(actual.face_materials[i] == expected.face_materials[i])
    assert_true(len(actual.face_lines) == len(expected.face_lines))
    for i in range(len(expected.face_lines)):
        assert_true(actual.face_lines[i] == expected.face_lines[i])
    for i in range(expected.index_count()):
        assert_true(actual.indices[i].p == expected.indices[i].p)
        assert_true(actual.indices[i].t == expected.indices[i].t)
        assert_true(actual.indices[i].n == expected.indices[i].n)
    assert_true(len(actual.objects) == len(expected.objects))
    assert_true(len(actual.groups) == len(expected.groups))
    for i in range(len(expected.objects)):
        assert_true(actual.objects[i].name == expected.objects[i].name)
        assert_true(
            actual.objects[i].face_count == expected.objects[i].face_count
        )
        assert_true(
            actual.objects[i].face_offset == expected.objects[i].face_offset
        )
        assert_true(
            actual.objects[i].index_offset == expected.objects[i].index_offset
        )
    for i in range(len(expected.groups)):
        assert_true(actual.groups[i].name == expected.groups[i].name)
        assert_true(
            actual.groups[i].face_count == expected.groups[i].face_count
        )
        assert_true(
            actual.groups[i].face_offset == expected.groups[i].face_offset
        )
        assert_true(
            actual.groups[i].index_offset == expected.groups[i].index_offset
        )
    assert_true(len(actual.materials) == len(expected.materials))
    for i in range(len(expected.materials)):
        assert_true(actual.materials[i].name == expected.materials[i].name)
        assert_true(
            actual.materials[i].fallback == expected.materials[i].fallback
        )
        assert_almost_equal(
            actual.materials[i].Kd[0], expected.materials[i].Kd[0]
        )
        assert_true(actual.materials[i].map_Kd == expected.materials[i].map_Kd)
    assert_true(len(actual.textures) == len(expected.textures))
    for i in range(len(expected.textures)):
        assert_true(actual.textures[i].name == expected.textures[i].name)
        assert_true(actual.textures[i].path == expected.textures[i].path)


def test_bulk_decoder_preserves_order_and_state_across_chunks() raises:
    var loader = MemoryAssetLoader()
    loader.add_file(
        "bulk/first.mtl", "newmtl first\nKd 0.2 0.3 0.4\nmap_Kd shared.png"
    )
    loader.add_file(
        "bulk/second.mtl", "newmtl second\nKd 0.8 0.3 0.4\nmap_Kd shared.png"
    )
    var text = String("usemtl first\nmtllib first.mtl\nv 1\nvt 0\nvn 1 0\n")
    for block in range(24):
        text += String(t"o object {block}\ng group {block}\n")
        if block % 2 == 0:
            text += "usemtl first\n"
        else:
            text += "usemtl second\n"
        text += "v 0 0 0\nv 1 0 0 0.25 0.5 0.75\nv 0 1 0 # comment\r\n"
        text += "vt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 1\n"
        text += "f -3/-3/-1 -2/-2/-1 -1/-1/-1\n"
        text += "f 1//1 2//1 3//1\nl 1/1 2/2\nf 1 2\nl 1\n"
    text += "mtllib second.mtl\ng\nf 1 2 3"
    var reference = _parse_obj_serial("bulk/mesh.obj", StringSpan(text), loader)
    var sequential = _parse_obj_bulk(
        "bulk/mesh.obj", StringSpan(text), loader, 1, 37
    )
    var parallel = _parse_obj_bulk(
        "bulk/mesh.obj", StringSpan(text), loader, 8, 37
    )
    _assert_mesh_equal(sequential, reference)
    _assert_mesh_equal(parallel, reference)


def test_bulk_decoder_retains_validation_and_unusual_input_behavior() raises:
    var loader = MemoryAssetLoader()
    for text in [
        "v 0 0 0\nf 2 1 1\nv 1 0 0",
        "v 0 0 0\nf -2 -1 -1",
        "v 0 0 0\nf 999999999999999999999 1 1",
        "v 0 0 0\nvt 0 0\nf 1/2 1/1 1/1",
        "v 0 0 0\nvn 0 0 1\nf 1//2 1//1 1//1",
        "v 0 0 0\nvt 0 0\nvn 0 0 1\nf 1/1/1 1/1 1/1/1",
        "v 1e309 0 0",
        "v 1e99999999999999999 0 0",
        "mtllib missing.mtl\nv 0 0 0",
    ]:
        with assert_raises():
            _ = _parse_obj_bulk(
                "bulk/mesh.obj", StringSpan(text), loader, 8, 13
            )
    for text in [
        "",
        "# comment",
        "v 1x",
        "v 1x\nv 1 2 3\nf 1 2 2",
        "v . . .",
        "v 1 2",
        "v 1 2 3 0.5",
        "g empty\nl\nf",
    ]:
        var reference = _parse_obj_serial(
            "bulk/mesh.obj", StringSpan(text), loader
        )
        var actual = _parse_obj_bulk(
            "bulk/mesh.obj", StringSpan(text), loader, 8, 1
        )
        _assert_mesh_equal(actual, reference)


def test_decimal_blocks_reject_every_nondigit_byte() raises:
    for lane in range(8):
        for byte in range(256):
            var bytes = List[UInt8](length=11, fill=UInt8(49))
            bytes[0] = UInt8(48)
            bytes[1] = UInt8(46)
            bytes[2 + lane] = UInt8(byte)
            bytes[10] = UInt8(32)
            var expected = F32ParseResult(0.0, 0)
            var reference_raised = False
            try:
                expected = parse_f32_at(Span(bytes), 0)
            except:
                reference_raised = True
            if reference_raised:
                with assert_raises():
                    _ = parse_f32_fast_at(Span(bytes), 0)
            else:
                var actual = parse_f32_fast_at(Span(bytes), 0)
                assert_true(actual.pos == expected.pos)
                assert_true(
                    bitcast[.uint32](actual.value)
                    == bitcast[.uint32](expected.value)
                )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
