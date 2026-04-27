from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_true,
    assert_false,
)


from bajo.obj import (
    ObjMesh,
    read_obj,
    parse_obj_text,
    triangulated_indices,
    MemoryObjTextLoader,
    read_obj_from_memory,
    read_obj_from_string,
    _read_mtl_text,
)


def test_basic_triangle() raises:
    var obj = String("")
    obj += "v 0 0 0\n"
    obj += "v 1 0 0\n"
    obj += "v 0 1 0\n"
    obj += "vt 0 0\n"
    obj += "vt 1 0\n"
    obj += "vt 0 1\n"
    obj += "vn 0 0 1\n"
    obj += "f 1/1/1 2/2/1 3/3/1"

    var mesh = parse_obj_text("triangle.obj", obj)
    assert_true(mesh.actual_position_count() == 3)
    assert_true(mesh.actual_texcoord_count() == 3)
    assert_true(mesh.actual_normal_count() == 1)
    assert_true(mesh.face_count() == 1)
    assert_true(mesh.index_count() == 3)
    assert_true(
        mesh.indices[0].p == 1
        and mesh.indices[0].t == 1
        and mesh.indices[0].n == 1
    )
    assert_true(
        mesh.indices[2].p == 3
        and mesh.indices[2].t == 3
        and mesh.indices[2].n == 1
    )


def test_negative_indices_and_triangulation() raises:
    var obj = String("")
    obj += "v 0 0 0\n"
    obj += "v 1 0 0\n"
    obj += "v 1 1 0\n"
    obj += "v 0 1 0\n"
    obj += "f -4 -3 -2 -1"

    var mesh = parse_obj_text("quad.obj", obj)
    var tris = triangulated_indices(mesh)
    assert_true(mesh.face_vertices[0] == 4)
    assert_true(mesh.indices[0].p == 1)
    assert_true(mesh.indices[3].p == 4)
    assert_true(len(tris) == 6)
    assert_true(tris[0].p == 1 and tris[1].p == 2 and tris[2].p == 3)
    assert_true(tris[3].p == 1 and tris[4].p == 3 and tris[5].p == 4)


def test_vertex_colors_lazy_fill() raises:
    var obj = String("")
    obj += "v 0 0 0\n"
    obj += "v 1 0 0 0.25 0.5 0.75\n"
    obj += "v 2 0 0"

    var mesh = parse_obj_text("colors.obj", obj)
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
    var obj = String("")
    obj += "o Cube\n"
    obj += "g Front\n"
    obj += "v 0 0 0\n"
    obj += "v 1 0 0\n"
    obj += "v 0 1 0\n"
    obj += "l 1 2\n"
    obj += "f 1 2 3\n"

    var mesh = parse_obj_text("groups.obj", obj)
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
    var mesh = ObjMesh()
    var fallback_idx = mesh.ensure_material("matA", fallback=True)

    var mtl = String("")
    mtl += "newmtl matA\n"
    mtl += "Ka 0.01 0.02 0.03\n"
    mtl += "Kd 0.1 0.2 0.3\n"
    mtl += "Ks 0.4 0.5 0.6\n"
    mtl += "Ke 0.7 0.8 0.9\n"
    mtl += "Ns 42\n"
    mtl += "Ni 1.45\n"
    mtl += "d 0.5\n"
    mtl += "illum 2\n"
    mtl += "map_Kd diffuse.png\n"
    mtl += "bump normal.png\n"

    _read_mtl_text(mesh, "assets/", mtl)
    var mat = mesh.materials[fallback_idx].copy()

    assert_true(mesh.material_count() == 1 and not mat.fallback)
    assert_almost_equal(mat.Kd0, 0.1)
    assert_almost_equal(mat.Kd1, 0.2)
    assert_almost_equal(mat.Kd2, 0.3)
    assert_almost_equal(mat.Ns, 42.0)
    assert_almost_equal(mat.Ni, 1.45)
    assert_almost_equal(mat.d, 0.5)
    assert_true(mat.illum == 2)
    assert_true(mesh.texture_count() == 3)
    assert_true(mat.map_Kd == 1)
    assert_true(mesh.textures[1].name == "diffuse.png")
    assert_true(mesh.textures[1].path == "assets/diffuse.png")
    assert_true(mat.map_bump == 2)
    assert_true(mesh.textures[2].name == "normal.png")
    assert_true(mesh.textures[2].path == "assets/normal.png")


def test_memory_loader_mtllib_with_spaces_and_texture_dedup() raises:
    var loader = MemoryObjTextLoader()

    var obj = String("")
    obj += "mtllib material library.mtl\n"
    obj += "usemtl matB\n"
    obj += "v 0 0 0\n"
    obj += "v 1 0 0\n"
    obj += "v 0 1 0\n"
    obj += "f 1 2 3\n"

    var mtl = String("")
    mtl += "newmtl matB\n"
    mtl += "map_Ka shared.png\n"
    mtl += "map_Kd shared.png\n"
    mtl += "map_bump -bm 0.5 shared.png\n"

    loader.add_file("models/model.obj", obj)
    loader.add_file("models/material library.mtl", mtl)

    var mesh = read_obj_from_memory("models/model.obj", loader)
    var mat_idx = mesh.face_materials[0]
    var mat = mesh.materials[mat_idx].copy()

    assert_true(mesh.face_count() == 1 and mesh.actual_position_count() == 3)
    assert_true(mesh.material_count() == 1 and mat.name == "matB")
    assert_true(mesh.texture_count() == 2)
    assert_true(mat.map_Ka == 1 and mat.map_Kd == 1 and mat.map_bump == 1)
    assert_true(mesh.textures[1].path == "models/shared.png")


def test_read_obj_from_string() raises:
    var obj = "v 0 0 0\n"
    obj += "v 1 0 0\n"
    obj += "v 0 1 0\n"
    obj += "f 1 2 3\n"

    var mesh = read_obj_from_string(obj)
    assert_true(mesh.face_count() == 1)
    assert_true(mesh.index_count() == 3)


def test_comments_blank_lines_and_crlf() raises:
    var obj = String("")
    obj += "# comment\r\n"
    obj += "\r\n"
    obj += "v 0 0 0 # inline comment\r\n"
    obj += "v 1 0 0\r\n"
    obj += "v 0 1 0\r\n"
    obj += "f 1 2 3\r\n"

    var mesh = parse_obj_text("comments.obj", obj)
    assert_true(mesh.actual_position_count() == 3)
    assert_true(mesh.face_count() == 1)
    assert_true(mesh.index_count() == 3)


def test_missing_texcoord_face_form() raises:
    var obj = String("")
    obj += "v 0 0 0\n"
    obj += "v 1 0 0\n"
    obj += "v 0 1 0\n"
    obj += "vn 0 0 1\n"
    obj += "f 1//1 2//1 3//1\n"

    var mesh = parse_obj_text("missing_texcoord.obj", obj)
    assert_true(mesh.face_count() == 1)
    assert_true(mesh.indices[0].p == 1)
    assert_true(mesh.indices[0].t == 0)
    assert_true(mesh.indices[0].n == 1)
    assert_true(mesh.indices[2].p == 3)
    assert_true(mesh.indices[2].t == 0)
    assert_true(mesh.indices[2].n == 1)


def test_multiple_groups_flush_offsets() raises:
    var obj = String("")
    obj += "g A\n"
    obj += "v 0 0 0\n"
    obj += "v 1 0 0\n"
    obj += "v 0 1 0\n"
    obj += "v 1 1 0\n"
    obj += "f 1 2 3\n"
    obj += "g B\n"
    obj += "f 2 4 3\n"

    var mesh = parse_obj_text("groups2.obj", obj)
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
    var mesh = ObjMesh()
    var mtl = String("")
    mtl += "newmtl glass\n"
    mtl += "Tr 0.25\n"

    _read_mtl_text(mesh, "", mtl)
    var mat = mesh.materials[0].copy()
    assert_almost_equal(mat.d, 0.75)


def test_d_overrides_tr_order() raises:
    var mesh = ObjMesh()
    var mtl = String("")
    mtl += "newmtl mat\n"
    mtl += "d 0.4\n"
    mtl += "Tr 0.9\n"

    _read_mtl_text(mesh, "", mtl)
    var mat = mesh.materials[0].copy()
    assert_almost_equal(mat.d, 0.4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
