from .types import ObjMesh
from .mtl import _read_mtl_file
from .primitives import _is_ws_or_line_cut
from .constants import *
from .cursor import ObjIndexLimit, ObjLineCursor
from .loaders import ObjTextLoader


comptime _MAX_OBJ_INDEX = Int(0x7FFFFFFF)


@always_inline
def _word_ends_here(bytes: ImmSpan[UInt8, _], p: Int) -> Bool:
    if p >= len(bytes):
        return True
    return _is_ws_or_line_cut(bytes.unsafe_get(p))


def _parse_v_cursor(mut mesh: ObjMesh, mut cur: ObjLineCursor) raises:
    cur.skip_ws()
    if cur.pos >= len(cur.bytes):
        return
    var x = cur._next_f32_at_pos()

    cur.skip_ws()
    if cur.pos >= len(cur.bytes):
        return
    var y = cur._next_f32_at_pos()

    cur.skip_ws()
    if cur.pos >= len(cur.bytes):
        return
    var z = cur._next_f32_at_pos()

    mesh.positions.append(x)
    mesh.positions.append(y)
    mesh.positions.append(z)
    debug_assert["safe", _use_compiler_assume=True](
        mesh.position_count(include_dummy=False) <= _MAX_OBJ_INDEX,
        "OBJ position count exceeds the supported index range",
    )

    cur.skip_ws()
    if cur.pos < len(cur.bytes):
        var r = cur._next_f32_at_pos()
        cur.skip_ws()
        if cur.pos < len(cur.bytes):
            var g = cur._next_f32_at_pos()
            cur.skip_ws()
            if cur.pos < len(cur.bytes):
                var b = cur._next_f32_at_pos()
                mesh._push_color(r, g, b)


def _parse_vt_cursor(mut mesh: ObjMesh, mut cur: ObjLineCursor) raises:
    cur.skip_ws()
    if cur.pos >= len(cur.bytes):
        return
    var u = cur._next_f32_at_pos()
    cur.skip_ws()
    if cur.pos >= len(cur.bytes):
        return
    var v = cur._next_f32_at_pos()
    mesh.texcoords.append(u)
    mesh.texcoords.append(v)
    debug_assert["safe", _use_compiler_assume=True](
        mesh.texcoord_count(include_dummy=False) <= _MAX_OBJ_INDEX,
        "OBJ texture coordinate count exceeds the supported index range",
    )


def _parse_vn_cursor(mut mesh: ObjMesh, mut cur: ObjLineCursor) raises:
    cur.skip_ws()
    if cur.pos >= len(cur.bytes):
        return
    var x = cur._next_f32_at_pos()
    cur.skip_ws()
    if cur.pos >= len(cur.bytes):
        return
    var y = cur._next_f32_at_pos()
    cur.skip_ws()
    if cur.pos >= len(cur.bytes):
        return
    var z = cur._next_f32_at_pos()
    mesh.normals.append(x)
    mesh.normals.append(y)
    mesh.normals.append(z)
    debug_assert["safe", _use_compiler_assume=True](
        mesh.normal_count(include_dummy=False) <= _MAX_OBJ_INDEX,
        "OBJ normal count exceeds the supported index range",
    )


def _finish_face_parse(
    mut mesh: ObjMesh,
    index_start: Int,
    count: Int,
    is_line: Bool,
):
    if not is_line:
        if count >= 3:
            mesh._push_element_meta(count, is_line=False)
        else:
            mesh.indices.shrink(index_start)
    else:
        if count >= 2:
            mesh._push_element_meta(count, is_line=True)
        else:
            mesh.indices.shrink(index_start)


def _parse_face_cursor(
    mut mesh: ObjMesh, mut cur: ObjLineCursor, is_line: Bool = False
) raises:
    var index_start = len(mesh.indices)
    var count = 0
    var position_limit = ObjIndexLimit(mesh.position_count())
    var texcoord_limit = ObjIndexLimit(mesh.texcoord_count())
    var normal_limit = ObjIndexLimit(mesh.normal_count())

    cur.skip_ws()
    if cur.pos >= len(cur.bytes):
        _finish_face_parse(mesh, index_start, count, is_line)
        return

    var first = cur.next_first_face_index_at_token(
        position_limit,
        texcoord_limit,
        normal_limit,
    )
    var shape = first.shape
    mesh.indices.append(first.idx)
    count += 1

    if shape == 1:
        while True:
            cur.skip_ws()
            if cur.pos >= len(cur.bytes):
                break
            mesh.indices.append(cur.next_index_p_only_at_token(position_limit))
            count += 1
    elif shape == 2:
        while True:
            cur.skip_ws()
            if cur.pos >= len(cur.bytes):
                break
            mesh.indices.append(
                cur.next_index_p_t_at_token(position_limit, texcoord_limit)
            )
            count += 1
    elif shape == 3:
        while True:
            cur.skip_ws()
            if cur.pos >= len(cur.bytes):
                break
            mesh.indices.append(
                cur.next_index_p_n_at_token(position_limit, normal_limit)
            )
            count += 1
    else:
        while True:
            cur.skip_ws()
            if cur.pos >= len(cur.bytes):
                break
            mesh.indices.append(
                cur.next_index_p_t_n_at_token(
                    position_limit,
                    texcoord_limit,
                    normal_limit,
                )
            )
            count += 1

    _finish_face_parse(mesh, index_start, count, is_line)


def _parse_obj[
    Loader: ObjTextLoader
](path: String, text: String, loader: Loader) raises -> ObjMesh:
    return _parse_obj(path, StringSpan(text), loader)


def _parse_obj[
    Loader: ObjTextLoader
](path: String, text: ImmStringSpan, loader: Loader) raises -> ObjMesh:
    var mesh = ObjMesh()
    var est_elements = text.byte_length() / 15
    mesh.positions.reserve(est_elements * 3)
    mesh.texcoords.reserve(est_elements * 2)
    mesh.normals.reserve(est_elements * 3)
    mesh.indices.reserve(est_elements * 6)
    mesh.face_vertices.reserve(est_elements)
    mesh.face_materials.reserve(est_elements)

    var text_len = text.byte_length()
    var bytes = Span(unsafe_ptr=text.unsafe_ptr(), length=text.byte_length())
    var line_start = 0

    while line_start < text_len:
        var line_end = text.find("\n", line_start)
        if line_end == -1:
            line_end = text_len

        var cur = ObjLineCursor(bytes[line_start:line_end])
        cur.skip_ws()
        if cur.pos < len(cur.bytes):
            var p = cur.pos
            var c0 = cur.bytes.unsafe_get(p)
            if c0 == CHAR_v:
                if _word_ends_here(cur.bytes, p + 1):
                    cur.pos = p + 1
                    _parse_v_cursor(mesh, cur)
                elif (
                    p + 1 < len(cur.bytes)
                    and cur.bytes.unsafe_get(p + 1) == CHAR_t
                ):
                    if _word_ends_here(cur.bytes, p + 2):
                        cur.pos = p + 2
                        _parse_vt_cursor(mesh, cur)
                elif (
                    p + 1 < len(cur.bytes)
                    and cur.bytes.unsafe_get(p + 1) == CHAR_n
                ):
                    if _word_ends_here(cur.bytes, p + 2):
                        cur.pos = p + 2
                        _parse_vn_cursor(mesh, cur)
            elif c0 == CHAR_f and _word_ends_here(cur.bytes, p + 1):
                cur.pos = p + 1
                _parse_face_cursor(mesh, cur, is_line=False)
            elif c0 == CHAR_l and _word_ends_here(cur.bytes, p + 1):
                cur.pos = p + 1
                _parse_face_cursor(mesh, cur, is_line=True)
            elif c0 == CHAR_g and _word_ends_here(cur.bytes, p + 1):
                cur.pos = p + 1
                mesh._begin_group(cur.joined_rest_of_line())
            elif c0 == CHAR_o and _word_ends_here(cur.bytes, p + 1):
                cur.pos = p + 1
                mesh._begin_object(cur.joined_rest_of_line())
            else:
                var tag_start = p
                var tag_end = p
                while tag_end < len(cur.bytes) and not _is_ws_or_line_cut(
                    cur.bytes.unsafe_get(tag_end)
                ):
                    tag_end += 1
                var tag = StringSpan(
                    unsafe_from_utf8=cur.bytes[tag_start:tag_end]
                )
                cur.pos = tag_end
                if tag == "usemtl":
                    var name = cur.joined_rest_of_line()
                    if name.byte_length() > 0:
                        mesh._current_material = mesh._ensure_material(
                            name, fallback=True
                        )
                elif tag == "mtllib":
                    var mtl_name = cur.joined_rest_of_line()
                    if mtl_name.byte_length() > 0:
                        _read_mtl_file(mesh, path, mtl_name, loader)
        line_start = line_end + 1

    if len(mesh.colors) > 0:
        while len(mesh.colors) < len(mesh.positions):
            mesh.colors.append(1.0)

    mesh._finish()
    return mesh^
