"""Count OBJ chunks, decode disjoint ranges, then resolve ordered metadata."""

from max.algorithm import parallelize
from std.bit import count_trailing_zeros, pop_count
from std.math import ceildiv
from std.memory import pack_bits
from std.sys import simd_width_of
from .types import ObjMesh, ObjGroup, ObjIndex
from .lines import _ObjLines
from .cursor import ObjIndexLimit, ObjLineCursor
from .constants import *
from .primitives import _is_ws_or_line_cut
from .serial import _ObjParserState, _word_ends_here, _parse_obj_serial
from .mtl import _read_mtl_file
from bajo.parser.asset_loader import AssetLoader


@fieldwise_init
struct _MetadataEvent(Copyable):
    var kind: Int
    var name: String
    var face: Int
    var index: Int


struct _ObjChunk(Copyable):
    var begin: Int
    var end: Int
    var positions: Int
    var texcoords: Int
    var normals: Int
    var faces: Int
    var indices: Int
    var has_colors: Bool
    var has_lines: Bool
    var position_offset: Int
    var texcoord_offset: Int
    var normal_offset: Int
    var face_offset: Int
    var index_offset: Int
    var events: List[_MetadataEvent]

    def __init__(out self):
        self.begin = 0
        self.end = 0
        self.positions = 0
        self.texcoords = 0
        self.normals = 0
        self.faces = 0
        self.indices = 0
        self.has_colors = False
        self.has_lines = False
        self.position_offset = 0
        self.texcoord_offset = 0
        self.normal_offset = 0
        self.face_offset = 0
        self.index_offset = 0
        self.events = List[_MetadataEvent]()


@always_inline
def _record_kind(mut cur: ObjLineCursor) -> Int:
    # 1=v, 2=vt, 3=vn, 4=f, 5=l; 6=o, 7=g, 8=usemtl, 9=mtllib.
    cur.skip_ws()
    if cur.pos >= len(cur.bytes):
        return 0
    var p = cur.pos
    var first = cur.bytes.unsafe_get(p)
    if first == CHAR_v:
        if _word_ends_here(cur.bytes, p + 1):
            cur.pos = p + 1
            return 1
        if p + 1 < len(cur.bytes) and _word_ends_here(cur.bytes, p + 2):
            var second = cur.bytes.unsafe_get(p + 1)
            cur.pos = p + 2
            if second == CHAR_t:
                return 2
            if second == CHAR_n:
                return 3
        return 0
    if _word_ends_here(cur.bytes, p + 1):
        cur.pos = p + 1
        # Character literals are numeric byte patterns; names in a case
        # pattern would bind a variable instead of comparing the constant.
        __match first:
            case 102:
                return 4
            case 108:
                return 5
            case 111:
                return 6
            case 103:
                return 7
            case _:
                return 0
    var tag = cur.next_word()
    if tag == "usemtl":
        return 8
    if tag == "mtllib":
        return 9
    return 0


@always_inline
def _word_count(bytes: ImmSpan[UInt8, _], begin: Int, end: Int) -> Int:
    # Loads may extend past this line, but never past the complete input span.
    # Masking the extra lanes lets short OBJ records use SIMD as well.
    comptime width = min(64, simd_width_of[UInt8]())
    var count = 0
    var pos = begin
    var previous_token = UInt64(0)
    while pos < end and len(bytes) - pos >= width:
        var block = (
            bytes.unsafe_ptr()
            .unsafe_offset(pos)
            .unsafe_load[width=width, alignment=1]()
        )
        var valid = min(width, end - pos)
        var cuts = UInt64(pack_bits(block.eq(CR) | block.eq(HASH)))
        var stop = min(valid, Int(count_trailing_zeros(cuts)))
        if stop == 0:
            return count
        var active = ~UInt64(0) >> UInt64(64 - stop)
        var whitespace = UInt64(pack_bits(block.eq(SPACE) | block.eq(TAB)))
        var tokens = ~whitespace & active
        var starts = tokens & ~((tokens << 1) | previous_token)
        count += Int(pop_count(starts))
        if stop < valid:
            return count
        previous_token = (tokens >> UInt64(valid - 1)) & 1
        pos += valid
    while pos < end:
        var byte = bytes.unsafe_get(pos)
        if byte == CR or byte == HASH:
            break
        var token = UInt64(byte != SPACE and byte != TAB)
        count += Int(token & ~previous_token)
        previous_token = token
        pos += 1
    return count


def _count_chunk(text: ImmStringSpan, begin: Int, end: Int) -> _ObjChunk:
    var chunk = _ObjChunk()
    chunk.begin = begin
    chunk.end = end
    var bytes = text.as_bytes()
    var start = begin
    var lines = _ObjLines(bytes, begin, end)
    while start < end:
        var stop = lines.next_end()
        var cur = ObjLineCursor(bytes[start:stop])
        var kind = _record_kind(cur)
        if kind >= 1 and kind <= 5:
            var count = _word_count(bytes, start + cur.pos, stop)
            __match kind:
                case 1:
                    if count >= 3:
                        chunk.positions += 1
                        chunk.has_colors = chunk.has_colors or count >= 6
                case 2:
                    if count >= 2:
                        chunk.texcoords += 1
                case 3:
                    if count >= 3:
                        chunk.normals += 1
                case _:
                    if count >= (2 if kind == 5 else 3):
                        chunk.faces += 1
                        chunk.indices += count
                        chunk.has_lines = chunk.has_lines or kind == 5
        elif kind >= 6:
            var name = cur.joined_rest_of_line()
            if kind <= 7 or name.byte_length() > 0:
                chunk.events.append(
                    _MetadataEvent(kind, name^, chunk.faces, chunk.indices)
                )
        start = stop + 1
    return chunk^


@always_inline
def _next_corner[
    shape: Int
](
    mut cur: ObjLineCursor,
    positions: ObjIndexLimit,
    texcoords: ObjIndexLimit,
    normals: ObjIndexLimit,
) raises -> ObjIndex:
    comptime __match shape:
        case 1:
            return cur.next_index_p_only_at_token(positions)
        case 2:
            return cur.next_index_p_t_at_token(positions, texcoords)
        case 3:
            return cur.next_index_p_n_at_token(positions, normals)
        case _:
            return cur.next_index_p_t_n_at_token(positions, texcoords, normals)


def _decode_corners[
    shape: Int
](
    mut cur: ObjLineCursor,
    first: ObjIndex,
    positions: ObjIndexLimit,
    texcoords: ObjIndexLimit,
    normals: ObjIndexLimit,
    output: MutSpan[ObjIndex, _],
    offset: Int,
    is_line: Bool,
) raises -> Int:
    # Delay the first writes until the element meets its minimum size. Short
    # elements still validate their indices just as the scalar parser does.
    if not cur.has_next():
        return 0
    var second = _next_corner[shape](cur, positions, texcoords, normals)
    var count = 2
    var third = ObjIndex(0, 0, 0)
    if not is_line:
        if not cur.has_next():
            return 0
        third = _next_corner[shape](cur, positions, texcoords, normals)
        count = 3
    if offset + count > len(output):
        raise Error("OBJ structural count mismatch")
    output[offset] = first
    output[offset + 1] = second
    if not is_line:
        output[offset + 2] = third
    while cur.has_next():
        if offset + count >= len(output):
            raise Error("OBJ structural count mismatch")
        output[offset + count] = _next_corner[shape](
            cur, positions, texcoords, normals
        )
        count += 1
    return count


def _decode_face(
    mut cur: ObjLineCursor,
    positions: ObjIndexLimit,
    texcoords: ObjIndexLimit,
    normals: ObjIndexLimit,
    output: MutSpan[ObjIndex, _],
    offset: Int,
    is_line: Bool,
) raises -> Int:
    if not cur.has_next():
        return 0
    var first = cur.next_first_face_index_at_token(
        positions, texcoords, normals
    )
    __match first.shape:
        case 1:
            return _decode_corners[1](
                cur,
                first.idx,
                positions,
                texcoords,
                normals,
                output,
                offset,
                is_line,
            )
        case 2:
            return _decode_corners[2](
                cur,
                first.idx,
                positions,
                texcoords,
                normals,
                output,
                offset,
                is_line,
            )
        case 3:
            return _decode_corners[3](
                cur,
                first.idx,
                positions,
                texcoords,
                normals,
                output,
                offset,
                is_line,
            )
        case _:
            return _decode_corners[4](
                cur,
                first.idx,
                positions,
                texcoords,
                normals,
                output,
                offset,
                is_line,
            )


def _decode_numeric(
    mut cur: ObjLineCursor,
    kind: Int,
    output: MutSpan[Float32, _],
    base: Int,
    colors: MutSpan[Float32, _],
) raises -> Bool:
    if not cur.has_next():
        return False
    var x = cur._next_f32_at_pos()
    if not cur.has_next():
        return False
    var y = cur._next_f32_at_pos()
    var z = Float32(0.0)
    if kind != 2:
        if not cur.has_next():
            return False
        z = cur._next_f32_at_pos()
    var width = 2 if kind == 2 else 3
    if base + width > len(output):
        raise Error("OBJ structural count mismatch")
    output[base] = x
    output[base + 1] = y
    if kind != 2:
        output[base + 2] = z
    if kind == 1 and cur.has_next():
        var r = cur._next_f32_at_pos()
        if cur.has_next():
            var g = cur._next_f32_at_pos()
            if cur.has_next():
                var b = cur._next_f32_at_pos()
                if base + 3 > len(colors):
                    raise Error("OBJ structural color count mismatch")
                colors[base] = r
                colors[base + 1] = g
                colors[base + 2] = b
    return True


def _decode_chunk(
    text: ImmStringSpan,
    chunk: _ObjChunk,
    mut mesh: ObjMesh,
) raises:
    var position = chunk.position_offset
    var texcoord = chunk.texcoord_offset
    var normal = chunk.normal_offset
    var face = chunk.face_offset
    var index = chunk.index_offset
    var bytes = text.as_bytes()
    var start = chunk.begin
    # A malformed record must never write into another worker's range.
    var position_begin = 3 * (chunk.position_offset + 1)
    var texcoord_begin = 2 * (chunk.texcoord_offset + 1)
    var normal_begin = 3 * (chunk.normal_offset + 1)
    var positions = Span(mesh.positions)[
        position_begin : position_begin + 3 * chunk.positions
    ]
    var texcoords = Span(mesh.texcoords)[
        texcoord_begin : texcoord_begin + 2 * chunk.texcoords
    ]
    var normals = Span(mesh.normals)[
        normal_begin : normal_begin + 3 * chunk.normals
    ]
    var colors = Span(mesh.colors)
    if len(colors) > 0:
        colors = colors[position_begin : position_begin + 3 * chunk.positions]
    var indices = Span(mesh.indices)[
        chunk.index_offset : chunk.index_offset + chunk.indices
    ]
    var lines = _ObjLines(bytes, chunk.begin, chunk.end)
    while start < chunk.end:
        var stop = lines.next_end()
        var cur = ObjLineCursor(bytes[start:stop])
        var kind = _record_kind(cur)
        __match kind:
            case 1:
                if _decode_numeric(
                    cur,
                    kind,
                    positions,
                    3 * (position - chunk.position_offset),
                    colors,
                ):
                    position += 1
            case 2:
                if _decode_numeric(
                    cur,
                    kind,
                    texcoords,
                    2 * (texcoord - chunk.texcoord_offset),
                    colors,
                ):
                    texcoord += 1
            case 3:
                if _decode_numeric(
                    cur,
                    kind,
                    normals,
                    3 * (normal - chunk.normal_offset),
                    colors,
                ):
                    normal += 1
            case 4 | 5:
                var count = _decode_face(
                    cur,
                    ObjIndexLimit(position + 1),
                    ObjIndexLimit(texcoord + 1),
                    ObjIndexLimit(normal + 1),
                    indices,
                    index - chunk.index_offset,
                    kind == 5,
                )
                if count > 0:
                    if face >= chunk.face_offset + chunk.faces:
                        raise Error("OBJ structural face count mismatch")
                    mesh.face_vertices[face] = count
                    if len(mesh.face_lines) > 0:
                        mesh.face_lines[face] = UInt8(
                            1
                        ) if kind == 5 else UInt8(0)
                    face += 1
                    index += count
            case _:
                pass
        start = stop + 1
    if (
        position != chunk.position_offset + chunk.positions
        or texcoord != chunk.texcoord_offset + chunk.texcoords
        or normal != chunk.normal_offset + chunk.normals
        or face != chunk.face_offset + chunk.faces
        or index != chunk.index_offset + chunk.indices
    ):
        raise Error("OBJ structural count mismatch")


def _finish_metadata[
    Loader: AssetLoader
](
    path: String,
    chunks: List[_ObjChunk],
    loader: Loader,
    mut mesh: ObjMesh,
) raises:
    var state = _ObjParserState()
    var previous_face = 0
    for chunk in chunks:
        for event in chunk.events:
            var face = chunk.face_offset + event.face
            var index = chunk.index_offset + event.index
            for output_face in range(previous_face, face):
                mesh.face_materials[output_face] = state.current_material
            previous_face = face
            __match event.kind:
                case 6:
                    state.current_object.face_count = (
                        face - state.current_object.face_offset
                    )
                    if state.current_object.face_count > 0:
                        mesh.objects.append(state.current_object^)
                    state.current_object = ObjGroup(event.name, 0, face, index)
                case 7:
                    state.current_group.face_count = (
                        face - state.current_group.face_offset
                    )
                    if state.current_group.face_count > 0:
                        mesh.groups.append(state.current_group^)
                    state.current_group = ObjGroup(event.name, 0, face, index)
                case 8:
                    state.ensure_material(mesh, event.name, fallback=True)
                case 9:
                    _read_mtl_file(mesh, path, event.name, loader)
                case _:
                    pass
    for face in range(previous_face, len(mesh.face_vertices)):
        mesh.face_materials[face] = state.current_material
    state.current_group.face_count = (
        len(mesh.face_vertices) - state.current_group.face_offset
    )
    if state.current_group.face_count > 0:
        mesh.groups.append(state.current_group^)
    state.current_group = ObjGroup(
        "", 0, len(mesh.face_vertices), len(mesh.indices)
    )
    state.current_object.face_count = (
        len(mesh.face_vertices) - state.current_object.face_offset
    )
    if state.current_object.face_count > 0:
        mesh.objects.append(state.current_object^)
    state.current_object = ObjGroup(
        "", 0, len(mesh.face_vertices), len(mesh.indices)
    )


def _parse_obj_bulk[
    Loader: AssetLoader
](
    path: String,
    text: ImmStringSpan,
    loader: Loader,
    workers: Int,
    chunk_bytes: Int = 256 * 1024,
) raises -> ObjMesh:
    var task_count = min(
        max(1, workers) * 4,
        max(1, ceildiv(text.byte_length(), max(1, chunk_bytes))),
    )
    var chunks = List[_ObjChunk](length=task_count, fill=_ObjChunk())

    def count_worker(task: Int) {text, task_count, mut chunks}:
        var begin = 0
        var end = text.byte_length()
        if task > 0:
            begin = text.find("\n", text.byte_length() * task // task_count - 1)
            begin = text.byte_length() if begin == -1 else begin + 1
        if task + 1 < task_count:
            end = text.find(
                "\n", text.byte_length() * (task + 1) // task_count - 1
            )
            end = text.byte_length() if end == -1 else end + 1
        chunks[task] = _count_chunk(text, begin, end)

    if workers > 1:
        parallelize(count_worker, task_count, min(workers, task_count))
    else:
        for task in range(task_count):
            count_worker(task)
    var positions = 0
    var texcoords = 0
    var normals = 0
    var faces = 0
    var indices = 0
    var colors = False
    var lines = False
    for task in range(task_count):
        ref chunk = chunks[task]
        chunk.position_offset = positions
        chunk.texcoord_offset = texcoords
        chunk.normal_offset = normals
        chunk.face_offset = faces
        chunk.index_offset = indices
        positions += chunk.positions
        texcoords += chunk.texcoords
        normals += chunk.normals
        faces += chunk.faces
        indices += chunk.indices
        colors = colors or chunk.has_colors
        lines = lines or chunk.has_lines
    if positions > 0x7FFFFFFF or texcoords > 0x7FFFFFFF or normals > 0x7FFFFFFF:
        return _parse_obj_serial(path, text, loader)
    var mesh = ObjMesh()
    mesh.positions.resize(unsafe_uninit_length=3 * (positions + 1))
    mesh.texcoords.resize(unsafe_uninit_length=2 * (texcoords + 1))
    mesh.normals.resize(unsafe_uninit_length=3 * (normals + 1))
    mesh.indices.resize(unsafe_uninit_length=indices)
    mesh.face_vertices.resize(unsafe_uninit_length=faces)
    mesh.face_materials.resize(unsafe_uninit_length=faces)
    if colors:
        mesh.colors = List[Float32](length=3 * (positions + 1), fill=1.0)
    if lines:
        mesh.face_lines.resize(unsafe_uninit_length=faces)
    var failed = List[UInt8](length=task_count, fill=0)

    def decode_worker(task: Int) {text, chunks, mut mesh, mut failed}:
        try:
            _decode_chunk(text, chunks[task], mesh)
        except:
            failed[task] = 1

    if workers > 1:
        parallelize(decode_worker, task_count, min(workers, task_count))
    else:
        for task in range(task_count):
            decode_worker(task)
    for task in range(task_count):
        if failed[task] != 0:
            # Invalid or structurally unusual input uses the scalar path to
            # retain its validation behavior and ordering of loader errors.
            return _parse_obj_serial(path, text, loader)
    _finish_metadata(path, chunks, loader, mesh)
    return mesh^
