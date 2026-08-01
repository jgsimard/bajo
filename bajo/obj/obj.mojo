from .types import ObjIndex, ObjMesh
from .mtl import _read_mtl_file
from .f32 import parse_f32_at
from .primitives import (
    _fix_index,
    _is_digit,
    _is_line_cut,
    _is_ws,
    _is_ws_or_line_cut,
)
from .constants import *
from .loaders import ObjTextLoader

comptime _MAX_OBJ_INDEX = Int(0x7FFFFFFF)


@fieldwise_init
struct FirstFaceIndex(TrivialRegisterPassable):
    var idx: ObjIndex
    var shape: Int


struct ObjIndexLimit(TrivialRegisterPassable):
    var count_with_dummy: Int
    var max_magnitude: Int

    @always_inline
    def __init__(out self, count_with_dummy: Int):
        self.count_with_dummy = count_with_dummy
        self.max_magnitude = count_with_dummy - 1


# OBJ parsing
@always_inline
def _word_ends_here(bytes: Span[mut=False, UInt8, _], p: Int) -> Bool:
    if p >= len(bytes):
        return True
    return _is_ws_or_line_cut(bytes.unsafe_get(p))


struct ObjLineCursor[origin: ImmOrigin]:
    var bytes: Span[mut=False, UInt8, Self.origin]
    var pos: Int

    @always_inline
    def __init__(out self, bytes: Span[mut=False, UInt8, Self.origin]):
        self.bytes = bytes
        self.pos = 0

    @always_inline
    def skip_ws(mut self):
        while self.pos < len(self.bytes):
            var b = self.bytes.unsafe_get(self.pos)
            if _is_line_cut(b):
                self.bytes = self.bytes[: self.pos]
                break
            if not _is_ws(b):
                break
            self.pos += 1

    def has_next(mut self) -> Bool:
        self.skip_ws()
        return self.pos < len(self.bytes)

    @always_inline
    def _next_f32_at_pos(mut self) raises -> Float32:
        var parsed = parse_f32_at(self.bytes, self.pos)
        self.pos = parsed.pos
        return parsed.value

    def next_f32(mut self) raises -> Float32:
        self.skip_ws()
        return self._next_f32_at_pos()

    def _parse_index_int(
        mut self, slash_terminates: Bool, limit: ObjIndexLimit
    ) raises -> Int:
        if self.pos >= len(self.bytes):
            raise String("missing OBJ index")

        var sign = 1
        var output = 0
        var b = self.bytes.unsafe_get(self.pos)
        if b == MINUS:
            sign = -1
            self.pos += 1
        elif b == PLUS:
            self.pos += 1

        while self.pos < len(self.bytes):
            b = self.bytes.unsafe_get(self.pos)
            if _is_digit(b):
                var digit = Int(b - ZERO)
                output = output * 10 + digit
                if output > limit.max_magnitude:
                    raise String(
                        "OBJ index exceeds the available element count"
                    )
                self.pos += 1
                continue

            break

        if output == 0:
            raise String("missing or zero OBJ index")

        if self.pos < len(self.bytes):
            b = self.bytes.unsafe_get(self.pos)
            if not ((slash_terminates and b == SLASH) or _is_ws_or_line_cut(b)):
                raise String("invalid character in OBJ index")

        return sign * output

    @always_inline
    def _parse_positive_index_int(
        mut self, slash_terminates: Bool, limit: ObjIndexLimit
    ) raises -> Int:
        # Fast path for the common OBJ case: positive decimal indices.
        # Falls back to signed parsing only when a sign is actually present.
        if self.pos >= len(self.bytes):
            raise String("missing OBJ index")

        if self.pos < len(self.bytes):
            var first = self.bytes.unsafe_get(self.pos)
            if first == MINUS or first == PLUS:
                return self._parse_index_int(slash_terminates, limit)

        var output = 0
        while self.pos < len(self.bytes):
            var b = self.bytes.unsafe_get(self.pos)
            if _is_digit(b):
                var digit = Int(b - ZERO)
                output = output * 10 + digit
                if output > limit.max_magnitude:
                    raise String(
                        "OBJ index exceeds the available element count"
                    )
                self.pos += 1
                continue

            break

        if output == 0:
            raise String("missing or zero OBJ index")

        if self.pos < len(self.bytes):
            var b = self.bytes.unsafe_get(self.pos)
            if not ((slash_terminates and b == SLASH) or _is_ws_or_line_cut(b)):
                raise String("invalid character in OBJ index")

        return output

    @always_inline
    def _at_signed_index(self) -> Bool:
        if self.pos >= len(self.bytes):
            return False
        var b = self.bytes.unsafe_get(self.pos)
        return b == MINUS or b == PLUS

    @always_inline
    def next_index_p_only_at_token(
        mut self, position_limit: ObjIndexLimit
    ) raises -> ObjIndex:
        var needs_fix = self._at_signed_index()
        var p_raw = self._parse_positive_index_int(
            slash_terminates=False,
            limit=position_limit,
        )
        if needs_fix:
            return ObjIndex(
                _fix_index(p_raw, position_limit.count_with_dummy), 0, 0
            )
        return ObjIndex(p_raw, 0, 0)

    def next_index_p_t_at_token(
        mut self, position_limit: ObjIndexLimit, texcoord_limit: ObjIndexLimit
    ) raises -> ObjIndex:
        var needs_fix = self._at_signed_index()
        var p_raw = self._parse_positive_index_int(
            slash_terminates=True,
            limit=position_limit,
        )
        var t_raw = 0

        if (
            self.pos < len(self.bytes)
            and self.bytes.unsafe_get(self.pos) == SLASH
        ):
            self.pos += 1
            if self._at_signed_index():
                needs_fix = True
            t_raw = self._parse_positive_index_int(
                slash_terminates=False,
                limit=texcoord_limit,
            )

        if t_raw == 0:
            raise String("missing OBJ texture index")
        if not needs_fix:
            return ObjIndex(p_raw, t_raw, 0)

        return ObjIndex(
            _fix_index(p_raw, position_limit.count_with_dummy),
            _fix_index(t_raw, texcoord_limit.count_with_dummy),
            0,
        )

    def next_index_p_n_at_token(
        mut self, position_limit: ObjIndexLimit, normal_limit: ObjIndexLimit
    ) raises -> ObjIndex:
        var needs_fix = self._at_signed_index()
        var p_raw = self._parse_positive_index_int(
            slash_terminates=True,
            limit=position_limit,
        )
        var n_raw = 0

        if (
            self.pos < len(self.bytes)
            and self.bytes.unsafe_get(self.pos) == SLASH
        ):
            self.pos += 1
            if (
                self.pos < len(self.bytes)
                and self.bytes.unsafe_get(self.pos) == SLASH
            ):
                self.pos += 1
                if self._at_signed_index():
                    needs_fix = True
                n_raw = self._parse_positive_index_int(
                    slash_terminates=False,
                    limit=normal_limit,
                )

        if n_raw == 0:
            raise String("missing OBJ normal index")
        if not needs_fix:
            return ObjIndex(p_raw, 0, n_raw)

        return ObjIndex(
            _fix_index(p_raw, position_limit.count_with_dummy),
            0,
            _fix_index(n_raw, normal_limit.count_with_dummy),
        )

    def next_index_p_t_n_at_token(
        mut self,
        position_limit: ObjIndexLimit,
        texcoord_limit: ObjIndexLimit,
        normal_limit: ObjIndexLimit,
    ) raises -> ObjIndex:
        var needs_fix = self._at_signed_index()
        var p_raw = self._parse_positive_index_int(
            slash_terminates=True,
            limit=position_limit,
        )
        var t_raw = 0
        var n_raw = 0

        if (
            self.pos < len(self.bytes)
            and self.bytes.unsafe_get(self.pos) == SLASH
        ):
            self.pos += 1
            if self._at_signed_index():
                needs_fix = True
            t_raw = self._parse_positive_index_int(
                slash_terminates=True,
                limit=texcoord_limit,
            )

            if (
                self.pos < len(self.bytes)
                and self.bytes.unsafe_get(self.pos) == SLASH
            ):
                self.pos += 1
                if self._at_signed_index():
                    needs_fix = True
                n_raw = self._parse_positive_index_int(
                    slash_terminates=False,
                    limit=normal_limit,
                )

        if t_raw == 0 or n_raw == 0:
            raise String("missing OBJ texture or normal index")
        if not needs_fix:
            return ObjIndex(p_raw, t_raw, n_raw)

        return ObjIndex(
            _fix_index(p_raw, position_limit.count_with_dummy),
            _fix_index(t_raw, texcoord_limit.count_with_dummy),
            _fix_index(n_raw, normal_limit.count_with_dummy),
        )

    def next_index_generic_at_token(
        mut self,
        position_limit: ObjIndexLimit,
        texcoord_limit: ObjIndexLimit,
        normal_limit: ObjIndexLimit,
    ) raises -> ObjIndex:
        var p_raw = self._parse_index_int(
            slash_terminates=True,
            limit=position_limit,
        )
        var t_raw = 0
        var n_raw = 0

        if (
            self.pos < len(self.bytes)
            and self.bytes.unsafe_get(self.pos) == SLASH
        ):
            self.pos += 1
            t_raw = self._parse_index_int(
                slash_terminates=True,
                limit=texcoord_limit,
            )

            if (
                self.pos < len(self.bytes)
                and self.bytes.unsafe_get(self.pos) == SLASH
            ):
                self.pos += 1
                n_raw = self._parse_index_int(
                    slash_terminates=False,
                    limit=normal_limit,
                )

        return ObjIndex(
            _fix_index(p_raw, position_limit.count_with_dummy),
            _fix_index(t_raw, texcoord_limit.count_with_dummy),
            _fix_index(n_raw, normal_limit.count_with_dummy),
        )

    def next_first_face_index_at_token(
        mut self,
        position_limit: ObjIndexLimit,
        texcoord_limit: ObjIndexLimit,
        normal_limit: ObjIndexLimit,
    ) raises -> FirstFaceIndex:
        # Parses the first face token once and infers the shape while parsing.
        #
        # shape:
        #   1 = p
        #   2 = p/t
        #   3 = p//n
        #   4 = p/t/n

        var shape = 1
        var needs_fix = self._at_signed_index()
        var p_raw = self._parse_positive_index_int(
            slash_terminates=True,
            limit=position_limit,
        )
        var t_raw = 0
        var n_raw = 0

        if (
            self.pos < len(self.bytes)
            and self.bytes.unsafe_get(self.pos) == SLASH
        ):
            self.pos += 1

            if (
                self.pos < len(self.bytes)
                and self.bytes.unsafe_get(self.pos) == SLASH
            ):
                # p//n
                self.pos += 1
                shape = 3

                if self._at_signed_index():
                    needs_fix = True
                n_raw = self._parse_positive_index_int(
                    slash_terminates=False,
                    limit=normal_limit,
                )

            else:
                # p/t or p/t/n
                shape = 2

                if self._at_signed_index():
                    needs_fix = True
                t_raw = self._parse_positive_index_int(
                    slash_terminates=True,
                    limit=texcoord_limit,
                )

                if (
                    self.pos < len(self.bytes)
                    and self.bytes.unsafe_get(self.pos) == SLASH
                ):
                    self.pos += 1
                    shape = 4

                    if self._at_signed_index():
                        needs_fix = True
                    n_raw = self._parse_positive_index_int(
                        slash_terminates=False,
                        limit=normal_limit,
                    )

        if not needs_fix:
            return FirstFaceIndex(ObjIndex(p_raw, t_raw, n_raw), shape)

        var p = _fix_index(p_raw, position_limit.count_with_dummy)
        var t = t_raw
        var n = n_raw
        if t_raw != 0:
            t = _fix_index(t_raw, texcoord_limit.count_with_dummy)
        if n_raw != 0:
            n = _fix_index(n_raw, normal_limit.count_with_dummy)

        return FirstFaceIndex(ObjIndex(p, t, n), shape)

    def joined_rest_of_line(mut self) -> String:
        self.skip_ws()
        if self.pos >= len(self.bytes):
            return ""

        var start = self.pos
        var logical_end = len(self.bytes)

        # Only string-like OBJ/MTL commands use this rare path. Keep comments
        # lazy here instead of pre-scanning every geometry line.
        var p = start
        while p < logical_end:
            var b = self.bytes.unsafe_get(p)
            if _is_line_cut(b):
                logical_end = p
                self.bytes = self.bytes[:logical_end]
                break
            p += 1

        var end_pos = logical_end - 1

        # Trim trailing whitespace.
        while end_pos >= start:
            var b = self.bytes.unsafe_get(end_pos)
            if not _is_ws(b):
                break
            end_pos -= 1

        self.pos = len(self.bytes)

        if end_pos < start:
            return ""

        return String(
            StringSlice[Self.origin](
                unsafe_from_utf8=self.bytes[start : end_pos + 1]
            )
        )

    def next_word(mut self) -> StringSlice[Self.origin]:
        self.skip_ws()
        if self.pos >= len(self.bytes):
            return StringSlice[Self.origin]()

        var start = self.pos
        while self.pos < len(self.bytes):
            var b = self.bytes.unsafe_get(self.pos)
            if _is_ws_or_line_cut(b):
                if _is_line_cut(b):
                    self.bytes = self.bytes[: self.pos]
                break
            self.pos += 1

        return StringSlice[Self.origin](
            unsafe_from_utf8=self.bytes[start : self.pos]
        )


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

    # Optional vertex color: v x y z r g b.
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

    # Counts are constant for this face.
    var position_limit = ObjIndexLimit(mesh.position_count())
    var texcoord_limit = ObjIndexLimit(mesh.texcoord_count())
    var normal_limit = ObjIndexLimit(mesh.normal_count())

    # Parse first token once.
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
        # f p p p ...
        while True:
            cur.skip_ws()
            if cur.pos >= len(cur.bytes):
                break

            mesh.indices.append(cur.next_index_p_only_at_token(position_limit))
            count += 1

    elif shape == 2:
        # f p/t p/t p/t ...
        while True:
            cur.skip_ws()
            if cur.pos >= len(cur.bytes):
                break

            mesh.indices.append(
                cur.next_index_p_t_at_token(
                    position_limit,
                    texcoord_limit,
                )
            )
            count += 1

    elif shape == 3:
        # f p//n p//n p//n ...
        while True:
            cur.skip_ws()
            if cur.pos >= len(cur.bytes):
                break

            mesh.indices.append(
                cur.next_index_p_n_at_token(
                    position_limit,
                    normal_limit,
                )
            )
            count += 1

    else:
        # f p/t/n p/t/n p/t/n ...
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
    return _parse_obj(path, StringSlice(text), loader)


def _parse_obj[
    Loader: ObjTextLoader
](
    path: String, text: StringSlice[mut=False, _], loader: Loader
) raises -> ObjMesh:
    var mesh = ObjMesh()

    # Same rough heuristic as before, with reserves for the other hot arrays too.
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

        # Do not pre-scan the whole line for CR/comments here. ObjLineCursor
        # treats CR and # as lazy logical line-end markers while parsing.
        var cur = ObjLineCursor(bytes[line_start:line_end])
        cur.skip_ws()

        if cur.pos < len(cur.bytes):
            var p = cur.pos
            var c0 = cur.bytes.unsafe_get(p)

            # Hot one/two-byte tag dispatch.
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
                # Rare multi-char tags.
                var tag_start = p
                var tag_end = p
                while tag_end < len(cur.bytes) and not _is_ws_or_line_cut(
                    cur.bytes.unsafe_get(tag_end)
                ):
                    tag_end += 1

                var tag = StringSlice(
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
