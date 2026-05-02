from bajo.obj.types import ObjIndex
from bajo.obj.mtl import _read_mtl_file


comptime TAB = UInt8(ord("\t"))
comptime SPACE = UInt8(ord(" "))
comptime CR = UInt8(ord("\r"))
comptime LF = UInt8(ord("\n"))

comptime ZERO = UInt8(ord("0"))
comptime NINE = UInt8(ord("9"))
comptime PLUS = UInt8(ord("+"))
comptime MINUS = UInt8(ord("-"))
comptime SLASH = UInt8(ord("/"))
comptime DOT = UInt8(ord("."))
comptime HASH = UInt8(ord("#"))

comptime CHAR_e = UInt8(ord("e"))
comptime CHAR_E = UInt8(ord("E"))
comptime CHAR_v = UInt8(ord("v"))
comptime CHAR_t = UInt8(ord("t"))
comptime CHAR_n = UInt8(ord("n"))
comptime CHAR_f = UInt8(ord("f"))
comptime CHAR_l = UInt8(ord("l"))
comptime CHAR_g = UInt8(ord("g"))
comptime CHAR_o = UInt8(ord("o"))


# parsing primitive
@always_inline
def _is_ws(b: UInt8) -> Bool:
    return b == SPACE or b == TAB or b == CR


@always_inline
def _is_line_cut(b: UInt8) -> Bool:
    return b == CR or b == HASH


@always_inline
def _is_ws_or_line_cut(b: UInt8) -> Bool:
    return _is_ws(b) or b == HASH


@always_inline
def _is_digit(b: UInt8) -> Bool:
    return b >= ZERO and b <= NINE


@always_inline
def _fix_index(raw: Int, count_with_dummy: Int) -> Int:
    if raw > 0:
        return raw
    if raw < 0:
        return count_with_dummy + raw
    return 0


# OBJ parsing
@always_inline
def _word_ends_here[
    o: Origin
](ptr: UnsafePointer[UInt8, o], p: Int, end: Int) -> Bool:
    if p >= end:
        return True
    return _is_ws_or_line_cut(ptr.load(p))


struct ObjLineCursor[o: Origin]:
    var text: StringSlice[Self.o]
    var ptr: UnsafePointer[UInt8, Self.o]
    var pos: Int
    var end: Int

    def __init__(out self, text: StringSlice[Self.o], start: Int, end: Int):
        self.text = text
        self.ptr = text.unsafe_ptr()
        self.pos = start
        self.end = end

    @always_inline
    def skip_ws(mut self):
        while self.pos < self.end:
            var b = self.ptr.load(self.pos)
            if _is_line_cut(b):
                self.end = self.pos
                break
            if not _is_ws(b):
                break
            self.pos += 1

    @always_inline
    def has_next(mut self) -> Bool:
        self.skip_ws()
        return self.pos < self.end

    @always_inline
    def _next_f32_at_pos(mut self) -> Float32:
        if self.pos >= self.end:
            return 0.0

        var p = self.pos
        var sign: Float64 = 1.0

        if self.ptr.load(p) == MINUS:
            sign = -1.0
            p += 1
        elif self.ptr.load(p) == PLUS:
            p += 1

        var num: Float64 = 0.0
        while p < self.end:
            var b = self.ptr.load(p)
            if _is_digit(b):
                num = num * 10.0 + Float64(Int(b - ZERO))
                p += 1
            else:
                break

        if p < self.end and self.ptr.load(p) == DOT:
            p += 1
            var fra: Float64 = 0.0
            var div: Float64 = 1.0

            while p < self.end:
                var b = self.ptr.load(p)
                if _is_digit(b):
                    fra = fra * 10.0 + Float64(Int(b - ZERO))
                    div *= 10.0
                    p += 1
                else:
                    break

            num += fra / div

        if p < self.end:
            var b = self.ptr.load(p)
            if b == CHAR_e or b == CHAR_E:
                p += 1
                var exp_sign = 1

                if p < self.end and self.ptr.load(p) == MINUS:
                    exp_sign = -1
                    p += 1
                elif p < self.end and self.ptr.load(p) == PLUS:
                    p += 1

                var eval = 0
                while p < self.end:
                    var eb = self.ptr.load(p)
                    if _is_digit(eb):
                        eval = eval * 10 + Int(eb - ZERO)
                        p += 1
                    else:
                        break

                if eval > 0:
                    var power: Float64 = 1.0
                    for _ in range(eval):
                        power *= 10.0

                    if exp_sign == 1:
                        num *= power
                    else:
                        num /= power

        self.pos = p
        return Float32(sign * num)

    @always_inline
    def next_f32(mut self) -> Float32:
        self.skip_ws()
        return self._next_f32_at_pos()

    @always_inline
    def _parse_index_int(mut self, slash_terminates: Bool) -> Int:
        var sign = 1
        var output = 0

        if self.pos < self.end and self.ptr.load(self.pos) == MINUS:
            sign = -1
            self.pos += 1
        elif self.pos < self.end and self.ptr.load(self.pos) == PLUS:
            self.pos += 1

        while self.pos < self.end:
            var b = self.ptr.load(self.pos)
            if slash_terminates and b == SLASH:
                break
            if _is_ws_or_line_cut(b):
                break
            if not _is_digit(b):
                break
            output = output * 10 + Int(b - ZERO)
            self.pos += 1

        return sign * output

    @always_inline
    def _parse_positive_index_int(mut self, slash_terminates: Bool) -> Int:
        # Fast path for the common OBJ case: positive decimal indices.
        # Falls back to signed parsing only when a sign is actually present.
        if self.pos < self.end:
            var first = self.ptr.load(self.pos)
            if first == MINUS or first == PLUS:
                return self._parse_index_int(slash_terminates)

        var output = 0
        while self.pos < self.end:
            var b = self.ptr.load(self.pos)
            if slash_terminates and b == SLASH:
                break
            if _is_ws_or_line_cut(b):
                break
            if not _is_digit(b):
                break

            output = output * 10 + Int(b - ZERO)
            self.pos += 1

        return output

    @always_inline
    def _finish_index_token(mut self):
        while self.pos < self.end:
            var b = self.ptr.load(self.pos)
            if _is_ws(b):
                break
            if _is_line_cut(b):
                self.end = self.pos
                break
            self.pos += 1

    @always_inline
    def _peek_face_shape(mut self) -> Int:
        # Returns:
        #   0 = generic / weird
        #   1 = p
        #   2 = p/t
        #   3 = p//n
        #   4 = p/t/n
        self.skip_ws()

        var p = self.pos
        var slash_count = 0
        var first_slash = -1

        while p < self.end:
            var b = self.ptr.load(p)
            if _is_ws_or_line_cut(b):
                break
            if b == SLASH:
                if slash_count == 0:
                    first_slash = p
                slash_count += 1
            p += 1

        if slash_count == 0:
            return 1
        if slash_count == 1:
            return 2
        if slash_count == 2:
            if (
                first_slash + 1 < self.end
                and self.ptr.load(first_slash + 1) == SLASH
            ):
                return 3
            return 4

        return 0

    @always_inline
    def _at_signed_index(mut self) -> Bool:
        if self.pos >= self.end:
            return False
        var b = self.ptr.load(self.pos)
        return b == MINUS or b == PLUS

    @always_inline
    def next_index_p_only_at_token(mut self, position_count: Int) -> ObjIndex:
        # Common path: positive OBJ indices are already absolute 1-based indices.
        # Only signed/relative tokens need _fix_index().
        var needs_fix = self._at_signed_index()
        var p_raw = self._parse_positive_index_int(slash_terminates=False)
        self._finish_index_token()

        if needs_fix:
            return ObjIndex(_fix_index(p_raw, position_count), 0, 0)
        return ObjIndex(p_raw, 0, 0)

    @always_inline
    def next_index_p_t_at_token(
        mut self, position_count: Int, texcoord_count: Int
    ) -> ObjIndex:
        var needs_fix = self._at_signed_index()
        var p_raw = self._parse_positive_index_int(slash_terminates=True)
        var t_raw = 0

        if self.pos < self.end and self.ptr.load(self.pos) == SLASH:
            self.pos += 1
            if self._at_signed_index():
                needs_fix = True
            t_raw = self._parse_positive_index_int(slash_terminates=False)

        self._finish_index_token()

        if needs_fix:
            return ObjIndex(
                _fix_index(p_raw, position_count),
                _fix_index(t_raw, texcoord_count),
                0,
            )

        return ObjIndex(p_raw, t_raw, 0)

    @always_inline
    def next_index_p_n_at_token(
        mut self, position_count: Int, normal_count: Int
    ) -> ObjIndex:
        var needs_fix = self._at_signed_index()
        var p_raw = self._parse_positive_index_int(slash_terminates=True)
        var n_raw = 0

        if self.pos < self.end and self.ptr.load(self.pos) == SLASH:
            self.pos += 1
            if self.pos < self.end and self.ptr.load(self.pos) == SLASH:
                self.pos += 1
                if self._at_signed_index():
                    needs_fix = True
                n_raw = self._parse_positive_index_int(slash_terminates=False)

        self._finish_index_token()

        if needs_fix:
            return ObjIndex(
                _fix_index(p_raw, position_count),
                0,
                _fix_index(n_raw, normal_count),
            )

        return ObjIndex(p_raw, 0, n_raw)

    @always_inline
    def next_index_p_t_n_at_token(
        mut self,
        position_count: Int,
        texcoord_count: Int,
        normal_count: Int,
    ) -> ObjIndex:
        var needs_fix = self._at_signed_index()
        var p_raw = self._parse_positive_index_int(slash_terminates=True)
        var t_raw = 0
        var n_raw = 0

        if self.pos < self.end and self.ptr.load(self.pos) == SLASH:
            self.pos += 1
            if self._at_signed_index():
                needs_fix = True
            t_raw = self._parse_positive_index_int(slash_terminates=True)

            if self.pos < self.end and self.ptr.load(self.pos) == SLASH:
                self.pos += 1
                if self._at_signed_index():
                    needs_fix = True
                n_raw = self._parse_positive_index_int(slash_terminates=False)

        self._finish_index_token()

        if needs_fix:
            return ObjIndex(
                _fix_index(p_raw, position_count),
                _fix_index(t_raw, texcoord_count),
                _fix_index(n_raw, normal_count),
            )

        return ObjIndex(p_raw, t_raw, n_raw)

    @always_inline
    def next_index_generic_at_token(
        mut self,
        position_count: Int,
        texcoord_count: Int,
        normal_count: Int,
    ) -> ObjIndex:
        var p_raw = self._parse_index_int(slash_terminates=True)
        var t_raw = 0
        var n_raw = 0

        if self.pos < self.end and self.ptr.load(self.pos) == SLASH:
            self.pos += 1
            t_raw = self._parse_index_int(slash_terminates=True)

            if self.pos < self.end and self.ptr.load(self.pos) == SLASH:
                self.pos += 1
                n_raw = self._parse_index_int(slash_terminates=False)

        self._finish_index_token()

        return ObjIndex(
            _fix_index(p_raw, position_count),
            _fix_index(t_raw, texcoord_count),
            _fix_index(n_raw, normal_count),
        )

    def joined_rest_of_line(mut self) -> String:
        self.skip_ws()
        if self.pos >= self.end:
            return ""

        var start = self.pos
        var logical_end = self.end

        # Only string-like OBJ/MTL commands use this rare path. Keep comments
        # lazy here instead of pre-scanning every geometry line.
        var p = start
        while p < logical_end:
            var b = self.ptr.load(p)
            if _is_line_cut(b):
                logical_end = p
                break
            p += 1

        var end_pos = logical_end - 1

        # Trim trailing whitespace.
        while end_pos >= start:
            var b = self.ptr.load(end_pos)
            if not _is_ws(b):
                break
            end_pos -= 1

        self.pos = self.end

        if end_pos < start:
            return ""

        return String(self.text[byte = start : end_pos + 1])

    @always_inline
    def next_word(mut self) -> StringSlice[Self.o]:
        self.skip_ws()
        if self.pos >= self.end:
            return StringSlice[Self.o]()

        var start = self.pos
        while self.pos < self.end:
            var b = self.ptr.load(self.pos)
            if _is_ws_or_line_cut(b):
                if _is_line_cut(b):
                    self.end = self.pos
                break
            self.pos += 1

        return self.text[byte = start : self.pos]


@always_inline
def _parse_v_cursor[o: Origin](mut mesh: ObjMesh, mut cur: ObjLineCursor[o]):
    cur.skip_ws()
    if cur.pos >= cur.end:
        return
    var x = cur._next_f32_at_pos()

    cur.skip_ws()
    if cur.pos >= cur.end:
        return
    var y = cur._next_f32_at_pos()

    cur.skip_ws()
    if cur.pos >= cur.end:
        return
    var z = cur._next_f32_at_pos()

    mesh.positions.append(x)
    mesh.positions.append(y)
    mesh.positions.append(z)

    # Optional vertex color: v x y z r g b.
    cur.skip_ws()
    if cur.pos < cur.end:
        var r = cur._next_f32_at_pos()

        cur.skip_ws()
        if cur.pos < cur.end:
            var g = cur._next_f32_at_pos()

            cur.skip_ws()
            if cur.pos < cur.end:
                var b = cur._next_f32_at_pos()
                mesh._push_color(r, g, b)


@always_inline
def _parse_vt_cursor[o: Origin](mut mesh: ObjMesh, mut cur: ObjLineCursor[o]):
    cur.skip_ws()
    if cur.pos >= cur.end:
        return
    var u = cur._next_f32_at_pos()

    cur.skip_ws()
    if cur.pos >= cur.end:
        return
    var v = cur._next_f32_at_pos()

    mesh.texcoords.append(u)
    mesh.texcoords.append(v)


@always_inline
def _parse_vn_cursor[o: Origin](mut mesh: ObjMesh, mut cur: ObjLineCursor[o]):
    cur.skip_ws()
    if cur.pos >= cur.end:
        return
    var x = cur._next_f32_at_pos()

    cur.skip_ws()
    if cur.pos >= cur.end:
        return
    var y = cur._next_f32_at_pos()

    cur.skip_ws()
    if cur.pos >= cur.end:
        return
    var z = cur._next_f32_at_pos()

    mesh.normals.append(x)
    mesh.normals.append(y)
    mesh.normals.append(z)


@always_inline
def _finish_face_parse(
    mut mesh: ObjMesh,
    index_start: Int,
    count: Int,
    valid: Bool,
    is_line: Bool,
):
    if valid:
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
    else:
        mesh.indices.shrink(index_start)


@always_inline
def _parse_face_cursor[
    o: Origin
](mut mesh: ObjMesh, mut cur: ObjLineCursor[o], is_line: Bool = False):
    var index_start = len(mesh.indices)
    var count = 0
    var valid = True

    # Counts are constant for this face. Do not recompute per vertex token.
    var position_count = mesh.position_count()
    var texcoord_count = mesh.texcoord_count()
    var normal_count = mesh.normal_count()

    # Detect p / p/t / p//n / p/t/n once from the first token of the face.
    # Real OBJ faces almost always use one shape consistently for the whole line.
    var shape = cur._peek_face_shape()

    if shape == 1:
        # f p p p ...
        while True:
            cur.skip_ws()
            if cur.pos >= cur.end:
                break

            var idx = cur.next_index_p_only_at_token(position_count)
            if idx.p == 0:
                valid = False
                break

            mesh.indices.append(idx)
            count += 1

    elif shape == 2:
        # f p/t p/t p/t ...
        while True:
            cur.skip_ws()
            if cur.pos >= cur.end:
                break

            var idx = cur.next_index_p_t_at_token(
                position_count, texcoord_count
            )
            if idx.p == 0:
                valid = False
                break

            mesh.indices.append(idx)
            count += 1

    elif shape == 3:
        # f p//n p//n p//n ...
        while True:
            cur.skip_ws()
            if cur.pos >= cur.end:
                break

            var idx = cur.next_index_p_n_at_token(position_count, normal_count)
            if idx.p == 0:
                valid = False
                break

            mesh.indices.append(idx)
            count += 1

    elif shape == 4:
        # f p/t/n p/t/n p/t/n ...
        while True:
            cur.skip_ws()
            if cur.pos >= cur.end:
                break

            var idx = cur.next_index_p_t_n_at_token(
                position_count, texcoord_count, normal_count
            )
            if idx.p == 0:
                valid = False
                break

            mesh.indices.append(idx)
            count += 1

    else:
        # Weird or malformed first token: preserve the old generic behavior.
        while True:
            cur.skip_ws()
            if cur.pos >= cur.end:
                break

            var idx = cur.next_index_generic_at_token(
                position_count, texcoord_count, normal_count
            )
            if idx.p == 0:
                valid = False
                break

            mesh.indices.append(idx)
            count += 1

    _finish_face_parse(mesh, index_start, count, valid, is_line)


def _parse_obj_text[
    Loader: ObjTextLoader
](path: String, text: String, loader: Loader) raises -> ObjMesh:
    return _parse_obj_slice(path, StringSlice(text), loader)


def _parse_obj_slice[
    o: Origin, Loader: ObjTextLoader
](path: String, text_slice: StringSlice[o], loader: Loader) raises -> ObjMesh:
    var mesh = ObjMesh()

    # Same rough heuristic as before, with reserves for the other hot arrays too.
    var est_elements = text_slice.byte_length() // 15
    mesh.positions.reserve(est_elements * 3)
    mesh.texcoords.reserve(est_elements * 2)
    mesh.normals.reserve(est_elements * 3)
    mesh.indices.reserve(est_elements * 6)
    mesh.face_vertices.reserve(est_elements)
    mesh.face_materials.reserve(est_elements)

    var ptr = text_slice.unsafe_ptr()
    var text_len = text_slice.byte_length()
    var line_start = 0

    while line_start < text_len:
        var line_end = text_slice.find("\n", line_start)
        if line_end == -1:
            line_end = text_len

        # Do not pre-scan the whole line for CR/comments here. ObjLineCursor
        # treats CR and # as lazy logical line-end markers while parsing.
        var end = line_end
        var cur = ObjLineCursor(text_slice, line_start, end)
        cur.skip_ws()

        if cur.pos < cur.end:
            var p = cur.pos
            var c0 = ptr.load(p)

            # Hot one/two-byte tag dispatch.
            if c0 == CHAR_v:
                if _word_ends_here(ptr, p + 1, end):
                    cur.pos = p + 1
                    _parse_v_cursor(mesh, cur)
                elif p + 1 < end and ptr.load(p + 1) == CHAR_t:
                    if _word_ends_here(ptr, p + 2, end):
                        cur.pos = p + 2
                        _parse_vt_cursor(mesh, cur)
                elif p + 1 < end and ptr.load(p + 1) == CHAR_n:
                    if _word_ends_here(ptr, p + 2, end):
                        cur.pos = p + 2
                        _parse_vn_cursor(mesh, cur)

            elif c0 == CHAR_f and _word_ends_here(ptr, p + 1, end):
                cur.pos = p + 1
                _parse_face_cursor(mesh, cur, is_line=False)

            elif c0 == CHAR_l and _word_ends_here(ptr, p + 1, end):
                cur.pos = p + 1
                _parse_face_cursor(mesh, cur, is_line=True)

            elif c0 == CHAR_g and _word_ends_here(ptr, p + 1, end):
                cur.pos = p + 1
                mesh._begin_group(cur.joined_rest_of_line())

            elif c0 == CHAR_o and _word_ends_here(ptr, p + 1, end):
                cur.pos = p + 1
                mesh._begin_object(cur.joined_rest_of_line())

            else:
                # Rare multi-char tags.
                var tag_start = p
                var tag_end = p
                while tag_end < end and not _is_ws_or_line_cut(
                    ptr.load(tag_end)
                ):
                    tag_end += 1

                var tag = text_slice[byte=tag_start:tag_end]
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
                        try:
                            _read_mtl_file(mesh, path, mtl_name, loader)
                        except:
                            pass

        line_start = line_end + 1

    if len(mesh.colors) > 0:
        while len(mesh.colors) < len(mesh.positions):
            mesh.colors.append(1.0)

    mesh._finish()
    return mesh^
