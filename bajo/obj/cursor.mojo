from .types import ObjIndex
from .f32 import parse_f32_at
from .primitives import (
    _fix_index,
    _is_digit,
    _is_line_cut,
    _is_ws,
    _is_ws_or_line_cut,
)
from .constants import MINUS, PLUS, SLASH, ZERO


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


struct ObjLineCursor[origin: ImmOrigin]:
    """Shared allocation-free line cursor for OBJ and MTL parsing."""

    var bytes: ImmSpan[UInt8, Self.origin]
    var pos: Int

    @always_inline
    def __init__(out self, bytes: ImmSpan[UInt8, Self.origin]):
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
        if self.pos >= len(self.bytes):
            raise String("missing OBJ index")
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
        var p_raw = self._parse_positive_index_int(False, position_limit)
        if needs_fix:
            return ObjIndex(
                _fix_index(p_raw, position_limit.count_with_dummy), 0, 0
            )
        return ObjIndex(p_raw, 0, 0)

    def next_index_p_t_at_token(
        mut self, position_limit: ObjIndexLimit, texcoord_limit: ObjIndexLimit
    ) raises -> ObjIndex:
        var needs_fix = self._at_signed_index()
        var p_raw = self._parse_positive_index_int(True, position_limit)
        var t_raw = 0
        if (
            self.pos < len(self.bytes)
            and self.bytes.unsafe_get(self.pos) == SLASH
        ):
            self.pos += 1
            if self._at_signed_index():
                needs_fix = True
            t_raw = self._parse_positive_index_int(False, texcoord_limit)
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
        var p_raw = self._parse_positive_index_int(True, position_limit)
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
                n_raw = self._parse_positive_index_int(False, normal_limit)
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
        var p_raw = self._parse_positive_index_int(True, position_limit)
        var t_raw = 0
        var n_raw = 0
        if (
            self.pos < len(self.bytes)
            and self.bytes.unsafe_get(self.pos) == SLASH
        ):
            self.pos += 1
            if self._at_signed_index():
                needs_fix = True
            t_raw = self._parse_positive_index_int(True, texcoord_limit)
            if (
                self.pos < len(self.bytes)
                and self.bytes.unsafe_get(self.pos) == SLASH
            ):
                self.pos += 1
                if self._at_signed_index():
                    needs_fix = True
                n_raw = self._parse_positive_index_int(False, normal_limit)
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
        var p_raw = self._parse_index_int(True, position_limit)
        var t_raw = 0
        var n_raw = 0
        if (
            self.pos < len(self.bytes)
            and self.bytes.unsafe_get(self.pos) == SLASH
        ):
            self.pos += 1
            t_raw = self._parse_index_int(True, texcoord_limit)
            if (
                self.pos < len(self.bytes)
                and self.bytes.unsafe_get(self.pos) == SLASH
            ):
                self.pos += 1
                n_raw = self._parse_index_int(False, normal_limit)
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
        var shape = 1
        var needs_fix = self._at_signed_index()
        var p_raw = self._parse_positive_index_int(True, position_limit)
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
                self.pos += 1
                shape = 3
                if self._at_signed_index():
                    needs_fix = True
                n_raw = self._parse_positive_index_int(False, normal_limit)
            else:
                shape = 2
                if self._at_signed_index():
                    needs_fix = True
                t_raw = self._parse_positive_index_int(True, texcoord_limit)
                if (
                    self.pos < len(self.bytes)
                    and self.bytes.unsafe_get(self.pos) == SLASH
                ):
                    self.pos += 1
                    shape = 4
                    if self._at_signed_index():
                        needs_fix = True
                    n_raw = self._parse_positive_index_int(False, normal_limit)
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
        var p = start
        while p < logical_end:
            var b = self.bytes.unsafe_get(p)
            if _is_line_cut(b):
                logical_end = p
                self.bytes = self.bytes[:logical_end]
                break
            p += 1
        var end_pos = logical_end - 1
        while end_pos >= start:
            var b = self.bytes.unsafe_get(end_pos)
            if not _is_ws(b):
                break
            end_pos -= 1
        self.pos = len(self.bytes)
        if end_pos < start:
            return ""
        return String(
            StringSpan[Self.origin](
                unsafe_from_utf8=self.bytes[start : end_pos + 1]
            )
        )

    def next_word(mut self) -> StringSpan[Self.origin]:
        self.skip_ws()
        if self.pos >= len(self.bytes):
            return StringSpan[Self.origin]()
        var start = self.pos
        while self.pos < len(self.bytes):
            var b = self.bytes.unsafe_get(self.pos)
            if _is_ws_or_line_cut(b):
                if _is_line_cut(b):
                    self.bytes = self.bytes[: self.pos]
                break
            self.pos += 1
        return StringSpan[Self.origin](
            unsafe_from_utf8=self.bytes[start : self.pos]
        )
