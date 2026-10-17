from .types import ObjIndex
from bajo.parser.number import parse_f32_fast_at
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
    def __init__(out self, count_with_dummy: Int) raises:
        if count_with_dummy < 1 or count_with_dummy - 1 > 0x7FFFFFFF:
            raise Error("OBJ element count exceeds signed 32-bit index storage")
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
        var parsed = parse_f32_fast_at(self.bytes, self.pos)
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
    def _parse_resolved_index(
        mut self, slash_terminates: Bool, limit: ObjIndexLimit
    ) -> Int:
        # Zero reports an invalid field. The corner parser raises once after
        # validating its fields, keeping error construction out of this loop.
        if self.pos >= len(self.bytes):
            return 0
        var first = self.bytes.unsafe_get(self.pos)
        var negative = first == MINUS
        if negative or first == PLUS:
            self.pos += 1

        var output = 0
        while self.pos < len(self.bytes):
            var b = self.bytes.unsafe_get(self.pos)
            if _is_digit(b):
                var digit = Int(b - ZERO)
                output = output * 10 + digit
                if output > limit.max_magnitude:
                    return 0
                self.pos += 1
                continue
            break

        if output == 0:
            return 0
        if self.pos < len(self.bytes):
            var b = self.bytes.unsafe_get(self.pos)
            if not ((slash_terminates and b == SLASH) or _is_ws_or_line_cut(b)):
                return 0
        return limit.count_with_dummy - output if negative else output

    @always_inline
    def next_index_p_only_at_token(
        mut self, position_limit: ObjIndexLimit
    ) raises -> ObjIndex:
        var p = self._parse_resolved_index(False, position_limit)
        if p == 0:
            raise Error("invalid or out-of-range OBJ position index")
        return ObjIndex._from_validated(p, 0, 0)

    def next_index_p_t_at_token(
        mut self, position_limit: ObjIndexLimit, texcoord_limit: ObjIndexLimit
    ) raises -> ObjIndex:
        var p = self._parse_resolved_index(True, position_limit)
        var t = 0
        if (
            self.pos < len(self.bytes)
            and self.bytes.unsafe_get(self.pos) == SLASH
        ):
            self.pos += 1
            t = self._parse_resolved_index(False, texcoord_limit)
        if p == 0 or t == 0:
            raise Error("invalid or out-of-range OBJ position/texture index")
        return ObjIndex._from_validated(p, t, 0)

    def next_index_p_n_at_token(
        mut self, position_limit: ObjIndexLimit, normal_limit: ObjIndexLimit
    ) raises -> ObjIndex:
        var p = self._parse_resolved_index(True, position_limit)
        var n = 0
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
                n = self._parse_resolved_index(False, normal_limit)
        if p == 0 or n == 0:
            raise Error("invalid or out-of-range OBJ position/normal index")
        return ObjIndex._from_validated(p, 0, n)

    def next_index_p_t_n_at_token(
        mut self,
        position_limit: ObjIndexLimit,
        texcoord_limit: ObjIndexLimit,
        normal_limit: ObjIndexLimit,
    ) raises -> ObjIndex:
        var p = self._parse_resolved_index(True, position_limit)
        var t = 0
        var n = 0
        if (
            self.pos < len(self.bytes)
            and self.bytes.unsafe_get(self.pos) == SLASH
        ):
            self.pos += 1
            t = self._parse_resolved_index(True, texcoord_limit)
            if (
                self.pos < len(self.bytes)
                and self.bytes.unsafe_get(self.pos) == SLASH
            ):
                self.pos += 1
                n = self._parse_resolved_index(False, normal_limit)
        if p == 0 or t == 0 or n == 0:
            raise Error("invalid or out-of-range OBJ face index")
        return ObjIndex._from_validated(p, t, n)

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
        return ObjIndex._from_validated(
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
        var p = self._parse_resolved_index(True, position_limit)
        var valid = p != 0
        var t = 0
        var n = 0
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
                n = self._parse_resolved_index(False, normal_limit)
                valid = valid and n != 0
            else:
                shape = 2
                t = self._parse_resolved_index(True, texcoord_limit)
                valid = valid and t != 0
                if (
                    self.pos < len(self.bytes)
                    and self.bytes.unsafe_get(self.pos) == SLASH
                ):
                    self.pos += 1
                    shape = 4
                    n = self._parse_resolved_index(False, normal_limit)
                    valid = valid and n != 0
        if not valid:
            raise Error("invalid or out-of-range OBJ face index")
        return FirstFaceIndex(ObjIndex._from_validated(p, t, n), shape)

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
