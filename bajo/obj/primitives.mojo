from .constants import SPACE, TAB, CR, HASH, ZERO, NINE


def _is_ws(b: UInt8) -> Bool:
    return b == SPACE or b == TAB or b == CR


def _is_line_cut(b: UInt8) -> Bool:
    return b == CR or b == HASH


def _is_ws_or_line_cut(b: UInt8) -> Bool:
    return _is_ws(b) or b == HASH


def _is_digit(b: UInt8) -> Bool:
    return b >= ZERO and b <= NINE


@always_inline
def _fix_index(raw: Int, count_with_dummy: Int) raises -> Int:
    var resolved = 0
    if raw > 0:
        resolved = raw
    elif raw < 0:
        resolved = count_with_dummy + raw

    if resolved <= 0 or resolved >= count_with_dummy:
        raise String(
            t"OBJ index {raw} is out of range for "
            t"{count_with_dummy - 1} available elements"
        )

    return resolved
