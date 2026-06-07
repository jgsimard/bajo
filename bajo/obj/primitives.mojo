from .constants import SPACE, TAB, CR, HASH, ZERO, NINE


def _is_ws(b: UInt8) -> Bool:
    return b == SPACE or b == TAB or b == CR


def _is_line_cut(b: UInt8) -> Bool:
    return b == CR or b == HASH


def _is_ws_or_line_cut(b: UInt8) -> Bool:
    return _is_ws(b) or b == HASH


def _is_digit(b: UInt8) -> Bool:
    return b >= ZERO and b <= NINE


def _fix_index(raw: Int, count_with_dummy: Int) -> Int:
    if raw > 0:
        return raw
    if raw < 0:
        return count_with_dummy + raw
    return 0
