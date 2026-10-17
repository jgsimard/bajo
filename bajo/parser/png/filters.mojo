"""SIMD reconstruction of noninterlaced, eight-bit RGB PNG scanlines.

Filter definitions: https://www.w3.org/TR/png-3/#9Filters
Rows include their leading filter byte and one accessible padding byte after
the last row. Four-byte pixel loads preserve the overlapping fourth byte.
"""

from std.math import abs
from std.memory import bitcast
from std.sys import simd_width_of


def _undo_sub(row: MutSpan[UInt8, _]):
    comptime zero = SIMD[.uint8, 16](0)
    var carry = zero
    var end = len(row) // 16 * 16
    for x in range(0, end, 16):
        var ptr = row.unsafe_ptr().unsafe_offset(x)
        var values = ptr.unsafe_load[width=16]()
        # Three interleaved inclusive scans, one per RGB channel.
        values += values.shuffle[
            16, 16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        ](zero)
        values += values.shuffle[
            16, 16, 16, 16, 16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        ](zero)
        values += values.shuffle[
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 1, 2, 3
        ](zero)
        values += carry.shuffle[
            13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13
        ]()
        ptr.unsafe_store[width=16](values)
        carry = values
    for x in range(max(3, end), len(row)):
        row[x] += row[x - 3]


def _undo_up(row: MutSpan[UInt8, _], above: ImmSpan[UInt8, _]):
    comptime lanes = simd_width_of[UInt8]()
    var end = len(row) // lanes * lanes
    for x in range(0, end, lanes):
        var ptr = row.unsafe_ptr().unsafe_offset(x)
        ptr.unsafe_store[width=lanes](
            ptr.unsafe_load[width=lanes]()
            + above.unsafe_ptr().unsafe_offset(x).unsafe_load[width=lanes]()
        )
    for x in range(end, len(row)):
        row[x] += above[x]


def _undo_predictive[
    paeth: Bool
](row: MutSpan[UInt8, _], above: ImmSpan[UInt8, _]):
    var left = SIMD[.int16, 4](0)
    var upper_left = SIMD[.int16, 4](0)
    for x in range(0, len(row), 3):
        var ptr = row.unsafe_ptr().unsafe_offset(x)
        var filtered = ptr.unsafe_load[width=4]()
        var up = (
            above.unsafe_ptr()
            .unsafe_offset(x)
            .unsafe_load[width=4]()
            .cast[.int16]()
        )
        var predictor: SIMD[.int16, 4]
        comptime if paeth:
            var a = abs(up - upper_left)
            var b = abs(left - upper_left)
            var c = abs(left + up - 2 * upper_left)
            predictor = (a.le(b) & a.le(c)).select(
                left, b.le(c).select(up, upper_left)
            )
        else:
            predictor = (left + up) >> 1
        var value = filtered + predictor.cast[.uint8]()
        # Lane three overlaps the next pixel (or row's filter byte).
        ptr.unsafe_store[width=4](value.shuffle[0, 1, 2, 7](filtered))
        left = value.cast[.int16]()
        upper_left = up


@always_inline
def _average_diagonal[
    steady: Bool
](
    raw: MutSpan[UInt8, _],
    first_row: Int,
    width: Int,
    diagonal: Int,
    mut left: SIMD[.int16, 16],
):
    var stride = width * 3 + 1
    var packed = SIMD[.uint32, 4](0)
    comptime for row in range(4):
        var x = diagonal - row
        if steady or (x >= 0 and x < width):
            packed[row] = (
                raw.unsafe_ptr()
                .unsafe_offset((first_row + row) * stride + 1 + x * 3)
                .unsafe_bitcast[UInt32]()
                .unsafe_load[alignment=1]()
            )
    var above = UInt32(0)
    if steady or diagonal < width:
        above = (
            raw.unsafe_ptr()
            .unsafe_offset((first_row - 1) * stride + 1 + diagonal * 3)
            .unsafe_bitcast[UInt32]()
            .unsafe_load[alignment=1]()
        )
    var top = bitcast[.uint8, 16](SIMD[.uint32, 4](above, 0, 0, 0)).cast[
        .int16
    ]()
    # A diagonal makes both left and up available from the previous step.
    # The four RGB pixels occupy independent four-lane groups (RGB + padding).
    var up = left.shuffle[16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11](
        top
    )
    var value = bitcast[.uint8, 16](packed) + ((left + up) >> 1).cast[.uint8]()
    left = value.cast[.int16]()
    var output = (bitcast[.uint32, 4](value) & UInt32(0xFFFFFF)) | (
        packed & UInt32(0xFF000000)
    )
    comptime for row in range(4):
        var x = diagonal - row
        if steady or (x >= 0 and x < width):
            raw.unsafe_ptr().unsafe_offset(
                (first_row + row) * stride + 1 + x * 3
            ).unsafe_bitcast[UInt32]().unsafe_store[alignment=1](output[row])


def _undo_average_four(raw: MutSpan[UInt8, _], first_row: Int, width: Int):
    var left = SIMD[.int16, 16](0)
    for diagonal in range(3):
        _average_diagonal[False](raw, first_row, width, diagonal, left)
    for diagonal in range(3, width):
        _average_diagonal[True](raw, first_row, width, diagonal, left)
    for diagonal in range(width, width + 3):
        _average_diagonal[False](raw, first_row, width, diagonal, left)


def _unfilter_row(
    row: MutSpan[UInt8, _], above: ImmSpan[UInt8, _], filter: UInt8
) raises:
    match filter:
        case 0:
            pass
        case 1:
            _undo_sub(row)
        case 2:
            _undo_up(row, above)
        case 3:
            _undo_predictive[False](row, above)
        case 4:
            _undo_predictive[True](row, above)
        case _:
            raise Error("invalid PNG filter type")


def _unfilter_rgb(mut raw: List[UInt8], width: Int, height: Int) raises:
    var row_bytes = 3 * width
    var stride = row_bytes + 1
    var zero = List[UInt8](length=row_bytes + 1, fill=0)
    var y = 0
    while y < height:
        var offset = y * stride
        if width >= 4 and y > 0 and y + 4 <= height:
            if (
                raw[offset] == 3
                and raw[offset + stride] == 3
                and raw[offset + 2 * stride] == 3
                and raw[offset + 3 * stride] == 3
            ):
                _undo_average_four(Span(raw), y, width)
                y += 4
                continue
        var row = Span(raw)[offset + 1 : offset + stride]
        # The immutable previous row is disjoint from the mutable current row.
        # Keep both owning lists alive through the call, including SIMD padding.
        var above_ptr = (
            ImmSpan(zero).unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()
        )
        if y != 0:
            above_ptr = (
                ImmSpan(raw)
                .unsafe_ptr()
                .unsafe_offset(offset + 1 - stride)
                .unsafe_origin_cast[ImmUntrackedOrigin]()
            )
        var above = ImmSpan[UInt8, ImmUntrackedOrigin](
            unsafe_ptr=above_ptr, length=row_bytes
        )
        _unfilter_row(row, above, raw[offset])
        y += 1
    _ = len(zero)
