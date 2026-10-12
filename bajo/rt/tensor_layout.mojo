"""Backend-neutral field and record TileTensor views.

Field count is static. The item extent can be static for CPU SIMD packets or
dynamic for GPU queues; both expose the same logical [field, item] coordinates.
"""

from layout import TileTensor, Coord, Idx, CoordLike
from layout.tile_layout import row_major


comptime RtFieldLayout[fields: Int, Extent: CoordLike] = type_of(
    row_major(Coord(Idx[fields], Extent()))
)
comptime RtFieldTile[
    dtype: DType, fields: Int, Extent: CoordLike, origin: Origin
] = TileTensor[dtype, RtFieldLayout[fields, Extent], origin]


@always_inline
def rt_field_tile[
    mut: Bool,
    origin: Origin[mut=mut],
    //,
    dtype: DType,
    fields: Int,
    Extent: CoordLike,
](storage: Pointer[Scalar[dtype], origin], extent: Extent) -> RtFieldTile[
    dtype, fields, Extent, origin
]:
    var layout = row_major(Coord(Idx[fields], extent))
    return TileTensor(
        Span(unsafe_ptr=storage, length=Int(layout.size())), layout
    )


comptime RtRecordLayout[stride: Int] = type_of(
    row_major(Coord(Int(0), Idx[stride]))
)
comptime RtRecordTile[dtype: DType, stride: Int, origin: Origin] = TileTensor[
    dtype, RtRecordLayout[stride], origin
]


@always_inline
def rt_record_tile[
    mut: Bool, origin: Origin[mut=mut], //, dtype: DType, stride: Int
](storage: Pointer[Scalar[dtype], origin], count: Int) -> RtRecordTile[
    dtype, stride, origin
]:
    return TileTensor(
        Span(unsafe_ptr=storage, length=stride * count),
        row_major(Coord(count, Idx[stride])),
    )
