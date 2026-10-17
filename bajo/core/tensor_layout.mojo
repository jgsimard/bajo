"""Borrowed field and record TileTensor views for CPU and GPU buffers.

Field count is static. The item extent can be static for CPU SIMD packets or
dynamic for GPU queues; both expose the same logical [field, item] coordinates.
"""

from layout import TileTensor, TensorLayout, Coord, Idx, CoordLike
from layout.tile_layout import row_major, col_major
from .vec import Vec3
from .frame import Frame


comptime FieldLayout[fields: Int, Extent: CoordLike] = type_of(
    row_major(Coord(Idx[fields], Extent()))
)
comptime FieldTile[
    dtype: DType, fields: Int, Extent: CoordLike, origin: Origin
] = TileTensor[dtype, FieldLayout[fields, Extent], origin]


@always_inline
def field_tile[
    mut: Bool,
    origin: Origin[mut=mut],
    //,
    dtype: DType,
    fields: Int,
    Extent: CoordLike,
](storage: Pointer[Scalar[dtype], origin], extent: Extent) -> FieldTile[
    dtype, fields, Extent, origin
]:
    var layout = row_major(Coord(Idx[fields], extent))
    return TileTensor(
        Span(unsafe_ptr=storage, length=Int(layout.size())), layout
    )


comptime RecordLayout[stride: Int, column_major: Bool = False] = (
    type_of(col_major(Coord(Int(0), Idx[stride]))) if column_major else type_of(
        row_major(Coord(Int(0), Idx[stride]))
    )
)
comptime RecordTile[
    dtype: DType, stride: Int, origin: Origin, column_major: Bool = False
] = TileTensor[dtype, RecordLayout[stride, column_major], origin]


@always_inline
def record_tile[
    mut: Bool,
    origin: Origin[mut=mut],
    //,
    dtype: DType,
    stride: Int,
    column_major: Bool = False,
](storage: Pointer[Scalar[dtype], origin], count: Int) -> RecordTile[
    dtype, stride, origin, column_major
]:
    var span = Span(unsafe_ptr=storage, length=stride * count)
    comptime if column_major:
        return rebind[RecordTile[dtype, stride, origin, column_major]](
            TileTensor(span, col_major(Coord(count, Idx[stride])))
        )
    else:
        return rebind[RecordTile[dtype, stride, origin, column_major]](
            TileTensor(span, row_major(Coord(count, Idx[stride])))
        )


@always_inline
def gather_record_field[
    mut: Bool,
    origin: Origin[mut=mut],
    Layout: TensorLayout,
    //,
    field: Int,
    dtype: DType,
    index_dtype: DType,
    length: SIMDLength,
](
    records: TileTensor[dtype, Layout, origin],
    ids: SIMD[index_dtype, length],
    uniform: Bool = False,
) -> SIMD[dtype, length]:
    """Read a field across valid record IDs in a flat affine [item, field] view.

    Layouts may put items or fields adjacent. Irregular IDs still need gathers;
    scalar and uniform accesses use one ordinary TileTensor load.
    """
    comptime assert Layout.rank == 2 and Layout.flat_rank == 2
    comptime if length == 1:
        return SIMD[dtype, length](records[(Int(ids[0]), field)])
    else:
        if uniform:
            return SIMD[dtype, length](records[(Int(ids[0]), field)])
        var stride = Int(records.layout(Coord(1, 0)))
        var offset = Int(records.layout(Coord(0, field)))
        return records.ptr.unsafe_gather(ids.cast[.int]() * stride + offset)


@always_inline
def gather_record_vec3[
    mut: Bool,
    origin: Origin[mut=mut],
    Layout: TensorLayout,
    dtype: DType,
    index_dtype: DType,
    length: SIMDLength,
    //,
    frame: Frame = .WORLD,
    field: Int = 0,
](
    records: TileTensor[dtype, Layout, origin],
    ids: SIMD[index_dtype, length],
    uniform: Bool = False,
) -> Vec3[dtype, frame, length]:
    return Vec3[dtype, frame, length](
        gather_record_field[field](records, ids, uniform),
        gather_record_field[field + 1](records, ids, uniform),
        gather_record_field[field + 2](records, ids, uniform),
    )
