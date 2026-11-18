"""SIMD record gathers and Vec3 loads from Modular TileTensor views.

TileTensor.load reads consecutive storage. Irregular record IDs need a gather;
these helpers map those IDs through the tensor's existing affine layout.
"""

from layout import TileTensor, TensorLayout, Coord
from .vec import Vec3
from .frame import Frame


@inline(.always)
def gather_record_field[
    Layout: TensorLayout,
    //,
    field: Int,
    dtype: DType,
    index_dtype: DType,
    length: SIMDLength,
](
    records: TileTensor[dtype, Layout, _],
    ids: SIMD[index_dtype, length],
    uniform: Bool = False,
) -> SIMD[dtype, length]:
    return gather_record_field(records, ids, field, uniform)


@inline(.always)
def gather_record_field[
    Layout: TensorLayout,
    dtype: DType,
    index_dtype: DType,
    length: SIMDLength,
](
    records: TileTensor[dtype, Layout, _],
    ids: SIMD[index_dtype, length],
    field: Int,
    uniform: Bool = False,
) -> SIMD[dtype, length]:
    """Gather an existing record field; callers supply valid record IDs."""
    comptime assert Layout.rank == 2 and Layout.flat_rank == 2
    comptime if length == 1:
        return SIMD[dtype, length](records[(Int(ids[0]), field)])
    else:
        if uniform:
            return SIMD[dtype, length](records[(Int(ids[0]), field)])
        var stride = Int(records.layout(Coord(1, 0)))
        var offset = Int(records.layout(Coord(0, field)))
        return records.ptr.unsafe_gather(ids.cast[.int]() * stride + offset)


@inline(.always)
def gather_record_vec3[
    Layout: TensorLayout,
    dtype: DType,
    index_dtype: DType,
    length: SIMDLength,
    //,
    frame: Frame = .WORLD,
    field: Int = 0,
](
    records: TileTensor[dtype, Layout, _],
    ids: SIMD[index_dtype, length],
    uniform: Bool = False,
) -> Vec3[dtype, frame, length]:
    return Vec3[dtype, frame, length](
        gather_record_field[field](records, ids, uniform),
        gather_record_field[field + 1](records, ids, uniform),
        gather_record_field[field + 2](records, ids, uniform),
    )
