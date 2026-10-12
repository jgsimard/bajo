"""Typed TileTensor views and plane contracts for GPU render stages.

Views borrow existing storage. Their layouts preserve the field-major queue
ABI and record-major material ABI; constructing a view never copies records.
"""
from layout import TileTensor, Coord, Idx
from layout.tile_layout import row_major
from bajo.rt.wavefront_contract import (
    WavePathFloatAbi,
    WaveShadeFloatAbi,
    WaveShadowFloatAbi,
)

# Deferred coatings retain hit UVs after traversal. The shared CPU ABI remains
# the four normal/distance planes; these extra planes belong to the GPU arena.
comptime GPU_SHADE_U = WaveShadeFloatAbi.PLANES
comptime GPU_SHADE_V = GPU_SHADE_U + 1
comptime GPU_SHADE_BOUNDARY = GPU_SHADE_V + 1
comptime GPU_SHADE_GNX = GPU_SHADE_BOUNDARY + 1
comptime GPU_SHADE_GNY = GPU_SHADE_GNX + 1
comptime GPU_SHADE_GNZ = GPU_SHADE_GNY + 1
comptime GPU_SHADE_TX = GPU_SHADE_GNZ + 1
comptime GPU_SHADE_TY = GPU_SHADE_TX + 1
comptime GPU_SHADE_TZ = GPU_SHADE_TY + 1
comptime GPU_SHADE_PLANES = GPU_SHADE_TZ + 1


comptime GpuRtFieldLayout[planes: Int] = type_of(
    row_major(Coord(Idx[planes], Int(0)))
)
comptime GpuRtFieldTile[planes: Int, origin: Origin] = TileTensor[
    .float32, GpuRtFieldLayout[planes], origin
]
comptime GpuRtPathTile[origin: Origin] = GpuRtFieldTile[
    WavePathFloatAbi.PLANES, origin
]
comptime GpuRtShadeTile[origin: Origin] = GpuRtFieldTile[
    GPU_SHADE_PLANES, origin
]
comptime GpuRtShadowTile[origin: Origin] = GpuRtFieldTile[
    WaveShadowFloatAbi.PLANES, origin
]


@always_inline
def gpu_rt_field_tile[
    mut: Bool, origin: Origin[mut=mut], //, planes: Int
](fields: Pointer[Float32, origin], capacity: Int) -> GpuRtFieldTile[
    planes, origin
]:
    return TileTensor(
        Span(unsafe_ptr=fields, length=planes * capacity),
        row_major(Coord(Idx[planes], capacity)),
    )


@always_inline
def gpu_rt_path_tile[
    mut: Bool, origin: Origin[mut=mut], //
](fields: Pointer[Float32, origin], capacity: Int) -> GpuRtPathTile[origin]:
    return gpu_rt_field_tile[WavePathFloatAbi.PLANES](fields, capacity)


@always_inline
def gpu_rt_shade_tile[
    mut: Bool, origin: Origin[mut=mut], //
](fields: Pointer[Float32, origin], capacity: Int) -> GpuRtShadeTile[origin]:
    return gpu_rt_field_tile[GPU_SHADE_PLANES](fields, capacity)


@always_inline
def gpu_rt_shadow_tile[
    mut: Bool, origin: Origin[mut=mut], //
](fields: Pointer[Float32, origin], capacity: Int) -> GpuRtShadowTile[origin]:
    return gpu_rt_field_tile[WaveShadowFloatAbi.PLANES](fields, capacity)


comptime GpuRtRecordLayout[stride: Int] = type_of(
    row_major(Coord(Int(0), Idx[stride]))
)
comptime GpuRtRecordTile[stride: Int, origin: Origin] = TileTensor[
    .float32, GpuRtRecordLayout[stride], origin
]


@always_inline
def gpu_rt_record_tile[
    mut: Bool, origin: Origin[mut=mut], //, stride: Int
](records: Pointer[Float32, origin], count: Int) -> GpuRtRecordTile[
    stride, origin
]:
    return TileTensor(
        Span(unsafe_ptr=records, length=stride * count),
        row_major(Coord(count, Idx[stride])),
    )
