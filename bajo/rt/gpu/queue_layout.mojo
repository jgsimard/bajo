"""Typed TileTensor views and plane contracts for GPU render stages.

Views borrow existing storage. Their layouts preserve the field-major queue
ABI and record-major material ABI; constructing a view never copies records.
"""

from layout import TileTensor, Coord, Idx, row_major
from bajo.rt.wavefront_contract import (
    WavePathFloatAbi,
    WaveShadeFloatAbi,
    WaveShadowFloatAbi,
    WaveSampleFloatAbi,
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


comptime GpuRtPathTile[origin: Origin] = TileTensor[
    .float32,
    type_of(row_major(Coord(Idx[WavePathFloatAbi.PLANES], Int(0)))),
    origin,
]
comptime GpuRtShadeTile[origin: Origin] = TileTensor[
    .float32, type_of(row_major(Coord(Idx[GPU_SHADE_PLANES], Int(0)))), origin
]
comptime GpuRtShadowTile[origin: Origin] = TileTensor[
    .float32,
    type_of(row_major(Coord(Idx[WaveShadowFloatAbi.PLANES], Int(0)))),
    origin,
]
comptime GpuRtRadianceTile[origin: Origin] = TileTensor[
    .float32,
    type_of(row_major(Coord(Idx[WaveSampleFloatAbi.PLANES], Int(0)))),
    origin,
]


def gpu_rt_path_tile[
    mut: Bool, origin: Origin[mut=mut], //
](fields: Pointer[Float32, origin], capacity: Int) -> GpuRtPathTile[origin]:
    return TileTensor(
        ptr=fields,
        layout=row_major(Coord(Idx[WavePathFloatAbi.PLANES], capacity)),
    )


def gpu_rt_shade_tile[
    mut: Bool, origin: Origin[mut=mut], //
](fields: Pointer[Float32, origin], capacity: Int) -> GpuRtShadeTile[origin]:
    return TileTensor(
        ptr=fields, layout=row_major(Coord(Idx[GPU_SHADE_PLANES], capacity))
    )


def gpu_rt_shadow_tile[
    mut: Bool, origin: Origin[mut=mut], //
](fields: Pointer[Float32, origin], capacity: Int) -> GpuRtShadowTile[origin]:
    return TileTensor(
        ptr=fields,
        layout=row_major(Coord(Idx[WaveShadowFloatAbi.PLANES], capacity)),
    )


def gpu_rt_radiance_tile[
    mut: Bool, origin: Origin[mut=mut], //
](fields: Pointer[Float32, origin], capacity: Int) -> GpuRtRadianceTile[origin]:
    return TileTensor(
        ptr=fields,
        layout=row_major(Coord(Idx[WaveSampleFloatAbi.PLANES], capacity)),
    )
