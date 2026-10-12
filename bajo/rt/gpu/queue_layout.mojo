"""Typed TileTensor views and plane contracts for GPU render stages.

Views borrow existing storage. Their layouts preserve the field-major queue
ABI and record-major material ABI; constructing a view never copies records.
"""
from bajo.rt.tensor_layout import (
    RtFieldLayout,
    RtFieldTile,
    rt_field_tile,
    RtRecordLayout,
    RtRecordTile,
    rt_record_tile,
)
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


comptime GpuRtFieldLayout[planes: Int] = RtFieldLayout[planes, Int]
comptime GpuRtFieldTile[planes: Int, origin: Origin] = RtFieldTile[
    .float32, planes, Int, origin
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

comptime GpuRtRadianceTile[origin: Origin] = GpuRtFieldTile[
    WaveSampleFloatAbi.PLANES, origin
]


@always_inline
def gpu_rt_field_tile[
    mut: Bool, origin: Origin[mut=mut], //, planes: Int
](fields: Pointer[Float32, origin], capacity: Int) -> GpuRtFieldTile[
    planes, origin
]:
    return rt_field_tile[.float32, planes](fields, capacity)


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


@always_inline
def gpu_rt_radiance_tile[
    mut: Bool, origin: Origin[mut=mut], //
](fields: Pointer[Float32, origin], capacity: Int) -> GpuRtRadianceTile[origin]:
    return gpu_rt_field_tile[WaveSampleFloatAbi.PLANES](fields, capacity)


comptime GpuRtRecordLayout[stride: Int] = RtRecordLayout[stride]
comptime GpuRtRecordTile[stride: Int, origin: Origin] = RtRecordTile[
    .float32, stride, origin
]


@always_inline
def gpu_rt_record_tile[
    mut: Bool, origin: Origin[mut=mut], //, stride: Int
](records: Pointer[Float32, origin], count: Int) -> GpuRtRecordTile[
    stride, origin
]:
    return rt_record_tile[.float32, stride](records, count)
