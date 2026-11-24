"""Packed material layouts shared by host upload and device shading.

Offsets are in Float32 elements. Texture-table offsets are in UInt32 elements.
"""

from layout import TileTensor, Coord, Idx, row_major
from bajo.rt.material_parameters import CoatedParameters


from bajo.rt.material_storage import (
    RgbMaterialLayout as GpuRgbMaterialLayout,
    MetalLayout as GpuMetalLayout,
    CoatedDiffuseLayout as GpuCoatedDiffuseLayout,
    DielectricLayout as GpuDielectricLayout,
    TextureDescLayout as GpuTextureDescLayout,
    load_coated_parameters,
)


comptime _GpuCoatedDiffuseParameters = CoatedParameters[1]


def _load_gpu_coated_parameters(
    data: ImmPointer[Float32, _], material_idx: Int
) -> _GpuCoatedDiffuseParameters:
    var records = TileTensor(
        ptr=data,
        layout=row_major(
            Coord(material_idx + 1, Idx[GpuCoatedDiffuseLayout.STRIDE])
        ),
    )
    return load_coated_parameters(records, UInt32(material_idx))
