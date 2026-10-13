"""Packed material layouts shared by host upload and device shading.

Offsets are in Float32 elements. Texture-table offsets are in UInt32 elements.
"""

from bajo.rt.material_parameters import CoatedParameters
from bajo.rt.tensor_layout import rt_record_tile


from bajo.rt.material_storage import (
    RgbMaterialLayout as GpuRgbMaterialLayout,
    MetalLayout as GpuMetalLayout,
    CoatedDiffuseLayout as GpuCoatedDiffuseLayout,
    DielectricLayout as GpuDielectricLayout,
    TextureDescLayout as GpuTextureDescLayout,
    load_coated_parameters,
)


comptime _GpuCoatedDiffuseParameters = CoatedParameters[1]


@always_inline
def _load_gpu_coated_parameters(
    data: Pointer[Float32, ImmutAnyOrigin], material_idx: Int
) -> _GpuCoatedDiffuseParameters:
    var records = rt_record_tile[.float32, GpuCoatedDiffuseLayout.STRIDE](
        data, material_idx + 1
    )
    return load_coated_parameters(records, UInt32(material_idx))
