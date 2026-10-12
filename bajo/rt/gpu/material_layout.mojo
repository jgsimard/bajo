"""Packed material layouts shared by host upload and device shading.

Offsets are in Float32 elements. Texture-table offsets are in UInt32 elements.
"""

from bajo.rt.render_types import Color
from bajo.rt.gpu.queue_layout import gpu_rt_record_tile


struct GpuRgbMaterialLayout:
    comptime STRIDE = 3
    comptime R = 0
    comptime G = 1
    comptime B = 2


struct GpuMetalLayout:
    comptime STRIDE = 4
    comptime ALBEDO = 0
    comptime FUZZ = 3


struct GpuCoatedDiffuseLayout:
    comptime STRIDE = 26
    comptime ALBEDO = 0
    comptime ROUGHNESS = 3
    comptime ETA = 4
    comptime THICKNESS = 5
    comptime LAYER_ALBEDO = 6
    comptime G = 9
    comptime MAX_DEPTH = 10
    comptime N_SAMPLES = 11
    comptime DISPLACEMENT_SCALE = 12
    comptime DISPLACEMENT_U_SCALE = 13
    comptime DISPLACEMENT_V_SCALE = 14
    comptime CONDUCTOR_ETA = 15
    comptime CONDUCTOR_K = 18
    comptime CONDUCTOR_ROUGHNESS = 21
    comptime ROUGHNESS_SCALE = 22
    comptime ROUGHNESS_U_SCALE = 23
    comptime ROUGHNESS_V_SCALE = 24
    comptime REMAP_ROUGHNESS = 25

    comptime TEXTURE_STRIDE = 3
    comptime ALBEDO_TEXTURE = 0
    comptime DISPLACEMENT_TEXTURE = 1
    comptime ROUGHNESS_TEXTURE = 2


struct GpuDielectricLayout:
    comptime STRIDE = 5
    comptime ETA = 0
    comptime DISPLACEMENT_SCALE = 1
    comptime DISPLACEMENT_U_SCALE = 2
    comptime DISPLACEMENT_V_SCALE = 3
    comptime DISPLACEMENT_TEXTURE = 4


struct GpuTextureDescLayout:
    comptime STRIDE = 3
    comptime PIXEL_OFFSET = 0
    comptime WIDTH = 1
    comptime HEIGHT = 2


@fieldwise_init
struct _GpuCoatedDiffuseParameters(TrivialRegisterPassable):
    """BSDF fields only; albedo and displacement use separate texture paths."""

    var roughness: Float32
    var eta: Float32
    var thickness: Float32
    var layer_albedo: Color
    var g: Float32
    var max_depth: Float32
    var n_samples: Float32
    var conductor_eta: Color
    var conductor_k: Color
    var conductor_roughness: Float32


@always_inline
def _load_gpu_coated_parameters(
    data: Pointer[Float32, ImmutAnyOrigin], material_idx: Int
) -> _GpuCoatedDiffuseParameters:
    var base = material_idx * GpuCoatedDiffuseLayout.STRIDE
    var records = gpu_rt_record_tile[GpuCoatedDiffuseLayout.STRIDE](
        data, material_idx + 1
    )
    return _GpuCoatedDiffuseParameters(
        records[(material_idx, GpuCoatedDiffuseLayout.ROUGHNESS)],
        records[(material_idx, GpuCoatedDiffuseLayout.ETA)],
        records[(material_idx, GpuCoatedDiffuseLayout.THICKNESS)],
        Color(data.unsafe_offset(base + GpuCoatedDiffuseLayout.LAYER_ALBEDO)),
        records[(material_idx, GpuCoatedDiffuseLayout.G)],
        records[(material_idx, GpuCoatedDiffuseLayout.MAX_DEPTH)],
        records[(material_idx, GpuCoatedDiffuseLayout.N_SAMPLES)],
        Color(data.unsafe_offset(base + GpuCoatedDiffuseLayout.CONDUCTOR_ETA)),
        Color(data.unsafe_offset(base + GpuCoatedDiffuseLayout.CONDUCTOR_K)),
        records[(material_idx, GpuCoatedDiffuseLayout.CONDUCTOR_ROUGHNESS)],
    )
