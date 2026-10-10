"""Packed material layouts shared by host upload and device shading.

Offsets are in Float32 elements. Texture-table offsets are in UInt32 elements.
"""

from bajo.rt.render_types import Color


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
    comptime STRIDE = 15
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

    comptime TEXTURE_STRIDE = 2
    comptime ALBEDO_TEXTURE = 0
    comptime DISPLACEMENT_TEXTURE = 1


struct GpuTextureDescLayout:
    comptime STRIDE = 3
    comptime PIXEL_OFFSET = 0
    comptime WIDTH = 1
    comptime HEIGHT = 2


@always_inline
def _load_gpu_material_color(
    data: Pointer[Float32, ImmutAnyOrigin], base: Int
) -> Color:
    return Color(
        data[unsafe_offset=base + GpuRgbMaterialLayout.R],
        data[unsafe_offset=base + GpuRgbMaterialLayout.G],
        data[unsafe_offset=base + GpuRgbMaterialLayout.B],
    )


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


@always_inline
def _load_gpu_coated_parameters(
    data: Pointer[Float32, ImmutAnyOrigin], material_idx: Int
) -> _GpuCoatedDiffuseParameters:
    var base = material_idx * GpuCoatedDiffuseLayout.STRIDE
    return _GpuCoatedDiffuseParameters(
        data[unsafe_offset=base + GpuCoatedDiffuseLayout.ROUGHNESS],
        data[unsafe_offset=base + GpuCoatedDiffuseLayout.ETA],
        data[unsafe_offset=base + GpuCoatedDiffuseLayout.THICKNESS],
        _load_gpu_material_color(
            data, base + GpuCoatedDiffuseLayout.LAYER_ALBEDO
        ),
        data[unsafe_offset=base + GpuCoatedDiffuseLayout.G],
        data[unsafe_offset=base + GpuCoatedDiffuseLayout.MAX_DEPTH],
        data[unsafe_offset=base + GpuCoatedDiffuseLayout.N_SAMPLES],
    )
