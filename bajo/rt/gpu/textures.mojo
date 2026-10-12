"""Device-safe access to GPU material textures and shading normals."""

from std.math import sqrt
from bajo.rt.texture_math import (
    image_pixel_offset,
    texture_luminance,
    bump_normal,
)
from std.memory import bitcast
from bajo.core import Vec3f32
from bajo.rt.render_types import Color
from bajo.rt.material_types import NO_TEXTURE, SurfaceId
from bajo.rt.gpu.material_layout import (
    GpuRgbMaterialLayout,
    GpuDielectricLayout,
    _GpuCoatedDiffuseParameters,
    _load_gpu_coated_parameters,
    GpuCoatedDiffuseLayout,
    GpuTextureDescLayout,
)


@always_inline
def _sample_gpu_image_texture(
    texture_idx: UInt32,
    u: Float32,
    v: Float32,
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
) -> Color:
    var desc_base = GpuTextureDescLayout.STRIDE * Int(texture_idx)
    var pixel_offset = Int(
        texture_descs[
            unsafe_offset=desc_base + GpuTextureDescLayout.PIXEL_OFFSET
        ]
    )
    var width = Int(
        texture_descs[unsafe_offset=desc_base + GpuTextureDescLayout.WIDTH]
    )
    var height = Int(
        texture_descs[unsafe_offset=desc_base + GpuTextureDescLayout.HEIGHT]
    )
    var base = pixel_offset + image_pixel_offset(width, height, u, v)
    return Color(texture_pixels.unsafe_offset(base))


@always_inline
def _sample_gpu_lambertian(
    surface_value: UInt32,
    u: Float32,
    v: Float32,
    lambertians: Pointer[Float32, ImmutAnyOrigin],
    texture_indices: Pointer[UInt32, ImmutAnyOrigin],
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
) -> Color:
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var material_base = GpuRgbMaterialLayout.STRIDE * material_idx
    var albedo = Color(lambertians.unsafe_offset(material_base))
    var texture_idx = texture_indices[unsafe_offset=material_idx]
    if texture_idx == NO_TEXTURE:
        return albedo
    var value = albedo * _sample_gpu_image_texture(
        texture_idx, u, v, texture_descs, texture_pixels
    )
    return Color(
        value.x.clamp(0.0, 1.0),
        value.y.clamp(0.0, 1.0),
        value.z.clamp(0.0, 1.0),
    )


@always_inline
def _sample_gpu_coated_albedo(
    surface_value: UInt32,
    u: Float32,
    v: Float32,
    coated_diffuses: Pointer[Float32, ImmutAnyOrigin],
    texture_indices: Pointer[UInt32, ImmutAnyOrigin],
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
) -> Color:
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var material_base = GpuCoatedDiffuseLayout.STRIDE * material_idx
    var albedo = Color(
        coated_diffuses.unsafe_offset(
            material_base + GpuCoatedDiffuseLayout.ALBEDO
        )
    )
    var texture_idx = texture_indices[
        unsafe_offset=GpuCoatedDiffuseLayout.TEXTURE_STRIDE * material_idx
        + GpuCoatedDiffuseLayout.ALBEDO_TEXTURE
    ]
    if texture_idx == NO_TEXTURE:
        return albedo
    var value = albedo * _sample_gpu_image_texture(
        texture_idx, u, v, texture_descs, texture_pixels
    )
    return Color(
        value.x.clamp(0.0, 1.0),
        value.y.clamp(0.0, 1.0),
        value.z.clamp(0.0, 1.0),
    )


@always_inline
def _gpu_coated_shading_normal(
    surface_value: UInt32,
    normal: Vec3f32[.WORLD],
    u: Float32,
    v: Float32,
    coated_diffuses: Pointer[Float32, ImmutAnyOrigin],
    texture_indices: Pointer[UInt32, ImmutAnyOrigin],
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
) -> Vec3f32[.WORLD]:
    if SurfaceId.kind_from_raw(surface_value) != .COATED_DIFFUSE:
        return normal
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var material_base = GpuCoatedDiffuseLayout.STRIDE * material_idx
    var texture_idx = texture_indices[
        unsafe_offset=GpuCoatedDiffuseLayout.TEXTURE_STRIDE * material_idx
        + GpuCoatedDiffuseLayout.DISPLACEMENT_TEXTURE
    ]
    var scale = coated_diffuses[
        unsafe_offset=material_base + GpuCoatedDiffuseLayout.DISPLACEMENT_SCALE
    ]
    if texture_idx == NO_TEXTURE or scale == 0.0:
        return normal
    var u_scale = coated_diffuses[
        unsafe_offset=material_base
        + GpuCoatedDiffuseLayout.DISPLACEMENT_U_SCALE
    ]
    var v_scale = coated_diffuses[
        unsafe_offset=material_base
        + GpuCoatedDiffuseLayout.DISPLACEMENT_V_SCALE
    ]
    return _gpu_bump_normal(
        normal,
        u,
        v,
        texture_idx,
        scale,
        u_scale,
        v_scale,
        texture_descs,
        texture_pixels,
    )


@always_inline
def _gpu_bump_normal(
    normal: Vec3f32[.WORLD],
    u: Float32,
    v: Float32,
    texture_idx: UInt32,
    scale: Float32,
    u_scale: Float32,
    v_scale: Float32,
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
) -> Vec3f32[.WORLD]:
    if texture_idx == NO_TEXTURE or scale == 0.0:
        return normal
    var desc_base = GpuTextureDescLayout.STRIDE * Int(texture_idx)
    var width = Float32(
        texture_descs[unsafe_offset=desc_base + GpuTextureDescLayout.WIDTH]
    )
    var height = Float32(
        texture_descs[unsafe_offset=desc_base + GpuTextureDescLayout.HEIGHT]
    )
    var texture_u = u * u_scale
    var texture_v = v * v_scale
    var color_u0 = _sample_gpu_image_texture(
        texture_idx,
        texture_u - Float32(0.5) / width,
        texture_v,
        texture_descs,
        texture_pixels,
    )
    var color_u1 = _sample_gpu_image_texture(
        texture_idx,
        texture_u + Float32(0.5) / width,
        texture_v,
        texture_descs,
        texture_pixels,
    )
    var color_v0 = _sample_gpu_image_texture(
        texture_idx,
        texture_u,
        texture_v - Float32(0.5) / height,
        texture_descs,
        texture_pixels,
    )
    var color_v1 = _sample_gpu_image_texture(
        texture_idx,
        texture_u,
        texture_v + Float32(0.5) / height,
        texture_descs,
        texture_pixels,
    )
    var value_u0 = texture_luminance(color_u0)
    var value_u1 = texture_luminance(color_u1)
    var value_v0 = texture_luminance(color_v0)
    var value_v1 = texture_luminance(color_v1)
    var slope_u = (value_u1 - value_u0) * width * u_scale * scale
    var slope_v = (value_v1 - value_v0) * height * v_scale * scale
    return bump_normal(normal, slope_u, slope_v)


@always_inline
def _gpu_dielectric_shading_normal(
    surface_value: UInt32,
    normal: Vec3f32[.WORLD],
    u: Float32,
    v: Float32,
    data: Pointer[Float32, ImmutAnyOrigin],
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
) -> Vec3f32[.WORLD]:
    var base = GpuDielectricLayout.STRIDE * Int(
        SurfaceId.index_from_raw(surface_value)
    )
    return _gpu_bump_normal(
        normal,
        u,
        v,
        bitcast[.uint32](
            data[unsafe_offset=base + GpuDielectricLayout.DISPLACEMENT_TEXTURE]
        ),
        data[unsafe_offset=base + GpuDielectricLayout.DISPLACEMENT_SCALE],
        data[unsafe_offset=base + GpuDielectricLayout.DISPLACEMENT_U_SCALE],
        data[unsafe_offset=base + GpuDielectricLayout.DISPLACEMENT_V_SCALE],
        texture_descs,
        texture_pixels,
    )


@always_inline
def _sample_gpu_coated_parameters(
    data: Pointer[Float32, ImmutAnyOrigin],
    material_idx: Int,
    u: Float32,
    v: Float32,
    texture_indices: Pointer[UInt32, ImmutAnyOrigin],
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
) -> _GpuCoatedDiffuseParameters:
    var value = _load_gpu_coated_parameters(data, material_idx)
    var texture_idx = texture_indices[
        unsafe_offset=GpuCoatedDiffuseLayout.TEXTURE_STRIDE * material_idx
        + GpuCoatedDiffuseLayout.ROUGHNESS_TEXTURE
    ]
    if texture_idx != NO_TEXTURE:
        var base = GpuCoatedDiffuseLayout.STRIDE * material_idx
        var color = _sample_gpu_image_texture(
            texture_idx,
            u
            * data[
                unsafe_offset=base + GpuCoatedDiffuseLayout.ROUGHNESS_U_SCALE
            ],
            v
            * data[
                unsafe_offset=base + GpuCoatedDiffuseLayout.ROUGHNESS_V_SCALE
            ],
            texture_descs,
            texture_pixels,
        )
        var roughness = (
            (0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z)
            * data[unsafe_offset=base + GpuCoatedDiffuseLayout.ROUGHNESS_SCALE]
        ).clamp(0.0, 1.0)
        value.conductor_roughness = (
            sqrt(roughness) if data[
                unsafe_offset=base + GpuCoatedDiffuseLayout.REMAP_ROUGHNESS
            ]
            != 0.0 else roughness
        )
    return value
