"""Device-safe access to GPU material textures and shading normals."""

from std.math import abs, floor
from bajo.core import Vec3f32, cross, normalize
from bajo.rt.render_types import Color
from bajo.rt.material_types import NO_TEXTURE, SurfaceId
from bajo.rt.gpu.material_layout import (
    GpuRgbMaterialLayout,
    GpuCoatedDiffuseLayout,
    GpuTextureDescLayout,
    _load_gpu_material_color,
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
    var wrapped_u = u - floor(u)
    var wrapped_v = v - floor(v)
    var x = min(Int(wrapped_u * Float32(width)), width - 1)
    var y = min(Int((Float32(1.0) - wrapped_v) * Float32(height)), height - 1)
    var base = pixel_offset + 3 * (y * width + x)
    return Color(
        texture_pixels[unsafe_offset=base],
        texture_pixels[unsafe_offset=base + 1],
        texture_pixels[unsafe_offset=base + 2],
    )


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
    var albedo = _load_gpu_material_color(lambertians, material_base)
    var texture_idx = texture_indices[unsafe_offset=material_idx]
    if texture_idx == NO_TEXTURE:
        return albedo
    return albedo * _sample_gpu_image_texture(
        texture_idx, u, v, texture_descs, texture_pixels
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
    var albedo = _load_gpu_material_color(
        coated_diffuses, material_base + GpuCoatedDiffuseLayout.ALBEDO
    )
    var texture_idx = texture_indices[
        unsafe_offset=GpuCoatedDiffuseLayout.TEXTURE_STRIDE * material_idx
        + GpuCoatedDiffuseLayout.ALBEDO_TEXTURE
    ]
    if texture_idx == NO_TEXTURE:
        return albedo
    return albedo * _sample_gpu_image_texture(
        texture_idx, u, v, texture_descs, texture_pixels
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
    var value_u0 = (
        Float32(0.2126) * color_u0.x
        + Float32(0.7152) * color_u0.y
        + Float32(0.0722) * color_u0.z
    )
    var value_u1 = (
        Float32(0.2126) * color_u1.x
        + Float32(0.7152) * color_u1.y
        + Float32(0.0722) * color_u1.z
    )
    var value_v0 = (
        Float32(0.2126) * color_v0.x
        + Float32(0.7152) * color_v0.y
        + Float32(0.0722) * color_v0.z
    )
    var value_v1 = (
        Float32(0.2126) * color_v1.x
        + Float32(0.7152) * color_v1.y
        + Float32(0.0722) * color_v1.z
    )
    var slope_u = (value_u1 - value_u0) * width * u_scale * scale
    var slope_v = (value_v1 - value_v0) * height * v_scale * scale
    var helper = Vec3f32[.WORLD](0.0, 1.0, 0.0)
    if abs(normal.y) > 0.99:
        helper = Vec3f32[.WORLD](1.0, 0.0, 0.0)
    var tangent = normalize(cross(helper, normal))
    var bitangent = cross(normal, tangent)
    return normalize(normal - tangent * slope_u - bitangent * slope_v)
