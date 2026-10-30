"""Device-safe access to GPU material textures and shading normals."""

from bajo.rt.texture_math import (
    sample_image_texture,
    albedo_from_texture,
    roughness_from_texture,
    bump_slope,
    bump_normal,
)
from std.memory import bitcast
from bajo.core import Vec3f32
from bajo.core.tensor_layout import record_tile
from bajo.core.color import Color
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
    var descs = record_tile[.uint32, GpuTextureDescLayout.STRIDE](
        texture_descs, Int(texture_idx) + 1
    )
    var pixel_offset = Int(
        descs[Int(texture_idx), GpuTextureDescLayout.PIXEL_OFFSET]
    )
    var width = Int(descs[Int(texture_idx), GpuTextureDescLayout.WIDTH])
    var height = Int(descs[Int(texture_idx), GpuTextureDescLayout.HEIGHT])
    return sample_image_texture(
        record_tile[.float32, 3](
            texture_pixels.unsafe_offset(pixel_offset), width * height
        ),
        width,
        height,
        u,
        v,
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
    var albedo = Color(lambertians.unsafe_offset(material_base))
    var texture_idx = texture_indices[unsafe_offset=material_idx]
    if texture_idx == NO_TEXTURE:
        return albedo
    return albedo_from_texture(
        albedo,
        _sample_gpu_image_texture(
            texture_idx, u, v, texture_descs, texture_pixels
        ),
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
    return albedo_from_texture(
        albedo,
        _sample_gpu_image_texture(
            texture_idx, u, v, texture_descs, texture_pixels
        ),
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
    var records = record_tile[.float32, GpuCoatedDiffuseLayout.STRIDE](
        coated_diffuses, material_idx + 1
    )
    var textures = record_tile[.uint32, GpuCoatedDiffuseLayout.TEXTURE_STRIDE](
        texture_indices, material_idx + 1
    )
    var texture_idx = textures[
        material_idx, GpuCoatedDiffuseLayout.DISPLACEMENT_TEXTURE
    ]
    var scale = records[material_idx, GpuCoatedDiffuseLayout.DISPLACEMENT_SCALE]
    if texture_idx == NO_TEXTURE or scale == 0.0:
        return normal
    var u_scale = records[
        material_idx, GpuCoatedDiffuseLayout.DISPLACEMENT_U_SCALE
    ]
    var v_scale = records[
        material_idx, GpuCoatedDiffuseLayout.DISPLACEMENT_V_SCALE
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
    var descs = record_tile[.uint32, GpuTextureDescLayout.STRIDE](
        texture_descs, Int(texture_idx) + 1
    )
    var width = Float32(descs[Int(texture_idx), GpuTextureDescLayout.WIDTH])
    var height = Float32(descs[Int(texture_idx), GpuTextureDescLayout.HEIGHT])
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
    var slope_u = bump_slope(color_u0, color_u1, width, u_scale, scale)
    var slope_v = bump_slope(color_v0, color_v1, height, v_scale, scale)
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
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var records = record_tile[.float32, GpuDielectricLayout.STRIDE](
        data, material_idx + 1
    )
    return _gpu_bump_normal(
        normal,
        u,
        v,
        bitcast[.uint32](
            records[material_idx, GpuDielectricLayout.DISPLACEMENT_TEXTURE]
        ),
        records[material_idx, GpuDielectricLayout.DISPLACEMENT_SCALE],
        records[material_idx, GpuDielectricLayout.DISPLACEMENT_U_SCALE],
        records[material_idx, GpuDielectricLayout.DISPLACEMENT_V_SCALE],
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
    var textures = record_tile[.uint32, GpuCoatedDiffuseLayout.TEXTURE_STRIDE](
        texture_indices, material_idx + 1
    )
    var texture_idx = textures[
        material_idx, GpuCoatedDiffuseLayout.ROUGHNESS_TEXTURE
    ]
    if texture_idx != NO_TEXTURE:
        var records = record_tile[.float32, GpuCoatedDiffuseLayout.STRIDE](
            data, material_idx + 1
        )
        var color = _sample_gpu_image_texture(
            texture_idx,
            u * records[material_idx, GpuCoatedDiffuseLayout.ROUGHNESS_U_SCALE],
            v * records[material_idx, GpuCoatedDiffuseLayout.ROUGHNESS_V_SCALE],
            texture_descs,
            texture_pixels,
        )
        value.conductor_roughness = roughness_from_texture(
            color,
            records[material_idx, GpuCoatedDiffuseLayout.ROUGHNESS_SCALE],
            records[material_idx, GpuCoatedDiffuseLayout.REMAP_ROUGHNESS].ne(
                0.0
            ),
        )
    return value
