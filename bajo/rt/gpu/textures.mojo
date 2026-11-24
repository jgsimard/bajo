"""Device-safe access to GPU material textures and shading normals."""

from layout import TileTensor, Coord, Idx, row_major
from bajo.core.image import (
    sample_image_texture,
    sample_image_texture_bilinear,
    image_pixel_index,
)
from bajo.core.geometry import normal_from_slopes as bump_normal

from bajo.rt.texture_math import (
    mapped_normal,
    albedo_from_texture,
    roughness_from_texture,
    bump_slope,
)
from std.memory import bitcast
from std.math import floor
from bajo.core import Vec3f32
from bajo.core.color import Color
from bajo.core.optics import conductor_k_from_reflectance
from bajo.rt.material_types import NO_TEXTURE, SurfaceId
from bajo.rt.gpu.material_layout import (
    GpuRgbMaterialLayout,
    GpuDielectricLayout,
    _GpuCoatedDiffuseParameters,
    _load_gpu_coated_parameters,
    GpuCoatedDiffuseLayout,
    GpuTextureDescLayout,
)


@fieldwise_init
struct _GpuImageTextureView[origin: ImmOrigin](TrivialRegisterPassable):
    """Borrowed pixels and dimensions decoded from one texture descriptor."""

    var pixels: Pointer[Float32, Self.origin]
    var width: Int
    var height: Int
    var format: UInt32

    @staticmethod
    def load(
        texture_idx: UInt32,
        texture_descs: ImmPointer[UInt32, _],
        texture_pixels: Pointer[Float32, Self.origin],
    ) -> Self:
        var index = Int(texture_idx)
        var descs = TileTensor(
            ptr=texture_descs,
            layout=row_major(
                Coord(index + 1, Idx[GpuTextureDescLayout.STRIDE])
            ),
        )
        var pixel_offset = Int(descs[index, GpuTextureDescLayout.PIXEL_OFFSET])
        return {
            texture_pixels.unsafe_offset(pixel_offset),
            Int(descs[index, GpuTextureDescLayout.WIDTH]),
            Int(descs[index, GpuTextureDescLayout.HEIGHT]),
            descs[index, GpuTextureDescLayout.FORMAT],
        }

    def texel(self, index: Int) -> Color:
        if self.format == GpuTextureDescLayout.PALETTE_RGB8:
            var indices = self.pixels.unsafe_bitcast[UInt32]()[
                unsafe_offset=256 + index
            ]
            return Color(
                self.pixels[unsafe_offset=Int(indices & UInt32(255))],
                self.pixels[
                    unsafe_offset=Int((indices >> UInt32(8)) & UInt32(255))
                ],
                self.pixels[
                    unsafe_offset=Int((indices >> UInt32(16)) & UInt32(255))
                ],
            )
        var texels = TileTensor(
            ptr=self.pixels, layout=row_major(self.width * self.height, 3)
        )
        return Color(texels.ptr_at_offset((index, 0)))

    def sample(self, u: Float32, v: Float32) -> Color:
        if self.format == GpuTextureDescLayout.PALETTE_RGB8:
            return self.texel(
                Int(image_pixel_index(self.width, self.height, u, v))
            )
        return sample_image_texture(
            TileTensor(
                ptr=self.pixels,
                layout=row_major(Coord(self.width * self.height, Idx[3])),
            ),
            self.width,
            self.height,
            u,
            v,
        )

    def sample_bilinear(self, u: Float32, v: Float32) -> Color:
        if self.format == GpuTextureDescLayout.PALETTE_RGB8:
            var x = (u - floor(u)) * Float32(self.width) - 0.5
            var y = (1.0 - (v - floor(v))) * Float32(self.height) - 0.5
            var fx = floor(x)
            var fy = floor(y)
            var dx = x - fx
            var dy = y - fy
            var x0 = Int(fx)
            var y0 = Int(fy)
            var x1 = ((x0 + 1) % self.width + self.width) % self.width
            var y1 = ((y0 + 1) % self.height + self.height) % self.height
            x0 = (x0 % self.width + self.width) % self.width
            y0 = (y0 % self.height + self.height) % self.height
            var top = (
                self.texel(y0 * self.width + x0) * (1.0 - dx)
                + self.texel(y0 * self.width + x1) * dx
            )
            var bottom = (
                self.texel(y1 * self.width + x0) * (1.0 - dx)
                + self.texel(y1 * self.width + x1) * dx
            )
            return top * (1.0 - dy) + bottom * dy
        return sample_image_texture_bilinear(
            TileTensor(
                ptr=self.pixels,
                layout=row_major(Coord(self.width * self.height, Idx[3])),
            ),
            self.width,
            self.height,
            u,
            v,
        )

    def environment_texel(self, index: Int) -> Color:
        var value = self.texel(index)
        return Color(max(value.x, 0.0), max(value.y, 0.0), max(value.z, 0.0))

    def sample_environment(self, u: Float32, v: Float32) -> Color:
        var x = min(max(Int(u * Float32(self.width)), 0), self.width - 1)
        var y = min(max(Int(v * Float32(self.height)), 0), self.height - 1)
        return self.environment_texel(y * self.width + x)


def _gpu_normal_map(
    surface: UInt32,
    normal: Vec3f32[.WORLD],
    tangent: Vec3f32[.WORLD],
    u: Float32,
    v: Float32,
    front_face: Bool,
    maps: ImmPointer[UInt32, _],
    count: Int,
    texture_descs: ImmPointer[UInt32, _],
    texture_pixels: ImmPointer[Float32, _],
) -> Vec3f32[.WORLD]:
    var records = TileTensor(ptr=maps, layout=row_major(count, 2))
    var begin = 0
    var end = count
    while begin < end:
        var middle = (begin + end) // 2
        var key = records[middle, 0]
        if key < surface:
            begin = middle + 1
        else:
            end = middle
    if begin >= count or records[begin, 0] != surface:
        return normal
    var texture = _GpuImageTextureView.load(
        records[begin, 1], texture_descs, texture_pixels
    )
    var outward = normal if front_face else -normal
    var result = mapped_normal(outward, tangent, texture.sample_bilinear(u, v))
    return result if front_face else -result


def _sample_gpu_image_texture(
    texture_idx: UInt32,
    u: Float32,
    v: Float32,
    texture_descs: ImmPointer[UInt32, _],
    texture_pixels: ImmPointer[Float32, _],
) -> Color:
    return _GpuImageTextureView.load(
        texture_idx, texture_descs, texture_pixels
    ).sample(u, v)


def _sample_gpu_lambertian(
    surface_value: UInt32,
    u: Float32,
    v: Float32,
    lambertians: ImmPointer[Float32, _],
    texture_indices: ImmPointer[UInt32, _],
    texture_descs: ImmPointer[UInt32, _],
    texture_pixels: ImmPointer[Float32, _],
) -> Color:
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var records = TileTensor(
        ptr=lambertians,
        layout=row_major(material_idx + 1, GpuRgbMaterialLayout.STRIDE),
    )
    var albedo = Color(records.ptr_at_offset((material_idx, 0)))
    var texture_idx = texture_indices[unsafe_offset=material_idx]
    if texture_idx == NO_TEXTURE:
        return albedo
    return albedo_from_texture(
        albedo,
        _sample_gpu_image_texture(
            texture_idx, u, v, texture_descs, texture_pixels
        ),
    )


def _sample_gpu_coated_albedo(
    surface_value: UInt32,
    u: Float32,
    v: Float32,
    coated_diffuses: ImmPointer[Float32, _],
    texture_indices: ImmPointer[UInt32, _],
    texture_descs: ImmPointer[UInt32, _],
    texture_pixels: ImmPointer[Float32, _],
) -> Color:
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var records = TileTensor(
        ptr=coated_diffuses,
        layout=row_major(material_idx + 1, GpuCoatedDiffuseLayout.STRIDE),
    )
    var textures = TileTensor(
        ptr=texture_indices,
        layout=row_major(
            material_idx + 1, GpuCoatedDiffuseLayout.TEXTURE_STRIDE
        ),
    )
    var albedo = Color(
        records.ptr_at_offset((material_idx, GpuCoatedDiffuseLayout.ALBEDO))
    )
    var texture_idx = textures[
        material_idx, GpuCoatedDiffuseLayout.ALBEDO_TEXTURE
    ]
    if texture_idx == NO_TEXTURE:
        return albedo
    return albedo_from_texture(
        albedo,
        _sample_gpu_image_texture(
            texture_idx, u, v, texture_descs, texture_pixels
        ),
    )


def _gpu_coated_shading_normal(
    surface_value: UInt32,
    normal: Vec3f32[.WORLD],
    u: Float32,
    v: Float32,
    coated_diffuses: ImmPointer[Float32, _],
    texture_indices: ImmPointer[UInt32, _],
    texture_descs: ImmPointer[UInt32, _],
    texture_pixels: ImmPointer[Float32, _],
) -> Vec3f32[.WORLD]:
    if SurfaceId.kind_from_raw(surface_value) != .COATED_DIFFUSE:
        return normal
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var records = TileTensor(
        ptr=coated_diffuses,
        layout=row_major(
            Coord(material_idx + 1, Idx[GpuCoatedDiffuseLayout.STRIDE])
        ),
    )
    var textures = TileTensor(
        ptr=texture_indices,
        layout=row_major(
            Coord(material_idx + 1, Idx[GpuCoatedDiffuseLayout.TEXTURE_STRIDE])
        ),
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


def _gpu_bump_normal(
    normal: Vec3f32[.WORLD],
    u: Float32,
    v: Float32,
    texture_idx: UInt32,
    scale: Float32,
    u_scale: Float32,
    v_scale: Float32,
    texture_descs: ImmPointer[UInt32, _],
    texture_pixels: ImmPointer[Float32, _],
) -> Vec3f32[.WORLD]:
    if texture_idx == NO_TEXTURE or scale == 0.0:
        return normal
    var texture = _GpuImageTextureView.load(
        texture_idx, texture_descs, texture_pixels
    )
    var width = Float32(texture.width)
    var height = Float32(texture.height)
    var texture_u = u * u_scale
    var texture_v = v * v_scale
    var color_u0 = texture.sample(texture_u - Float32(0.5) / width, texture_v)
    var color_u1 = texture.sample(texture_u + Float32(0.5) / width, texture_v)
    var color_v0 = texture.sample(texture_u, texture_v - Float32(0.5) / height)
    var color_v1 = texture.sample(texture_u, texture_v + Float32(0.5) / height)
    var slope_u = bump_slope(color_u0, color_u1, width, u_scale, scale)
    var slope_v = bump_slope(color_v0, color_v1, height, v_scale, scale)
    return bump_normal(normal, slope_u, slope_v)


def _gpu_dielectric_shading_normal(
    surface_value: UInt32,
    normal: Vec3f32[.WORLD],
    u: Float32,
    v: Float32,
    data: ImmPointer[Float32, _],
    texture_descs: ImmPointer[UInt32, _],
    texture_pixels: ImmPointer[Float32, _],
) -> Vec3f32[.WORLD]:
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var records = TileTensor(
        ptr=data,
        layout=row_major(
            Coord(material_idx + 1, Idx[GpuDielectricLayout.STRIDE])
        ),
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


def _sample_gpu_coated_parameters(
    data: ImmPointer[Float32, _],
    material_idx: Int,
    u: Float32,
    v: Float32,
    texture_indices: ImmPointer[UInt32, _],
    texture_descs: ImmPointer[UInt32, _],
    texture_pixels: ImmPointer[Float32, _],
) -> _GpuCoatedDiffuseParameters:
    var value = _load_gpu_coated_parameters(data, material_idx)
    var textures = TileTensor(
        ptr=texture_indices,
        layout=row_major(
            Coord(material_idx + 1, Idx[GpuCoatedDiffuseLayout.TEXTURE_STRIDE])
        ),
    )
    var texture_idx = textures[
        material_idx, GpuCoatedDiffuseLayout.ROUGHNESS_TEXTURE
    ]
    if texture_idx != NO_TEXTURE:
        var records = TileTensor(
            ptr=data,
            layout=row_major(
                Coord(material_idx + 1, Idx[GpuCoatedDiffuseLayout.STRIDE])
            ),
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
    var records = TileTensor(
        ptr=data,
        layout=row_major(
            Coord(material_idx + 1, Idx[GpuCoatedDiffuseLayout.STRIDE])
        ),
    )
    if (
        records[material_idx, GpuCoatedDiffuseLayout.CONDUCTOR_REFLECTANCE]
        != 0.0
    ):
        var reflectance = _sample_gpu_coated_albedo(
            SurfaceId.pack_raw(.COATED_DIFFUSE, UInt32(material_idx)),
            u,
            v,
            data,
            texture_indices,
            texture_descs,
            texture_pixels,
        )
        value.conductor_eta = Color(1.0) / value.eta
        value.conductor_k = (
            conductor_k_from_reflectance(reflectance) / value.eta
        )
    return value
