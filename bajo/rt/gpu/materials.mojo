"""Prepared GPU material and texture table ownership."""

from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from bajo.rt.material_storage import pack_coated_materials
from bajo.rt.material_types import NO_TEXTURE
from bajo.rt.scene_data import SceneData
from bajo.rt.gpu.upload import _upload_nonempty
from bajo.rt.gpu.material_layout import (
    GpuRgbMaterialLayout,
    GpuDielectricLayout,
    GpuMetalLayout,
    GpuCoatedDiffuseLayout,
    GpuTextureDescLayout,
)


def _flatten_lambertians(world: SceneData) -> List[Float32]:
    var out = List[Float32](
        capacity=len(world.surfaces().lambertians) * GpuRgbMaterialLayout.STRIDE
    )
    for material in world.surfaces().lambertians:
        out.append(material.albedo.x)
        out.append(material.albedo.y)
        out.append(material.albedo.z)
    return out^


def _flatten_lambertian_texture_indices(world: SceneData) -> List[UInt32]:
    var out = List[UInt32](capacity=len(world.surfaces().lambertians))
    for material in world.surfaces().lambertians:
        out.append(material.texture_index)
    return out^


def _flatten_texture_descs(world: SceneData) -> List[UInt32]:
    var out = List[UInt32](
        length=len(world.surfaces().image_textures)
        * GpuTextureDescLayout.STRIDE,
        fill=0,
    )
    var pixel_offset = UInt32(0)
    for texture_idx in range(len(world.surfaces().image_textures)):
        ref texture = world.surfaces().image_textures[texture_idx]
        var base = texture_idx * GpuTextureDescLayout.STRIDE
        out[base + GpuTextureDescLayout.PIXEL_OFFSET] = pixel_offset
        out[base + GpuTextureDescLayout.WIDTH] = UInt32(texture.width)
        out[base + GpuTextureDescLayout.HEIGHT] = UInt32(texture.height)
        pixel_offset += UInt32(len(texture.pixels))
    return out^


def _flatten_texture_pixels(world: SceneData) -> List[Float32]:
    var count = 0
    for texture_idx in range(len(world.surfaces().image_textures)):
        count += len(world.surfaces().image_textures[texture_idx].pixels)
    var out = List[Float32](capacity=count)
    for texture_idx in range(len(world.surfaces().image_textures)):
        ref texture = world.surfaces().image_textures[texture_idx]
        for value in texture.pixels:
            out.append(value)
    return out^


def _flatten_metals(world: SceneData) -> List[Float32]:
    var out = List[Float32](
        length=len(world.surfaces().metals) * GpuMetalLayout.STRIDE, fill=0.0
    )
    for material_idx in range(len(world.surfaces().metals)):
        ref material = world.surfaces().metals[material_idx]
        var base = material_idx * GpuMetalLayout.STRIDE
        out[
            base + GpuMetalLayout.ALBEDO + GpuRgbMaterialLayout.R
        ] = material.albedo.x
        out[
            base + GpuMetalLayout.ALBEDO + GpuRgbMaterialLayout.G
        ] = material.albedo.y
        out[
            base + GpuMetalLayout.ALBEDO + GpuRgbMaterialLayout.B
        ] = material.albedo.z
        out[base + GpuMetalLayout.FUZZ] = material.fuzz
    return out^


def _flatten_dielectrics(world: SceneData) -> List[Float32]:
    var out = List[Float32](capacity=len(world.surfaces().dielectrics))
    for material in world.surfaces().dielectrics:
        out.append(material.refraction_index)
        out.append(material.displacement_scale)
        out.append(material.displacement_u_scale)
        out.append(material.displacement_v_scale)
        out.append(bitcast[.float32](material.displacement_texture_index))
    return out^


def _flatten_emissives(world: SceneData) -> List[Float32]:
    var out = List[Float32](
        capacity=len(world.surfaces().emissives) * GpuRgbMaterialLayout.STRIDE
    )
    for material in world.surfaces().emissives:
        out.append(material.radiance.x)
        out.append(material.radiance.y)
        out.append(material.radiance.z)
    return out^


def _flatten_coated_diffuses(world: SceneData) -> List[Float32]:
    return pack_coated_materials[False](world.surfaces())


def _flatten_coated_diffuse_texture_indices(world: SceneData) -> List[UInt32]:
    var out = List[UInt32](
        length=len(world.surfaces().coated_diffuses)
        * GpuCoatedDiffuseLayout.TEXTURE_STRIDE,
        fill=NO_TEXTURE,
    )
    for material_idx in range(len(world.surfaces().coated_diffuses)):
        ref material = world.surfaces().coated_diffuses[material_idx]
        var base = material_idx * GpuCoatedDiffuseLayout.TEXTURE_STRIDE
        out[
            base + GpuCoatedDiffuseLayout.ALBEDO_TEXTURE
        ] = material.texture_index
        out[
            base + GpuCoatedDiffuseLayout.DISPLACEMENT_TEXTURE
        ] = material.displacement_texture_index
        out[
            base + GpuCoatedDiffuseLayout.ROUGHNESS_TEXTURE
        ] = material.roughness_texture_index
    return out^


@fieldwise_init
struct GpuRtShadeStages(ImplicitlyCopyable):
    """Runtime launch requirements for a fixed set of shading kernels."""

    var metal_dielectric: Bool
    var coated_diffuse: Bool

    def needs_source_paths(self) -> Bool:
        return self.metal_dielectric or self.coated_diffuse


struct GpuRtMaterials:
    """Flattened device material tables shared by all GPU geometry backends."""

    var lambertians: DeviceBuffer[.float32]
    var metals: DeviceBuffer[.float32]
    var dielectrics: DeviceBuffer[.float32]
    var emissives: DeviceBuffer[.float32]
    var coated_diffuses: DeviceBuffer[.float32]
    var lambertian_texture_indices: DeviceBuffer[.uint32]
    var coated_diffuse_texture_indices: DeviceBuffer[.uint32]
    var texture_descs: DeviceBuffer[.uint32]
    var texture_pixels: DeviceBuffer[.float32]
    var mix_surfaces: DeviceBuffer[.uint32]
    var mix_amounts: DeviceBuffer[.float32]
    var mix_texture_indices: DeviceBuffer[.uint32]
    var mix_uv_scales: DeviceBuffer[.float32]
    var medium_absorption: DeviceBuffer[.float32]
    var sphere_media: DeviceBuffer[.uint32]
    var triangle_media: DeviceBuffer[.uint32]
    var instance_media: DeviceBuffer[.uint32]
    var sphere_media_count: Int32
    var triangle_media_count: Int32
    var instance_media_count: Int32
    var stages: GpuRtShadeStages

    def __init__(
        out self,
        mut ctx: DeviceContext,
        world: SceneData,
    ) raises:
        for material in world.surfaces().metals:
            if material.texture_index != NO_TEXTURE:
                raise Error("GPU metal image textures are not yet supported")
        self.lambertians = _upload_nonempty(ctx, _flatten_lambertians(world))
        self.metals = _upload_nonempty(ctx, _flatten_metals(world))
        self.dielectrics = _upload_nonempty(ctx, _flatten_dielectrics(world))
        self.emissives = _upload_nonempty(ctx, _flatten_emissives(world))
        self.coated_diffuses = _upload_nonempty(
            ctx, _flatten_coated_diffuses(world)
        )
        self.lambertian_texture_indices = _upload_nonempty(
            ctx, _flatten_lambertian_texture_indices(world)
        )
        self.coated_diffuse_texture_indices = _upload_nonempty(
            ctx, _flatten_coated_diffuse_texture_indices(world)
        )
        self.texture_descs = _upload_nonempty(
            ctx, _flatten_texture_descs(world)
        )
        self.texture_pixels = _upload_nonempty(
            ctx, _flatten_texture_pixels(world)
        )
        self.mix_surfaces = _upload_nonempty(ctx, world.surfaces().mix_surfaces)
        self.mix_amounts = _upload_nonempty(ctx, world.surfaces().mix_amounts)
        self.mix_texture_indices = _upload_nonempty(
            ctx, world.surfaces().mix_texture_indices
        )
        self.mix_uv_scales = _upload_nonempty(
            ctx, world.surfaces().mix_uv_scales
        )
        var absorption = List[Float32](
            capacity=3 * len(world.media().absorption)
        )
        for value in world.media().absorption:
            absorption.append(value.x)
            absorption.append(value.y)
            absorption.append(value.z)
        self.medium_absorption = _upload_nonempty(ctx, absorption)
        self.sphere_media = _upload_nonempty(ctx, world.media().spheres)
        self.triangle_media = _upload_nonempty(ctx, world.media().triangles)
        self.instance_media = _upload_nonempty(ctx, world.media().instances)
        self.sphere_media_count = Int32(len(world.media().spheres))
        self.triangle_media_count = Int32(len(world.media().triangles))
        self.instance_media_count = Int32(len(world.media().instances))
        self.stages = GpuRtShadeStages(
            len(world.surfaces().metals) > 0
            or len(world.surfaces().dielectrics) > 0,
            len(world.surfaces().coated_diffuses) > 0,
        )
