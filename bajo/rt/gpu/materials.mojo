"""Prepared GPU material and texture table ownership."""

from max.gpu.host import DeviceBuffer, DeviceContext
from bajo.rt.material_types import NO_TEXTURE
from bajo.rt.scene_data import SceneData
from bajo.rt.gpu.upload import _upload_nonempty
from bajo.rt.gpu.material_layout import (
    GpuRgbMaterialLayout,
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
    var out = List[Float32](
        length=len(world.surfaces().coated_diffuses)
        * GpuCoatedDiffuseLayout.STRIDE,
        fill=0.0,
    )
    for material_idx in range(len(world.surfaces().coated_diffuses)):
        ref material = world.surfaces().coated_diffuses[material_idx]
        var base = material_idx * GpuCoatedDiffuseLayout.STRIDE
        out[
            base + GpuCoatedDiffuseLayout.ALBEDO + GpuRgbMaterialLayout.R
        ] = material.albedo.x
        out[
            base + GpuCoatedDiffuseLayout.ALBEDO + GpuRgbMaterialLayout.G
        ] = material.albedo.y
        out[
            base + GpuCoatedDiffuseLayout.ALBEDO + GpuRgbMaterialLayout.B
        ] = material.albedo.z
        out[base + GpuCoatedDiffuseLayout.ROUGHNESS] = material.roughness
        out[base + GpuCoatedDiffuseLayout.ETA] = material.eta
        out[base + GpuCoatedDiffuseLayout.THICKNESS] = material.thickness
        out[
            base + GpuCoatedDiffuseLayout.LAYER_ALBEDO + GpuRgbMaterialLayout.R
        ] = material.layer_albedo.x
        out[
            base + GpuCoatedDiffuseLayout.LAYER_ALBEDO + GpuRgbMaterialLayout.G
        ] = material.layer_albedo.y
        out[
            base + GpuCoatedDiffuseLayout.LAYER_ALBEDO + GpuRgbMaterialLayout.B
        ] = material.layer_albedo.z
        out[base + GpuCoatedDiffuseLayout.G] = material.g
        out[base + GpuCoatedDiffuseLayout.MAX_DEPTH] = Float32(
            material.max_depth
        )
        out[base + GpuCoatedDiffuseLayout.N_SAMPLES] = Float32(
            material.n_samples
        )
        out[
            base + GpuCoatedDiffuseLayout.DISPLACEMENT_SCALE
        ] = material.displacement_scale
        out[
            base + GpuCoatedDiffuseLayout.DISPLACEMENT_U_SCALE
        ] = material.displacement_u_scale
        out[
            base + GpuCoatedDiffuseLayout.DISPLACEMENT_V_SCALE
        ] = material.displacement_v_scale
    return out^


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
        self.stages = GpuRtShadeStages(
            len(world.surfaces().metals) > 0
            or len(world.surfaces().dielectrics) > 0,
            len(world.surfaces().coated_diffuses) > 0,
        )
