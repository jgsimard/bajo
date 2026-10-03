"""Prepared GPU material and texture table ownership."""

from max.gpu.host import DeviceBuffer, DeviceContext
from bajo.rt.material_types import NO_TEXTURE
from bajo.rt.scene_data import SceneData
from bajo.rt.gpu.upload import _upload_nonempty


def _flatten_lambertians(world: SceneData) -> List[Float32]:
    var out = List[Float32](capacity=len(world.surfaces().lambertians) * 3)
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
    var out = List[UInt32](capacity=len(world.surfaces().image_textures) * 3)
    var pixel_offset = UInt32(0)
    for texture_idx in range(len(world.surfaces().image_textures)):
        ref texture = world.surfaces().image_textures[texture_idx]
        out.append(pixel_offset)
        out.append(UInt32(texture.width))
        out.append(UInt32(texture.height))
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
    var out = List[Float32](capacity=len(world.surfaces().metals) * 4)
    for material in world.surfaces().metals:
        out.append(material.albedo.x)
        out.append(material.albedo.y)
        out.append(material.albedo.z)
        out.append(material.fuzz)
    return out^


def _flatten_dielectrics(world: SceneData) -> List[Float32]:
    var out = List[Float32](capacity=len(world.surfaces().dielectrics))
    for material in world.surfaces().dielectrics:
        out.append(material.refraction_index)
    return out^


def _flatten_emissives(world: SceneData) -> List[Float32]:
    var out = List[Float32](capacity=len(world.surfaces().emissives) * 3)
    for material in world.surfaces().emissives:
        out.append(material.radiance.x)
        out.append(material.radiance.y)
        out.append(material.radiance.z)
    return out^


def _flatten_coated_diffuses(world: SceneData) -> List[Float32]:
    var out = List[Float32](capacity=len(world.surfaces().coated_diffuses) * 15)
    for material in world.surfaces().coated_diffuses:
        out.append(material.albedo.x)
        out.append(material.albedo.y)
        out.append(material.albedo.z)
        out.append(material.roughness)
        out.append(material.eta)
        out.append(material.thickness)
        out.append(material.layer_albedo.x)
        out.append(material.layer_albedo.y)
        out.append(material.layer_albedo.z)
        out.append(material.g)
        out.append(Float32(material.max_depth))
        out.append(Float32(material.n_samples))
        out.append(material.displacement_scale)
        out.append(material.displacement_u_scale)
        out.append(material.displacement_v_scale)
    return out^


def _flatten_coated_diffuse_texture_indices(
    world: SceneData,
) -> List[UInt32]:
    var out = List[UInt32](capacity=len(world.surfaces().coated_diffuses) * 2)
    for material in world.surfaces().coated_diffuses:
        out.append(material.texture_index)
        out.append(material.displacement_texture_index)
    return out^


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
    var has_non_lambertian: Bool

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
        self.has_non_lambertian = (
            len(world.surfaces().metals) > 0
            or len(world.surfaces().dielectrics) > 0
        )
