"""Material record schemas shared by CPU preparation and GPU upload.

Logical coordinates are [material, field]. GPU records keep fields adjacent;
CPU tables keep materials adjacent so SIMD lanes can gather one field.
"""

from layout import TileTensor, TensorLayout
from bajo.core.tensor_layout import (
    record_tile,
    gather_record_field,
    gather_record_vec3,
)
from bajo.rt.material_parameters import CoatedParameters
from bajo.rt.material_types import SurfaceStore, CoatedDiffuse


struct RgbMaterialLayout:
    comptime STRIDE = 3
    comptime R = 0
    comptime G = 1
    comptime B = 2


struct MetalLayout:
    comptime STRIDE = 4
    comptime ALBEDO = 0
    comptime FUZZ = 3


struct CoatedDiffuseLayout:
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


struct DielectricLayout:
    comptime STRIDE = 5
    comptime ETA = 0
    comptime DISPLACEMENT_SCALE = 1
    comptime DISPLACEMENT_U_SCALE = 2
    comptime DISPLACEMENT_V_SCALE = 3
    comptime DISPLACEMENT_TEXTURE = 4


struct TextureDescLayout:
    comptime STRIDE = 3
    comptime PIXEL_OFFSET = 0
    comptime WIDTH = 1
    comptime HEIGHT = 2


@always_inline
def load_coated_parameters[
    mut: Bool,
    origin: Origin[mut=mut],
    Layout: TensorLayout,
    //,
    length: SIMDLength,
](
    records: TileTensor[.float32, Layout, origin],
    ids: SIMD[.uint32, length],
    uniform: Bool = False,
) -> CoatedParameters[length]:
    """Decode the same fields from scalar records or SIMD material columns.

    Callers replace inactive IDs with a valid ID before gathering. Texture
    sampling stays in each backend and replaces conductor roughness afterwards.
    """
    return CoatedParameters[length](
        gather_record_field[CoatedDiffuseLayout.ROUGHNESS](
            records, ids, uniform
        ),
        gather_record_field[CoatedDiffuseLayout.ETA](records, ids, uniform),
        gather_record_field[CoatedDiffuseLayout.THICKNESS](
            records, ids, uniform
        ),
        gather_record_vec3[.WORLD, CoatedDiffuseLayout.LAYER_ALBEDO](
            records, ids, uniform
        ),
        gather_record_field[CoatedDiffuseLayout.G](records, ids, uniform),
        gather_record_field[CoatedDiffuseLayout.MAX_DEPTH](
            records, ids, uniform
        ),
        gather_record_field[CoatedDiffuseLayout.N_SAMPLES](
            records, ids, uniform
        ),
        gather_record_vec3[.WORLD, CoatedDiffuseLayout.CONDUCTOR_ETA](
            records, ids, uniform
        ),
        gather_record_vec3[.WORLD, CoatedDiffuseLayout.CONDUCTOR_K](
            records, ids, uniform
        ),
        gather_record_field[CoatedDiffuseLayout.CONDUCTOR_ROUGHNESS](
            records, ids, uniform
        ),
    )


def store_coated_material[
    origin: Origin[mut=True], Layout: TensorLayout
](
    records: TileTensor[.float32, Layout, origin],
    idx: Int,
    material: CoatedDiffuse,
):
    records[(idx, CoatedDiffuseLayout.ALBEDO)] = material.albedo.x
    records[(idx, CoatedDiffuseLayout.ALBEDO + 1)] = material.albedo.y
    records[(idx, CoatedDiffuseLayout.ALBEDO + 2)] = material.albedo.z
    records[(idx, CoatedDiffuseLayout.ROUGHNESS)] = material.roughness
    records[(idx, CoatedDiffuseLayout.ETA)] = material.eta
    records[(idx, CoatedDiffuseLayout.THICKNESS)] = material.thickness
    records[(idx, CoatedDiffuseLayout.LAYER_ALBEDO)] = material.layer_albedo.x
    records[
        (idx, CoatedDiffuseLayout.LAYER_ALBEDO + 1)
    ] = material.layer_albedo.y
    records[
        (idx, CoatedDiffuseLayout.LAYER_ALBEDO + 2)
    ] = material.layer_albedo.z
    records[(idx, CoatedDiffuseLayout.G)] = material.g
    records[(idx, CoatedDiffuseLayout.MAX_DEPTH)] = Float32(material.max_depth)
    records[(idx, CoatedDiffuseLayout.N_SAMPLES)] = Float32(material.n_samples)
    records[
        (idx, CoatedDiffuseLayout.DISPLACEMENT_SCALE)
    ] = material.displacement_scale
    records[
        (idx, CoatedDiffuseLayout.DISPLACEMENT_U_SCALE)
    ] = material.displacement_u_scale
    records[
        (idx, CoatedDiffuseLayout.DISPLACEMENT_V_SCALE)
    ] = material.displacement_v_scale
    records[(idx, CoatedDiffuseLayout.CONDUCTOR_ETA)] = material.conductor_eta.x
    records[
        (idx, CoatedDiffuseLayout.CONDUCTOR_ETA + 1)
    ] = material.conductor_eta.y
    records[
        (idx, CoatedDiffuseLayout.CONDUCTOR_ETA + 2)
    ] = material.conductor_eta.z
    records[(idx, CoatedDiffuseLayout.CONDUCTOR_K)] = material.conductor_k.x
    records[(idx, CoatedDiffuseLayout.CONDUCTOR_K + 1)] = material.conductor_k.y
    records[(idx, CoatedDiffuseLayout.CONDUCTOR_K + 2)] = material.conductor_k.z
    records[
        (idx, CoatedDiffuseLayout.CONDUCTOR_ROUGHNESS)
    ] = material.conductor_roughness
    records[
        (idx, CoatedDiffuseLayout.ROUGHNESS_SCALE)
    ] = material.roughness_scale
    records[
        (idx, CoatedDiffuseLayout.ROUGHNESS_U_SCALE)
    ] = material.roughness_u_scale
    records[
        (idx, CoatedDiffuseLayout.ROUGHNESS_V_SCALE)
    ] = material.roughness_v_scale
    records[(idx, CoatedDiffuseLayout.REMAP_ROUGHNESS)] = Float32(
        1.0 if material.remap_roughness else 0.0
    )


def pack_coated_materials[
    COLUMN_MAJOR: Bool
](surfaces: SurfaceStore) -> List[Float32]:
    var count = len(surfaces.coated_diffuses)
    var out = List[Float32](length=count * CoatedDiffuseLayout.STRIDE, fill=0.0)
    var records = record_tile[
        .float32, CoatedDiffuseLayout.STRIDE, COLUMN_MAJOR
    ](out.unsafe_ptr(), count)
    for idx in range(count):
        store_coated_material(records, idx, surfaces.coated_diffuses[idx])
    return out^
