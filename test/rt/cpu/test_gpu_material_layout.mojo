"""CPU/GPU material layouts retain field offsets and decoder semantics."""

from layout import TileTensor, Coord, Idx
from layout.tile_layout import row_major
from bajo.rt.material_types import SurfaceStore
from bajo.rt.material_storage import (
    CoatedDiffuseLayout,
    pack_coated_materials,
    load_coated_parameters,
)
from bajo.core.tensor_layout import record_tile

from std.memory import bitcast
from std.testing import TestSuite, assert_equal

from bajo.core import Point3f32
from bajo.rt.material_types import ImageTexture, NO_TEXTURE, SurfaceId
from bajo.rt.render_types import Color
from bajo.rt.scene_data import SceneBuilder, SceneData
from bajo.rt.gpu.material_layout import _load_gpu_coated_parameters
from bajo.rt.gpu.materials import (
    _flatten_coated_diffuse_texture_indices,
    _flatten_metals,
    _flatten_texture_descs,
    _flatten_texture_pixels,
)
from bajo.rt.gpu.textures import _sample_gpu_coated_albedo


def _world() raises -> SceneData:
    var builder = SceneBuilder()
    _ = builder.surfaces.add_image_texture(
        ImageTexture(1, 1, [0.5, 0.25, 0.75])
    )
    _ = builder.surfaces.add_image_texture(ImageTexture(1, 1, [0.25, 0.5, 1.0]))
    _ = builder.surfaces.add_coated_diffuse(
        Color(0.5, 0.75, 0.25),
        0.125,
        1.5,
        texture_index=0,
        displacement_texture_index=1,
        displacement_scale=0.25,
        displacement_u_scale=2.0,
        displacement_v_scale=3.0,
        thickness=0.0625,
        layer_albedo=Color(0.125, 0.25, 0.5),
        g=-0.25,
        max_depth=7,
        n_samples=2,
    )
    _ = builder.surfaces.add_coated_diffuse(
        Color(0.25, 0.5, 0.75),
        0.25,
        1.25,
        texture_index=1,
        displacement_scale=0.5,
        displacement_u_scale=4.0,
        displacement_v_scale=5.0,
        thickness=0.125,
        layer_albedo=Color(0.5, 0.25, 0.125),
        g=0.5,
        max_depth=9,
        n_samples=3,
    )
    var surface = builder.add_metal(Color(0.25, 0.5, 0.75), 0.125)
    _ = builder.add_metal(Color(0.75, 0.5, 0.25), 0.5)
    builder.add_sphere(Point3f32[.WORLD](0.0, 0.0, 0.0), 1.0, surface)
    return builder^.finish()


def test_material_upload_keeps_packed_abi() raises:
    var world = _world()
    var coated = pack_coated_materials[False](world.surfaces())
    # Fixed records intentionally do not use layout constants: changing the
    # schema must not silently change the established upload representation.
    var expected: List[Float32] = [
        0.5,
        0.75,
        0.25,
        0.125,
        1.5,
        0.0625,
        0.125,
        0.25,
        0.5,
        -0.25,
        7.0,
        2.0,
        0.25,
        2.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.25,
        0.5,
        0.75,
        0.25,
        1.25,
        0.125,
        0.5,
        0.25,
        0.125,
        0.5,
        9.0,
        3.0,
        0.5,
        4.0,
        5.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert_equal(len(coated), len(expected))
    for index in range(len(expected)):
        assert_equal(
            bitcast[.uint32](coated[index]), bitcast[.uint32](expected[index])
        )

    var metals = _flatten_metals(world)
    expected = [0.25, 0.5, 0.75, 0.125, 0.75, 0.5, 0.25, 0.5]
    assert_equal(len(metals), len(expected))
    for index in range(len(expected)):
        assert_equal(
            bitcast[.uint32](metals[index]), bitcast[.uint32](expected[index])
        )

    var parameters = _load_gpu_coated_parameters(
        coated.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        1,
    )
    assert_equal(parameters.roughness, Float32(0.25))
    assert_equal(parameters.eta, Float32(1.25))
    assert_equal(parameters.thickness, Float32(0.125))
    assert_equal(parameters.layer_albedo.x, Float32(0.5))
    assert_equal(parameters.layer_albedo.y, Float32(0.25))
    assert_equal(parameters.layer_albedo.z, Float32(0.125))
    assert_equal(parameters.g, Float32(0.5))
    assert_equal(parameters.max_depth, Float32(9.0))
    assert_equal(parameters.n_samples, Float32(3.0))


def test_material_texture_tables_address_second_record() raises:
    var world = _world()
    var coated = pack_coated_materials[False](world.surfaces())
    var indices = _flatten_coated_diffuse_texture_indices(world)
    var descs = _flatten_texture_descs(world)
    var pixels = _flatten_texture_pixels(world)
    var expected_indices: List[UInt32] = [
        0,
        1,
        NO_TEXTURE,
        1,
        NO_TEXTURE,
        NO_TEXTURE,
    ]
    assert_equal(len(indices), len(expected_indices))
    for index in range(len(expected_indices)):
        assert_equal(indices[index], expected_indices[index])
    var expected_descs: List[UInt32] = [0, 1, 1, 3, 1, 1]
    assert_equal(len(descs), len(expected_descs))
    for index in range(len(expected_descs)):
        assert_equal(descs[index], expected_descs[index])
    var color = _sample_gpu_coated_albedo(
        SurfaceId.pack_raw(.COATED_DIFFUSE, 1),
        0.0,
        0.0,
        coated.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        indices.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        descs.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        pixels.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
    )
    assert_equal(color.x, Float32(0.0625))
    assert_equal(color.y, Float32(0.25))
    assert_equal(color.z, Float32(0.75))


def test_conductor_upload_preserves_optics_and_substrate_roughness() raises:
    var builder = SceneBuilder()
    var surface = builder.surfaces.add_coated_conductor(
        Color(1.2, 0.9, 0.6), Color(7.2, 6.6, 5.4), 0.2, 0.1, 1.5
    )
    builder.add_sphere(Point3f32[.WORLD](0.0), 1.0, surface)
    var world = builder^.finish()
    var records = pack_coated_materials[False](world.surfaces())
    var parameters = _load_gpu_coated_parameters(
        records.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        0,
    )
    assert_equal(parameters.conductor_eta.x[0], Float32(1.2) / Float32(1.5))
    assert_equal(parameters.conductor_k.z[0], Float32(5.4) / Float32(1.5))
    assert_equal(parameters.conductor_roughness, Float32(0.2))
    assert_equal(parameters.roughness, Float32(0.1))


def _surfaces() -> SurfaceStore:
    var surfaces = SurfaceStore()
    for idx in range(5):
        var f = Float32(idx + 1)
        _ = surfaces.add_coated_diffuse(
            Color(0.1 * f),
            0.1 * f,
            1.0 + 0.1 * f,
            thickness=0.01 * f,
            layer_albedo=Color(0.02 * f, 0.03 * f, 0.04 * f),
            g=0.1 * f,
            max_depth=idx + 2,
            n_samples=idx + 1,
        )
        surfaces.coated_diffuses[idx].conductor_eta = Color(f, f + 1.0, f + 2.0)
        surfaces.coated_diffuses[idx].conductor_k = Color(
            f + 3.0, f + 4.0, f + 5.0
        )
        surfaces.coated_diffuses[idx].conductor_roughness = 0.05 * f
    return surfaces^


def test_coated_material_tiles_keep_the_same_logical_fields() raises:
    var surfaces = _surfaces()
    var rows = pack_coated_materials[False](surfaces)
    var columns = pack_coated_materials[True](surfaces)
    var gpu = record_tile[.float32, CoatedDiffuseLayout.STRIDE](
        rows.unsafe_ptr(), 5
    )
    var cpu = record_tile[.float32, CoatedDiffuseLayout.STRIDE, True](
        columns.unsafe_ptr(), 5
    )
    for material in range(5):
        for field in range(CoatedDiffuseLayout.STRIDE):
            assert_equal(cpu[(material, field)], gpu[(material, field)])
    # Distinct physical layouts, with independently specified addresses.
    assert_equal(rows[26 + 4], surfaces.coated_diffuses[1].eta)
    assert_equal(columns[4 * 5 + 1], surfaces.coated_diffuses[1].eta)


def _gathers[length: SIMDLength]() raises:
    var surfaces = _surfaces()
    var rows = pack_coated_materials[False](surfaces)
    var columns = pack_coated_materials[True](surfaces)
    var gpu = record_tile[.float32, CoatedDiffuseLayout.STRIDE](
        rows.unsafe_ptr(), 5
    )
    var cpu = record_tile[.float32, CoatedDiffuseLayout.STRIDE, True](
        columns.unsafe_ptr(), 5
    )
    var ids = SIMD[.uint32, length](0)
    for lane in range(Int(length)):
        ids[lane] = UInt32((lane * 3 + 2) % 5)
    var vector = load_coated_parameters(cpu, ids)
    # Static extents use the same affine field decoder as dynamic row and
    # column views; they must retain the same logical record coordinates.
    var static_records = TileTensor(
        Span(rows), row_major(Coord(Idx[5], Idx[26]))
    )
    var general = load_coated_parameters(static_records, ids)
    var uniform = load_coated_parameters(cpu, SIMD[.uint32, length](3), True)
    for lane in range(Int(length)):
        var scalar = load_coated_parameters(gpu, ids[lane])
        assert_equal(general.eta[lane], scalar.eta)
        assert_equal(general.layer_albedo.z[lane], scalar.layer_albedo.z)
        assert_equal(general.conductor_k.z[lane], scalar.conductor_k.z)
        assert_equal(vector.roughness[lane], scalar.roughness)
        assert_equal(vector.eta[lane], scalar.eta)
        assert_equal(vector.thickness[lane], scalar.thickness)
        assert_equal(vector.layer_albedo.x[lane], scalar.layer_albedo.x)
        assert_equal(vector.layer_albedo.y[lane], scalar.layer_albedo.y)
        assert_equal(vector.layer_albedo.z[lane], scalar.layer_albedo.z)
        assert_equal(vector.g[lane], scalar.g)
        assert_equal(vector.max_depth[lane], scalar.max_depth)
        assert_equal(vector.n_samples[lane], scalar.n_samples)
        assert_equal(vector.conductor_eta.x[lane], scalar.conductor_eta.x)
        assert_equal(vector.conductor_eta.y[lane], scalar.conductor_eta.y)
        assert_equal(vector.conductor_eta.z[lane], scalar.conductor_eta.z)
        assert_equal(vector.conductor_k.x[lane], scalar.conductor_k.x)
        assert_equal(vector.conductor_k.y[lane], scalar.conductor_k.y)
        assert_equal(vector.conductor_k.z[lane], scalar.conductor_k.z)
        assert_equal(
            vector.conductor_roughness[lane], scalar.conductor_roughness
        )
        assert_equal(uniform.eta[lane], surfaces.coated_diffuses[3].eta)


def test_coated_decoder_matches_scalar_records_and_uniform_broadcasts() raises:
    _gathers[1]()
    _gathers[4]()
    _gathers[8]()
    _gathers[16]()
    _gathers[32]()


def test_empty_coated_tables_are_valid() raises:
    var surfaces = SurfaceStore()
    assert_equal(len(pack_coated_materials[False](surfaces)), 0)
    assert_equal(len(pack_coated_materials[True](surfaces)), 0)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
