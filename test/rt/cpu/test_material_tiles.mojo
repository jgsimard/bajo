"""Material decoding is independent of CPU/GPU physical layout."""

from std.testing import TestSuite, assert_equal
from layout import TileTensor, Coord, Idx
from layout.tile_layout import row_major
from bajo.rt.material_types import SurfaceStore
from bajo.rt.render_types import Color
from bajo.rt.material_storage import (
    CoatedDiffuseLayout,
    pack_coated_materials,
    coated_column_tile,
    load_coated_parameters,
)
from bajo.rt.tensor_layout import rt_record_tile


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
    var gpu = rt_record_tile[.float32, CoatedDiffuseLayout.STRIDE](
        rows.unsafe_ptr(), 5
    )
    var cpu = coated_column_tile(columns.unsafe_ptr(), 5)
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
    var gpu = rt_record_tile[.float32, CoatedDiffuseLayout.STRIDE](
        rows.unsafe_ptr(), 5
    )
    var cpu = coated_column_tile(columns.unsafe_ptr(), 5)
    var ids = SIMD[.uint32, length](0)
    for lane in range(Int(length)):
        ids[lane] = UInt32((lane * 3 + 2) % 5)
    var vector = load_coated_parameters(cpu, ids)
    # A fully static layout exercises the general coordinate-map decoder,
    # independently of the two optimized preparation layouts.
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
