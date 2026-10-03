"""Exercise the shared device traversal on host memory with known geometry."""

from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_almost_equal

from bajo.bvh.gpu.compressed_bounds_bvh import (
    CWBVH_NODE_WORDS,
    CWBVH_EXP_IMASK,
    CWBVH_META_BASE,
    CWBVH_QUANTIZED_BASE,
)
from bajo.bvh.gpu.tlas_diagnostics import _trace_cwbvh8_with_stats_in_frame
from bajo.bvh.gpu.triangle_trace import trace_cwbvh8_triangles
from bajo.core import Point3f32, Rayf32, Vec3f32


def test_cwbvh_shared_traversal_hit_miss_and_counters() raises:
    # One leaf in lane zero, with unit quantization and bounds [-1,1]^2 at z=1.
    var nodes = List[Float32](length=CWBVH_NODE_WORDS, fill=0.0)
    nodes[0] = -1.0
    nodes[1] = -1.0
    nodes[CWBVH_EXP_IMASK] = bitcast[.float32](UInt32(0x007F7F7F))
    nodes[CWBVH_META_BASE] = bitcast[.float32](UInt32(0x20))
    nodes[CWBVH_QUANTIZED_BASE + 4] = bitcast[.float32](UInt32(1))
    nodes[CWBVH_QUANTIZED_BASE + 6] = bitcast[.float32](UInt32(2))
    nodes[CWBVH_QUANTIZED_BASE + 8] = bitcast[.float32](UInt32(2))
    nodes[CWBVH_QUANTIZED_BASE + 10] = bitcast[.float32](UInt32(1))
    # SIMD records guarantee the alignment required by triangle loads.
    var records = List[SIMD[.float32, 4]]()
    records.append(SIMD[.float32, 4](2.0, 0.0, 0.0, 0.0))
    records.append(SIMD[.float32, 4](0.0, 2.0, 0.0, 0.0))
    records.append(
        SIMD[.float32, 4](-1.0, -1.0, 1.0, bitcast[.float32](UInt32(7)))
    )
    var triangles = records.unsafe_ptr().unsafe_bitcast[Float32]()
    for ray_kind in range(4):
        var x = Float32(-0.25)
        var y = Float32(-0.25)
        var t_max = Float32(10.0)
        if ray_kind == 1:
            x = 3.0
            y = 3.0
        elif ray_kind == 2:
            x = 0.5
            y = 0.5
        elif ray_kind == 3:
            t_max = 0.5
        var ray = Rayf32[.WORLD](
            Point3f32[.WORLD](x, y, 0.0),
            Vec3f32[.WORLD](0.0, 0.0, 1.0),
            0.0,
            t_max,
        )
        var measured = _trace_cwbvh8_with_stats_in_frame[.WORLD](
            nodes.unsafe_ptr(), triangles, UInt32(0), ray
        )
        var closest = trace_cwbvh8_triangles[.WORLD, .CLOSEST_HIT](
            nodes.unsafe_ptr(), triangles, UInt32(0), ray
        )
        var shadow = trace_cwbvh8_triangles[.WORLD, .ANY_HIT](
            nodes.unsafe_ptr(), triangles, UInt32(0), ray
        )
        assert_equal(measured.hit.is_hit(), ray_kind == 0)
        assert_equal(closest.is_hit(), ray_kind == 0)
        assert_equal(shadow.is_occluded(), ray_kind == 0)
        assert_equal(measured.stats.node_visits, UInt32(1))
        assert_equal(measured.stats.max_stack_depth, UInt32(0))
        var expected_tests = UInt32(
            1
        ) if ray_kind == 0 or ray_kind == 2 else UInt32(0)
        assert_equal(measured.stats.leaf_blocks, expected_tests)
        assert_equal(measured.stats.primitive_tests, expected_tests)
        if ray_kind == 0:
            assert_equal(measured.hit.prim, UInt32(7))
            assert_equal(closest.prim, UInt32(7))
            assert_almost_equal(measured.hit.t, 1.0)
            assert_almost_equal(closest.t, 1.0)


def test_cwbvh_shared_traversal_pending_node_stack() raises:
    # The root has two intersected internal children, forcing one pending task.
    var nodes = List[Float32](length=3 * CWBVH_NODE_WORDS, fill=0.0)
    for node_idx in range(3):
        var base = node_idx * CWBVH_NODE_WORDS
        nodes[base] = -1.0
        nodes[base + 1] = -1.0
        nodes[base + CWBVH_EXP_IMASK] = bitcast[.float32](UInt32(0x007F7F7F))
        nodes[base + CWBVH_META_BASE] = bitcast[.float32](UInt32(0x20))
        var z = UInt32(1) if node_idx < 2 else UInt32(2)
        nodes[base + CWBVH_QUANTIZED_BASE + 4] = bitcast[.float32](z)
        nodes[base + CWBVH_QUANTIZED_BASE + 6] = bitcast[.float32](UInt32(2))
        nodes[base + CWBVH_QUANTIZED_BASE + 8] = bitcast[.float32](UInt32(2))
        nodes[base + CWBVH_QUANTIZED_BASE + 10] = bitcast[.float32](z)
    nodes[CWBVH_EXP_IMASK] = bitcast[.float32](UInt32(0x037F7F7F))
    nodes[4] = bitcast[.float32](UInt32(1))  # internal child base
    nodes[CWBVH_META_BASE] = bitcast[.float32](UInt32(0x3938))
    nodes[CWBVH_QUANTIZED_BASE + 4] = bitcast[.float32](UInt32(0x0101))
    nodes[CWBVH_QUANTIZED_BASE + 6] = bitcast[.float32](UInt32(0x0202))
    nodes[CWBVH_QUANTIZED_BASE + 8] = bitcast[.float32](UInt32(0x0202))
    nodes[CWBVH_QUANTIZED_BASE + 10] = bitcast[.float32](UInt32(0x0202))
    nodes[2 * CWBVH_NODE_WORDS + 5] = bitcast[.float32](UInt32(1))
    var records = List[SIMD[.float32, 4]]()
    for triangle_idx in range(2):
        records.append(SIMD[.float32, 4](2.0, 0.0, 0.0, 0.0))
        records.append(SIMD[.float32, 4](0.0, 2.0, 0.0, 0.0))
        records.append(
            SIMD[.float32, 4](
                -1.0,
                -1.0,
                Float32(triangle_idx + 1),
                bitcast[.float32](UInt32(triangle_idx + 7)),
            )
        )
    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](-0.25, -0.25, 0.0),
        Vec3f32[.WORLD](0.0, 0.0, 1.0),
    )
    var triangles = records.unsafe_ptr().unsafe_bitcast[Float32]()
    var result = _trace_cwbvh8_with_stats_in_frame[.WORLD](
        nodes.unsafe_ptr(), triangles, UInt32(0), ray
    )
    assert_equal(result.hit.prim, UInt32(7))
    assert_almost_equal(result.hit.t, 1.0)
    assert_equal(result.stats.node_visits, UInt32(3))
    assert_equal(result.stats.leaf_blocks, UInt32(1))
    assert_equal(result.stats.primitive_tests, UInt32(1))
    assert_equal(result.stats.max_stack_depth, UInt32(1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
