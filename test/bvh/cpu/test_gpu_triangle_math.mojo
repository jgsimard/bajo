"""Exercise GPU triangle traversal math on the host without a device launch."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from std.memory import bitcast
from std.math import ceil, clamp, floor, fma
from bajo.bvh.constants import EMPTY_LANE, f32_max
from bajo.bvh.wide_meta import _pack_wide_meta
from bajo.bvh.gpu.compressed_bounds_bvh import (
    _encode_cwbvh8_node,
    _scale_from_exponent,
    _quantize_lower,
    _quantize_upper,
)
from bajo.bvh.gpu.triangle_trace import (
    _intersect_triangle_leaf,
    trace_cwbvh8_triangles,
    trace_cwbvh8_indexed_triangles,
)
from bajo.bvh.types import Hit
from bajo.core import Point3W, Rayf32, Vec3W


def test_gpu_triangle_nearest_hit_normal_and_occlusion() raises:
    for reverse in [False, True]:
        var vertices = List[Float32]()
        var packed = List[Float32](length=36, fill=0.0)
        var wide_leaf = List[Float32](length=48, fill=0.0)
        var wide = List[Float32](length=56, fill=0.0)
        var indices: List[UInt32] = [0, 1, 2, EMPTY_LANE]
        for lane in range(8):
            for field in range(3):
                wide[field * 8 + lane] = -1.0 if lane == 0 else f32_max
                wide[(field + 3) * 8 + lane] = 5.0 if lane == 0 else -f32_max
            wide[6 * 8 + lane] = bitcast[.float32](
                _pack_wide_meta(0, 3) if lane == 0 else EMPTY_LANE
            )
        for triangle in range(3):
            var depth = Float32(1 << (2 - triangle if reverse else triangle))
            var triangle_vertices: List[Float32] = [
                -1.0,
                -1.0,
                depth - 0.5,
                2.0,
                -1.0,
                depth - 0.5,
                -1.0,
                2.0,
                depth + 1.0,
            ]
            for value in triangle_vertices:
                vertices.append(value)
            var record: List[Float32] = [
                3.0,
                0.0,
                0.0,
                0.0,
                0.0,
                3.0,
                1.5,
                0.0,
                -1.0,
                -1.0,
                depth - 0.5,
                bitcast[.float32](UInt32(triangle)),
            ]
            for field in range(12):
                packed[triangle * 12 + field] = record[field]
            # Ordinary wide layout is v0/primitive/e1/padding/e2/padding.
            for field in range(3):
                wide_leaf[field * 4 + triangle] = record[8 + field]
                wide_leaf[(4 + field) * 4 + triangle] = record[field]
                wide_leaf[(8 + field) * 4 + triangle] = record[4 + field]
            wide_leaf[3 * 4 + triangle] = record[11]
        wide_leaf[3 * 4 + 3] = bitcast[.float32](EMPTY_LANE)
        var nodes = List[Float32](length=20, fill=0.0)
        var ids = List[UInt32](length=3, fill=EMPTY_LANE)
        var counter = List[UInt32](length=1, fill=0)
        _encode_cwbvh8_node[4](
            wide.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            indices.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            nodes.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            ids.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            counter.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            0,
        )
        var ray = Rayf32[.WORLD](Point3W(0.0, 0.0, 0.0), Vec3W(0.0, 0.0, 1.0))
        var ordinary = Hit[.WORLD].miss(ray.t_max)
        assert_true(
            _intersect_triangle_leaf[.WORLD, 4, .CLOSEST_HIT](
                wide_leaf.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                0,
                3,
                ray,
                ordinary,
            )
        )
        var compressed = trace_cwbvh8_triangles[.WORLD, .CLOSEST_HIT](
            nodes.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            packed.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            0,
            ray,
        )
        var indexed = trace_cwbvh8_indexed_triangles[.WORLD, .CLOSEST_HIT](
            nodes.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            ids.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            vertices.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            0,
            ray,
        )
        for hit in [ordinary, compressed, indexed]:
            assert_equal(hit.prim, UInt32(2 if reverse else 0))
            assert_almost_equal(hit.t, 1.0)
            assert_almost_equal(hit.normal.x, 0.0)
            assert_almost_equal(hit.normal.y, -0.4472136)
            assert_almost_equal(hit.normal.z, 0.8944272)
        assert_true(
            trace_cwbvh8_triangles[.WORLD, .ANY_HIT](
                nodes.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                packed.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                0,
                ray,
            ).is_occluded()
        )
        assert_true(
            trace_cwbvh8_indexed_triangles[.WORLD, .ANY_HIT](
                nodes.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                ids.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                vertices.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                0,
                ray,
            ).is_occluded()
        )
        var miss = Rayf32[.WORLD](
            Point3W(100.0, 100.0, 0.0), Vec3W(0.0, 0.0, 1.0)
        )
        assert_true(
            not trace_cwbvh8_triangles[.WORLD, .CLOSEST_HIT](
                nodes.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                packed.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                0,
                miss,
            ).is_hit()
        )


def test_cwbvh_reciprocal_quantization_preserves_conservative_bounds() raises:
    # Include the smallest normal scale and reciprocals in the subnormal range.
    for exponent in range(1, 255):
        var scale = _scale_from_exponent(UInt32(exponent))
        var inverse = Float32(1) / scale
        for fraction in [Float32(0.125), Float32(0.5), Float32(1.0)]:
            var value = fraction * scale
            var bits = bitcast[.uint32](value)
            for delta in [-1, 0, 1]:
                var edge = bitcast[.float32](UInt32(Int(bits) + delta))
                var lower = _quantize_lower[True](edge, 0, scale, inverse)
                var upper = _quantize_upper[True](edge, 0, scale, inverse)
                assert_equal(lower, _quantize_lower(edge, 0, scale))
                assert_equal(upper, _quantize_upper(edge, 0, scale))
                assert_true(fma(Float32(lower), scale, Float32(0)) <= edge)
                # A decoded upper bound may overflow to infinity at large scales;
                # it must still contain the finite source value.
                assert_true(fma(Float32(upper), scale, Float32(0)) >= edge)


def test_cwbvh_quantization_matches_integer_reference_at_byte_boundaries() raises:
    # Exercise every byte boundary, adjacent floats, translations, and both
    # saturation limits against the previous 64-bit integer implementation.
    for exponent in [1, 32, 64, 96, 127, 160, 192, 224, 246]:
        var scale = _scale_from_exponent(UInt32(exponent))
        var inverse = Float32(1) / scale
        for translation in [Float32(-8), Float32(0), Float32(8)]:
            var start = translation * scale
            for boundary in range(-1, 257):
                var value = fma(Float32(boundary), scale, start)
                var bits = bitcast[.uint32](value)
                for delta in [-1, 0, 1]:
                    # Skip signed-zero bit patterns that would become NaNs.
                    if (bits & UInt32(0x7FFFFFFF)) == 0 and delta < 0:
                        continue
                    var edge = bitcast[.float32](UInt32(Int(bits) + delta))
                    for reciprocal in [False, True]:
                        var scaled = (
                            (edge - start)
                            * inverse if reciprocal else (edge - start)
                            / scale
                        )
                        var lower = clamp(Int(floor(scaled)), 0, 255)
                        var upper = clamp(Int(ceil(scaled)), 0, 255)
                        if (
                            fma(Float32(lower), scale, start) > edge
                            and lower > 0
                        ):
                            lower -= 1
                        if (
                            fma(Float32(upper), scale, start) < edge
                            and upper < 255
                        ):
                            upper += 1
                        if reciprocal:
                            assert_equal(
                                _quantize_lower[True](
                                    edge, start, scale, inverse
                                ),
                                UInt32(lower),
                            )
                            assert_equal(
                                _quantize_upper[True](
                                    edge, start, scale, inverse
                                ),
                                UInt32(upper),
                            )
                        else:
                            assert_equal(
                                _quantize_lower(edge, start, scale),
                                UInt32(lower),
                            )
                            assert_equal(
                                _quantize_upper(edge, start, scale),
                                UInt32(upper),
                            )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
