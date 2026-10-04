"""Segment isolation across conservative and exact packed BLAS assembly."""

from std.testing import TestSuite, assert_equal

from bajo.bvh.constants import (
    PrimitiveKind,
    WideNode,
    CPU_TRI_LEAF_PACKED_STRIDE,
    SPHERE_LEAF_PACKED_STRIDE,
)
from bajo.bvh.cpu import (
    CpuBlasSet,
    build_cpu_triangle_blas_set,
    build_cpu_sphere_blas_set,
    trace_blas_set,
)
from bajo.bvh.types import BlasDesc, Sphere
from bajo.core import Point3f32, Rayf32, Vec3f32


def _assert_segments[
    kind: PrimitiveKind, leaf_width: SIMDLength
](blases: CpuBlasSet[kind, 4, leaf_width], large_count: Int,) raises:
    assert_equal(blases.blas_count, 4)
    var node_base = 0
    var leaf_base = 0
    comptime leaf_stride = (
        CPU_TRI_LEAF_PACKED_STRIDE if kind
        == .TRIANGLE else SPHERE_LEAF_PACKED_STRIDE
    )
    for idx in range(4):
        var desc = BlasDesc.load(blases.descs.unsafe_ptr(), UInt32(idx))
        assert_equal(Int(desc.node_f32_base), node_base)
        assert_equal(Int(desc.leaf_f32_base), leaf_base)
        var count = 1 if idx == 0 else large_count if idx == 2 else 0
        assert_equal(Int(desc.prim_count), count)
        if count == 0:
            assert_equal(desc.node_count, UInt32(0))
            assert_equal(desc.leaf_block_count, UInt32(0))
        node_base += Int(desc.node_count) * 4 * WideNode.CHILD_STRIDE
        leaf_base += Int(desc.leaf_block_count) * leaf_width * leaf_stride

    assert_equal(node_base, len(blases.nodes))
    assert_equal(leaf_base, len(blases.leaves))


def _mesh(count: Int, offset: Float32) -> List[Point3f32[.LOCAL]]:
    var vertices = List[Point3f32[.LOCAL]](capacity=count * 3)
    for idx in range(count):
        var x = offset + Float32(idx) * 2.0
        vertices.append(Point3f32[.LOCAL](x - 0.5, -0.5, 2.0))
        vertices.append(Point3f32[.LOCAL](x + 0.5, -0.5, 2.0))
        vertices.append(Point3f32[.LOCAL](x, 0.5, 2.0))
    return vertices^


def _test_triangles(count: Int) raises:
    var meshes: List[List[Point3f32[.LOCAL]]] = [
        _mesh(1, 0.0),
        List[Point3f32[.LOCAL]](),
        _mesh(count, 10000.0),
        List[Point3f32[.LOCAL]](),
    ]
    # Different node/leaf widths exercise both strides during relocation.
    var blases = build_cpu_triangle_blas_set[4, 2](meshes)

    _assert_segments(blases, count)
    for idx in range(4):
        for sample_idx in range(2):
            var x = Float32(sample_idx) * 10000.0
            var ray = Rayf32[.LOCAL](
                Point3f32[.LOCAL](x, 0.0, 0.0),
                Vec3f32[.LOCAL](0.0, 0.0, 1.0),
            )
            var expected = (idx == 0 and sample_idx == 0) or (
                idx == 2 and sample_idx == 1
            )
            var hit = trace_blas_set[4, 2, .CLOSEST_HIT, .LOCAL](
                blases, UInt32(idx), ray
            )
            assert_equal(Bool(hit.is_hit()), expected)
            assert_equal(
                Bool(
                    trace_blas_set[4, 2, .ANY_HIT, .LOCAL](
                        blases,
                        UInt32(idx),
                        ray,
                    ).is_occluded()
                ),
                expected,
            )
            if expected:
                assert_equal(hit.prim, UInt32(0))
                assert_equal(hit.t, Float32(2.0))


def _spheres(count: Int, offset: Float32) -> List[Sphere[.LOCAL]]:
    var spheres = List[Sphere[.LOCAL]](capacity=count)
    for idx in range(count):
        spheres.append(
            Sphere[.LOCAL](
                Point3f32[.LOCAL](offset + Float32(idx) * 2.0, 0.0, 2.0),
                0.5,
            )
        )
    return spheres^


def _test_spheres(count: Int) raises:
    var sets: List[List[Sphere[.LOCAL]]] = [
        _spheres(1, 0.0),
        List[Sphere[.LOCAL]](),
        _spheres(count, 10000.0),
        List[Sphere[.LOCAL]](),
    ]
    var blases = build_cpu_sphere_blas_set[4](sets)

    _assert_segments(blases, count)
    for idx in range(4):
        for sample_idx in range(2):
            var x = Float32(sample_idx) * 10000.0
            var ray = Rayf32[.LOCAL](
                Point3f32[.LOCAL](x, 0.0, 0.0),
                Vec3f32[.LOCAL](0.0, 0.0, 1.0),
            )
            var expected = (idx == 0 and sample_idx == 0) or (
                idx == 2 and sample_idx == 1
            )
            var hit = trace_blas_set[4, 4, .CLOSEST_HIT, .LOCAL](
                blases, UInt32(idx), ray
            )
            assert_equal(Bool(hit.is_hit()), expected)
            assert_equal(
                Bool(
                    trace_blas_set[4, 4, .ANY_HIT, .LOCAL](
                        blases,
                        UInt32(idx),
                        ray,
                    ).is_occluded()
                ),
                expected,
            )
            if expected:
                assert_equal(hit.prim, UInt32(0))
                assert_equal(hit.t, Float32(1.5))


def test_conservative_triangle_segments() raises:
    _test_triangles(8)


def test_exact_triangle_segments() raises:
    _test_triangles(4096)


def test_conservative_sphere_segments() raises:
    _test_spheres(8)


def test_exact_sphere_segments() raises:
    _test_spheres(4096)


def test_mixed_triangle_builds_preserve_parallel_small_segments() raises:
    var meshes = List[List[Point3f32[.LOCAL]]]()
    for i in range(17):
        meshes.append(_mesh(4096 if i == 8 else 31, Float32(i) * 10000.0))
    var blases = build_cpu_triangle_blas_set[4, 2](meshes)
    for i in range(len(meshes)):
        var ray = Rayf32[.LOCAL](
            Point3f32[.LOCAL](Float32(i) * 10000.0, 0.0, 0.0),
            Vec3f32[.LOCAL](0.0, 0.0, 1.0),
        )
        var hit = trace_blas_set[4, 2, .CLOSEST_HIT, .LOCAL](
            blases, UInt32(i), ray
        )
        assert_equal(hit.prim, UInt32(0))
        assert_equal(hit.t, 2.0)
        var desc = BlasDesc.load(blases.descs.unsafe_ptr(), UInt32(i))
        assert_equal(desc.prim_count, UInt32(4096 if i == 8 else 31))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
