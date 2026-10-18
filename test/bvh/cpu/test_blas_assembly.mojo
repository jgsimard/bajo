"""Segment isolation across conservative and exact packed BLAS assembly."""

from std.testing import TestSuite, assert_equal
from std.memory import bitcast

from bajo.bvh.constants import (
    PrimitiveKind,
    WideNode,
    CPU_TRI_LEAF_PACKED_STRIDE,
    SPHERE_LEAF_PACKED_STRIDE,
)
from bajo.bvh.cpu import (
    CpuBlasSet,
    CpuBvhBuildMethod,
    build_cpu_triangle_blas_set,
    build_cpu_sphere_blas_set,
    trace_blas_set,
    trace_blas_set_packet,
    trace_blas_set_packet_any_hit,
)
from bajo.bvh.gpu.triangle_bvh import _flatten_triangle_sets
from bajo.bvh.types import BlasDesc, Sphere, SphereLeafBlock, Hit
from bajo.core import Point3, Point3f32, Ray, Rayf32, Vec3, Vec3f32


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
    return List[Sphere[.LOCAL]](
        length=count,
        fill_with=lambda (i: Int) -> Sphere[.LOCAL]: Sphere[.LOCAL](
            Point3f32[.LOCAL](offset + Float32(i) * 2.0, 0.0, 2.0), 0.5
        ),
    )


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


def _test_borrowed_triangle_mesh[method: CpuBvhBuildMethod]() raises:
    var vertices = _mesh(31, 0.0)
    var meshes: List[List[Point3f32[.LOCAL]]] = [vertices.copy()]
    var borrowed = build_cpu_triangle_blas_set[4, 2, method](vertices)
    var batched = build_cpu_triangle_blas_set[4, 2, method](meshes)
    assert_equal(len(vertices), 93)
    assert_equal(borrowed.blas_count, 1)
    assert_equal(len(borrowed.descs), len(batched.descs))
    assert_equal(len(borrowed.nodes), len(batched.nodes))
    assert_equal(len(borrowed.leaves), len(batched.leaves))
    for i in range(len(borrowed.descs)):
        assert_equal(borrowed.descs[i], batched.descs[i])
    for i in range(len(borrowed.nodes)):
        assert_equal(
            bitcast[.uint32](borrowed.nodes[i]),
            bitcast[.uint32](batched.nodes[i]),
        )
    for i in range(len(borrowed.leaves)):
        assert_equal(
            bitcast[.uint32](borrowed.leaves[i]),
            bitcast[.uint32](batched.leaves[i]),
        )

    # Packed output must own its geometry after the borrowed input is released.
    vertices.clear()
    var ray = Rayf32[.LOCAL](
        Point3f32[.LOCAL](0.0), Vec3f32[.LOCAL](0.0, 0.0, 1.0)
    )
    var hit = trace_blas_set[4, 2, .CLOSEST_HIT, .LOCAL](
        borrowed, UInt32(0), ray
    )
    assert_equal(hit.prim, UInt32(0))
    assert_equal(hit.t, 2.0)


def test_borrowed_triangle_mesh_matches_batch() raises:
    _test_borrowed_triangle_mesh[.SAH]()
    _test_borrowed_triangle_mesh[.LBVH]()
    _test_borrowed_triangle_mesh[.HPLOC]()


def test_empty_borrowed_triangle_mesh() raises:
    var vertices = List[Point3f32[.LOCAL]]()
    var blases = build_cpu_triangle_blas_set[4, 2](vertices)
    assert_equal(blases.blas_count, 1)
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), UInt32(0))
    assert_equal(desc.prim_count, UInt32(0))
    assert_equal(len(blases.nodes), 0)
    assert_equal(len(blases.leaves), 0)


def test_borrowed_triangle_upload_matches_batch() raises:
    var vertices: List[Point3f32[.WORLD]] = [
        Point3f32[.WORLD](-1.0, -1.0, 2.0),
        Point3f32[.WORLD](1.0, -1.0, 2.0),
        Point3f32[.WORLD](0.0, 1.0, 2.0),
    ]
    var meshes: List[List[Point3f32[.WORLD]]] = [vertices.copy()]
    var borrowed = _flatten_triangle_sets(vertices)
    var batched = _flatten_triangle_sets(meshes)
    assert_equal(len(vertices), 3)
    assert_equal(borrowed.segments.segment_count(), 1)
    assert_equal(borrowed.segments.item_count(), 1)
    assert_equal(len(borrowed.packed_vertices), 9)
    for i in range(9):
        assert_equal(borrowed.packed_vertices[i], batched.packed_vertices[i])
    vertices.clear()
    assert_equal(borrowed.packed_vertices[0], Float32(-1.0))


def test_empty_borrowed_triangle_upload() raises:
    var vertices = List[Point3f32[.WORLD]]()
    var packed = _flatten_triangle_sets(vertices)
    assert_equal(packed.segments.segment_count(), 1)
    assert_equal(packed.segments.item_count(), 0)
    assert_equal(len(packed.packed_vertices), 0)


def test_packed_views_preserve_identifier_bits_and_record_boundaries() raises:
    var words = SIMD[.uint32, 4](0, 0x80000000, 0x7FC01234, 0xFFFFFFFF)
    var block = SphereLeafBlock[.WORLD, 4]()
    block.center.x = 1.0
    block.radius = 2.0
    block.prim_indices = words
    var leaves = List[Float32](length=3 * 5 * 4, fill=-17.0)
    block.store(leaves.unsafe_ptr(), 1)
    var loaded = SphereLeafBlock[.WORLD, 4].load(leaves.unsafe_ptr(), 1)
    assert_equal(loaded.prim_indices, words)
    assert_equal(loaded.center.x, block.center.x)
    assert_equal(loaded.radius, block.radius)
    assert_equal(leaves[19], Float32(-17.0))
    assert_equal(leaves[40], Float32(-17.0))
    var hits = List[Float32](length=3 * 8, fill=-17.0)
    var hit = Hit[.WORLD].miss()
    hit.prim = words[2]
    hit.inst = words[3]
    hit.store(hits, 1)
    var restored = Hit[.WORLD].load(hits, 1)
    assert_equal(restored.prim, hit.prim)
    assert_equal(restored.inst, hit.inst)
    assert_equal(hits[7], Float32(-17.0))
    assert_equal(hits[16], Float32(-17.0))


def _segment_packet_rays() -> Ray[.float32, .LOCAL, 4]:
    return Ray[.float32, .LOCAL, 4](
        Point3[.float32, .LOCAL, 4](
            SIMD[.float32, 4](0.0, 10000.0, 10002.0, 0.0), 0.0, 0.0
        ),
        Vec3[.float32, .LOCAL, 4](0.0, 0.0, 1.0),
    )


def _assert_packet_segment_hits(
    idx: Int, hits: Hit[.LOCAL, 4], occluded: SIMD[.bool, 4], distance: Float32
) raises:
    comptime for lane in range(4):
        var expected = (idx == 0 and lane == 0) or (
            idx == 2 and (lane == 1 or lane == 2)
        )
        assert_equal(Bool(hits.is_hit()[lane]), expected)
        assert_equal(Bool(occluded[lane]), expected)
        if expected:
            assert_equal(hits.prim[lane], UInt32(1 if lane == 2 else 0))
            assert_equal(hits.t[lane], distance)


def test_triangle_packets_preserve_segment_offsets_and_empty_segments() raises:
    var meshes: List[List[Point3f32[.LOCAL]]] = [
        _mesh(1, 0.0),
        [],
        _mesh(8, 10000.0),
        [],
    ]
    var blases = build_cpu_triangle_blas_set[4, 2](meshes)
    var rays = _segment_packet_rays()
    var valid = SIMD[.bool, 4](True, True, True, False)
    for idx in range(4):
        var hits = trace_blas_set_packet(blases, UInt32(idx), rays, valid)
        var occluded = trace_blas_set_packet_any_hit(
            blases, UInt32(idx), rays, valid
        )
        _assert_packet_segment_hits(idx, hits, occluded, 2.0)


def test_sphere_packets_preserve_segment_offsets_and_empty_segments() raises:
    var sets: List[List[Sphere[.LOCAL]]] = [
        _spheres(1, 0.0),
        [],
        _spheres(8, 10000.0),
        [],
    ]
    var blases = build_cpu_sphere_blas_set[4](sets)
    var rays = _segment_packet_rays()
    var valid = SIMD[.bool, 4](True, True, True, False)
    for idx in range(4):
        var hits = trace_blas_set_packet(blases, UInt32(idx), rays, valid)
        var occluded = trace_blas_set_packet_any_hit(
            blases, UInt32(idx), rays, valid
        )
        _assert_packet_segment_hits(idx, hits, occluded, 1.5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
