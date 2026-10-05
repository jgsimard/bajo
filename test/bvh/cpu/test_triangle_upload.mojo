"""Host packing of borrowed triangle meshes for GPU preparation."""

from std.testing import TestSuite, assert_equal
from bajo.bvh.gpu.triangle_bvh import _flatten_triangle_sets
from bajo.core import Point3f32


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
