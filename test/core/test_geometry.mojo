"""Surface sampling and normals in arbitrary coordinate frames."""

from std.testing import TestSuite, assert_equal, assert_true
from bajo.core import Point3
from bajo.core.geometry import sample_triangle_surface


def test_triangle_surface_sampling_retains_endpoints_and_degenerate_masks() raises:
    var p0 = Point3[.float32, .LOCAL, 4](0)
    var p1 = Point3[.float32, .LOCAL, 4](2, 0, 0)
    var p2 = Point3[.float32, .LOCAL, 4](0, 4, 0)
    p2.y[3] = 0
    var sample = sample_triangle_surface(
        p0,
        p1,
        p2,
        SIMD[.float32, 4](0, 1, 1, 0.25),
        SIMD[.float32, 4](0, 0, 1, 0.5),
    )
    assert_equal(sample.valid, SIMD[.bool, 4](True, True, True, False))
    assert_equal(sample.point.x[0], Float32(0))
    assert_equal(sample.point.x[1], Float32(2))
    assert_equal(sample.point.y[2], Float32(4))
    assert_equal(sample.normal.z[0], Float32(1))
    assert_equal(sample.normal.y[3], Float32(1))
    assert_true(sample.normal.is_finite().reduce_and())


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
