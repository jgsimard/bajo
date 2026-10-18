from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from bajo.bvh.host_utils import compute_bounds
from bajo.core import Point3W, cross, dot, length
from bajo.rt import Color, SceneBuilder


def test_box_has_closed_outward_faces_bounds_and_surfaces() raises:
    var builder = SceneBuilder()
    _ = builder.add_lambertian(Color(0.1))
    var surface = builder.add_lambertian(Color(0.8))
    var minimum = Point3W(-2.0, 1.0, -4.0)
    var maximum = Point3W(3.0, 3.0, -1.0)
    builder.add_box(minimum, maximum, surface)
    var scene = builder^.finish()
    ref vertices = scene.triangle_vertices()
    assert_equal(len(vertices), 36)
    assert_equal(len(scene.triangle_surfaces()), 12)
    var bounds = compute_bounds(vertices)
    var center = bounds.centroid()
    for axis in range(3):
        assert_equal(bounds._min[axis], minimum[axis])
        assert_equal(bounds._max[axis], maximum[axis])
    var area = Float32(0.0)
    for triangle in range(12):
        ref a = vertices[3 * triangle]
        ref b = vertices[3 * triangle + 1]
        ref c = vertices[3 * triangle + 2]
        var normal = cross(b - a, c - a)
        assert_true(dot(normal, a - center) > 0.0)
        # Every triangle lies on one of the six axis-aligned boundary planes.
        var boundary_face = False
        for axis in range(3):
            boundary_face |= (
                a[axis] == b[axis]
                and b[axis] == c[axis]
                and (a[axis] == minimum[axis] or a[axis] == maximum[axis])
            )
        assert_true(boundary_face)
        area += length(normal) * 0.5
        assert_equal(scene.triangle_surfaces()[triangle].value, surface.value)
    assert_almost_equal(area, Float32(2 * (5 * 2 + 5 * 3 + 2 * 3)))


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
