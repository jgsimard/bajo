from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_true,
    assert_false,
    assert_almost_equal,
)

from bajo.core import (
    AABB,
    Quat,
    Vec3f32,
    assert_vec_equal,
    Affine3,
    Point3f32,
    Frame,
)
from bajo.core.utils import degrees_to_radians


def test_logic() raises:
    box = AABB[Frame.WORLD](Point3f32(0), Point3f32(2, 2, 2))

    assert_true(box.contains_point(Point3f32(1)))
    assert_false(box.contains_point(Point3f32(3, 1, 1)))

    box_overlap = AABB[Frame.WORLD](Point3f32(1), Point3f32(3))
    box_far = AABB[Frame.WORLD](Point3f32(4), Point3f32(5))
    assert_true(box.overlaps(box_overlap))
    assert_false(box.overlaps(box_far))


def test_merge() raises:
    a = AABB[Frame.WORLD](Point3f32(0), Point3f32(1))
    b = AABB[Frame.WORLD](Point3f32(-1), Point3f32(0.5))
    merged = AABB.merge(a, b)
    assert_vec_equal(merged._min, Point3f32(-1))
    assert_vec_equal(merged._max, Point3f32(1))


def test_apply_trs_rotated() raises:
    box = AABB[Frame.WORLD](Point3f32(-1), Point3f32(1))

    # Rotate 45 degrees around Z
    angle = degrees_to_radians(Float32(45))
    r = Quat.from_axis_angle(Vec3f32(0, 0, 1), angle)
    t = Vec3f32(0)
    s = Vec3f32(1)
    transform = Affine3.from_rotation_scale_translation(r, s, t)

    new_box = box.apply_transform(transform)

    # the corners should move to ±sqrt(2)
    sqrt_2 = Float32(sqrt(2.0))
    assert_vec_equal(new_box._min, Point3f32(-sqrt_2, -sqrt_2, -1.0))
    assert_vec_equal(new_box._max, Point3f32(sqrt_2, sqrt_2, 1.0))


def test_aabb_store6_with_nonzero_base() raises:
    var data = List[Float32](length=18, fill=-1.0)
    var b = AABB[Frame.WORLD](
        Point3f32(1.0, 2.0, 3.0), Point3f32(4.0, 5.0, 6.0)
    )

    b.store6(data.unsafe_ptr(), 6)

    assert_almost_equal(data[6], 1.0)
    assert_almost_equal(data[7], 2.0)
    assert_almost_equal(data[8], 3.0)
    assert_almost_equal(data[9], 4.0)
    assert_almost_equal(data[10], 5.0)
    assert_almost_equal(data[11], 6.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
