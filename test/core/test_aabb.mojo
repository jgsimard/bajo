from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_true,
    assert_false,
)

from bajo.core.aabb import AABB
from bajo.core.conversion import degrees_to_radians
from bajo.core.quat import Quat
from bajo.core.vec import Vec3f32, assert_vec_equal


def test_logic() raises:
    box = AABB(Vec3f32(0), Vec3f32(2, 2, 2))

    assert_true(box.contains(Vec3f32(1)))
    assert_false(box.contains(Vec3f32(3, 1, 1)))

    box_overlap = AABB(Vec3f32(1), Vec3f32(3))
    box_far = AABB(Vec3f32(4), Vec3f32(5))
    assert_true(box.overlaps(box_overlap))
    assert_false(box.overlaps(box_far))

    # Ray Intersect
    ray_o = Vec3f32(-1, 1, 1)
    ray_d = Vec3f32(1, 0, 0)
    # Diag3.inv handles the 1/d calculation used in ray_intersects
    inv_d = 1.0 / ray_d

    assert_true(box.ray_intersects(ray_o, inv_d, 0.0, 10.0))


def test_merge() raises:
    a = AABB(Vec3f32(0), Vec3f32(1))
    b = AABB(Vec3f32(-1), Vec3f32(0.5))
    merged = AABB.merge(a, b)
    assert_vec_equal(merged._min, Vec3f32(-1))
    assert_vec_equal(merged._max, Vec3f32(1))


def test_apply_trs_rotated() raises:
    box = AABB(Vec3f32(-1), Vec3f32(1))

    # Rotate 45 degrees around Z
    angle = degrees_to_radians(Float32(45))
    r = Quat.from_axis_angle(Vec3f32(0, 0, 1), angle)
    t = Vec3f32(0)
    s = Vec3f32(1)

    new_box = box.apply_trs(t, r, s)

    # the corners should move to ±sqrt(2)
    sqrt_2 = Float32(sqrt(2.0))
    assert_vec_equal(new_box._min, Vec3f32(-sqrt_2, -sqrt_2, -1.0))
    assert_vec_equal(new_box._max, Vec3f32(sqrt_2, sqrt_2, 1.0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
