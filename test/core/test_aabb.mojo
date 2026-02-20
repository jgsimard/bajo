from testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
    assert_false,
)

from bajo.core.aabb import AABB
from bajo.core.conversion import degrees_to_radians
from bajo.core.quat import Quat
from bajo.core.vec import Vec3f32, assert_vec_equal


fn test_logic() raises:
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


fn test_merge() raises:
    a = AABB(Vec3f32(0), Vec3f32(1))
    b = AABB(Vec3f32(-1), Vec3f32(0.5))
    merged = AABB.merge(a, b)
    assert_vec_equal(merged.min, Vec3f32(-1))
    assert_vec_equal(merged.max, Vec3f32(1))


fn test_apply_trs_rotated() raises:
    box = AABB(Vec3f32(-1), Vec3f32(1))

    # Rotate 45 degrees around Z
    angle = degrees_to_radians(Float32(45))
    r = Quat.angle_axis(angle, Vec3f32(0, 0, 1))
    t = Vec3f32(0)
    s = Vec3f32(1, 1, 1)

    new_box = box.apply_trs(t, r, s)

    # After 45 deg rotation, the corners move to ±sqrt(2)
    # The new AABB should expand to fit the diamond shape
    expected_limit = Float32(1.41421356)
    assert_almost_equal(new_box.max.x(), expected_limit, atol=1e-5)
    assert_almost_equal(new_box.max.y(), expected_limit, atol=1e-5)
    # Z should remain 1.0
    assert_almost_equal(new_box.max.z(), 1.0, atol=1e-5)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
