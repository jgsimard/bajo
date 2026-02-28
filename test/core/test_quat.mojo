from testing import (
    TestSuite,
    assert_almost_equal,
)

from bajo.core.conversion import degrees_to_radians
from bajo.core.quat import Quat
from bajo.core.vec import Vec3f32, assert_vec_equal


def assert_quat_equal(a: Quat, b: Quat) raises:
    assert_almost_equal(a.data, b.data, atol=1e-6)


def test_from_axis_angle_mul() raises:
    q1 = Quat.from_axis_angle(Vec3f32(0, 1, 0), 0)
    assert_quat_equal(q1, Quat(0, 0, 0, 1))

    angle = degrees_to_radians(Float32(45))
    q2 = Quat.from_axis_angle(Vec3f32(0, 1, 0), angle)
    assert_quat_equal(q2, Quat(0, 0.3826834, 0, 0.9238795))

    q3 = Quat.from_axis_angle(Vec3f32(1, 0, 0), angle)
    assert_quat_equal(q3, Quat(0.3826834, 0, 0, 0.9238795))

    m1 = q2 * q3
    assert_quat_equal(m1, Quat(0.353553, 0.353553, -0.146447, 0.853553))


def test_mul_rotate() raises:
    # Rotate 90 X then 90 Y
    angle = degrees_to_radians(Float32(90))
    qx = Quat.from_axis_angle(Vec3f32(1, 0, 0), angle)
    qy = Quat.from_axis_angle(Vec3f32(0, 1, 0), angle)

    q_combined = qy * qx  # Note: Order matters (Local vs World)

    # Rotate a vector (0, 0, 1)
    # 1. Rotate 90 around X -> (0, -1, 0)
    # 2. Rotate 90 around Y -> (0, -1, 0) ... Y doesn't affect it
    v = Vec3f32(0, 0, 1)
    result = q_combined.rotate(v)
    assert_vec_equal(result, Vec3f32(0, -1, 0))


def test_rotate() raises:
    # Rotate (1, 0, 0) 90 degrees around Y axis -> should be (0, 0, -1)
    axis = Vec3f32(0, 1, 0)
    angle = degrees_to_radians(Float32(90))
    q = Quat.from_axis_angle(axis, angle)
    v = Vec3f32(1, 0, 0)
    rotated = q.rotate(v)
    assert_vec_equal(rotated, Vec3f32(0, 0, -1))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
