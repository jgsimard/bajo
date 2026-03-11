from std.testing import (
    TestSuite,
    assert_almost_equal,
)

from bajo.core.conversion import degrees_to_radians
from bajo.core.quat import Quaternion
from bajo.core.vec import Vec3, assert_vec_equal


def assert_quat_equal[
    T: DType where T.is_floating_point()
](a: Quaternion[T], b: Quaternion[T]) raises:
    assert_almost_equal(a.data, b.data, atol=1e-8)


def _test_from_axis_angle_mul[T: DType where T.is_floating_point()]() raises:
    comptime S = Scalar[T]
    comptime Q = Quaternion[T]

    q1 = Q.from_axis_angle(Vec3[T](0, 1, 0), 0)
    assert_quat_equal(q1, Q(0, 0, 0, 1))

    angle = degrees_to_radians(S(45))
    q2 = Q.from_axis_angle(Vec3[T](0, 1, 0), angle)
    assert_quat_equal(q2, Q(0, 0.3826834, 0, 0.9238795))

    q3 = Q.from_axis_angle(Vec3[T](1, 0, 0), angle)
    assert_quat_equal(q3, Q(0.3826834, 0, 0, 0.9238795))

    m1 = q2 * q3
    assert_quat_equal(m1, Q(0.353553, 0.353553, -0.146447, 0.853553))


def test_from_axis_angle_mul() raises:
    # _test_from_axis_angle_mul[DType.float16]() # does not pass !
    _test_from_axis_angle_mul[DType.float32]()
    _test_from_axis_angle_mul[DType.float64]()


def test_mul_rotate() raises:
    comptime T = DType.float32
    comptime S = Scalar[T]

    # Rotate 90 X then 90 Y
    angle = degrees_to_radians(S(90))
    qx = Quaternion[T].from_axis_angle(Vec3[T](1, 0, 0), angle)
    qy = Quaternion[T].from_axis_angle(Vec3[T](0, 1, 0), angle)

    q_combined = qy * qx  # Note: Order matters (Local vs World)

    # Rotate a vector (0, 0, 1)
    # 1. Rotate 90 around X -> (0, -1, 0)
    # 2. Rotate 90 around Y -> (0, -1, 0) ... Y doesn't affect it
    v = Vec3[T](0, 0, 1)
    result = q_combined.rotate(v)
    assert_vec_equal(result, Vec3[T](0, -1, 0))


def _test_rotate[T: DType where T.is_floating_point()]() raises:
    comptime S = Scalar[T]

    # Rotate (1, 0, 0) 90 degrees around Y axis -> should be (0, 0, -1)
    axis = Vec3[T](0, 1, 0)
    angle = degrees_to_radians(S(90))
    q = Quaternion[T].from_axis_angle(axis, angle)
    v = Vec3[T](1, 0, 0)
    rotated = q.rotate(v)
    assert_vec_equal(rotated, Vec3[T](0, 0, -1))


def test_rotate() raises:
    _test_rotate[DType.float16]()
    _test_rotate[DType.float32]()
    _test_rotate[DType.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
