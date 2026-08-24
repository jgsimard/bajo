from std.testing import (
    TestSuite,
    assert_almost_equal,
)

from bajo.core.utils import degrees_to_radians
from bajo.core.quat import Quaternion
from bajo.core.vec import Vec3, assert_vec_equal
from bajo.core.frame import Frame


def assert_quat_equal[
    T: DType
](a: Quaternion[T, _], b: Quaternion[T, _]) raises:
    assert_almost_equal(a.x, b.x, atol=1e-8)
    assert_almost_equal(a.y, b.y, atol=1e-8)
    assert_almost_equal(a.z, b.z, atol=1e-8)
    assert_almost_equal(a.w, b.w, atol=1e-8)


def _test_from_axis_angle_mul[T: DType]() raises where T.is_floating_point():
    comptime F = Frame.WORLD
    comptime S = Scalar[T]
    comptime Q = Quaternion[T, F]

    var q1 = Q.from_axis_angle(Vec3[T, F](0, 1, 0), 0)
    assert_quat_equal(q1, Q(0, 0, 0, 1))

    var angle = degrees_to_radians(S(45))
    var q2 = Q.from_axis_angle(Vec3[T, F](0, 1, 0), angle)
    assert_quat_equal(q2, Q(0, 0.3826834, 0, 0.9238795))

    var q3 = Q.from_axis_angle(Vec3[T, F](1, 0, 0), angle)
    assert_quat_equal(q3, Q(0.3826834, 0, 0, 0.9238795))

    var m1 = q2 * q3
    assert_quat_equal(m1, Q(0.353553, 0.353553, -0.146447, 0.853553))


def test_from_axis_angle_mul() raises:
    # _test_from_axis_angle_mul[.float16]() # does not pass !
    _test_from_axis_angle_mul[.float32]()
    _test_from_axis_angle_mul[.float64]()


def test_mul_rotate() raises:
    comptime F = Frame.WORLD
    comptime T = DType.float32
    comptime S = Scalar[T]

    # Rotate 90 X then 90 Y
    var angle = degrees_to_radians(S(90))
    var qx = Quaternion[T].from_axis_angle(Vec3[T, F](1, 0, 0), angle)
    var qy = Quaternion[T].from_axis_angle(Vec3[T, F](0, 1, 0), angle)

    var q_combined = qy * qx  # Note: Order matters (Local vs World)

    # Rotate a vector (0, 0, 1)
    # 1. Rotate 90 around X -> (0, -1, 0)
    # 2. Rotate 90 around Y -> (0, -1, 0) ... Y doesn't affect it
    var v = Vec3[T, F](0, 0, 1)
    var result = q_combined.rotate(v)
    assert_vec_equal(result, Vec3[T, F](0, -1, 0))


def _test_rotate[T: DType]() raises where T.is_floating_point():
    comptime F = Frame.WORLD
    comptime S = Scalar[T]

    # Rotate (1, 0, 0) 90 degrees around Y axis -> should be (0, 0, -1)
    var axis = Vec3[T, F](0, 1, 0)
    var angle = degrees_to_radians(S(90))
    var q = Quaternion[T].from_axis_angle(axis, angle)
    var v = Vec3[T, F](1, 0, 0)
    var rotated = q.rotate(v)
    assert_vec_equal(rotated, Vec3[T, F](0, 0, -1))


def test_rotate() raises:
    _test_rotate[.float16]()
    _test_rotate[.float32]()
    _test_rotate[.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
