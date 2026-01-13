from testing import TestSuite, assert_equal, assert_almost_equal
from bajo.bmath import Quat, deg_to_radians, cross, Vec3f


fn assert_quat_equal(a: Quat, b: Quat) raises:
    assert_almost_equal(a.wxyz, b.wxyz, atol=1e-4)


fn test_quaternions() raises:
    var q1 = Quat.angle_axis(0, Vec3f(0, 1, 0))
    assert_quat_equal(q1, Quat(1, 0, 0, 0))

    var q2 = Quat.angle_axis(deg_to_radians(Scalar[DType.float32](45)), Vec3f(0, 1, 0))
    assert_quat_equal(q2, Quat(0.9238795, 0, 0.3826834, 0))

    var q3 = Quat.angle_axis(deg_to_radians(Scalar[DType.float32](45)), Vec3f(1, 0, 0))
    assert_quat_equal(q3, Quat(0.9238795, 0.3826834, 0, 0))

    var m1 = q2 * q3
    assert_quat_equal(m1, Quat(0.853553, 0.353553, 0.353553, -0.146447))


fn test_vec3_operations() raises:
    var v1 = Vec3f(1, 2, 3)
    var v2 = Vec3f(4, 5, 6)

    var v_add = v1 + v2
    assert_almost_equal(v_add.x(), 5.0)

    var c = cross(v1, v2)

    # 1,2,3 x 4,5,6 = -3, 6, -3
    assert_almost_equal(c.x(), -3.0)
    assert_almost_equal(c.y(), 6.0)
    assert_almost_equal(c.z(), -3.0)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
