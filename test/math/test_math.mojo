from testing import TestSuite, assert_equal, assert_almost_equal
from bajo.math import Quat, Vector3, deg_to_radians


fn assert_quat_equal[dtype: DType](a: Quat[dtype], b: Quat[dtype]) raises:
    assert_almost_equal(a.wxyz, b.wxyz, atol=1e-4)


fn test_quaternions() raises:
    comptime f32 = DType.float32
    comptime vec3 = Vector3[f32]
    comptime quat = Quat[f32]

    var q1 = quat.angle_axis(0, vec3(0, 1, 0))
    assert_quat_equal(q1, quat(1, 0, 0, 0))

    var q2 = quat.angle_axis(deg_to_radians(Scalar[f32](45)), vec3(0, 1, 0))
    assert_quat_equal(q2, quat(0.9238795, 0, 0.3826834, 0))

    var q3 = quat.angle_axis(deg_to_radians(Scalar[f32](45)), vec3(1, 0, 0))
    assert_quat_equal(q3, quat(0.9238795, 0.3826834, 0, 0))

    var m1 = q2 * q3
    assert_quat_equal(m1, quat(0.853553, 0.353553, 0.353553, -0.146447))


fn test_vec3_operations() raises:
    comptime FloatType = DType.float32
    var v1 = Vector3[FloatType](1, 2, 3)
    var v2 = Vector3[FloatType](4, 5, 6)

    var v_add = v1 + v2
    assert_almost_equal(v_add.x, 5.0)

    var c = v1.cross(v2)

    # 1,2,3 x 4,5,6 = -3, 6, -3
    assert_almost_equal(c.x, -3.0)
    assert_almost_equal(c.y, 6.0)
    assert_almost_equal(c.z, -3.0)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
