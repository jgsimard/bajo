from testing import TestSuite, assert_equal, assert_almost_equal
from src.math import Quat, Vector3, to_radians

fn assert_quat_equal[type: DType](a: Quat[type], b: Quat[type]) raises:
    assert_almost_equal(a.wxyz, b.wxyz, atol=1e-4)

fn test_quaternions() raises:
    alias FloatType = DType.float32
    var q1 = Quat[FloatType].angle_axis(0, Vector3[FloatType](0, 1, 0))
    assert_quat_equal(q1, Quat[FloatType](1, 0, 0, 0))

    var q2 = Quat[FloatType].angle_axis(to_radians(Scalar[FloatType](45)), Vector3[FloatType](0, 1, 0))
    assert_quat_equal(q2, Quat[FloatType](0.9238795, 0, 0.3826834, 0))

    var q3 = Quat[FloatType].angle_axis(to_radians(Scalar[FloatType](45)), Vector3[FloatType](1, 0, 0))
    assert_quat_equal(q3, Quat[FloatType](0.9238795, 0.3826834, 0, 0))

    var m1 = q2 * q3
    assert_quat_equal(m1, Quat[FloatType](0.853553, 0.353553, 0.353553, -0.146447))

fn test_vec3_operations() raises:
    alias FloatType = DType.float32
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
