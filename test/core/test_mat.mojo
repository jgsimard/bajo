from testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
)
from bajo.core.mat import Mat33f32, determinant, _matvec
from bajo.core.vec import (
    Vec3f32,
    assert_vec_equal,
)
from bajo.core.conversion import degrees_to_radians
from bajo.core.quat import Quat


fn test_basics() raises:
    # Transpose check
    # fmt: off
    m = Mat33f32(
        1, 2, 3, 
        4, 5, 6, 
        7, 8, 9
    )
    mt = Mat33f32(
        1, 4, 7, 
        2, 5, 8, 
        3, 6, 9
    )
    # fmt: on
    assert_equal(m.transpose(), mt)

    # Determinant of Identity
    assert_almost_equal(determinant(Mat33f32.identity()), 1.0)


fn test_Mat3f_from_quat() raises:
    angle = degrees_to_radians(Float32(30))
    r = Quat.from_axis_angle(Vec3f32(1, 0, 0), angle)
    m = r.to_matrix()
    # Rotating a vector by the matrix should match rotating by the quaternion
    v = Vec3f32(0, 1, 0)
    assert_vec_equal(_matvec(m, v), r.rotate(v))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
