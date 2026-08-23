from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
)

from bajo.core import Vec3W, assert_vec_equal, Quat, Mat22, Mat33, Mat44
from bajo.core.mat import (
    inverse,
    determinant,
    _matmul,
    _matvec,
    assert_mat_equal,
)
from bajo.core.utils import degrees_to_radians
from bajo.core.frame import Frame


def test_basics() raises:
    comptime T = DType.float32
    # Transpose check
    # fmt: off
    var m = Mat33[T, .WORLD](
        1, 2, 3, 
        4, 5, 6, 
        7, 8, 9
    )
    var mt = Mat33[T, .WORLD](
        1, 4, 7, 
        2, 5, 8, 
        3, 6, 9
    )
    # fmt: on
    assert_mat_equal(m.transpose(), mt)

    assert_almost_equal(determinant(Mat33[T, .WORLD].identity()), 1.0)

    var m2 = Mat22[T, .WORLD](1, 2, 3, 4)
    assert_almost_equal(determinant(m2), -2.0)
    assert_almost_equal(determinant(_matmul(m2, inverse(m2))), 1.0)

    var m3 = Mat33[T, .WORLD](1, 2, 3, 4, 5, 6, 7, 8, 10)
    assert_almost_equal(determinant(m3), -3.0)
    assert_almost_equal(determinant(_matmul(m3, inverse(m3))), 1.0)

    var m4 = Mat44[T, .WORLD](
        1, 3, 5, 9, 1, 3, 1, 7, 4, 3, 9, 7, 5, 2, 0, 9
    )
    assert_almost_equal(determinant(m4), -376.0)
    assert_almost_equal(determinant(_matmul(m4, inverse(m4))), 1.0)


def test_Mat3f_from_quat() raises:
    var axis = Vec3W(1, 0, 0)
    var angle = degrees_to_radians(Float32(30))
    var q = Quat.from_axis_angle(axis, angle)
    var m = q.to_matrix()
    var v = Vec3W(0, 1, 0)
    assert_vec_equal(_matvec(m, v), q.rotate(v))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
