from testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
)
from bajo.core.mat2 import Mat33f32, determinant
from bajo.core.mat import Mat3f, Mat3x4f
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


fn test_trs_composition_decomposition() raises:
    t = Vec3f32(10, 20, 30)

    angle = degrees_to_radians(Float32(45))
    r = Quat.angle_axis(angle, Vec3f32(0, 0, 1))
    s = Vec3f32(2, 2, 2)

    # Create Mat3x4 from TRS
    mat = Mat3x4f.from_trs(t, r, s)

    # Transform a point
    p = Vec3f32(1, 0, 0)
    p_tx = mat.txfm_point(p)
    # Expected: Scaled(2,0,0) -> Rotated 45deg on Z(1.414, 1.414, 0) -> Translated(11.414, 21.414, 30)
    assert_vec_equal(p_tx, Vec3f32(11.41421, 21.41421, 30), atol=1e-4)

    # Decompose back
    decomposed = mat.decompose()
    dec_t = decomposed[0].copy()
    dec_s = decomposed[2].copy()

    assert_vec_equal(dec_t, t)
    assert_almost_equal(dec_s[0], 2.0)


fn test_composition() raises:
    parent_mat = Mat3x4f.from_trs(
        Vec3f32(10, 0, 0), Quat.identity(), Vec3f32(1)
    )
    child_mat = Mat3x4f.from_trs(Vec3f32(5, 0, 0), Quat.identity(), Vec3f32(1))

    combined = parent_mat.compose(child_mat)

    p = Vec3f32(0, 0, 0)
    result = combined.txfm_point(p)

    # Should be at 15 (10 + 5)
    assert_almost_equal(result.x(), 15.0)


fn test_Mat3f_from_quat() raises:
    angle = degrees_to_radians(Float32(30))
    r = Quat.angle_axis(angle, Vec3f32(1, 0, 0))
    m = Mat3f.from_quat(r)
    # Rotating a vector by the matrix should match rotating by the quaternion
    v = Vec3f32(0, 1, 0)
    assert_vec_equal(m * v, r.rotate_vec(v))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
