"""Assertions shared by the Mojo test suites."""

from std.testing import assert_almost_equal

from bajo.core.frame import Frame
from bajo.core.mat import Mat
from bajo.core.vec import Geo3, GeoKind


def assert_vec_equal[
    dtype: DType, kind: GeoKind, frame: Frame, width: SIMDLength
](
    a: Geo3[dtype, kind, frame, width],
    b: Geo3[dtype, kind, frame, width],
    atol: Float64 = 1e-5,
) raises:
    assert_almost_equal(a.x, b.x, msg="x", atol=atol)
    assert_almost_equal(a.y, b.y, msg="y", atol=atol)
    assert_almost_equal(a.z, b.z, msg="z", atol=atol)


def assert_mat_equal[
    dtype: DType,
    rows: Int,
    cols: Int,
    frame: Frame,
    width: SIMDLength,
](
    a: Mat[dtype, rows, cols, frame, width],
    b: Mat[dtype, rows, cols, frame, width],
    atol: Float64 = 1e-5,
) raises:
    for i in range(rows):
        for j in range(cols):
            for lane in range(width):
                assert_almost_equal(
                    a[i][j][lane],
                    b[i][j][lane],
                    msg=String(t"[{i}][{j}][{lane}]"),
                    atol=atol,
                )
