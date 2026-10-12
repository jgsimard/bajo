"""Pixel comparisons with size, pixel index, and component diagnostics."""

from std.testing import assert_almost_equal, assert_equal

from bajo.rt.render_types import Color


def assert_pixels_close(
    actual: List[Color],
    expected: List[Color],
    atol: Float64 = 1e-5,
    rtol: Optional[Float64] = None,
) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(expected)):
        comptime for axis in range(3):
            if rtol:
                assert_almost_equal(
                    actual[i][axis],
                    expected[i][axis],
                    atol=atol,
                    rtol=rtol.value(),
                    msg=String(t"pixel {i}, component {axis}"),
                )
            else:
                assert_almost_equal(
                    actual[i][axis],
                    expected[i][axis],
                    atol=atol,
                    msg=String(t"pixel {i}, component {axis}"),
                )


def assert_pixels_equal(actual: List[Color], expected: List[Color]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(expected)):
        comptime for axis in range(3):
            assert_equal(
                actual[i][axis],
                expected[i][axis],
                msg=String(t"pixel {i}, component {axis}"),
            )
