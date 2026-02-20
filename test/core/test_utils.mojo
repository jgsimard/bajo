from testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
    assert_false,
)

from bajo.core.utils import QuadraticSolutions, solve_quadratic


fn test_solve_quadratic() raises:
    # Lane 0: x^2 - 5x + 6 = 0 -> roots 2, 3
    # Lane 1: x^2 + 0x + 1 = 0 -> no real roots
    comptime T = SIMD[DType.float32, 2]
    a = T(1.0, 1.0)
    b = T(-5.0, 0.0)
    c = T(6.0, 1.0)

    res = solve_quadratic(a, b, c)

    # Lane 0
    assert_true(res.mask[0])
    assert_almost_equal(res.roots_0[0], 2.0)
    assert_almost_equal(res.roots_1[0], 3.0)

    # Lane 1
    assert_false(res.mask[1])


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
