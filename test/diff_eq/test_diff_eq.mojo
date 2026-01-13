from testing import TestSuite, assert_equal, assert_almost_equal
from math import exp
from layout.layout_tensor import LayoutTensor
from layout.layout import Layout

from bajo.diff_eq import ODEProblem, solve

comptime dtype = DType.float64
comptime decay_layout = Layout.row_major(1)
comptime LT = LayoutTensor[dtype, decay_layout, MutAnyOrigin]


fn decay_system(mut dy: LT, y: LT, t: Scalar[dtype]):
    dy[0] = -y[0]


fn test_correctness() raises:
    var u0 = LT.stack_allocation()
    u0[0] = 1.0

    comptime t_end = 2.0
    var dt = 0.01
    var prob = ODEProblem[decay_system](u0, 0.0, t_end, dt)

    var result = solve(prob, dt=dt)
    var expected = exp(-t_end)

    assert_almost_equal(result[0], expected)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
