from std.testing import TestSuite, assert_equal, assert_almost_equal
from std.math import exp

from layout.layout_tensor import LayoutTensor
from layout.layout import Layout

from bajo.diff_eq import ODEProblem, solve

comptime dtype = DType.float64
comptime decay_layout = Layout.row_major(1)


fn decay_system(
    dy: LayoutTensor[dtype, decay_layout, MutAnyOrigin],
    y: LayoutTensor[dtype, decay_layout, ImmutAnyOrigin],
    t: Scalar[dtype],
):
    dy[0] = -y[0]


def test_correctness() raises:
    u0 = LayoutTensor[dtype, decay_layout, MutAnyOrigin].stack_allocation()
    u0[0] = 1.0

    comptime t_end = 2.0
    dt = 0.01
    prob = ODEProblem[decay_system](u0, 0.0, t_end, dt)

    result = solve(prob, dt=dt)
    expected = exp(-t_end)

    assert_almost_equal(result[0], expected)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
