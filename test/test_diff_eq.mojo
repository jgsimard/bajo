from testing import TestSuite, assert_equal, assert_almost_equal
from math import exp
from layout.layout_tensor import LayoutTensor
from layout.layout import Layout

from src.diff_eq import ODEProblem, solve

comptime float_type = DType.float64

# 1. Define the system
comptime decay_layout = Layout.row_major(1)


fn decay_system(
    mut dy: LayoutTensor[float_type, decay_layout, MutAnyOrigin],
    y: LayoutTensor[float_type, decay_layout, MutAnyOrigin],
    t: Float64,
):
    dy[0] = -y[0]


fn test_correctness() raises:
    var u0 = LayoutTensor[
        float_type, decay_layout, MutAnyOrigin
    ].stack_allocation()
    u0[0] = 1.0

    comptime t_end = 2.0
    var prob = ODEProblem[decay_system](u0, 0.0, t_end)

    var dt = 0.01
    var result = solve(prob, dt=dt)
    var expected = exp(-t_end)

    assert_almost_equal(result[0], expected)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
