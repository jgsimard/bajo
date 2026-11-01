"""Numeric conversions preserve enclosure even at Float32 underflow."""
from std.testing import TestSuite, assert_true
from bajo.core.numerics import round_float32_outward


def test_round_float32_outward_conversion_encloses_signed_endpoints() raises:
    for value in [
        Float64(0),
        Float64(1e-50),
        Float64(-1e-50),
        Float64(1.00000001),
        Float64(-1.00000001),
        Float64(1e30),
        Float64(-1e30),
    ]:
        assert_true(Float64(round_float32_outward(value, False)) <= value)
        assert_true(Float64(round_float32_outward(value, True)) >= value)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
