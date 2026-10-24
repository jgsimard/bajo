"""Sampler IDs are checked when they enter the typed sampling API."""

from std.testing import TestSuite, assert_equal, assert_raises
from bajo.core.random import Sampler


def test_sampler_sequence_ids() raises:
    var cases = [
        Sampler.INDEPENDENT,
        Sampler.HALTON,
        Sampler.R2,
        Sampler.OWEN_SOBOL,
        Sampler.SZ,
        Sampler.STBN,
    ]
    for id in range(len(cases)):
        assert_equal(Sampler(id), cases[id])
        assert_equal(Sampler(UInt32(id)), cases[id])
        assert_equal(Int(cases[id]), id)


def test_sampler_rejects_unknown_sequence_ids() raises:
    for id in [Int.MIN, -1, 6, Int.MAX]:
        with assert_raises():
            _ = Sampler(id)
    for id in [UInt32(6), UInt32.MAX]:
        with assert_raises():
            _ = Sampler(id)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
