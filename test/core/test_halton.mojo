"""Radical-inverse arithmetic preserves digits across the UInt32 boundary."""
from std.testing import TestSuite, assert_equal
from bajo.core.random import _HALTON_PRIMES, _radical_inverse


def _reference(index: UInt64, base: UInt64) -> Float32:
    var value = Float32(0)
    var inverse = Float32(1) / Float32(base)
    var factor = inverse
    var digits = index
    while digits > 0:
        value += Float32(digits % base) * factor
        digits /= base
        factor *= inverse
    return value


def test_halton_prime_dimensions_and_single_digit_boundary() raises:
    var primes = materialize[_HALTON_PRIMES]()
    for prime in primes:
        var base = UInt64(prime)
        for index in range(1024):
            assert_equal(
                _radical_inverse(UInt64(index), base),
                _reference(UInt64(index), base),
            )
        assert_equal(
            _radical_inverse(base - 1, base), _reference(base - 1, base)
        )
        assert_equal(_radical_inverse(base, base), _reference(base, base))
        assert_equal(
            _radical_inverse(base * base + 1, base),
            _reference(base * base + 1, base),
        )


def test_halton_large_index_fallback() raises:
    var indices: List[UInt64] = [
        UInt64(UInt32.MAX) - 1,
        UInt64(UInt32.MAX),
        UInt64(UInt32.MAX) + 1,
        UInt64.MAX,
    ]
    var primes = materialize[_HALTON_PRIMES]()
    for index in indices:
        for prime in primes:
            var base = UInt64(prime)
            assert_equal(_radical_inverse(index, base), _reference(index, base))
    var large_base = UInt64(UInt32.MAX) + 32
    assert_equal(
        _radical_inverse(UInt64.MAX, large_base),
        _reference(UInt64.MAX, large_base),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
