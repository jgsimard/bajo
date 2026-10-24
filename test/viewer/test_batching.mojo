"""A batch must cover each sample once and clamp the last partial batch."""

from std.testing import TestSuite, assert_equal
from bajo.viewer.batching import samples_for_batch, smaller_batch, larger_batch


def test_batches_partition_non_divisible_sample_budget() raises:
    var offset = 0
    var batches = 0
    var sizes = List[Int]()
    while offset < 10:
        var size = samples_for_batch(4, offset, 10)
        sizes.append(size)
        offset += size
        batches += 1
    assert_equal(offset, 10)
    assert_equal(batches, 3)
    assert_equal(sizes[0], 4)
    assert_equal(sizes[1], 4)
    assert_equal(sizes[2], 2)
    assert_equal(samples_for_batch(4, offset, 10), 0)
    assert_equal(samples_for_batch(100, 8, 10), 2)


def test_minus_one_renders_all_remaining_samples_even_while_moving() raises:
    assert_equal(samples_for_batch(-1, 0, 37), 37)
    assert_equal(samples_for_batch(-1, 7, 37), 30)
    assert_equal(samples_for_batch(-1, 7, 37, True, 1), 30)
    assert_equal(samples_for_batch(-1, 37, 37), 0)


def test_moving_preview_never_exceeds_fixed_batch_size_or_budget() raises:
    assert_equal(samples_for_batch(8, 0, 37, True, 2), 2)
    assert_equal(samples_for_batch(2, 0, 37, True, 8), 2)
    assert_equal(samples_for_batch(8, 36, 37, True, 2), 1)


def test_batch_controls_reach_all_and_return_to_fixed_size() raises:
    assert_equal(larger_batch(4, 10), 8)
    assert_equal(larger_batch(8, 10), 10)
    assert_equal(larger_batch(10, 10), -1)
    assert_equal(larger_batch(-1, 10), -1)
    assert_equal(smaller_batch(-1, 10), 5)
    assert_equal(smaller_batch(5, 10), 2)
    assert_equal(smaller_batch(1, 10), 1)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def main() raises:
    run_tests()
