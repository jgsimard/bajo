"""Stage accounting, estimates for uneven batches, and display formatting."""
from std.testing import TestSuite, assert_equal, assert_true
from bajo.viewer.timing import ViewerTimings, format_duration


def test_stage_accounting_and_refresh_do_not_count_samples_twice() raises:
    var t = ViewerTimings(100, 200)
    assert_equal(t.expected_total_ns(0, 0, 10, 4), -1)
    t.begin_request(1000)
    assert_equal(t.total_ns(1050), 350)
    t.finish_request(50, 400, 20, 480, 4)
    assert_equal(t.preparation_ns, 250)
    assert_equal(t.total_ns(0), 780)
    # Six remaining samples: 600 tracing + 2 presentation operations.
    assert_equal(t.expected_total_ns(0, 4, 10, 4), 1420)
    assert_equal(t.expected_total_ns(0, 4, 10, -1), 1400)
    t.begin_request(2000)
    t.finish_request(0, 0, 30, 30, 0)
    assert_equal(t.batch_ns, 400)
    assert_equal(t.traced_samples, 4)
    assert_equal(t.postprocess_ns, 30)
    assert_equal(t.total_ns(0), 810)
    assert_equal(t.expected_total_ns(0, 4, 10, 4), 1470)
    assert_equal(t.expected_total_ns(0, 10, 10, 4), 810)


def test_estimate_inflight_reset_and_stale_generation() raises:
    var t = ViewerTimings(100, 200)
    t.finish_request(0, 100, 10, 110, 1)
    t.begin_request(1000)
    assert_equal(t.expected_total_ns(1100, 1, 2, 1), 520)
    assert_true(t.expected_total_ns(2000, 1, 2, 1) >= t.total_ns(2000))
    t.reset_render()
    assert_equal(t.total_ns(5000), 300)
    assert_equal(t.expected_total_ns(5000, 0, 2, 1), -1)
    # A stale batch can still have prepared reusable scene acceleration.
    t.finish_request(40, 1000, 20, 1060, 1, False)
    assert_equal(t.preparation_ns, 240)
    assert_equal(t.total_ns(0), 340)
    assert_equal(t.traced_samples, 0)


def test_timing_labels_wrap_and_format_long_durations() raises:
    assert_equal(format_duration(-1), String("--"))
    assert_equal(format_duration(1500000000), String("1.5 s"))
    assert_equal(format_duration(61000000000), String("1m 1s"))
    assert_equal(format_duration(3660000000000), String("1h 1m"))
    var t = ViewerTimings()
    assert_equal(len(t.lines(960, 0, 0, 10, 1)), 1)
    assert_equal(len(t.lines(640, 0, 0, 10, 1)), 2)
    assert_equal(len(t.lines(320, 0, 0, 10, 1)), 3)
    assert_true("Est. total --" in t.lines(960, 0, 0, 10, 1)[0])


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
