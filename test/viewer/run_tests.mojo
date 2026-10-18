"""Run the viewer Mojo tests in one process."""

from test.runner import TestRunner, module_selected

import test.viewer.test_linear_frame as test_linear_frame


def main() raises:
    var tests = TestRunner()
    comptime if module_selected["test_linear_frame"]():
        tests.run(
            "test/viewer/test_linear_frame.mojo",
            test_linear_frame.run_tests[True],
        )
    tests^.finish()
