"""Run the viewer Mojo tests in one process."""

from test.runner import TestRunner, module_selected

import test.viewer.test_core as test_core
import test.viewer.test_batching as test_batching
import test.viewer.test_native as test_native


def main() raises:
    var tests = TestRunner()
    comptime if module_selected["test_core"]():
        tests.run("test/viewer/test_core.mojo", test_core.run_tests[True])
    comptime if module_selected["test_native"]():
        tests.run("test/viewer/test_native.mojo", test_native.run_tests[True])
    comptime if module_selected["test_batching"]():
        tests.run(
            "test/viewer/test_batching.mojo", test_batching.run_tests[True]
        )
    tests^.finish()
