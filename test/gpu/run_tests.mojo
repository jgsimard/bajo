"""Run the gpu Mojo tests in one process."""

from test.runner import TestRunner, module_selected

import test.gpu.test_buffers as test_buffers


def main() raises:
    var tests = TestRunner()
    comptime if module_selected["test_buffers"]():
        tests.run("test/gpu/test_buffers.mojo", test_buffers.run_tests[True])
    tests^.finish()
