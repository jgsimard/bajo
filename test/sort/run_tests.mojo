"""Run the sort Mojo tests in one process."""

from test.runner import TestRunner, module_selected

import test.sort.cpu.test_validation as cpu_test_validation
import test.sort.gpu.test_bitonic_sort as gpu_test_bitonic_sort
import test.sort.gpu.test_decoupled_fallback as gpu_test_decoupled_fallback
import test.sort.gpu.test_onesweep as gpu_test_onesweep
import test.sort.gpu.test_radix_sort as gpu_test_radix_sort
import test.sort.gpu.test_sort_contracts as gpu_test_sort_contracts


def main() raises:
    var tests = TestRunner()
    comptime if module_selected["cpu_test_validation"]():
        tests.run(
            "test/sort/cpu/test_validation.mojo",
            cpu_test_validation.run_tests[True],
        )
    comptime if module_selected["gpu_test_bitonic_sort"]():
        tests.run(
            "test/sort/gpu/test_bitonic_sort.mojo",
            gpu_test_bitonic_sort.run_tests[True],
        )
    comptime if module_selected["gpu_test_decoupled_fallback"]():
        tests.run(
            "test/sort/gpu/test_decoupled_fallback.mojo",
            gpu_test_decoupled_fallback.run_tests[True],
        )
    comptime if module_selected["gpu_test_onesweep"]():
        tests.run(
            "test/sort/gpu/test_onesweep.mojo",
            gpu_test_onesweep.run_tests[True],
        )
    comptime if module_selected["gpu_test_radix_sort"]():
        tests.run(
            "test/sort/gpu/test_radix_sort.mojo",
            gpu_test_radix_sort.run_tests[True],
        )
    comptime if module_selected["gpu_test_sort_contracts"]():
        tests.run(
            "test/sort/gpu/test_sort_contracts.mojo",
            gpu_test_sort_contracts.run_tests[True],
        )
    tests^.finish()
