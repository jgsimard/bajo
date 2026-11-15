"""Run CPU and device postprocess regression suites."""
from test.runner import TestRunner, module_selected
import test.postprocess.cpu.test_atrous as cpu_test_atrous
import test.postprocess.cpu.test_guides as cpu_test_guides
import test.postprocess.gpu.test_atrous as gpu_test_atrous
import test.postprocess.gpu.test_guides as gpu_test_guides


def main() raises:
    var tests = TestRunner()
    comptime if module_selected["cpu_test_atrous"]():
        tests.run(
            "test/postprocess/cpu/test_atrous.mojo",
            cpu_test_atrous.run_tests[True],
        )
    comptime if module_selected["gpu_test_atrous"]():
        tests.run(
            "test/postprocess/gpu/test_atrous.mojo",
            gpu_test_atrous.run_tests[True],
        )
    comptime if module_selected["gpu_test_guides"]():
        tests.run(
            "test/postprocess/gpu/test_guides.mojo",
            gpu_test_guides.run_tests[True],
        )
    comptime if module_selected["cpu_test_guides"]():
        tests.run(
            "test/postprocess/cpu/test_guides.mojo",
            cpu_test_guides.run_tests[True],
        )
    tests^.finish()
