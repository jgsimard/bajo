"""Run the core Mojo tests in one process."""

from test.runner import TestRunner, module_selected

import test.core.test_aabb as test_aabb
import test.core.test_halton as test_halton
import test.core.test_intersect as test_intersect
import test.core.test_mat as test_mat
import test.core.test_morton as test_morton
import test.core.test_quat as test_quat
import test.core.test_ray as test_ray
import test.core.test_segments as test_segments
import test.core.test_transform as test_transform
import test.core.test_vec as test_vec


def main() raises:
    var tests = TestRunner()
    comptime if module_selected["test_aabb"]():
        tests.run("test/core/test_aabb.mojo", test_aabb.run_tests[True])
    comptime if module_selected["test_halton"]():
        tests.run("test/core/test_halton.mojo", test_halton.run_tests[True])
    comptime if module_selected["test_intersect"]():
        tests.run(
            "test/core/test_intersect.mojo", test_intersect.run_tests[True]
        )
    comptime if module_selected["test_mat"]():
        tests.run("test/core/test_mat.mojo", test_mat.run_tests[True])
    comptime if module_selected["test_morton"]():
        tests.run("test/core/test_morton.mojo", test_morton.run_tests[True])
    comptime if module_selected["test_quat"]():
        tests.run("test/core/test_quat.mojo", test_quat.run_tests[True])
    comptime if module_selected["test_ray"]():
        tests.run("test/core/test_ray.mojo", test_ray.run_tests[True])
    comptime if module_selected["test_segments"]():
        tests.run("test/core/test_segments.mojo", test_segments.run_tests[True])
    comptime if module_selected["test_transform"]():
        tests.run(
            "test/core/test_transform.mojo", test_transform.run_tests[True]
        )
    comptime if module_selected["test_vec"]():
        tests.run("test/core/test_vec.mojo", test_vec.run_tests[True])
    tests^.finish()
