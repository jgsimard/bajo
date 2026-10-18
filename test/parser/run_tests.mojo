"""Run the parser Mojo tests in one process."""

from test.runner import TestRunner, module_selected

import test.parser.exr.test_exr as exr_test_exr
import test.parser.obj.test_obj as obj_test_obj
import test.parser.pbrt.test_bmw as pbrt_test_bmw
import test.parser.pbrt.test_crown as pbrt_test_crown
import test.parser.pbrt.test_pbrt as pbrt_test_pbrt
import test.parser.ply.test_ply as ply_test_ply
import test.parser.png.test_png as png_test_png
import test.parser.test_deflate as test_deflate


def main() raises:
    var tests = TestRunner()
    comptime if module_selected["exr_test_exr"]():
        tests.run("test/parser/exr/test_exr.mojo", exr_test_exr.run_tests[True])
    comptime if module_selected["obj_test_obj"]():
        tests.run("test/parser/obj/test_obj.mojo", obj_test_obj.run_tests[True])
    comptime if module_selected["pbrt_test_bmw"]():
        tests.run(
            "test/parser/pbrt/test_bmw.mojo", pbrt_test_bmw.run_tests[True]
        )
    comptime if module_selected["pbrt_test_crown"]():
        tests.run(
            "test/parser/pbrt/test_crown.mojo", pbrt_test_crown.run_tests[True]
        )
    comptime if module_selected["pbrt_test_pbrt"]():
        tests.run(
            "test/parser/pbrt/test_pbrt.mojo", pbrt_test_pbrt.run_tests[True]
        )
    comptime if module_selected["ply_test_ply"]():
        tests.run("test/parser/ply/test_ply.mojo", ply_test_ply.run_tests[True])
    comptime if module_selected["png_test_png"]():
        tests.run("test/parser/png/test_png.mojo", png_test_png.run_tests[True])
    comptime if module_selected["test_deflate"]():
        tests.run("test/parser/test_deflate.mojo", test_deflate.run_tests[True])
    tests^.finish()
