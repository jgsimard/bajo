"""Collect module failures while allowing the remaining tests to run."""

from std.sys import is_defined

# Module entry points use an unused compile-time parameter so that
# __functions_in_module() excludes the entry point from its own discovery.


def module_selected[name: StaticString]() -> Bool:
    return (
        not is_defined["BAJO_TEST_FILTERED"]()
        or is_defined["BAJO_TEST_" + name]()
    )


@explicit_destroy("Call tests^.finish() to report all test results")
struct TestRunner(Deinitable where False):
    var modules: Int
    var failures: Int

    def __init__(out self):
        self.modules = 0
        self.failures = 0

    def run[Fn: def() raises -> None](mut self, name: String, test: Fn):
        self.modules += 1
        print("\nRunning:", name)
        try:
            test()
        except error:
            self.failures += 1
            print("FAIL:", name, error)

    def finish(deinit self) raises:
        print("\nRan", self.modules, "test modules;", self.failures, "failed.")
        if self.modules == 0:
            raise Error("No Mojo test modules selected")
        if self.failures > 0:
            raise Error("Mojo tests failed")
