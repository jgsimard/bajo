from testing import (
    TestSuite,
    assert_almost_equal,
)
from bajo.core.mat import Mat3f, Mat3x4f
from bajo.core.vec import (
    Vec3f32,
    assert_vec_equal,
)
from bajo.core.conversion import degrees_to_radians
from bajo.core.quat import Quat


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
