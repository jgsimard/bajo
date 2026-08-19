from std.testing import TestSuite, assert_almost_equal

from bajo.core import Frame, Rayf32, Point3W, Vec3W, assert_vec_equal


def test_load_packed_ray_span() raises:
    var data: List[Float32] = [
        -1.0,
        -2.0,
        -3.0,
        0.1,
        1.0,
        0.0,
        0.0,
        4.0,
        2.0,
        3.0,
        4.0,
        0.25,
        0.0,
        1.0,
        0.0,
        8.0,
    ]

    var ray = Rayf32[Frame.WORLD](data, 1)

    assert_vec_equal(ray.o, Point3W(2.0, 3.0, 4.0))
    assert_almost_equal(ray.t_min, 0.25)
    assert_vec_equal(ray.d, Vec3W(0.0, 1.0, 0.0))
    assert_almost_equal(ray.t_max, 8.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
