from test.helpers import assert_vec_equal

from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_true,
    assert_false,
    assert_almost_equal,
)

from bajo.core import (
    AABB,
    Quat,
    Vec3W,
    Affine3,
    Point3W,
    Point3f32,
    Point3,
    AxisAlignedBoundingBox,
)
from bajo.core.utils import degrees_to_radians


def test_logic() raises:
    var box = AABB[.WORLD](Point3W(0), Point3W(2, 2, 2))

    assert_true(box.contains_point(Point3W(1)))
    assert_false(box.contains_point(Point3W(3, 1, 1)))

    var box_overlap = AABB[.WORLD](Point3W(1), Point3W(3))
    var box_far = AABB[.WORLD](Point3W(4), Point3W(5))
    assert_true(box.overlaps(box_overlap))
    assert_false(box.overlaps(box_far))


def test_merge() raises:
    var a = AABB[.WORLD](Point3W(0), Point3W(1))
    var b = AABB[.WORLD](Point3W(-1), Point3W(0.5))
    var merged = AABB.merge(a, b)
    assert_vec_equal(merged._min, Point3W(-1))
    assert_vec_equal(merged._max, Point3W(1))


def test_apply_trs_rotated() raises:
    var box = AABB[.WORLD](Point3W(-1), Point3W(1))

    # Rotate 45 degrees around Z
    var angle = degrees_to_radians(Float32(45))
    var r = Quat.from_axis_angle(Vec3W(0, 0, 1), angle)
    var t = Vec3W(0)
    var s = Vec3W(1)
    var transform = Affine3[
        .float32, .WORLD, .CAMERA
    ].from_rotation_scale_translation(r, s, t)

    var new_box = box.apply_transform(transform)

    # the corners should move to ±sqrt(2)
    var sqrt_2 = Float32(sqrt(2.0))
    assert_vec_equal(new_box._min, Point3f32[.CAMERA](-sqrt_2, -sqrt_2, -1.0))
    assert_vec_equal(new_box._max, Point3f32[.CAMERA](sqrt_2, sqrt_2, 1.0))


def test_aabb_load_store6_with_nonzero_base() raises:
    var data = List[Float32](length=18, fill=-1.0)
    var b = AABB[.WORLD](Point3W(1.0, 2.0, 3.0), Point3W(4.0, 5.0, 6.0))

    b.store6(data, 6)

    assert_almost_equal(data[6], 1.0)
    assert_almost_equal(data[7], 2.0)
    assert_almost_equal(data[8], 3.0)
    assert_almost_equal(data[9], 4.0)
    assert_almost_equal(data[10], 5.0)
    assert_almost_equal(data[11], 6.0)

    var checked = AABB[.WORLD].load6(data, 6)
    var pointer = AABB[.WORLD].load6(data.unsafe_ptr(), 6)
    var readonly = data.unsafe_ptr().unsafe_mut_cast[False]()
    var offset = AABB[.WORLD].load6(readonly.unsafe_offset(6))
    assert_vec_equal(checked._min, b._min)
    assert_vec_equal(checked._max, b._max)
    assert_vec_equal(pointer._min, b._min)
    assert_vec_equal(pointer._max, b._max)
    assert_vec_equal(offset._min, b._min)
    assert_vec_equal(offset._max, b._max)


def test_distance2_scalar_and_packets_at_extreme_scales() raises:
    for scale in List[Float32]([Float32(1e-30), Float32(1.0), Float32(1e30)]):
        var box = AABB[.WORLD](Point3W(0.0), Point3W(scale))
        var p = Point3W(-scale, scale * 0.5, scale * 2.0)
        var scale2 = Float64(scale) * Float64(scale)
        assert_almost_equal(box.distance2(p) / scale2, Float64(2.0), atol=1e-12)
        assert_almost_equal(
            box.distance2[True](p) / scale2, Float64(8.25), atol=1e-12
        )
        var boxes = AxisAlignedBoundingBox[.float32, .WORLD, 2](
            Point3[.float32, .WORLD, 2](0.0),
            Point3[.float32, .WORLD, 2](scale),
        )
        var nearest = boxes.distance2(p) / scale2
        var furthest = boxes.distance2[True](p) / scale2
        comptime for lane in range(2):
            assert_almost_equal(nearest[lane], Float64(2.0), atol=1e-12)
            assert_almost_equal(furthest[lane], Float64(8.25), atol=1e-12)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
