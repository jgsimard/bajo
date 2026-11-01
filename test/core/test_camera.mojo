"""Camera basis, view conversion and angular motion without a viewer."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_raises,
)
from bajo.core import Camera, Point3W, Vec3W, Affine3f32, length
from test.helpers import assert_vec_equal


def test_camera_view_roundtrip_preserves_scaled_sheared_basis() raises:
    var camera = Camera.from_yaw_pitch(Point3W(2, 3, 4), 30, 20, 60, 4.5)
    camera.right = camera.right * 2
    camera.up += camera.right * 0.2
    var restored = Camera.from_world_to_camera(
        camera.world_to_camera(), 60, 4.5
    )
    assert_vec_equal(restored.origin, camera.origin)
    assert_vec_equal(restored.forward, camera.forward)
    assert_vec_equal(restored.right, camera.right)
    assert_vec_equal(restored.up, camera.up)
    assert_almost_equal(restored.vertical_fov_degrees(), Float32(60), atol=1e-5)
    assert_almost_equal(restored.focus_dist, Float32(4.5))
    with assert_raises():
        _ = Camera.from_world_to_camera(Affine3f32[.WORLD, .CAMERA](0), 60)


def test_camera_supports_positive_z_view_conventions_and_portrait_film() raises:
    var view = Affine3f32[.WORLD, .CAMERA].from_look_at(
        Point3W(1, 2, 3), Point3W(1, 2, 4), Vec3W(0, 1, 0)
    )
    var camera = Camera.from_world_to_camera(view, 60, positive_z_forward=True)
    assert_vec_equal(camera.origin, Point3W(1, 2, 3))
    assert_vec_equal(camera.forward, Vec3W(0, 0, 1))
    var old_scale = camera.fov_scale
    camera.fit_fov_to_film(4, 8)
    assert_almost_equal(camera.fov_scale, old_scale * 2)
    camera.set_lens_radius(0.02)
    assert_almost_equal(length(camera.defocus_disk_u), Float32(0.02))
    assert_almost_equal(length(camera.defocus_disk_v), Float32(0.02))


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
