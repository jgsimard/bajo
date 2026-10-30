"""Host boundary errors must occur before launching any GPU traversal."""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
    assert_raises,
)
from max.gpu.host import DeviceContext

from bajo.core.camera import Camera
from bajo.bvh.types import Hit
from bajo.bvh.gpu.camera_launch import validate_camera_launch
from bajo.bvh.gpu.ray_launch import validate_ray_launch
from bajo.bvh.gpu.validate import validate_sorted_keys
from bajo.core import Rayf32


def test_ray_launch_rejects_invalid_counts_and_short_records() raises:
    with DeviceContext() as ctx:
        var rays = ctx.enqueue_create_buffer[.float32](Rayf32.STRIDE)
        var hits = ctx.enqueue_create_buffer[.float32](Hit.STRIDE)
        validate_ray_launch(rays, hits, 1)
        for count in [-1, 0, 2, Int.MAX]:
            with assert_raises():
                validate_ray_launch(rays, hits, count)
        var short_rays = ctx.enqueue_create_buffer[.float32](Rayf32.STRIDE - 1)
        var short_hits = ctx.enqueue_create_buffer[.float32](Hit.STRIDE - 1)
        with assert_raises():
            validate_ray_launch(short_rays, hits, 1)
        with assert_raises():
            validate_ray_launch(rays, short_hits, 1)


def test_camera_launch_rejects_invalid_shapes_and_partial_records() raises:
    with DeviceContext() as ctx:
        var camera = ctx.enqueue_create_buffer[.float32](Camera.STRIDE)
        var hits = ctx.enqueue_create_buffer[.float32](2 * Hit.STRIDE)
        validate_camera_launch(camera, hits, 2, 2, 1)
        for bad in [-1, 0, Int.MAX]:
            with assert_raises():
                validate_camera_launch(camera, hits, bad, 2, 1)
            with assert_raises():
                validate_camera_launch(camera, hits, 2, bad, 1)
            with assert_raises():
                validate_camera_launch(camera, hits, 2, 2, bad)
        # A second view is needed when there are more rays than pixels per view.
        with assert_raises():
            validate_camera_launch(camera, hits, 2, 1, 1)
        var short_camera = ctx.enqueue_create_buffer[.float32](
            Camera.STRIDE - 1
        )
        var short_hits = ctx.enqueue_create_buffer[.float32](2 * Hit.STRIDE - 1)
        with assert_raises():
            validate_camera_launch(short_camera, hits, 2, 2, 1)
        with assert_raises():
            validate_camera_launch(camera, short_hits, 2, 2, 1)


def test_sort_diagnostics_check_every_id_and_its_uniqueness() raises:
    with DeviceContext() as ctx:
        var size = 2049
        var keys = ctx.enqueue_create_buffer[.uint32](size)
        var ids = ctx.enqueue_create_buffer[.uint32](size)
        with keys.map_to_host() as k, ids.map_to_host() as v:
            for i in range(size):
                k[i] = UInt32(i)
                v[i] = UInt32(i)
        assert_true(validate_sorted_keys(keys, ids, size).values_ok)
        with ids.map_to_host() as v:
            v[1] = UInt32(size)
        var invalid = validate_sorted_keys(keys, ids, size)
        assert_false(invalid.values_ok)
        assert_equal(invalid.first_bad_value, 1)
        with ids.map_to_host() as v:
            v[1] = 0
        invalid = validate_sorted_keys(keys, ids, size)
        assert_false(invalid.values_ok)
        assert_equal(invalid.first_bad_value, 1)
        for bad_size in [-1, size + 1]:
            with assert_raises():
                _ = validate_sorted_keys(keys, ids, bad_size)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
