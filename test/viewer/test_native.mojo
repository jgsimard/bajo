"""Headless checks for the native viewer's scene conversion and worker."""

from std.ffi import external_call, c_int
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
)
from bajo.core import Affine3f32, Point3W, Vec3f32
from bajo.viewer.core import ViewerScene, viewer_camera
from bajo.viewer.scenes import load_viewer_scene
from bajo.viewer.raster import raster_camera, instance_matrix
from bajo.viewer.debug import object_boxes
from bajo.viewer.worker import RenderWorker


def test_raster_camera_and_mirrored_instance_keep_scene_coordinates() raises:
    var camera = viewer_camera(Point3W(2, 3, 4), 90, 20, 60)
    var raster = raster_camera(camera)
    assert_equal(raster.position.x, Float32(2))
    assert_almost_equal(raster.target.x - raster.position.x, camera.forward.x)
    assert_almost_equal(raster.target.y - raster.position.y, camera.forward.y)
    assert_almost_equal(raster.up.z, camera.up.z)
    assert_almost_equal(raster.fovy, Float32(60))
    var affine = Affine3f32[.LOCAL, .WORLD].from_scale(
        Vec3f32[.LOCAL](-2, 3, 4)
    )
    affine.tx = 5
    affine.ty = 6
    affine.tz = 7
    var matrix = instance_matrix(affine)
    assert_equal(matrix.m0, Float32(-2))
    assert_equal(matrix.m5, Float32(3))
    assert_equal(matrix.m10, Float32(4))
    assert_equal(matrix.m12, Float32(5))
    assert_equal(matrix.m13, Float32(6))
    assert_equal(matrix.m14, Float32(7))
    assert_equal(matrix.m15, Float32(1))


def _wait(worker: RenderWorker) raises:
    for _ in range(10000):
        if worker.ready():
            return
        _ = external_call["usleep", c_int](UInt32(1000))
    raise Error("native render worker did not complete")


def test_worker_publishes_frames_bounds_and_reuses_cpu_scene() raises:
    var scene = ViewerScene(load_viewer_scene(1))
    var camera = scene.state.camera
    var settings = scene.state.settings.copy()
    settings.image_width = 8
    settings.image_height = 6
    settings.samples_per_pixel = 1
    settings.sample_sequence_length = 4
    settings.max_depth = 2
    var objects = object_boxes(scene.scene_data(), 3)
    assert_equal(len(objects), 3)
    var worker = RenderWorker(scene^)
    worker.submit(camera, settings, 7, 3, 0, True, True)
    _wait(worker)
    assert_equal(worker.state[].error, String(""))
    assert_true(len(worker.state[].boxes) > 0)
    var result = worker.state[].result.take()
    assert_equal(result.generation, UInt64(7))
    assert_equal(result.samples, 1)
    assert_equal(len(result.pixels), 48)
    var nodes = Int(
        worker.state[].scene.cpu_scene().triangle_bvh.value().nodes.unsafe_ptr()
    )
    worker.acknowledge()
    camera.origin.x += 0.1
    settings.sample_offset = 1
    worker.submit(camera, settings, 8, 4, 0, True, False)
    _wait(worker)
    assert_equal(worker.state[].error, String(""))
    var next_result = worker.state[].result.take()
    assert_equal(next_result.generation, UInt64(8))
    assert_equal(len(next_result.pixels), 48)
    assert_equal(
        Int(
            worker.state[]
            .scene.cpu_scene()
            .triangle_bvh.value()
            .nodes.unsafe_ptr()
        ),
        nodes,
    )
    worker.acknowledge()


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def main() raises:
    run_tests()
