"""Headless checks for the native viewer's scene conversion and worker."""

from std.ffi import external_call, c_int
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
)
from bajo.bvh import Camera
from bajo.core import Affine3f32, Point3W, Vec3W, Vec3f32, cross, dot
from bajo.parser.asset_loader import MemoryAssetLoader
from bajo.viewer.core import ViewerScene, viewer_camera
from bajo.viewer.scenes import load_viewer_scene
from bajo.viewer.raster import (
    raster_camera,
    raster_view_matrix,
    instance_matrix,
)
from bajo.viewer.native import rotate_camera
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


def _assert_raster_projection_matches_camera(camera: Camera) raises:
    var view = raster_view_matrix(camera)
    for y in range(0, 12, 5):
        for x in range(0, 16, 7):
            var ray = camera.make_ray(x, y, 16, 12)
            var point = ray.o + ray.d * 3
            var vx = (
                view.m0 * point.x
                + view.m4 * point.y
                + view.m8 * point.z
                + view.m12
            )
            var vy = (
                view.m1 * point.x
                + view.m5 * point.y
                + view.m9 * point.z
                + view.m13
            )
            var vz = (
                view.m2 * point.x
                + view.m6 * point.y
                + view.m10 * point.z
                + view.m14
            )
            assert_true(vz < 0)
            assert_almost_equal(
                vx / (-vz * camera.fov_scale * (Float32(16) / 12)),
                2 * (Float32(x) + 0.5) / 16 - 1,
                atol=1e-5,
            )
            assert_almost_equal(
                vy / (-vz * camera.fov_scale),
                1 - 2 * (Float32(y) + 0.5) / 12,
                atol=1e-5,
            )


def test_raster_projection_matches_builtin_and_pbrt_camera_rays() raises:
    var origin = Point3W(-11, 0.8, 5)
    var target = Point3W(-2, -0.5, 0)
    var camera = Camera.from_vfov(origin, target, Vec3W(0, 1, 0), 30)
    _assert_raster_projection_matches_camera(camera)
    var loader = MemoryAssetLoader()
    loader.add_file(
        "bmw-camera.pbrt",
        (
            "LookAt -11 0.8 5  -2 -0.5 0  0 1 0\n"
            'Camera "perspective" "float fov" [30]\n'
            'WorldBegin\nShape "sphere"\nWorldEnd\n'
        ),
    )
    var scene = load_viewer_scene(9, "bmw-camera.pbrt", loader)
    assert_true(dot(scene.camera.right, camera.right) < 0)
    _assert_raster_projection_matches_camera(scene.camera)
    # Imported camera transforms can also scale or shear the basis.
    scene.camera.right = scene.camera.right * 2
    scene.camera.up += scene.camera.right * 0.2
    _assert_raster_projection_matches_camera(scene.camera)


def test_rotation_preserves_camera_handedness_and_pitch_direction() raises:
    for reflected in range(2):
        var camera = viewer_camera(Point3W(2, 3, 4), 20, 10, 60)
        if reflected:
            camera.right = -camera.right
        var up = camera.up
        var right = camera.right
        camera.defocus_disk_u = camera.right * 0.2
        camera.defocus_disk_v = camera.up * 0.3
        rotate_camera(camera, 0, 0)
        assert_almost_equal(dot(camera.up, up), Float32(1), atol=1e-6)
        assert_almost_equal(dot(camera.right, right), Float32(1), atol=1e-6)
        var previous_y = camera.forward.y
        rotate_camera(camera, 5, 10)
        assert_true(camera.forward.y > previous_y)
        var handedness = dot(cross(camera.right, camera.forward), camera.up)
        assert_true(handedness < 0 if reflected else handedness > 0)
        assert_almost_equal(
            dot(camera.up, camera.forward), Float32(0), atol=1e-6
        )
        assert_almost_equal(
            dot(camera.defocus_disk_u, camera.defocus_disk_u),
            Float32(0.04),
            atol=1e-6,
        )
        assert_almost_equal(
            dot(camera.defocus_disk_v, camera.defocus_disk_v),
            Float32(0.09),
            atol=1e-6,
        )


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
