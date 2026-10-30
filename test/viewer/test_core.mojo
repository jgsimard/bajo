"""Native scene loading, shared camera state, and ownership transitions."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)
from bajo.core.aabb import compute_bounds
from bajo.core import Affine3f32, Point3f32, Point3W, Vec3f32
from bajo.core.random import Sampler
from bajo.parser.asset_loader import MemoryAssetLoader
from bajo.rt.material_types import Integrator
from bajo.core.color import Color
from bajo.rt.render_types import RenderSettings
from bajo.rt.scene_data import SceneBuilder
from bajo.rt.scene_description import SceneDescription
from bajo.viewer import (
    ViewerScene,
    load_viewer_scene,
    viewer_camera,
    viewer_scene_index,
)


def test_camera_pose_matches_viewer_navigation_and_projection() raises:
    var camera = viewer_camera(Point3W(0, 1, 3), 0, 0, 90)
    assert_equal(camera.origin.y[0], Float32(1))
    assert_equal(camera.forward.z[0], Float32(-1))
    assert_equal(camera.right.x[0], Float32(1))
    assert_equal(camera.up.y[0], Float32(1))
    var center_ray = camera.make_ray(1, 1, 3, 3)
    assert_equal(center_ray.d.x[0], Float32(0))
    assert_equal(center_ray.d.y[0], Float32(0))
    assert_equal(center_ray.d.z[0], Float32(-1))
    assert_almost_equal(camera.fov_scale, Float32(1))
    var turned = viewer_camera(Point3W(0), 90, 0, 28)
    assert_almost_equal(turned.forward.x[0], Float32(1))
    assert_almost_equal(turned.forward.z[0], Float32(0), atol=1e-6)
    var elevated = viewer_camera(Point3W(0), 0, 30, 28)
    assert_almost_equal(elevated.forward.y[0], Float32(0.5))


def test_builtin_scene_loads_camera_and_render_settings() raises:
    var scene = load_viewer_scene(viewer_scene_index("cornell"))
    assert_true(len(scene.data.triangle_vertices()) > 0)
    assert_equal(scene.camera.origin.y[0], Float32(1))
    assert_equal(scene.camera.origin.z[0], Float32(3.2))
    assert_equal(scene.settings.image_width, 320)
    assert_equal(scene.settings.image_height, 214)
    assert_true(scene.integrator == Integrator.PATH)
    scene.settings.validate()
    with assert_raises():
        _ = viewer_scene_index("missing")
    with assert_raises():
        _ = load_viewer_scene(99)
    with assert_raises():
        _ = load_viewer_scene(9)


def test_pbrt_loading_preserves_authored_camera_and_settings() raises:
    var loader = MemoryAssetLoader()
    loader.add_file(
        "scene/native.pbrt",
        (
            "LookAt 2 3 4  2 3 0  0 1 0\n"
            'Camera "perspective" "float fov" [40]\n'
            'Film "rgb" "integer xresolution" [24] "integer yresolution" [16]\n'
            'Sampler "halton" "integer pixelsamples" [7]\n'
            'Integrator "path" "integer maxdepth" [4]\n'
            'WorldBegin\nShape "sphere"\nWorldEnd\n'
        ),
    )
    var scene = ViewerScene(load_viewer_scene(9, "scene/native.pbrt", loader))
    assert_false(scene.has_cpu_scene())
    assert_equal(len(scene.scene_data().spheres()), 1)
    assert_equal(scene.state.camera.origin.x[0], Float32(2))
    assert_equal(scene.state.camera.origin.y[0], Float32(3))
    assert_equal(scene.state.camera.origin.z[0], Float32(4))
    assert_equal(scene.state.camera.forward.z[0], Float32(-1))
    assert_equal(scene.state.settings.image_width, 24)
    assert_equal(scene.state.settings.image_height, 16)
    assert_equal(scene.state.settings.samples_per_pixel, 7)
    assert_equal(scene.state.settings.max_depth, 4)
    assert_true(scene.state.settings.sampler == Sampler.HALTON)
    scene.state.settings.validate()


def test_cpu_preparation_moves_geometry_and_retains_acceleration() raises:
    var scene = ViewerScene(load_viewer_scene(1))
    var geometry = Int(scene.scene_data().triangle_vertices().unsafe_ptr())
    assert_false(scene.has_cpu_scene())
    with assert_raises():
        _ = len(scene.cpu_scene().scene_data().triangle_vertices())
    scene.prepare_cpu()
    assert_true(scene.has_cpu_scene())
    assert_equal(
        Int(scene.scene_data().triangle_vertices().unsafe_ptr()), geometry
    )
    var nodes = Int(scene.cpu_scene().triangle_bvh.value().nodes.unsafe_ptr())
    scene.state.camera.origin = Point3W(1, 2, 3)
    scene.state.settings.max_depth = 5
    scene.prepare_cpu()
    assert_equal(
        Int(scene.cpu_scene().triangle_bvh.value().nodes.unsafe_ptr()), nodes
    )
    assert_equal(
        Int(scene.scene_data().triangle_vertices().unsafe_ptr()), geometry
    )
    with assert_raises():
        scene.prepare_cpu[.LBVH]()
    assert_equal(
        Int(scene.cpu_scene().triangle_bvh.value().nodes.unsafe_ptr()), nodes
    )


def test_native_owner_preserves_shared_meshes_and_instance_transforms() raises:
    var builder = SceneBuilder()
    var material = builder.add_lambertian(Color(0.5))
    var vertices: List[Point3f32[.LOCAL]] = [
        Point3f32[.LOCAL](-1, 0, 0),
        Point3f32[.LOCAL](1, 0, 0),
        Point3f32[.LOCAL](0, 1, 0),
    ]
    var bounds = compute_bounds(vertices)
    var mesh = builder.add_owned_triangle_mesh(
        vertices^, List[Float32](), List[Float32]()
    )
    var transform = Affine3f32[.LOCAL, .WORLD].from_scale(
        Vec3f32[.LOCAL](1, 2, 1)
    )
    transform.tx = -2
    builder.add_triangle_instance(mesh, transform, bounds, material)
    transform.m00 = -1
    transform.tx = 2
    builder.add_triangle_instance(mesh, transform, bounds, material)
    var scene = ViewerScene[4, 4](
        SceneDescription(
            builder^.finish(),
            viewer_camera(Point3W(0, 1, 5), 0, 0, 45),
            RenderSettings(8, 8, 1, UInt64(1234)),
            .PATH,
        )
    )
    var storage = Int(scene.scene_data().triangle_meshes()[0].unsafe_ptr())
    scene.prepare_cpu()
    assert_equal(len(scene.scene_data().triangle_meshes()), 1)
    assert_equal(len(scene.scene_data().triangle_instances()), 2)
    assert_equal(
        Int(scene.scene_data().triangle_meshes()[0].unsafe_ptr()), storage
    )
    ref instances = scene.scene_data().triangle_instances()
    assert_equal(instances[0].blas_idx, instances[1].blas_idx)
    assert_equal(instances[0].transform.tx[0], Float32(-2))
    assert_equal(instances[1].transform.tx[0], Float32(2))
    assert_true(instances[1].transform.reverses_orientation()[0])


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def main() raises:
    run_tests()
