from std.math import max

from bajo.core import AABB, Frame, Vec3f32
from bajo.bvh.camera import Camera
from bajo.bvh.types import Ray


def make_camera_rays_and_params(
    bounds: AABB[Frame.WORLD],
    width: Int,
    height: Int,
    views: Int,
    fov_scale: Float32 = 0.75,
) -> Tuple[List[Ray[Frame.WORLD]], List[Float32]]:
    var center = bounds.centroid()
    var extent = bounds.extent()

    var scene_w = max(max(extent.x, extent.y), extent.z)
    if scene_w < 1.0:
        scene_w = 1.0

    var rays = List[Ray[Frame.WORLD]](capacity=width * height * views)
    var params = List[Float32](capacity=views * Camera.STRIDE)

    for view in range(views):
        var view_offset = Float32(view) - Float32(views - 1) * 0.5
        var eye = center + Vec3f32[Frame.WORLD](
            view_offset * scene_w * 0.30,
            extent.y * 0.20,
            -scene_w * 2.50,
        )
        var camera = Camera(
            eye,
            center,
            Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
            fov_scale,
        )
        params.extend(camera.flatten())

        for py in range(height):
            for px in range(width):
                rays.append(camera.make_ray(px, py, width, height))

    return (rays^, params^)
