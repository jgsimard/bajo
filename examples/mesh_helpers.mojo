from std.math import max
from bajo.core import AABB, Affine3f32, Quat, Vec3f32


def normalized_instance_scale(
    bounds: AABB[.LOCAL],
    target_extent: Float32,
    variation: Float32,
) -> Vec3f32[.LOCAL]:
    var extent = bounds.extent()
    var local_extent = max(max(extent.x, extent.y), extent.z)
    if local_extent < Float32(1.0e-6):
        local_extent = Float32(1.0)

    return Vec3f32[.LOCAL](target_extent / local_extent * variation)


def bottom_centered_transform(
    bounds: AABB[.LOCAL],
    rotation: Quat,
    scale: Vec3f32[.LOCAL],
    bottom_center: Vec3f32[.WORLD],
) -> Affine3f32[.LOCAL, .WORLD]:
    var transform = Affine3f32[.LOCAL, .WORLD].from_rotation_scale(
        rotation, scale
    )
    var c = bounds.centroid()
    var local_anchor = Vec3f32[.LOCAL](c.x, bounds._min.y, c.z)
    var anchor_delta = transform.vector(local_anchor)
    transform.tx = bottom_center.x - anchor_delta.x
    transform.ty = bottom_center.y - anchor_delta.y
    transform.tz = bottom_center.z - anchor_delta.z
    return transform^
