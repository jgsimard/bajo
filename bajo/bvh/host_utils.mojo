from std.gpu import DeviceBuffer
from std.gpu.host import DeviceContext

from bajo.core.aabb import AABB
from bajo.core.vec import Vec3f32, vmin, vmax, cross, length, normalize
from bajo.bvh.types import Ray, RAY_FLAT_STRIDE


def compute_bounds(verts: List[Vec3f32]) -> AABB:
    var bounds = AABB.invalid()
    for i in range(len(verts)):
        bounds.grow(verts[i])
    return bounds


def flatten_vertices(verts: List[Vec3f32]) -> List[Float32]:
    var out = List[Float32](capacity=len(verts) * 3)
    for i in range(len(verts)):
        out.append(verts[i].x)
        out.append(verts[i].y)
        out.append(verts[i].z)
    return out^


def copy_list_to_device[
    dtype: DType
](mut ctx: DeviceContext, values: List[Scalar[dtype]]) raises -> DeviceBuffer[
    dtype
]:
    var buf = ctx.enqueue_create_buffer[dtype](len(values))
    with buf.map_to_host() as h:
        for i in range(len(values)):
            h[i] = values[i]
    return buf^


def flatten_rays(rays: List[Ray]) -> List[Float32]:
    var out = List[Float32](capacity=len(rays) * RAY_FLAT_STRIDE)
    for ray in rays:
        out.append(ray.o.x)
        out.append(ray.o.y)
        out.append(ray.o.z)
        out.append(ray.d.x)
        out.append(ray.d.y)
        out.append(ray.d.z)
        out.append(ray.rd.x)
        out.append(ray.rd.y)
        out.append(ray.rd.z)
        out.append(ray.t_min)
        out.append(ray.t_max)
        out.append(Float32(ray.mask))
    return out^


@always_inline
def hit_t_for_checksum(t: Float32) -> Float64:
    if t < 1.0e20:
        return Float64(t)
    return 0.0


def append_camera_rays(
    mut rays: List[Ray],
    origin: Vec3f32,
    target: Vec3f32,
    up_hint: Vec3f32,
    width: Int,
    height: Int,
    fov_scale: Float32,
):
    var forward = normalize(target - origin)
    var right = normalize(cross(forward, up_hint))
    var up = normalize(cross(right, forward))

    var aspect = Float32(width) / Float32(height)

    for y in range(height):
        for x in range(width):
            var sx = ((Float32(x) + 0.5) / Float32(width)) * 2.0 - 1.0
            var sy = 1.0 - ((Float32(y) + 0.5) / Float32(height)) * 2.0
            var dir = normalize(
                forward
                + right * (sx * aspect * fov_scale)
                + up * (sy * fov_scale)
            )
            rays.append(Ray(origin, dir))


def generate_primary_rays(
    bounds: AABB,
    width: Int,
    height: Int,
    views: Int,
    fov_scale: Float32 = 0.75,
) -> List[Ray]:
    var rays = List[Ray](capacity=width * height * views)

    var center = bounds.centroid()
    var extent = bounds.extent()
    var radius = length(extent) * 0.5
    if radius < 1.0:
        radius = 1.0
    var dist = radius * 2.8

    if views >= 1:
        append_camera_rays(
            rays,
            center + Vec3f32(0.0, 0.0, -dist),
            center,
            Vec3f32(0.0, 1.0, 0.0),
            width,
            height,
            fov_scale,
        )

    if views >= 2:
        append_camera_rays(
            rays,
            center + Vec3f32(-dist, 0.0, 0.0),
            center,
            Vec3f32(0.0, 1.0, 0.0),
            width,
            height,
            fov_scale,
        )

    if views >= 3:
        append_camera_rays(
            rays,
            center + Vec3f32(0.0, dist, 0.0),
            center,
            Vec3f32(0.0, 0.0, 1.0),
            width,
            height,
            fov_scale,
        )

    return rays^


def compute_centroid_bounds(verts: List[Vec3f32]) -> AABB:
    var centroid = AABB.invalid()

    for i in range(len(verts) / 3):
        ref v0 = verts[i * 3 + 0]
        ref v1 = verts[i * 3 + 1]
        ref v2 = verts[i * 3 + 2]
        var c = AABB.invalid()
        c.grow(v0, v1, v2)
        centroid.grow(c.centroid())

    return centroid


def append_camera_params(
    mut params: List[Float32],
    origin: Vec3f32,
    target: Vec3f32,
    up_hint: Vec3f32,
    fov: Float32,
):
    var forward = normalize(target - origin)
    var right = normalize(cross(forward, up_hint))
    var up = normalize(cross(right, forward))

    params.append(origin.x)
    params.append(origin.y)
    params.append(origin.z)
    params.append(forward.x)
    params.append(forward.y)
    params.append(forward.z)
    params.append(right.x)
    params.append(right.y)
    params.append(right.z)
    params.append(up.x)
    params.append(up.y)
    params.append(up.z)
    params.append(fov)


def generate_camera_params(
    bounds_min: Vec3f32,
    bounds_max: Vec3f32,
    views: Int,
) -> List[Float32]:
    var params = List[Float32](capacity=views * 12)

    var center = (bounds_min + bounds_max) * 0.5
    var extent = bounds_max - bounds_min
    var radius = length(extent) * 0.5
    if radius < 1.0:
        radius = 1.0
    var dist = radius * 2.8

    if views >= 1:
        append_camera_params(
            params,
            center + Vec3f32(0.0, 0.0, -dist),
            center,
            Vec3f32(0.0, 1.0, 0.0),
            0.75,
        )

    if views >= 2:
        append_camera_params(
            params,
            center + Vec3f32(-dist, 0.0, 0.0),
            center,
            Vec3f32(0.0, 1.0, 0.0),
            0.75,
        )

    if views >= 3:
        append_camera_params(
            params,
            center + Vec3f32(0.0, dist, 0.0),
            center,
            Vec3f32(0.0, 0.0, 1.0),
            0.75,
        )

    return params^
