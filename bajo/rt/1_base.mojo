from math import sqrt, tan, pi, sqrt, clamp
from random import random_float64
from algorithm import parallelize
from utils import Variant
from os import abort
from std.utils.numerics import max_finite, min_finite

from bajo.bmath import (
    Vec2f,
    Vec3f,
    length,
    length2,
    normalize,
    dot,
    cross,
    deg_to_radians,
)
from bajo.color import rgb_to_srgb

comptime Point3 = Vec3f
comptime Color = Vec3f


fn main() raises:
    print(" Ray Tracing in One Weekend - Part 1")

    comptime aspect_ratio = 16.0 / 9.0
    comptime image_width = 600
    comptime n_samples = 10
    comptime max_depth = 10

    var world = HitableList(
        [Sphere(Point3(0, -100.5, -1), 100), Sphere(Point3(0, 0, -1), 0.5)]
    )

    var cam = Camera(image_width, aspect_ratio, n_samples, max_depth)
    cam.render(world)

fn linear_to_gamma(color: Color) -> Color:
    return Color(sqrt(color.data))

fn write_color(mut f: FileHandle, color: Color):
    var out_color = linear_to_gamma(color).clamp(0.0, 0.999)

    var r = out_color.x()
    var g = out_color.y()
    var b = out_color.z()

    var ir = Int(255.99 * r)
    var ig = Int(255.99 * g)
    var ib = Int(255.99 * b)

    f.write("{} {} {}\n".format(ir, ig, ib))


@fieldwise_init
@register_passable("trivial")
struct Ray(Copyable, Writable):
    var origin: Point3
    var direction: Point3

    fn at(self, t: Float32) -> Point3:
        return self.origin + t * self.direction


@fieldwise_init
@register_passable("trivial")
struct HitRecord(Copyable):
    var p: Point3
    var normal: Vec3f
    var t: Float32
    var front_face: Bool

    fn set_face_normal(mut self, r: Ray, outward_normal: Vec3f):
        # Sets the hit record normal vector.
        # NOTE: the parameter `outward_normal` is assumed to have unit length.

        var front_face = dot(r.direction, outward_normal) < 0
        self.normal = outward_normal if front_face else -outward_normal


trait Hittable(Copyable):
    fn hit(
        self, ray: Ray, ray_t: Interval[DType.float32]
    ) -> Optional[HitRecord]:
        ...


@fieldwise_init
struct Sphere(Hittable, Writable):
    var center: Point3
    var radius: Float32

    fn hit(
        self, ray: Ray, ray_t: Interval[DType.float32]
    ) -> Optional[HitRecord]:
        var oc = self.center - ray.origin
        var a = length2(ray.direction)
        var h = dot(ray.direction, oc)
        var c = length2(oc) - self.radius * self.radius

        var discriminant = h * h - a * c
        if discriminant < 0:
            return None

        var sqrtd = sqrt(discriminant)

        # Find the nearest root that lies in the acceptable range.
        var root = (h - sqrtd) / a
        if not ray_t.surrounds(root):
            root = (h + sqrtd) / a
            if not ray_t.surrounds(root):
                return None

        var t = root
        var p = ray.at(t)
        var normal = (p - self.center) / self.radius
        var rec = HitRecord(p, normal, t, False)
        var outward_normal = (rec.p - self.center) / self.radius
        rec.set_face_normal(ray, outward_normal)

        return rec


comptime Hittables = Variant[Sphere]


@fieldwise_init
struct HitableList(Hittable, Writable):
    var objects: List[Hittables]

    fn hit(
        self, ray: Ray, ray_t: Interval[DType.float32]
    ) -> Optional[HitRecord]:
        var closest_so_far = ray_t.max
        var hit_anything: Optional[HitRecord] = None

        for obj in self.objects:
            var interval = Interval(ray_t.min, closest_so_far)
            var hit_res: Optional[HitRecord]
            if obj.isa[Sphere]():
                hit_res = obj[Sphere].hit(ray, interval)
            else:
                abort()

            if hit_res:
                var hit = hit_res.value()
                closest_so_far = hit.t
                hit_anything = Optional(hit)

        return hit_anything


# IMAGE_NB==6
@fieldwise_init
struct Interval[T: DType](
    Copyable,
    Representable,
    Stringable,
    Writable,
):
    var min: Scalar[Self.T]
    var max: Scalar[Self.T]

    fn __init__(out self):
        self.min = max_finite[Self.T]()
        self.max = min_finite[Self.T]()

    fn __contains__(self, other: Scalar[Self.T]) -> Bool:
        return self.min <= other <= self.max

    fn surrounds(self, other: Scalar[Self.T]) -> Bool:
        return self.min < other < self.max

    fn size(self) -> Scalar[Self.T]:
        return self.max - self.min

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        writer.write("({},{})".format(self.min, self.max))

    fn __str__(self) -> String:
        return String.write(self)

    fn __repr__(self) -> String:
        return String("Interval", self, "")


struct Camera(Copyable):
    var aspect_ratio: Float32
    """Ratio of image width over height."""
    var image_width: Int
    """Rendered image width in pixel count."""
    var image_height: Int
    """Rendered image height."""
    var center: Point3
    """Camera center."""
    var pixel00_loc: Point3
    """Location of pixel 0, 0."""
    var pixel_delta_u: Vec3f
    """Offset to pixel to the right."""
    var pixel_delta_v: Vec3f
    """Offset to pixel below."""
    var samples_per_pixel: Int
    """Count of random samples for each pixel."""
    var max_depth: Int 
    """Maximum number of ray bounces into scene."""

    fn __init__(
        out self,
        image_width: Int,
        aspect_ratio: Float32,
        samples_per_pixel: Int,
        max_depth: Int,
    ):
        self.samples_per_pixel = samples_per_pixel
        self.max_depth = max_depth
        self.aspect_ratio = aspect_ratio
        self.image_width = image_width
        self.image_height = max(1, Int(self.image_width / aspect_ratio))

        self.center = Point3.zero()

        # Camera
        var focal_length = Float32(1.0)
        var viewport_height = Float32(2.0)
        var viewport_width = (
            viewport_height
            * Float32(self.image_width)
            / Float32(self.image_height)
        )

        # Calculate the vectors across the horizontal and down the vertical viewport edges.
        var viewport_u = Vec3f(viewport_width, 0, 0)
        var viewport_v = Vec3f(0, -viewport_height, 0)

        # Calculate the horizontal and vertical delta vectors from pixel to pixel.
        self.pixel_delta_u = viewport_u / self.image_width
        self.pixel_delta_v = viewport_v / self.image_height

        # Calculate the location of the upper left pixel.
        var viewport_upper_left = (
            self.center
            - Vec3f(0, 0, focal_length)
            - viewport_u / 2
            - viewport_v / 2
        )

        self.pixel00_loc = viewport_upper_left + 0.5 * (
            self.pixel_delta_u + self.pixel_delta_v
        )

    fn ray_color(self, ray: Ray, depth: Int, world: HitableList) -> Color:
        if depth <= 0:
            return Color.zero()

        comptime infinity = max_finite[DType.float32]()
        var hit_res = world.hit(ray, Interval(Float32(0.001), infinity))
        if hit_res:
            var hit = hit_res.value()
            var direction  = hit.normal + random_in_unit_vector()
            var new_ray = Ray(hit.p, direction)
            return 0.5 * self.ray_color(new_ray, depth - 1, world)

        comptime start_value = Color(1.0, 1.0, 1.0)
        comptime end_value = Color(0.5, 0.7, 1.0)

        var unit_direction = ray.direction
        var a = 0.5 * (unit_direction.y() + 1.0)
        return (1.0 - a) * start_value + a * end_value

    fn render(self, world: HitableList) raises:
        with open("rtiaw_1.ppm", "w") as f:
            var header = "P3\n{} {}\n255\n".format(
                self.image_width, self.image_height
            )
            f.write(header)

            var factor = Float32(1.0 / self.samples_per_pixel)
            for j in range(self.image_height):
                for i in range(self.image_width):
                    var pixel_color = Color.zero()

                    for sample in range(self.samples_per_pixel):
                        var r = self.get_ray(i, j)
                         
                        pixel_color += factor * self.ray_color(r, self.max_depth, world)

                    write_color(f, pixel_color)

    fn get_ray(self, i: Int, j: Int) -> Ray:
        var offset = Vec2f.random(-0.5, 0.5)
        var pixel_sample = (
            self.pixel00_loc
            + ((i + offset.x()) * self.pixel_delta_u)
            + ((j + offset.y()) * self.pixel_delta_v)
        )

        var origin = self.center
        var direction = pixel_sample - origin

        return Ray(origin, direction)



# IMAGE_NB==7
# IMAGE_NB==8
# IMAGE_NB==9
# IMAGE_NB==10
# IMAGE_NB==11
# IMAGE_NB==12
# IMAGE_NB==13
# IMAGE_NB==14
# IMAGE_NB==15
# IMAGE_NB==16
# IMAGE_NB==17
# IMAGE_NB==18
# IMAGE_NB==19
# IMAGE_NB==20
# IMAGE_NB==21
# IMAGE_NB==22
# IMAGE_NB==23
fn random_in_unit_vector() -> Vec3f:
    while True:
        var p = Vec3f.random(-1.0, 1.0)
        if Float32(1e-160) <= dot(p, p) < 1.0:
            return normalize(p)

fn random_on_hemisphere(normal: Vec3f) -> Vec3f:
    var on_unit_sphere = random_in_unit_vector()
    if dot(on_unit_sphere, normal) > 0.0: # In the same hemisphere as the normal
        return on_unit_sphere
    else:
        return -on_unit_sphere


fn random_in_unit_disk() -> Vec3f:
    var unit = Vec3f(1.0, 1.0, 0.0)
    while True:
        r1 = Float32(random_float64())
        r2 = Float32(random_float64())
        var p = 2.0 * Vec3f(r1, r2, 0.0) - unit
        if dot(p, p) < 1.0:
            return p


fn random_in_unit_sphere() -> Vec3f:
    var unit = Vec3f(1.0, 1.0, 1.0)
    while True:
        r1 = Float32(random_float64())
        r2 = Float32(random_float64())
        r3 = Float32(random_float64())
        var p = 2.0 * Vec3f(r1, r2, r3) - unit
        if dot(p, p) < 1.0:
            return p


fn reflect(v: Vec3f, n: Vec3f) -> Vec3f:
    return v - n * dot(v, n) * 2.0


fn refract(v: Vec3f, n: Vec3f, ni_over_nt: Float32) -> Optional[Vec3f]:
    var uv = normalize(v)
    var dt = dot(uv, n)
    var discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt)
    if discriminant > 0.0:
        return ni_over_nt * (uv - n * dt) - n * sqrt(discriminant)
    return None


fn schlick(cosine: Float32, ref_idx: Float32) -> Float32:
    var r0 = pow(((1.0 - ref_idx) / (1.0 + ref_idx)), 2)
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5)


trait Material(Copyable):
    fn scatter(self, ray: Ray, hit: HitRecord) -> Optional[Tuple[Ray, Vec3f]]:
        ...


@fieldwise_init
struct Lambertian(Material, Writable):
    var albedo: Vec3f

    fn scatter(self, ray: Ray, hit: HitRecord) -> Optional[Tuple[Ray, Vec3f]]:
        var target = hit.p + hit.normal + random_in_unit_sphere()
        var scattered = Ray(hit.p, target - hit.p)
        return (scattered, self.albedo)


# fn random_scene() -> HitableList:
#     var list_spheres = List[Sphere]()
#     var list_mat = List[Lambertian]()
#     # var origin = Vec3f(4.0, 0.2, 0.0)

#     # Ground
#     list_spheres.append(Sphere(Vec3f(0.0, -1000.0, 0.0), 1000.0, len(list_mat)))
#     list_mat.append(Lambertian(Vec3f(0.5, 0.8, 0.5)))

#     # for a in range(-11, 11):
#     #     for b in range(-11, 11):
#     #         var choose_mat = random_float64()

#     # var center = Vec3f(
#     #     a + 0.9 * random_float64().cast[DType.float32](),
#     #     0.2,
#     #     b + 0.9 * random_float64().cast[DType.float32]()
#     # )

#     # if length(center - origin) > 0.9:
#     #     # if choose_mat < 0.8:  # Diffuse
#     #     var albedo = Vec3f(
#     #         random_float64().cast[DType.float32]() * random_float64().cast[DType.float32](),
#     #         random_float64().cast[DType.float32]() * random_float64().cast[DType.float32](),
#     #         random_float64().cast[DType.float32]() * random_float64().cast[DType.float32]()
#     #     )
#     #     var mat = Lambertian(albedo)
#     #     world.append(Sphere(center, 0.2, UnsafePointer(to=mat))
#     # elif choose_mat < 0.95: # Metal
#     #     let albedo = Vec3(
#     #         0.5 * (1.0 + random_float64().cast[DType.float32]()),
#     #         0.5 * (1.0 + random_float64().cast[DType.float32]()),
#     #         0.5 * (1.0 + random_float64().cast[DType.float32]())
#     #     )
#     #     let fuzz = 0.5 * random_float64().cast[DType.float32]()
#     #     world.push(Sphere(center, 0.2, Metal(albedo, fuzz)))
#     # else: # Glass
#     #     world.push(Sphere(center, 0.2, Dielectric(1.5)))

#     # world.append(Sphere(Vec3f(0.0, 1.0, 0.0), 1.0, Dielectric(1.5)))

#     list_spheres.append(Sphere(Vec3f(-4.0, 1.0, 0.0), 1.0, len(list_mat)))
#     list_mat.append(Lambertian(Vec3f(0.4, 0.2, 0.1)))
#     # world.append(Sphere(Vec3f(4.0, 1.0, 0.0), 1.0, Metal(Vec3(0.7, 0.6, 0.5), 0.0)))

#     return HitableList(list_spheres^, list_mat^)


# fn ray_color(ray: Ray, world: HitableList, depth: Int) -> Vec3f:
#     @parameter
#     if IMAGE_NB == 2:
#         comptime start_value = Color(1.0, 1.0, 1.0)
#         comptime end_value = Color(0.5, 0.7, 1.0)
#         var unit_direction = ray.direction
#         var a = 0.5 * (unit_direction.y() + 1.0)
#         return (1.0 - a) * start_value + a * end_value
#     else:
#         var hit_opt = world.hit(ray, 0.001, 3.4028235e38)

#         if hit_opt:
#             var hit = hit_opt.value()
#             if depth < 50:
#                 ref mat = world.list_mat[hit.material_id]
#                 var scatter_res = mat.scatter(ray, hit)
#                 if scatter_res:
#                     var scattered = scatter_res.value()[0]
#                     var attenuation = scatter_res.value()[1]
#                     return attenuation * ray_color(scattered, world, depth + 1)
#             return Vec3f.zero()
#         else:
#             var unit_direction = normalize(ray.direction)
#             var t = 0.5 * (unit_direction[1] + 1.0)
#             comptime start_value = Vec3f(1.0, 1.0, 1.0)
#             comptime end_value = Vec3f(0.5, 0.7, 1.0)
#             return (1.0 - t) * start_value + t * end_value
