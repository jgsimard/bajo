from math import sqrt, tan, pi, clamp, cos
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
    degrees_to_radians,
)

comptime Point3 = Vec3f
comptime Color = Vec3f


@fieldwise_init
struct Scene:
    var camera: Camera
    var world: HitableList


fn main() raises:
    print(" Ray Tracing in One Weekend - Part 1")
    scene = create_top_scene()
    # scene = create_basic_scene()
    # scene = create_random_scene()
    scene.camera.render(scene.world)


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


@register_passable("trivial")
struct HitRecord(Copyable):
    var p: Point3
    var normal: Vec3f
    var material_id: Int
    var t: Float32
    var front_face: Bool

    fn __init__(
        out self, p: Point3, normal: Vec3f, material_id: Int, t: Float32, r: Ray
    ):
        self.p = p
        self.material_id = material_id
        self.t = t
        self.front_face = dot(r.direction, normal) < 0
        self.normal = normal if self.front_face else -normal


trait Hittable(Copyable):
    fn hit(
        self, ray: Ray, ray_t: Interval[DType.float32]
    ) -> Optional[HitRecord]:
        ...


@fieldwise_init
struct Sphere(Hittable, Writable):
    var center: Point3
    var radius: Float32
    var material_id: Int

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
        return HitRecord(p, normal, self.material_id, t, ray)


comptime HittableVariant = Variant[Sphere]


@fieldwise_init
struct HitableList(Hittable, Writable):
    var objects: List[HittableVariant]
    var materials: List[MaterialVariant]

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
    var vfov: Float32
    """ Vertical view angle (field of view)."""
    var lookfrom: Point3
    # Point camera is looking from
    var lookat: Point3
    # // Point camera is looking at
    var vup: Vec3f
    # // Camera-relative "up" direction
    var u: Vec3f
    var v: Vec3f
    var w: Vec3f
    var defocus_angle: Float32
    """Variation angle of rays through each pixel."""
    var focus_dist: Float32
    """Distance from camera lookfrom point to plane of perfect focus."""
    var defocus_disk_u: Vec3f
    """Defocus disk horizontal radius."""
    var defocus_disk_v: Vec3f
    """Defocus disk vertical radius."""

    fn __init__(
        out self,
        image_width: Int,
        aspect_ratio: Float32,
        samples_per_pixel: Int,
        max_depth: Int,
        vfov: Float32,
        lookfrom: Point3,
        lookat: Point3,
        vup: Vec3f,
        defocus_angle: Float32,
        focus_dist: Float32,
    ):
        self.samples_per_pixel = samples_per_pixel
        self.max_depth = max_depth
        self.vfov = vfov
        self.lookfrom = lookfrom
        self.lookat = lookat
        self.vup = vup
        self.defocus_angle = defocus_angle
        self.focus_dist = focus_dist
        self.aspect_ratio = aspect_ratio
        self.image_width = image_width
        self.image_height = max(1, Int(self.image_width / aspect_ratio))

        self.center = lookfrom

        # Camera
        var theta = degrees_to_radians(vfov)
        var h = tan(theta / 2)
        var viewport_height = 2 * h * focus_dist
        var viewport_width = (
            viewport_height
            * Float32(self.image_width)
            / Float32(self.image_height)
        )

        # Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        self.w = normalize(lookfrom - lookat)
        self.u = normalize(cross(vup, self.w))
        self.v = cross(self.w, self.u)

        # Calculate the vectors across the horizontal and down the vertical viewport edges.
        var viewport_u = viewport_width * self.u
        var viewport_v = viewport_height * -self.v

        # Calculate the horizontal and vertical delta vectors from pixel to pixel.
        self.pixel_delta_u = viewport_u / self.image_width
        self.pixel_delta_v = viewport_v / self.image_height

        # Calculate the location of the upper left pixel.
        var viewport_upper_left = (
            self.center
            - (focus_dist * self.w)
            - viewport_u / 2
            - viewport_v / 2
        )

        self.pixel00_loc = viewport_upper_left + 0.5 * (
            self.pixel_delta_u + self.pixel_delta_v
        )

        # // Calculate the camera defocus disk basis vectors.
        var defocus_radius = focus_dist * tan(
            degrees_to_radians(defocus_angle / 2)
        )
        self.defocus_disk_u = self.u * defocus_radius
        self.defocus_disk_v = self.v * defocus_radius

    fn ray_color(self, ray: Ray, world: HitableList) -> Color:
        var cur_ray = ray
        var accumulated_attenuation = Color(1.0, 1.0, 1.0)

        for bounce in range(self.max_depth):
            comptime infinity = max_finite[DType.float32]()
            var hit_res = world.hit(cur_ray, Interval(Float32(0.001), infinity))

            if hit_res:
                var hit = hit_res.value()
                var material_id = hit.material_id

                var scatter_res: Optional[Tuple[Ray, Color]]
                ref material = world.materials[material_id]
                if material.isa[Lambertian]():
                    scatter_res = material[Lambertian].scatter(cur_ray, hit)
                elif material.isa[Metal]():
                    scatter_res = material[Metal].scatter(cur_ray, hit)
                elif material.isa[Dielectric]():
                    scatter_res = material[Dielectric].scatter(cur_ray, hit)
                else:
                    print("Material not implemented yet")
                    abort()

                if scatter_res:
                    var scatter = scatter_res.value()
                    var scattered_ray = scatter[0]
                    var attenuation = scatter[1]

                    cur_ray = scattered_ray
                    accumulated_attenuation *= attenuation
                else:
                    return Color.zeros()
            else:
                # RAY HIT THE SKY
                comptime start_value = Color(1.0, 1.0, 1.0)
                comptime end_value = Color(0.5, 0.7, 1.0)

                var unit_direction = cur_ray.direction
                var a = 0.5 * (unit_direction.y() + 1.0)
                var sky_color = (1.0 - a) * start_value + a * end_value
                # Final result is the sky color tinted by all previous bounces
                return accumulated_attenuation * sky_color

        # If we exceeded the depth without hitting the sky, return black
        return Color.zeros()

    fn render(self, world: HitableList) raises:
        with open("rtiaw_1.ppm", "w") as f:
            var header = "P3\n{} {}\n255\n".format(
                self.image_width, self.image_height
            )
            f.write(header)

            var factor = Float32(1.0 / self.samples_per_pixel)
            for j in range(self.image_height):
                for i in range(self.image_width):
                    var pixel_color = Color.zeros()

                    for sample in range(self.samples_per_pixel):
                        var r = self.get_ray(i, j)

                        pixel_color += factor * self.ray_color(r, world)

                    write_color(f, pixel_color)

    fn get_ray(self, i: Int, j: Int) -> Ray:
        var offset = Vec2f.random(-0.5, 0.5)
        var pixel_sample = (
            self.pixel00_loc
            + ((i + offset.x()) * self.pixel_delta_u)
            + ((j + offset.y()) * self.pixel_delta_v)
        )

        var origin = (
            self.center if self.defocus_angle
            <= 0 else self.defocus_disk_sample()
        )
        var direction = pixel_sample - origin

        return Ray(origin, direction)

    fn defocus_disk_sample(self) -> Vec3f:
        """Returns a random point in the camera defocus disk."""
        var p = random_in_unit_disk()
        return (
            self.center
            + (p[0] * self.defocus_disk_u)
            + (p[1] * self.defocus_disk_v)
        )


fn random_unit_vector() -> Vec3f:
    while True:
        var p = Vec3f.random(-1.0, 1.0)
        if Float32(1e-160) <= dot(p, p) < 1.0:
            return normalize(p)


fn random_on_hemisphere(normal: Vec3f) -> Vec3f:
    var on_unit_sphere = random_unit_vector()
    if (
        dot(on_unit_sphere, normal) > 0.0
    ):  # In the same hemisphere as the normal
        return on_unit_sphere
    else:
        return -on_unit_sphere


fn random_in_unit_disk() -> Vec3f:
    while True:
        r1 = Float32(random_float64(-1, 1))
        r2 = Float32(random_float64(-1, 1))
        var p = 2.0 * Vec3f(r1, r2, 0.0)
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
    return v - 2.0 * dot(v, n) * n


fn refract(uv: Vec3f, n: Vec3f, etai_over_etat: Float32) -> Vec3f:
    var cos_theta = min(dot(-uv, n), 1.0)
    var r_out_perp = etai_over_etat * (uv + cos_theta * n)
    var r_out_parallel = -sqrt(abs(1.0 - length2(r_out_perp))) * n
    return r_out_perp + r_out_parallel


trait Material(Copyable):
    fn scatter(self, ray: Ray, hit: HitRecord) -> Optional[Tuple[Ray, Color]]:
        ...


comptime MaterialVariant = Variant[Lambertian, Metal, Dielectric]


@fieldwise_init
struct Lambertian(Material, Writable):
    var albedo: Vec3f

    fn scatter(self, ray: Ray, hit: HitRecord) -> Optional[Tuple[Ray, Color]]:
        var scatter_direction = hit.normal + random_unit_vector()

        # Catch degenerate scatter direction
        if scatter_direction.near_zero():
            scatter_direction = hit.normal

        var scattered = Ray(hit.p, scatter_direction)
        return (scattered, self.albedo)


@fieldwise_init
struct Metal(Material, Writable):
    var albedo: Vec3f
    var fuzz: Float32

    fn scatter(self, ray: Ray, hit: HitRecord) -> Optional[Tuple[Ray, Color]]:
        var reflected = reflect(ray.direction, hit.normal)
        reflected = normalize(reflected) + (self.fuzz * random_unit_vector())
        var scattered = Ray(hit.p, reflected)

        if dot(scattered.direction, hit.normal) < 0:
            return None

        return (scattered, self.albedo)


@fieldwise_init
struct Dielectric(Material, Writable):
    var refraction_index: Float32

    fn scatter(self, ray: Ray, hit: HitRecord) -> Optional[Tuple[Ray, Color]]:
        var attenuation = Color.ones()
        var ri = (
            1.0
            / self.refraction_index if hit.front_face else self.refraction_index
        )

        var unit_direction = normalize(ray.direction)
        var cos_theta = min(dot(-unit_direction, hit.normal), 1.0)
        var sin_theta = sqrt(1.0 - cos_theta * cos_theta)

        var cannot_refract = ri * sin_theta > 1.0
        var direction: Vec3f

        # total internal reflection
        if cannot_refract or reflectance(cos_theta, ri) > Float32(
            random_float64()
        ):  # Must Reflect
            direction = reflect(unit_direction, hit.normal)
        else:  # can refract
            direction = refract(unit_direction, hit.normal, ri)

        scattered = Ray(hit.p, direction)
        return (scattered, attenuation)


fn reflectance(cosine: Float32, ref_idx: Float32) -> Float32:
    """Schlick's approximation for reflectance."""
    var r0 = pow(((1.0 - ref_idx) / (1.0 + ref_idx)), 2)
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5)


fn create_random_scene() -> Scene:
    var materials = List[MaterialVariant]()
    var objects = List[HittableVariant]()

    # Ground material
    materials.append(Lambertian(Color(0.5, 0.5, 0.5)))
    objects.append(Sphere(Point3(0, -1000, 0), 1000, 0))

    # Random small spheres
    for a in range(-11, 11):
        for b in range(-11, 11):
            var choose_mat = random_float64()
            var center = Point3(
                Float32(a) + 0.9 * Float32(random_float64()),
                0.2,
                Float32(b) + 0.9 * Float32(random_float64()),
            )

            if length(center - Point3(4, 0.2, 0)) > 0.9:
                if choose_mat < 0.8:
                    # Diffuse (Lambertian)
                    var albedo = Vec3f.random() * Vec3f.random()
                    materials.append(Lambertian(albedo))
                    objects.append(Sphere(center, 0.2, len(materials) - 1))

                elif choose_mat < 0.95:
                    # Metal
                    var albedo = Vec3f.random(0.5, 1.0)
                    var fuzz = Float32(random_float64(0, 0.5))
                    materials.append(Metal(albedo, fuzz))
                    objects.append(Sphere(center, 0.2, len(materials) - 1))

                else:
                    # Glass (Dielectric)
                    materials.append(Dielectric(1.5))
                    objects.append(Sphere(center, 0.2, len(materials) - 1))

    # Glass Sphere
    materials.append(Dielectric(1.5))
    objects.append(Sphere(Point3(0, 1, 0), 1.0, len(materials) - 1))

    # Matte Sphere
    materials.append(Lambertian(Color(0.4, 0.2, 0.1)))
    objects.append(Sphere(Point3(-4, 1, 0), 1.0, len(materials) - 1))

    # Metal Sphere
    materials.append(Metal(Color(0.7, 0.6, 0.5), 0.0))
    objects.append(Sphere(Point3(4, 1, 0), 1.0, len(materials) - 1))

    var world = HitableList(objects^, materials^)

    var cam = Camera(
        image_width=400,
        aspect_ratio=16.0 / 9.0,
        samples_per_pixel=10,
        max_depth=10,
        vfov=20,
        lookfrom=Point3(13, 2, 3),
        lookat=Point3(0, 0, 0),
        vup=Vec3f(0, 1, 0),
        defocus_angle=0.6,
        focus_dist=10.0,
    )

    return Scene(cam^, world^)


fn create_basic_scene() -> Scene:
    var world = HitableList(
        [
            Sphere(Point3(0, -100.5, -1), 100, 0),
            Sphere(Point3(0, 0, -1.2), 0.5, 1),
            Sphere(Point3(-1, 0, -1), 0.5, 2),
            Sphere(Point3(-1, 0, -1), 0.4, 3),
            Sphere(Point3(1, 0, -1), 0.5, 4),
        ],
        [
            Lambertian(Color(0.8, 0.8, 0.0)),  # ground
            Lambertian(Color(0.1, 0.2, 0.5)),  # center
            Dielectric(1.5),  # left
            Dielectric(1.0 / 1.5),  # bubble
            Metal(Color(0.8, 0.6, 0.2), 1.0),  # right
        ],
    )

    var cam = Camera(
        image_width=400,
        aspect_ratio=16.0 / 9.0,
        samples_per_pixel=10,
        max_depth=10,
        vfov=20,
        lookfrom=Point3(-2, 2, 1),
        lookat=Point3(0, 0, -1),
        vup=Vec3f(0, 1, 0),
        defocus_angle=10.0,
        focus_dist=3.4,
    )
    return Scene(cam^, world^)


fn create_top_scene() -> Scene:
    var R = Float32(cos(pi / 4))
    var world = HitableList(
        [Sphere(Point3(-R, 0, -1), R, 0), Sphere(Point3(R, 0, -1), R, 1)],
        [Lambertian(Color(0, 0, 1)), Lambertian(Color(1, 0, 0))],
    )

    var cam = Camera(
        image_width=400,
        aspect_ratio=16.0 / 9.0,
        samples_per_pixel=10,
        max_depth=10,
        vfov=90,
        lookfrom=Point3(0, 0, 0),
        lookat=Point3(0, 0, -1),
        vup=Vec3f(0, 1, 0),
        defocus_angle=0.0,
        focus_dist=10.0,
    )
    return Scene(cam^, world^)
