from std.algorithm import parallelize
from std.math import sqrt, tan, pi, cos, fma
from std.os import abort
from std.io.file_descriptor import FileDescriptor
from std.utils import Variant

from bajo.core.vec import (
    Vec3f32,
    length,
    length2,
    normalize,
    dot,
    cross,
)
from bajo.core.utils import degrees_to_radians
from bajo.bvh.constants import TRACE_CLOSEST_HIT, f32_max
from bajo.core.random import (
    random_unit_vector,
    random_in_unit_disk,
    Rng,
)
from bajo.bvh.types import Ray as bRay, Sphere as bSphere
from bajo.bvh.cpu.sphere_bvh import SphereBvh


comptime Point3 = Vec3f32
comptime Color = Vec3f32
comptime BVH_WIDTH = 8


@fieldwise_init
struct Scene:
    var camera: Camera
    var world: World


def main() raises:
    print("Ray Tracing in One Weekend")
    # scene = create_top_scene()
    # scene = create_basic_scene()
    scene = create_random_scene()
    scene.camera.render(scene.world)


def colorize(color: Color) -> Color:
    return Color(
        sqrt(color.x).clamp(0.0, 0.999) * 255.99,
        sqrt(color.y).clamp(0.0, 0.999) * 255.99,
        sqrt(color.z).clamp(0.0, 0.999) * 255.99,
    )


def write_ppm_from_colors(
    path: String,
    width: Int,
    height: Int,
    image_data: List[Color],
) raises:
    # Fast binary PPM writer.
    #
    # P3 writes formatted text for every channel.
    # P6 writes a tiny text header followed by raw RGB bytes.
    var pixel_count = width * height
    var byte_count = pixel_count * 3

    with open(path, "w") as f:
        var fd = FileDescriptor(f)
        fd.write(t"P6\n{width} {height}\n255\n")

        var _bytes = List[UInt8](length=byte_count, fill=0)
        var out = _bytes.unsafe_ptr()

        var j = 0
        for i in range(pixel_count):
            var out_color = colorize(image_data[i])

            out[j] = UInt8(out_color.x)
            out[j + 1] = UInt8(out_color.y)
            out[j + 2] = UInt8(out_color.z)

            j += 3

        fd.write_bytes(_bytes)


@fieldwise_init
struct Ray(Copyable, Writable):
    var origin: Point3
    var direction: Point3
    var inv_direction: Point3
    var time: Float32

    def at(self, t: Float32) -> Point3:
        return self.origin + t * self.direction


struct HitRecord(Copyable):
    var p: Point3
    var normal: Vec3f32
    var material_id: Int
    var t: Float32
    var u: Float32
    var v: Float32
    var front_face: Bool

    def __init__(
        out self,
        p: Point3,
        normal: Vec3f32,
        material_id: Int,
        t: Float32,
        r: Ray,
    ):
        self.p = p.copy()
        self.material_id = material_id
        self.t = t
        self.u = 0.0
        self.v = 0.0
        self.front_face = dot(r.direction, normal) < 0
        self.normal = normal * Float32(1.0 if self.front_face else -1.0)


@fieldwise_init
struct Sphere(Copyable, Writable):
    var center: Point3
    var radius: Float32
    var material_id: Int


@fieldwise_init
struct World(Copyable):
    var bvh: SphereBvh[BVH_WIDTH]
    var objects: List[Sphere]
    var materials: List[MaterialVariant]

    def __init__(
        out self,
        var objects: List[Sphere],
        var materials: List[MaterialVariant],
    ):
        self.objects = objects^
        self.materials = materials^

        var bvh_spheres = List[bSphere](capacity=len(self.objects))
        for obj in self.objects:
            bvh_spheres.append(bSphere(obj.center, obj.radius))

        self.bvh = SphereBvh[BVH_WIDTH].__init__["median"](
            bvh_spheres.unsafe_ptr(),
            UInt32(len(bvh_spheres)),
        )

    def hit(
        self,
        ray: Ray,
        ray_t_min: Float32,
        ray_t_max: Float32,
    ) -> Optional[HitRecord]:
        var origin = ray.at(ray_t_min)
        var tmax = ray_t_max - ray_t_min

        var bvh_ray = bRay(
            origin,
            Float32(0.0),
            ray.direction,
            tmax,
        )

        var bvh_hit = self.bvh.trace[TRACE_CLOSEST_HIT](bvh_ray)

        if not bvh_hit.is_hit():
            return None

        if bvh_hit.t >= tmax:
            return None

        var t = bvh_hit.t + ray_t_min
        if not (ray_t_min < t < ray_t_max):
            return None

        var sphere_idx = Int(bvh_hit.prim)
        ref obj = self.objects[sphere_idx]

        var p = ray.at(t)
        var normal = (p - obj.center) / obj.radius

        return HitRecord(p, normal, obj.material_id, t, ray)


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
    var pixel_delta_u: Vec3f32
    """Offset to pixel to the right."""
    var pixel_delta_v: Vec3f32
    """Offset to pixel below."""
    var samples_per_pixel: Int
    """Count of random samples for each pixel."""
    var max_depth: Int
    """Maximum number of ray bounces into scene."""
    var vfov: Float32
    """ Vertical view angle (field of view)."""
    var lookfrom: Point3
    """Point camera is looking from."""
    var lookat: Point3
    """Point camera is looking at."""
    var vup: Vec3f32
    """Camera-relative "up" direction."""
    var u: Vec3f32
    var v: Vec3f32
    var w: Vec3f32
    var defocus_angle: Float32
    """Variation angle of rays through each pixel."""
    var focus_dist: Float32
    """Distance from camera lookfrom point to plane of perfect focus."""
    var defocus_disk_u: Vec3f32
    """Defocus disk horizontal radius."""
    var defocus_disk_v: Vec3f32
    """Defocus disk vertical radius."""

    def __init__(
        out self,
        image_width: Int,
        aspect_ratio: Float32,
        samples_per_pixel: Int,
        max_depth: Int,
        vfov: Float32,
        lookfrom: Point3,
        lookat: Point3,
        vup: Vec3f32,
        defocus_angle: Float32,
        focus_dist: Float32,
    ):
        self.samples_per_pixel = samples_per_pixel
        self.max_depth = max_depth
        self.vfov = vfov
        self.lookfrom = lookfrom.copy()
        self.lookat = lookat.copy()
        self.vup = vup.copy()
        self.defocus_angle = defocus_angle
        self.focus_dist = focus_dist
        self.aspect_ratio = aspect_ratio
        self.image_width = image_width
        self.image_height = max(
            1, Int(Float32(self.image_width) / aspect_ratio)
        )

        self.center = lookfrom.copy()

        # Camera
        theta = degrees_to_radians(vfov)
        h = tan(theta / 2)
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
        self.pixel_delta_u = viewport_u / Float32(self.image_width)
        self.pixel_delta_v = viewport_v / Float32(self.image_height)

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

        # Calculate the camera defocus disk basis vectors.
        var defocus_radius = focus_dist * tan(
            degrees_to_radians(defocus_angle / 2)
        )
        self.defocus_disk_u = self.u * defocus_radius
        self.defocus_disk_v = self.v * defocus_radius

    def ray_color(self, ray: Ray, world: World, mut rng: Rng) -> Color:
        var cur_ray = ray.copy()
        var accumulated_attenuation = Color(1.0, 1.0, 1.0)

        for _bounce in range(self.max_depth):
            var hit_res = world.hit(cur_ray, 0.001, f32_max)

            if hit_res:
                ref hit = hit_res.value()
                var material_id = hit.material_id

                var scatter_res: Optional[Tuple[Ray, Color]]
                ref material = world.materials[material_id]
                if material.isa[Lambertian]():
                    scatter_res = material[Lambertian].scatter(
                        cur_ray, hit, rng
                    )
                elif material.isa[Metal]():
                    scatter_res = material[Metal].scatter(cur_ray, hit, rng)
                elif material.isa[Dielectric]():
                    scatter_res = material[Dielectric].scatter(
                        cur_ray, hit, rng
                    )
                else:
                    print("Material not implemented yet")
                    abort()

                if scatter_res:
                    ref scatter = scatter_res.value()
                    var scattered_ray = scatter[0].copy()
                    attenuation = scatter[1].copy()

                    cur_ray = scattered_ray.copy()
                    accumulated_attenuation *= attenuation
                else:
                    return Color(0)
            else:
                # RAY HIT THE SKY
                var start_value = Color(1.0, 1.0, 1.0)
                var end_value = Color(0.5, 0.7, 1.0)

                var unit_direction = normalize(cur_ray.direction)
                var a = 0.5 * (unit_direction.y + 1.0)
                var sky_color = (1.0 - a) * start_value + a * end_value
                # Final result is the sky color tinted by all previous bounces
                return accumulated_attenuation * sky_color

        # If we exceeded the depth without hitting the sky, return black
        return Color(0)

    def render(self, world: World) raises:
        var image_data = List[Color](
            length=self.image_width * self.image_height, fill=Color(0)
        )

        @parameter
        def worker(j: Int):
            rng = Rng(seed=123, id=UInt64(j))
            factor = Float32(1.0 / Float32(self.samples_per_pixel))
            for i in range(self.image_width):
                var pixel_color = Color(0)
                for _sample in range(self.samples_per_pixel):
                    var r = self.get_ray(i, j, rng)
                    pixel_color += self.ray_color(r, world, rng)

                image_data[j * self.image_width + i] = pixel_color * factor

        parallelize[worker](self.image_height, self.image_height)

        write_ppm_from_colors(
            "rtiaw.ppm",
            self.image_width,
            self.image_height,
            image_data,
        )

    def get_ray(self, i: Int, j: Int, mut rng: Rng) -> Ray:
        var r1 = rng.f32()
        var r2 = rng.f32()
        var pixel_sample = (
            self.pixel00_loc
            + ((Float32(i) + r1) * self.pixel_delta_u)
            + ((Float32(j) + r2) * self.pixel_delta_v)
        )

        origin = (
            self.center.copy() if self.defocus_angle
            <= 0 else self.defocus_disk_sample(rng)
        )
        direction = pixel_sample - origin
        time = rng.f32()

        return Ray(origin, direction, 1.0 / direction, time)

    def defocus_disk_sample(self, mut rng: Rng) -> Vec3f32:
        """Returns a random point in the camera defocus disk."""
        p = random_in_unit_disk(rng)
        return (
            self.center
            + (p[0] * self.defocus_disk_u)
            + (p[1] * self.defocus_disk_v)
        )


def reflect(v: Vec3f32, n: Vec3f32) -> Vec3f32:
    return v - 2.0 * dot(v, n) * n


def refract(uv: Vec3f32, n: Vec3f32, etai_over_etat: Float32) -> Vec3f32:
    var cos_theta = min(-dot(uv, n), 1.0)
    var r_out_perp = etai_over_etat * (uv + cos_theta * n)
    var r_out_parallel = -sqrt(abs(1.0 - length2(r_out_perp))) * n
    return r_out_perp + r_out_parallel


trait Material(Copyable):
    def scatter(
        self, ray: Ray, hit: HitRecord, mut rng: Rng
    ) -> Optional[Tuple[Ray, Color]]:
        ...


comptime MaterialVariant = Variant[Lambertian, Metal, Dielectric]


@fieldwise_init
struct Lambertian(Material, Writable):
    var albedo: Vec3f32

    def scatter(
        self, ray: Ray, hit: HitRecord, mut rng: Rng
    ) -> Optional[Tuple[Ray, Color]]:
        var scatter_direction = hit.normal + random_unit_vector(rng)

        # Catch degenerate scatter direction
        if scatter_direction.is_near_zero():
            scatter_direction = hit.normal.copy()

        scattered = Ray(
            hit.p, scatter_direction, 1.0 / scatter_direction, ray.time
        )
        return (scattered^, self.albedo.copy())


@fieldwise_init
struct Metal(Material, Writable):
    var albedo: Vec3f32
    var fuzz: Float32

    def scatter(
        self, ray: Ray, hit: HitRecord, mut rng: Rng
    ) -> Optional[Tuple[Ray, Color]]:
        reflected = reflect(ray.direction, hit.normal)
        reflected = normalize(reflected) + (self.fuzz * random_unit_vector(rng))
        scattered = Ray(hit.p, reflected, 1.0 / reflected, ray.time)

        if dot(scattered.direction, hit.normal) < 0:
            return None

        return (scattered^, self.albedo.copy())


@fieldwise_init
struct Dielectric(Material, Writable):
    var refraction_index: Float32

    def scatter(
        self, ray: Ray, hit: HitRecord, mut rng: Rng
    ) -> Optional[Tuple[Ray, Color]]:
        attenuation = Color(1)
        ri = (
            1.0
            / self.refraction_index if hit.front_face else self.refraction_index
        )

        var unit_direction = normalize(ray.direction)
        var cos_theta = min(dot(-unit_direction, hit.normal), 1.0)
        var sin_theta = sqrt(1.0 - cos_theta * cos_theta)

        var cannot_refract = ri * sin_theta > 1.0
        var direction: Vec3f32

        # total internal reflection
        var _rng = rng.f32()
        if cannot_refract or reflectance(cos_theta, ri) > _rng:
            direction = reflect(unit_direction, hit.normal)
        else:
            direction = refract(unit_direction, hit.normal, ri)

        scattered = Ray(hit.p, direction, 1.0 / direction, ray.time)
        return (scattered^, attenuation.copy())


def reflectance[
    dtype: DType, size: Int
](cosine: SIMD[dtype, size], ref_idx: SIMD[dtype, size]) -> SIMD[dtype, size]:
    """
    Schlick's approximation for reflectance.
    """
    # r0: ((1-n)/(1+n))^2
    var root = (1.0 - ref_idx) / (1.0 + ref_idx)
    var r0 = root * root

    # (1 - cosine)^5
    var x = 1.0 - cosine
    var x2 = x * x
    var x5 = x2 * x2 * x

    # r0 + (1.0 - r0) * x5
    return fma(1.0 - r0, x5, r0)


def create_random_scene() -> Scene:
    rng = Rng(123, 321)
    materials = List[MaterialVariant]()
    objects = List[Sphere]()

    # Ground material
    materials.append(Lambertian(Color(0.5, 0.5, 0.5)))
    objects.append(Sphere(Point3(0, -1000, 0), 1000, 0))

    # Random small spheres
    for a in range(-11, 11):
        for b in range(-11, 11):
            var choose_mat = rng.f32()
            center = Point3(
                Float32(a) + 0.9 * rng.f32(),
                0.2,
                Float32(b) + 0.9 * rng.f32(),
            )

            if length(center - Point3(4, 0.2, 0)) > 0.9:
                if choose_mat < 0.8:
                    # Diffuse (Lambertian)
                    vr1 = Vec3f32(rng.f32(), rng.f32(), rng.f32())
                    vr2 = Vec3f32(rng.f32(), rng.f32(), rng.f32())
                    albedo = vr1 * vr2
                    materials.append(Lambertian(albedo))
                    objects.append(Sphere(center, 0.2, len(materials) - 1))

                elif choose_mat < 0.95:
                    # Metal
                    albedo = Vec3f32(rng.f32(), rng.f32(), rng.f32()) + 0.5
                    fuzz = rng.f32() * 0.5
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

    cam = Camera(
        image_width=600,
        aspect_ratio=16.0 / 9.0,
        samples_per_pixel=10,
        max_depth=10,
        vfov=20,
        lookfrom=Point3(13, 2, 3),
        lookat=Point3(0, 0, 0),
        vup=Vec3f32(0, 1, 0),
        defocus_angle=0.6,
        focus_dist=10.0,
    )
    world = World(objects^, materials^)
    return Scene(cam^, world^)


def create_basic_scene() -> Scene:
    world = World(
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

    cam = Camera(
        image_width=600,
        aspect_ratio=16.0 / 9.0,
        samples_per_pixel=10,
        max_depth=10,
        vfov=20,
        lookfrom=Point3(-2, 2, 1),
        lookat=Point3(0, 0, -1),
        vup=Vec3f32(0, 1, 0),
        defocus_angle=10.0,
        focus_dist=3.4,
    )
    return Scene(cam^, world^)


def create_top_scene() -> Scene:
    R = Float32(cos(pi / 4))
    world = World(
        [Sphere(Point3(-R, 0, -1), R, 0), Sphere(Point3(R, 0, -1), R, 1)],
        [Lambertian(Color(0, 0, 1)), Lambertian(Color(1, 0, 0))],
    )

    cam = Camera(
        image_width=400,
        aspect_ratio=16.0 / 9.0,
        samples_per_pixel=10,
        max_depth=10,
        vfov=90,
        lookfrom=Point3(0, 0, 0),
        lookat=Point3(0, 0, -1),
        vup=Vec3f32(0, 1, 0),
        defocus_angle=0.0,
        focus_dist=10.0,
    )
    return Scene(cam^, world^)
