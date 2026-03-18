from std.algorithm import parallelize
from std.math import sqrt, tan, pi, clamp, cos
from std.os import abort
from std.random import random_float64, random_si64
from std.sys.info import size_of
from std.utils.numerics import max_finite, min_finite
from std.utils import Variant

# from bajo.core.vec_simd import (
from bajo.core.vec import (
    Vec2f32,
    Vec3f32,
    length,
    length2,
    normalize,
    dot,
    cross,
    vmin,
    vmax,
    longest_axis,
)
from bajo.core.conversion import degrees_to_radians

from bajo.core.random import (
    random_unit_vector,
    random_on_hemisphere,
    random_in_unit_disk,
    random_in_unit_sphere,
    # random_unit_vector_simd as random_unit_vector,
    # random_on_hemisphere_simd as random_on_hemisphere,
    # random_in_unit_disk_simd as random_in_unit_disk,
    # random_in_unit_sphere_simd as random_in_unit_sphere,
    PhiloxRNG,
)

comptime Point3 = Vec3f32
comptime Color = Vec3f32


@fieldwise_init
struct Scene:
    var camera: Camera
    var world: BVH


fn main() raises:
    print("Ray Tracing in One Weekend - Part 2")
    # scene = create_top_scene()
    # scene = create_basic_scene()
    scene = create_random_scene()
    scene.camera.render(scene.world)


fn colorize(color: Color) -> Color:
    out = Color(uninitialized=True)
    comptime for i in range(3):
        out.data[i] = sqrt(color[i]).clamp(0.0, 0.999) * 255.99
    return out^
    # return Color(sqrt(color.data).clamp(0.0, 0.999) * 255.99)


fn write_color(mut f: FileHandle, color: Color):
    var out_color = colorize(color)
    ir = Int(out_color.x())
    ig = Int(out_color.y())
    ib = Int(out_color.z())

    f.write(t"{ir} {ig} {ib}\n")


@fieldwise_init
struct Ray(Copyable, Writable):
    var origin: Point3  # 16 -> 16
    var direction: Point3  # 16 -> 32
    var inv_direction: Point3  # 16 -> 48
    var time: Float32  #  4 -> 52
    var _pad: InlineArray[Float32, 3]  # 3*4=12-> 64

    fn __init__(
        out self, origin: Point3, direction: Point3, time: Float32 = 0.0
    ):
        self.origin = origin.copy()
        self.direction = direction.copy()
        self.inv_direction = 1.0 / self.direction
        self.time = time
        self._pad = InlineArray[Float32, 3](fill=0.0)

    fn at(self, t: Float32) -> Point3:
        return self.origin + t * self.direction


struct HitRecord(Copyable):
    var p: Point3  # 4*4 = 16 => 16
    var normal: Vec3f32  # 4*4 = 16 => 32
    var material_id: Int  # 4 => 36
    var t: Float32  # 4 => 40
    var u: Float32  # 4 => 44
    var v: Float32  # 4 => 48
    var front_face: Bool  # 1 -> 4 => 52

    fn __init__(
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


trait Texture(Movable):
    fn value(self, u: Float32, v: Float32, p: Point3) -> Color:
        ...


comptime TextureVariant = Variant[SolidColor, CheckerTexture]


@fieldwise_init
struct SolidColor(Texture):
    var albedo: Color

    fn __init__(out self, r: Float32, g: Float32, b: Float32):
        self.albedo = Color(r, g, b)

    fn value(self, u: Float32, v: Float32, p: Point3) -> Color:
        return self.albedo.copy()


@fieldwise_init
struct CheckerTexture(Texture):
    var albedo: Color

    fn __init__(out self, r: Float32, g: Float32, b: Float32):
        self.albedo = Color(r, g, b)

    fn value(self, u: Float32, v: Float32, p: Point3) -> Color:
        return self.albedo.copy()


trait Hittable(Copyable):
    fn hit(
        self, ray: Ray, ray_t: Interval[DType.float32]
    ) -> Optional[HitRecord]:
        ...


comptime HittableVariant = Variant[Sphere]


@fieldwise_init
struct Sphere(Hittable, Writable):
    var center: Ray
    var radius: Float32
    var material_id: Int

    fn __init__(out self, center: Point3, radius: Float32, material_id: Int):
        self.center = Ray(center, Vec3f32(0), 0)
        self.radius = radius
        self.material_id = material_id

    fn hit(
        self, ray: Ray, ray_t: Interval[DType.float32]
    ) -> Optional[HitRecord]:
        current_center = self.center.at(ray.time)
        oc = current_center - ray.origin
        a = length2(ray.direction)
        h = dot(ray.direction, oc)
        c = length2(oc) - self.radius * self.radius

        discriminant = h * h - a * c
        if discriminant < 0:
            return None

        sqrtd = sqrt(discriminant)

        # Find the nearest root that lies in the acceptable range.
        root = (h - sqrtd) / a
        if not ray_t.surrounds(root):
            root = (h + sqrtd) / a
            if not ray_t.surrounds(root):
                return None

        t = root
        p = ray.at(t)
        normal = (p - current_center) / self.radius
        return HitRecord(p, normal, self.material_id, t, ray)

    fn bounding_box(self) -> AABB:
        rvec = Vec3f32(self.radius, self.radius, self.radius)

        # time = 0.0
        var center0 = self.center.at(0.0)
        var box0 = AABB(center0 - rvec, center0 + rvec)

        # time = 1.0
        var center1 = self.center.at(1.0)
        var box1 = AABB(center1 - rvec, center1 + rvec)

        return AABB(box0, box1)


@fieldwise_init
struct Interval[T: DType](
    Copyable,
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

    fn union(self, other: Self) -> Self:
        return Interval(min(self.min, other.min), max(self.max, other.max))

    fn expand(self, delta: Scalar[Self.T]) -> Self:
        padding = delta / 2
        return Interval(self.min - padding, self.max + padding)


@fieldwise_init
struct AABB(Copyable):
    """Axis Aligned Bounding Box."""

    var min: Point3
    var max: Point3

    fn __init__(out self, a: AABB, b: AABB):
        self.min = vmin(a.min, b.min)
        self.max = vmax(a.max, b.max)

    fn hit(self, ray: Ray, ray_t: Interval[DType.float32]) -> Bool:
        var t_lower = ray.inv_direction * (self.min - ray.origin)
        var t_upper = ray.inv_direction * (self.max - ray.origin)

        var t_min_vec = vmin(t_lower, t_upper)
        var t_max_vec = vmax(t_lower, t_upper)

        var t_box_min = max(
            t_min_vec.x(), t_min_vec.y(), t_min_vec.z(), ray_t.min
        )

        var t_box_max = min(
            t_max_vec.x(), t_max_vec.y(), t_max_vec.z(), ray_t.max
        )

        return t_box_min <= t_box_max

    fn merge(mut self, other: Self):
        self.min = vmin(self.min, other.min)
        self.max = vmax(self.max, other.max)

    fn edges(self) -> Point3:
        return self.max - self.min


@fieldwise_init
struct BVHNode(Copyable):
    """Bounding Volume Hierarchy Node."""

    var bbox: AABB
    var left_idx: Int
    """Index in 'nodes' list. -1 if leaf."""
    var right_idx: Int
    """Index in 'nodes' list. -1 if leaf."""
    var object_idx: Int
    """Index in 'objects' list. -1 if internal node."""


@fieldwise_init
struct InvalidHittableError(Movable, TrivialRegisterPassable, Writable):
    ...


fn get_bounding_box(obj: HittableVariant) -> AABB:
    if obj.isa[Sphere]():
        return obj[Sphere].bounding_box()
    print("InvalidHittableError")
    abort()
    # raise InvalidHittableError()


@fieldwise_init
struct BVH(Hittable):
    """Bounding Volume Hierarchy."""

    var nodes: List[BVHNode]
    var objects: List[HittableVariant]
    var materials: List[MaterialVariant]
    var root_idx: Int

    fn __init__(
        out self,
        var objects: List[HittableVariant],
        var materials: List[MaterialVariant],
    ):
        self.nodes = List[BVHNode]()
        self.objects = objects^
        self.materials = materials^
        self.root_idx = -1

        if len(self.objects) > 0:
            self.root_idx = self._build(0, len(self.objects))

    fn hit(
        self, ray: Ray, ray_t: Interval[DType.float32]
    ) -> Optional[HitRecord]:
        if self.root_idx == -1:
            return None

        var closest_so_far = ray_t.max
        var hit_anything: Optional[HitRecord] = None

        var node_stack = InlineArray[Int, 32](fill=0)
        var stack_ptr = 0

        # Push root
        node_stack[stack_ptr] = self.root_idx
        stack_ptr += 1

        while stack_ptr > 0:
            # Pop
            stack_ptr -= 1
            var node_idx = node_stack[stack_ptr]
            ref node = self.nodes[node_idx]

            # check AABB
            if not node.bbox.hit(ray, Interval(ray_t.min, closest_so_far)):
                continue

            # leaf node = object
            if node.object_idx != -1:
                ref obj = self.objects[node.object_idx]
                var hit_res: Optional[HitRecord]

                if obj.isa[Sphere]():
                    hit_res = obj[Sphere].hit(
                        ray, Interval(ray_t.min, closest_so_far)
                    )
                else:
                    print("ooooooops")
                    abort()

                if hit_res:
                    ref rec = hit_res.value()
                    closest_so_far = rec.t
                    hit_anything = hit_res^

            # internal node = check children
            else:
                # push children to stack
                node_stack[stack_ptr] = node.right_idx
                stack_ptr += 1
                node_stack[stack_ptr] = node.left_idx
                stack_ptr += 1

        return hit_anything^

    fn _build(mut self, start: Int, end: Int) -> Int:
        var span_len = end - start

        # leaf node
        if span_len == 1:
            box = get_bounding_box(self.objects[start])
            node = BVHNode(box^, -1, -1, start)
            self.nodes.append(node^)
            return len(self.nodes) - 1

        aabb = get_bounding_box(self.objects[start])
        for i in range(start + 1, end):
            aabb.merge(get_bounding_box(self.objects[i]))
        axis = longest_axis(aabb.edges())

        # internal node
        fn cmp_fn(a: HittableVariant, b: HittableVariant) capturing -> Bool:
            var box_a = get_bounding_box(a)
            var box_b = get_bounding_box(b)
            return box_a.min[axis] < box_b.min[axis]

        # SIMD version
        sort[cmp_fn=cmp_fn, stable=True](self.objects[start : start + span_len])
        # # not SIMD version
        # sort[cmp_fn=cmp_fn](self.objects[start : start + span_len])

        mid = start + span_len // 2

        # Recursively build children
        var left_idx = self._build(start, mid)
        var right_idx = self._build(mid, end)

        # Compute combined bounding box
        ref box_l = self.nodes[left_idx].bbox
        ref box_r = self.nodes[right_idx].bbox
        var combined_box = AABB(box_l, box_r)

        node = BVHNode(combined_box^, left_idx, right_idx, -1)
        self.nodes.append(node^)
        return len(self.nodes) - 1


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

    fn __init__(
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

    fn ray_color(self, ray: Ray, world: BVH, mut rng: PhiloxRNG) -> Color:
        var cur_ray = ray.copy()
        var accumulated_attenuation = Color(1.0, 1.0, 1.0)

        for _bounce in range(self.max_depth):
            comptime infinity = max_finite[DType.float32]()
            var hit_res = world.hit(cur_ray, Interval(Float32(0.001), infinity))

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

                var unit_direction = cur_ray.direction.copy()
                a = 0.5 * (unit_direction.y() + 1.0)
                var sky_color = (1.0 - a) * start_value + a * end_value
                # Final result is the sky color tinted by all previous bounces
                return accumulated_attenuation * sky_color

        # If we exceeded the depth without hitting the sky, return black
        return Color(0)

    fn render(self, world: BVH) raises:
        var image_data = List[Color](
            length=self.image_width * self.image_height, fill=Color(0)
        )

        @parameter
        fn worker(j: Int):
            rng = PhiloxRNG(seed=123, id=UInt64(j))
            factor = Float32(1.0 / Float32(self.samples_per_pixel))
            for i in range(self.image_width):
                var pixel_color = Color(0)
                for _sample in range(self.samples_per_pixel):
                    r = self.get_ray(i, j, rng)
                    pixel_color += self.ray_color(r, world, rng)

                image_data[j * self.image_width + i] = pixel_color * factor

        parallelize[worker](self.image_height, self.image_height)
        # parallelize[worker](self.image_height, 1)

        with open("rtiaw_2.ppm", "w") as f:
            f.write(t"P3\n{self.image_width} {self.image_height}\n255\n")
            for color in image_data:
                write_color(f, color)

    fn get_ray(self, i: Int, j: Int, mut rng: PhiloxRNG) -> Ray:
        var r1 = rng.next_f32()
        var r2 = rng.next_f32()
        offset = Vec2f32(r1, r2)
        var pixel_sample = (
            self.pixel00_loc
            + ((Float32(i) + offset.x()) * self.pixel_delta_u)
            + ((Float32(j) + offset.y()) * self.pixel_delta_v)
        )

        origin = (
            self.center.copy() if self.defocus_angle
            <= 0 else self.defocus_disk_sample(rng)
        )
        direction = pixel_sample - origin
        time = rng.next_f32()

        return Ray(origin, direction, time)

    fn defocus_disk_sample(self, mut rng: PhiloxRNG) -> Vec3f32:
        """Returns a random point in the camera defocus disk."""
        p = random_in_unit_disk(rng)
        return (
            self.center
            + (p[0] * self.defocus_disk_u)
            + (p[1] * self.defocus_disk_v)
        )


fn reflect(v: Vec3f32, n: Vec3f32) -> Vec3f32:
    return v - 2.0 * dot(v, n) * n


fn refract(uv: Vec3f32, n: Vec3f32, etai_over_etat: Float32) -> Vec3f32:
    var cos_theta = min(dot(-uv, n), 1.0)
    var r_out_perp = etai_over_etat * (uv + cos_theta * n)
    var r_out_parallel = -sqrt(abs(1.0 - length2(r_out_perp))) * n
    return r_out_perp + r_out_parallel


trait Material(Copyable):
    fn scatter(
        self, ray: Ray, hit: HitRecord, mut rng: PhiloxRNG
    ) -> Optional[Tuple[Ray, Color]]:
        ...


comptime MaterialVariant = Variant[Lambertian, Metal, Dielectric]


@fieldwise_init
struct Lambertian(Material, Writable):
    var albedo: Vec3f32

    fn scatter(
        self, ray: Ray, hit: HitRecord, mut rng: PhiloxRNG
    ) -> Optional[Tuple[Ray, Color]]:
        var scatter_direction = hit.normal + random_unit_vector(rng)

        # Catch degenerate scatter direction
        if scatter_direction.is_near_zero():
            scatter_direction = hit.normal.copy()

        scattered = Ray(hit.p, scatter_direction, ray.time)
        return (scattered^, self.albedo.copy())


@fieldwise_init
struct Metal(Material, Writable):
    var albedo: Vec3f32
    var fuzz: Float32

    fn scatter(
        self, ray: Ray, hit: HitRecord, mut rng: PhiloxRNG
    ) -> Optional[Tuple[Ray, Color]]:
        reflected = reflect(ray.direction, hit.normal)
        reflected = normalize(reflected) + (self.fuzz * random_unit_vector(rng))
        scattered = Ray(hit.p, reflected, ray.time)

        if dot(scattered.direction, hit.normal) < 0:
            return None

        return (scattered^, self.albedo.copy())


@fieldwise_init
struct Dielectric(Material, Writable):
    var refraction_index: Float32

    fn scatter(
        self, ray: Ray, hit: HitRecord, mut rng: PhiloxRNG
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
        var _rng = rng.next_f32()
        if cannot_refract or reflectance(cos_theta, ri) > _rng:
            direction = reflect(unit_direction, hit.normal)
        else:
            direction = refract(unit_direction, hit.normal, ri)

        scattered = Ray(hit.p, direction, ray.time)
        return (scattered^, attenuation.copy())


fn reflectance(cosine: Float32, ref_idx: Float32) -> Float32:
    """Schlick's approximation for reflectance."""
    var r0 = pow(((1.0 - ref_idx) / (1.0 + ref_idx)), 2)
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5)


fn create_random_scene() -> Scene:
    rng = PhiloxRNG(123, 321)
    materials = List[MaterialVariant]()
    objects = List[HittableVariant]()

    # Ground material
    materials.append(Lambertian(Color(0.5, 0.5, 0.5)))
    objects.append(Sphere(Point3(0, -1000, 0), 1000, 0))

    # Random small spheres
    for a in range(-11, 11):
        for b in range(-11, 11):
            var choose_mat = rng.next_f32()
            center = Point3(
                Float32(a) + 0.9 * rng.next_f32(),
                0.2,
                Float32(b) + 0.9 * rng.next_f32(),
            )

            if length(center - Point3(4, 0.2, 0)) > 0.9:
                if choose_mat < 0.8:
                    # Diffuse (Lambertian)
                    vr1 = Vec3f32(
                        rng.next_f32(), rng.next_f32(), rng.next_f32()
                    )
                    vr2 = Vec3f32(
                        rng.next_f32(), rng.next_f32(), rng.next_f32()
                    )
                    albedo = vr1 * vr2
                    materials.append(Lambertian(albedo^))
                    var center_dir = Vec3f32(0, rng.next_f32() * 0.5, 0)
                    var center_ray = Ray(center, center_dir, 0.2)
                    objects.append(Sphere(center_ray^, 0.2, len(materials) - 1))

                elif choose_mat < 0.95:
                    # Metal
                    albedo = (
                        Vec3f32(rng.next_f32(), rng.next_f32(), rng.next_f32())
                        + 0.5
                    )
                    fuzz = rng.next_f32() * 0.5
                    materials.append(Metal(albedo^, fuzz))
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
        image_width=400,
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
    world = BVH(objects^, materials^)
    return Scene(cam^, world^)


fn create_basic_scene() -> Scene:
    world = BVH(
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
        image_width=400,
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


fn create_top_scene() -> Scene:
    R = Float32(cos(pi / 4))
    world = BVH(
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
