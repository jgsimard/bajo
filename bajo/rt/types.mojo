from std.math import abs

from bajo.core import Vec3f32, dot
from bajo.bvh.constants import TRACE
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.types import Ray, Sphere


comptime Color = Vec3f32
comptime Point3 = Vec3f32
comptime BVH_WIDTH = 8

comptime MAT_LAMBERTIAN = UInt32(0)
comptime MAT_METAL = UInt32(1)
comptime MAT_DIELECTRIC = UInt32(2)


@fieldwise_init
struct RtSphere(Copyable, Writable):
    var center: Point3
    var radius: Float32
    var material_id: UInt32


@fieldwise_init
struct Material(Copyable, Writable):
    var kind: UInt32
    var albedo: Color
    var fuzz: Float32
    var refraction_index: Float32

    @staticmethod
    def lambertian(albedo: Color) -> Self:
        return Self(MAT_LAMBERTIAN, albedo, 0.0, 1.0)

    @staticmethod
    def metal(albedo: Color, fuzz: Float32) -> Self:
        debug_assert["safe"](fuzz >= 0.0)
        debug_assert["safe"](fuzz <= 1.0)
        return Self(MAT_METAL, albedo, fuzz, 1.0)

    @staticmethod
    def dielectric(refraction_index: Float32) -> Self:
        debug_assert["safe"](refraction_index > 0.0)
        return Self(MAT_DIELECTRIC, Color(1.0), 0.0, refraction_index)


@fieldwise_init
struct HitRecord(Copyable, Writable):
    var p: Point3
    var normal: Vec3f32
    var material_id: UInt32
    var t: Float32
    var front_face: Bool


@fieldwise_init
struct ScatterResult(Copyable, Writable):
    var ok: Bool
    var ray: Ray
    var attenuation: Color


struct RenderSettings(Copyable, Writable):
    var image_width: Int
    var image_height: Int
    var samples_per_pixel: Int
    var max_depth: Int
    var rng_seed: UInt64

    def __init__(
        out self,
        image_width: Int,
        image_height: Int,
        samples_per_pixel: Int,
        max_depth: Int,
        rng_seed: UInt64,
    ):
        debug_assert["safe"](image_width > 0, "image width must be positive")
        debug_assert["safe"](image_height > 0, "image height must be positive")
        debug_assert["safe"](
            samples_per_pixel > 0, "samples per pixel must be positive"
        )
        debug_assert["safe"](max_depth >= 0, "max depth must be non-negative")

        self.image_width = image_width
        self.image_height = image_height
        self.samples_per_pixel = samples_per_pixel
        self.max_depth = max_depth
        self.rng_seed = rng_seed


@fieldwise_init
struct RenderTimings(Copyable, Writable):
    var total_ns: Int
    var init_ns: Int
    var render_ns: Int
    var pixel_count: Int
    var sample_count: Int
    var max_depth: Int


struct RenderResult(Movable):
    var pixels: List[Color]
    var timings: RenderTimings

    def __init__(
        out self,
        var pixels: List[Color],
        timings: RenderTimings,
    ):
        self.pixels = pixels^
        self.timings = timings.copy()


struct World(Movable):
    var bvh: SphereBvh[BVH_WIDTH]
    var spheres: List[RtSphere]
    var materials: List[Material]

    def __init__(
        out self,
        var spheres: List[RtSphere],
        var materials: List[Material],
    ):
        debug_assert["safe"](
            len(spheres) > 0, "world requires at least one sphere"
        )
        debug_assert["safe"](
            len(materials) > 0, "world requires at least one material"
        )

        self.spheres = spheres^
        self.materials = materials^

        var bvh_spheres = List[Sphere](capacity=len(self.spheres))
        for s in self.spheres:
            debug_assert["safe"](
                s.radius != 0.0, "sphere radius must be non-zero"
            )
            debug_assert["safe"](
                s.material_id < UInt32(len(self.materials)),
                "sphere material_id is out of range",
            )
            bvh_spheres.append(Sphere(s.center, abs(s.radius)))

        self.bvh = SphereBvh[BVH_WIDTH].__init__["lbvh"](bvh_spheres^)

    def hit(self, ray: Ray) -> Optional[HitRecord]:
        var bvh_hit = self.bvh.trace[TRACE.CLOSEST_HIT](ray)
        if not bvh_hit.is_hit():
            return None

        var sphere_idx = Int(bvh_hit.prim)
        debug_assert["safe"](
            sphere_idx >= 0 and sphere_idx < len(self.spheres),
            "BVH returned an out-of-range sphere index",
        )
        ref sphere = self.spheres[sphere_idx]
        debug_assert["safe"](
            sphere.material_id < UInt32(len(self.materials)),
            "hit sphere material_id is out of range",
        )
        var p = ray_at(ray, bvh_hit.t)
        var outward_normal = (p - sphere.center) / sphere.radius
        var front_face = dot(ray.d, outward_normal) < 0.0
        var normal = outward_normal if front_face else -outward_normal

        return HitRecord(
            p,
            normal,
            sphere.material_id,
            bvh_hit.t,
            front_face,
        )


def ray_at(ray: Ray, t: Float32) -> Point3:
    return ray.o + t * ray.d


def add_material(mut materials: List[Material], material: Material) -> UInt32:
    var material_id = UInt32(len(materials))
    materials.append(material.copy())
    return material_id


def add_sphere(
    mut spheres: List[RtSphere],
    center: Point3,
    radius: Float32,
    material_id: UInt32,
):
    debug_assert["safe"](radius != 0.0, "sphere radius must be non-zero")
    spheres.append(RtSphere(center, radius, material_id))
