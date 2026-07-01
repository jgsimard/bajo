from std.algorithm import parallelize
from std.io.file_descriptor import FileDescriptor
from std.math import abs, fma, sqrt
from std.time import perf_counter_ns

from bajo.core import Vec3f32, dot, length2, normalize
from bajo.core.random import Rng, random_in_unit_disk, random_unit_vector
from bajo.bvh.camera import Camera
from bajo.bvh.constants import TRACE, f32_max
from bajo.bvh.types import Ray
from bajo.rt.types import (
    Color,
    HitRecord,
    MAT_DIELECTRIC,
    MAT_LAMBERTIAN,
    MAT_METAL,
    Material,
    RenderResult,
    RenderSettings,
    RenderTimings,
    ScatterResult,
    World,
)


@fieldwise_init
struct PathHit(Copyable, Writable):
    var hit: Bool
    var record: HitRecord

    @staticmethod
    def miss() -> Self:
        return Self(
            False,
            HitRecord(
                Vec3f32(0.0),
                Vec3f32(0.0),
                UInt32(0),
                f32_max,
                True,
            ),
        )


def reflect(v: Vec3f32, n: Vec3f32) -> Vec3f32:
    return v - 2.0 * dot(v, n) * n


def refract(uv: Vec3f32, n: Vec3f32, etai_over_etat: Float32) -> Vec3f32:
    var cos_theta = min(dot(-uv, n), 1.0)
    var r_out_perp = etai_over_etat * (uv + cos_theta * n)
    var r_out_parallel = -sqrt(abs(1.0 - length2(r_out_perp))) * n
    return r_out_perp + r_out_parallel


def reflectance(cosine: Float32, ref_idx: Float32) -> Float32:
    var root = (1.0 - ref_idx) / (1.0 + ref_idx)
    var r2 = root * root
    var x = 1.0 - cosine
    var x2 = x * x
    var x5 = x2 * x2 * x
    return fma(1.0 - r2, x5, r2)


def scatter_lambertian(
    material: Material,
    hit: HitRecord,
    mut rng: Rng,
) -> ScatterResult:
    debug_assert["safe"](material.kind == MAT_LAMBERTIAN)

    var scatter_direction = hit.normal + random_unit_vector(rng)
    if scatter_direction.is_near_zero():
        scatter_direction = hit.normal

    return ScatterResult(
        True,
        Ray(hit.p, normalize(scatter_direction), 0.001, f32_max),
        material.albedo,
    )


def scatter_metal(
    material: Material,
    ray: Ray,
    hit: HitRecord,
    mut rng: Rng,
) -> ScatterResult:
    debug_assert["safe"](material.kind == MAT_METAL)
    debug_assert["safe"](material.fuzz >= 0.0 and material.fuzz <= 1.0)

    var reflected = reflect(normalize(ray.d), hit.normal)
    reflected = normalize(reflected + material.fuzz * random_unit_vector(rng))
    var scattered = Ray(hit.p, reflected, 0.001, f32_max)
    return ScatterResult(
        dot(scattered.d, hit.normal) > 0.0,
        scattered,
        material.albedo,
    )


def scatter_dielectric(
    material: Material,
    ray: Ray,
    hit: HitRecord,
    mut rng: Rng,
) -> ScatterResult:
    debug_assert["safe"](material.kind == MAT_DIELECTRIC)
    debug_assert["safe"](material.refraction_index > 0.0)

    var ri = (
        1.0
        / material.refraction_index if hit.front_face else material.refraction_index
    )
    var unit_direction = normalize(ray.d)
    var cos_theta = min(dot(-unit_direction, hit.normal), 1.0)
    var sin_theta = sqrt(1.0 - cos_theta * cos_theta)

    var direction: Vec3f32
    if ri * sin_theta > 1.0 or reflectance(cos_theta, ri) > rng.f32():
        direction = reflect(unit_direction, hit.normal)
    else:
        direction = refract(unit_direction, hit.normal, ri)

    return ScatterResult(
        True,
        Ray(hit.p, normalize(direction), 0.001, f32_max),
        Color(1.0),
    )


def scatter(
    material: Material,
    ray: Ray,
    hit: HitRecord,
    mut rng: Rng,
) -> ScatterResult:
    if material.kind == MAT_LAMBERTIAN:
        return scatter_lambertian(material, hit, rng)
    elif material.kind == MAT_METAL:
        return scatter_metal(material, ray, hit, rng)
    elif material.kind == MAT_DIELECTRIC:
        return scatter_dielectric(material, ray, hit, rng)

    debug_assert["safe"](False, "unknown RT material kind")
    return ScatterResult(
        False, Ray(hit.p, hit.normal, 0.001, f32_max), Color(0.0)
    )


def _sky_color(ray: Ray) -> Color:
    var unit_direction = normalize(ray.d)
    var a = 0.5 * (unit_direction.y + 1.0)
    return (1.0 - a) * Color(1.0) + a * Color(0.5, 0.7, 1.0)


def _trace_world(world: World, ray: Ray) -> PathHit:
    var bvh_hit = world.bvh.trace[TRACE.CLOSEST_HIT](ray)
    if not bvh_hit.is_hit():
        return PathHit.miss()

    var sphere_idx = Int(bvh_hit.prim)
    debug_assert["safe"](
        sphere_idx >= 0 and sphere_idx < len(world.spheres),
        "BVH returned an out-of-range sphere index",
    )
    ref sphere = world.spheres[sphere_idx]
    debug_assert["safe"](
        sphere.material_id < UInt32(len(world.materials)),
        "hit sphere material_id is out of range",
    )

    var p = ray.o + bvh_hit.t * ray.d
    var outward_normal = (p - sphere.center) / sphere.radius
    var front_face = dot(ray.d, outward_normal) < 0.0
    var normal = outward_normal if front_face else -outward_normal

    return PathHit(
        True,
        HitRecord(
            p,
            normal,
            sphere.material_id,
            bvh_hit.t,
            front_face,
        ),
    )


def _init_pixel_rngs(settings: RenderSettings) -> List[Rng]:
    var pixel_count = settings.image_width * settings.image_height
    var rngs = List[Rng](capacity=pixel_count)
    for pixel_idx in range(pixel_count):
        rngs.append(Rng(seed=settings.rng_seed, id=UInt64(pixel_idx)))

    return rngs^


def _make_primary_ray(
    settings: RenderSettings,
    camera: Camera,
    px: Int,
    py: Int,
    mut rng: Rng,
) -> Ray:
    var lens = random_in_unit_disk(rng)
    return camera.make_ray_sampled(
        px,
        py,
        settings.image_width,
        settings.image_height,
        rng.f32(),
        rng.f32(),
        lens.x,
        lens.y,
        0.001,
    )


def _trace_path(
    settings: RenderSettings,
    world: World,
    ray: Ray,
    mut rng: Rng,
) -> Color:
    var cur_ray = ray
    var throughput = Color(1.0)

    for _bounce in range(settings.max_depth):
        var hit = _trace_world(world, cur_ray)
        if hit.hit:
            ref record = hit.record
            var material = world.materials[Int(record.material_id)].copy()
            var scattered = scatter(material, cur_ray, record, rng)
            if not scattered.ok:
                return Color(0.0)

            throughput *= scattered.attenuation
            cur_ray = scattered.ray
        else:
            return throughput * _sky_color(cur_ray)

    return Color(0.0)


def _render_pixel(
    settings: RenderSettings,
    camera: Camera,
    world: World,
    px: Int,
    py: Int,
    mut rng: Rng,
) -> Color:
    var pixel_color = Color(0.0)

    for _sample in range(settings.samples_per_pixel):
        var ray = _make_primary_ray(settings, camera, px, py, rng)
        pixel_color += _trace_path(settings, world, ray, rng)

    return pixel_color * (1.0 / Float32(settings.samples_per_pixel))


def render(
    settings: RenderSettings, camera: Camera, world: World
) -> RenderResult:
    var total_t0 = perf_counter_ns()
    var pixel_count = settings.image_width * settings.image_height
    var init_t0 = perf_counter_ns()
    var rng_states = _init_pixel_rngs(settings)
    var pixels = List[Color](length=pixel_count, fill=Color(0.0))
    var init_t1 = perf_counter_ns()

    @parameter
    def worker(py: Int):
        for px in range(settings.image_width):
            var pixel_idx = py * settings.image_width + px
            ref rng = rng_states[pixel_idx]
            pixels[pixel_idx] = _render_pixel(
                settings,
                camera,
                world,
                px,
                py,
                rng,
            )

    var render_t0 = perf_counter_ns()
    parallelize[worker](settings.image_height, settings.image_height)
    var render_t1 = perf_counter_ns()
    var total_t1 = perf_counter_ns()

    var timings = RenderTimings(
        Int(total_t1 - total_t0),
        Int(init_t1 - init_t0),
        Int(render_t1 - render_t0),
        pixel_count,
        pixel_count * settings.samples_per_pixel,
        settings.max_depth,
    )
    return RenderResult(pixels^, timings)


def linear_to_gamma(x: Float32) -> Float32:
    if x <= 0.0:
        return 0.0
    return sqrt(x)


def color_to_byte(x: Float32) -> UInt8:
    return UInt8((256.0 * linear_to_gamma(x).clamp(0.0, 0.999)))


def write_ppm_from_colors(
    path: String,
    width: Int,
    height: Int,
    pixels: List[Color],
) raises:
    with open(path, "w") as f:
        var fd = FileDescriptor(f)
        fd.write(t"P6\n{width} {height}\n255\n")

        var bytes = List[UInt8](length=width * height * 3, fill=0)
        var out = bytes.unsafe_ptr()
        var out_i = 0

        for p in pixels:
            out[out_i + 0] = color_to_byte(p.x)
            out[out_i + 1] = color_to_byte(p.y)
            out[out_i + 2] = color_to_byte(p.z)
            out_i += 3

        fd.write_bytes(bytes)
