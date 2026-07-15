from std.algorithm import parallelize
from std.io.file_descriptor import FileDescriptor
from std.math import abs, fma, sqrt
from std.time import perf_counter_ns

from bajo.core import (
    Vec3f32,
    dot,
    length2,
    normalize,
    Point3f32,
    Frame,
    Vec3,
    Rayf32,
)
from bajo.core.random import (
    Rng,
    random_in_unit_disk,
    random_on_hemisphere,
    random_unit_vector,
)
from bajo.bvh.camera import Camera
from bajo.bvh.constants import TRACE, f32_max
from bajo.rt.types import (
    Color,
    Dielectric,
    HitRecord,
    Lambertian,
    MAT_DIELECTRIC,
    MAT_LAMBERTIAN,
    MAT_METAL,
    Metal,
    PRIM_SPHERE,
    PrimitiveId,
    RENDER_AO,
    RENDER_MIS,
    RENDER_NEE,
    RENDER_NORMALS,
    RENDER_PATH,
    RenderResult,
    RenderSettings,
    RenderTimings,
    ScatterResult,
    SurfaceId,
    SurfaceStore,
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
                PrimitiveId(PRIM_SPHERE, UInt32(0)),
                Point3f32[Frame.WORLD](0.0),
                Vec3f32[Frame.WORLD](0.0),
                SurfaceId(MAT_LAMBERTIAN, UInt32(0)),
                f32_max,
                True,
            ),
        )


@fieldwise_init
struct PathState(Copyable, Writable):
    var path_id: UInt32
    var pixel_id: UInt32
    var ray: Rayf32[Frame.WORLD]
    var throughput: Color


@fieldwise_init
struct ShadeWork(Copyable, Writable):
    var path: PathState
    var hit: HitRecord


def reflect[
    dtype: DType, frame: Frame
](v: Vec3[dtype, frame], n: Vec3[dtype, frame]) -> Vec3[dtype, frame]:
    return v - 2.0 * dot(v, n) * n


def refract[
    dtype: DType, frame: Frame
](
    uv: Vec3[dtype, frame],
    n: Vec3[dtype, frame],
    etai_over_etat: Scalar[dtype],
) -> Vec3[dtype, frame]:
    var cos_theta = min(dot(-uv, n), 1.0)
    var r_out_perp = etai_over_etat * (uv + cos_theta * n)
    var r_out_parallel = -sqrt(abs(1.0 - length2(r_out_perp))) * n
    return r_out_perp + r_out_parallel


def reflectance[
    dtype: DType
](cosine: Scalar[dtype], ref_idx: Scalar[dtype]) -> Scalar[dtype]:
    var root = (1.0 - ref_idx) / (1.0 + ref_idx)
    var r2 = root * root
    var x = 1.0 - cosine
    var x2 = x * x
    var x5 = x2 * x2 * x
    return fma(1.0 - r2, x5, r2)


def scatter_lambertian(
    ref material: Lambertian,
    hit: HitRecord,
    mut rng: Rng,
) -> ScatterResult:
    var scatter_direction = hit.normal + random_unit_vector[Frame.WORLD](rng)
    if scatter_direction.is_near_zero():
        scatter_direction = hit.normal

    return ScatterResult(
        True,
        Rayf32[Frame.WORLD](
            hit.p, normalize(scatter_direction), 0.001, f32_max
        ),
        material.albedo,
    )


def scatter_metal(
    ref material: Metal,
    ray: Rayf32[Frame.WORLD],
    hit: HitRecord,
    mut rng: Rng,
) -> ScatterResult:
    debug_assert["safe"](material.fuzz >= 0.0 and material.fuzz <= 1.0)

    var reflected = reflect[DType.float32, Frame.WORLD](
        normalize(ray.d), hit.normal
    )
    reflected = normalize(
        reflected + material.fuzz * random_unit_vector[Frame.WORLD](rng)
    )
    var scattered = Rayf32[Frame.WORLD](hit.p, reflected, 0.001, f32_max)
    return ScatterResult(
        dot(scattered.d, hit.normal) > 0.0,
        scattered,
        material.albedo,
    )


def scatter_dielectric(
    ref material: Dielectric,
    ray: Rayf32[Frame.WORLD],
    hit: HitRecord,
    mut rng: Rng,
) -> ScatterResult:
    debug_assert["safe"](material.refraction_index > 0.0)

    var ri = (
        1.0
        / material.refraction_index if hit.front_face else material.refraction_index
    )
    var unit_direction = normalize(ray.d)
    var cos_theta = min(dot(-unit_direction, hit.normal), 1.0)
    var sin_theta = sqrt(1.0 - cos_theta * cos_theta)

    var direction: Vec3f32[Frame.WORLD]
    if ri * sin_theta > 1.0 or reflectance(cos_theta, ri) > rng.f32():
        direction = reflect[DType.float32, Frame.WORLD](
            unit_direction, hit.normal
        )
    else:
        direction = refract[DType.float32, Frame.WORLD](
            unit_direction, hit.normal, ri
        )

    return ScatterResult(
        True,
        Rayf32[Frame.WORLD](hit.p, normalize(direction), 0.001, f32_max),
        Color(1.0),
    )


def scatter(
    surface: SurfaceId,
    surfaces: SurfaceStore,
    ray: Rayf32[Frame.WORLD],
    hit: HitRecord,
    mut rng: Rng,
) -> ScatterResult:
    if surface.kind() == MAT_LAMBERTIAN:
        ref material = surfaces.lambertians[Int(surface.index())]
        return scatter_lambertian(material, hit, rng)
    elif surface.kind() == MAT_METAL:
        ref material = surfaces.metals[Int(surface.index())]
        return scatter_metal(material, ray, hit, rng)
    elif surface.kind() == MAT_DIELECTRIC:
        ref material = surfaces.dielectrics[Int(surface.index())]
        return scatter_dielectric(material, ray, hit, rng)

    debug_assert["safe"](False, "unknown RT surface kind")
    return ScatterResult(
        False,
        Rayf32[Frame.WORLD](hit.p, hit.normal, 0.001, f32_max),
        Color(0.0),
    )


def _sky_color(ray: Rayf32[Frame.WORLD]) -> Color:
    var unit_direction = normalize(ray.d)
    var a = 0.5 * (unit_direction.y + 1.0)
    return (1.0 - a) * Color(1.0) + a * Color(0.5, 0.7, 1.0)


def _trace_world(world: World, ray: Rayf32[Frame.WORLD]) -> PathHit:
    var hit = world.trace(ray)
    if not hit:
        return PathHit.miss()
    return PathHit(True, hit.value().copy())


def _init_pixel_rngs(settings: RenderSettings) -> List[Rng]:
    var pixel_count = settings.image_width * settings.image_height
    var rngs = List[Rng](capacity=pixel_count)
    for pixel_idx in range(pixel_count):
        rngs.append(Rng(seed=settings.rng_seed, id=UInt64(pixel_idx)))

    return rngs^


def _init_path_rngs(settings: RenderSettings) -> List[Rng]:
    var path_count = (
        settings.image_width
        * settings.image_height
        * settings.samples_per_pixel
    )
    var rngs = List[Rng](capacity=path_count)
    for path_idx in range(path_count):
        rngs.append(Rng(seed=settings.rng_seed, id=UInt64(path_idx)))

    return rngs^


def _make_primary_ray(
    settings: RenderSettings,
    camera: Camera,
    px: Int,
    py: Int,
    mut rng: Rng,
) -> Rayf32[Frame.WORLD]:
    var lens = random_in_unit_disk[Frame.WORLD](rng)
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


def _trace_path[
    MAX_DEPTH: Int
](
    settings: RenderSettings,
    world: World,
    ray: Rayf32[Frame.WORLD],
    mut rng: Rng,
) -> Color:
    comptime assert MAX_DEPTH >= 0, "max depth must be non-negative"

    var cur_ray = ray
    var throughput = Color(1.0)

    for _bounce in range(MAX_DEPTH):
        var hit = _trace_world(world, cur_ray)
        if hit.hit:
            ref record = hit.record
            var scattered = scatter(
                record.surface,
                world.surfaces,
                cur_ray,
                record,
                rng,
            )
            if not scattered.ok:
                return Color(0.0)

            throughput *= scattered.attenuation
            cur_ray = scattered.ray
        else:
            return throughput * _sky_color(cur_ray)

    return Color(0.0)


def _trace_normals(world: World, ray: Rayf32[Frame.WORLD]) -> Color:
    var hit = _trace_world(world, ray)
    if not hit.hit:
        return Color(0.0)

    ref record = hit.record
    return 0.5 * (record.normal + Color(1.0))


def _trace_ao(world: World, ray: Rayf32[Frame.WORLD], mut rng: Rng) -> Color:
    var hit = _trace_world(world, ray)
    if not hit.hit:
        return _sky_color(ray)

    ref record = hit.record
    var ao_dir = random_on_hemisphere[Frame.WORLD](rng, record.normal)
    var ao_ray = Rayf32[Frame.WORLD](record.p, normalize(ao_dir), 0.001, 4.0)
    var occluder = world.trace(ao_ray)
    if occluder:
        return Color(0.08)

    return Color(1.0)


def _trace_algorithm[
    ALGORITHM: UInt32, MAX_DEPTH: Int
](
    settings: RenderSettings,
    world: World,
    ray: Rayf32[Frame.WORLD],
    mut rng: Rng,
) -> Color:
    comptime if ALGORITHM == RENDER_PATH:
        return _trace_path[MAX_DEPTH](settings, world, ray, rng)
    elif ALGORITHM == RENDER_NORMALS:
        return _trace_normals(world, ray)
    elif ALGORITHM == RENDER_AO:
        return _trace_ao(world, ray, rng)
    elif ALGORITHM == RENDER_NEE:
        comptime assert False, "RENDER_NEE render algorithm not implemented yet"
    elif ALGORITHM == RENDER_MIS:
        comptime assert False, "RENDER_MIS render algorithm not implemented yet"
    else:
        comptime assert False, "unknown RT render algorithm"


def _render_pixel[
    ALGORITHM: UInt32, MAX_DEPTH: Int
](
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
        pixel_color += _trace_algorithm[ALGORITHM, MAX_DEPTH](
            settings, world, ray, rng
        )

    return pixel_color * (1.0 / Float32(settings.samples_per_pixel))


def _append_scattered_path(
    mut next_paths: List[PathState],
    work: ShadeWork,
    scattered: ScatterResult,
):
    if scattered.ok:
        next_paths.append(
            PathState(
                work.path.path_id,
                work.path.pixel_id,
                scattered.ray,
                work.path.throughput * scattered.attenuation,
            )
        )


def _shade_lambertian_queue(
    mut next_paths: List[PathState],
    queue: List[ShadeWork],
    surfaces: SurfaceStore,
    mut rng_states: List[Rng],
):
    for work in queue:
        ref hit = work.hit
        ref material = surfaces.lambertians[Int(hit.surface.index())]
        ref rng = rng_states[Int(work.path.path_id)]
        var scattered = scatter_lambertian(material, hit, rng)
        _append_scattered_path(next_paths, work, scattered)


def _shade_metal_queue(
    mut next_paths: List[PathState],
    queue: List[ShadeWork],
    surfaces: SurfaceStore,
    mut rng_states: List[Rng],
):
    for work in queue:
        ref hit = work.hit
        ref material = surfaces.metals[Int(hit.surface.index())]
        ref rng = rng_states[Int(work.path.path_id)]
        var scattered = scatter_metal(material, work.path.ray, hit, rng)
        _append_scattered_path(next_paths, work, scattered)


def _shade_dielectric_queue(
    mut next_paths: List[PathState],
    queue: List[ShadeWork],
    surfaces: SurfaceStore,
    mut rng_states: List[Rng],
):
    for work in queue:
        ref hit = work.hit
        ref material = surfaces.dielectrics[Int(hit.surface.index())]
        ref rng = rng_states[Int(work.path.path_id)]
        var scattered = scatter_dielectric(material, work.path.ray, hit, rng)
        _append_scattered_path(next_paths, work, scattered)


def _make_initial_paths(
    settings: RenderSettings,
    camera: Camera,
    mut rng_states: List[Rng],
) -> List[PathState]:
    var pixel_count = settings.image_width * settings.image_height
    var path_count = pixel_count * settings.samples_per_pixel
    var paths = List[PathState](capacity=path_count)

    for pixel_idx in range(pixel_count):
        var px = pixel_idx % settings.image_width
        var py = pixel_idx / settings.image_width
        for sample_idx in range(settings.samples_per_pixel):
            var path_idx = pixel_idx * settings.samples_per_pixel + sample_idx
            ref rng = rng_states[path_idx]
            var ray = _make_primary_ray(settings, camera, px, py, rng)
            paths.append(
                PathState(
                    UInt32(path_idx),
                    UInt32(pixel_idx),
                    ray,
                    Color(1.0),
                )
            )

    return paths^


def render_wavefront[
    ALGORITHM: UInt32 = RENDER_PATH, MAX_DEPTH: Int = 8
](settings: RenderSettings, camera: Camera, world: World) -> RenderResult:
    comptime assert MAX_DEPTH >= 0, "max depth must be non-negative"

    comptime if ALGORITHM != RENDER_PATH:
        debug_assert["safe"](False, "wavefront currently supports RENDER_PATH")

    var total_t0 = perf_counter_ns()
    var pixel_count = settings.image_width * settings.image_height
    var init_t0 = perf_counter_ns()
    var rng_states = _init_path_rngs(settings)
    var pixels = List[Color](length=pixel_count, fill=Color(0.0))
    var active_paths = _make_initial_paths(settings, camera, rng_states)
    var init_t1 = perf_counter_ns()

    var render_t0 = perf_counter_ns()
    for _bounce in range(MAX_DEPTH):
        if len(active_paths) == 0:
            break

        var lambertian_queue = List[ShadeWork]()
        var metal_queue = List[ShadeWork]()
        var dielectric_queue = List[ShadeWork]()
        var next_paths = List[PathState](capacity=len(active_paths))

        for path in active_paths:
            var hit = _trace_world(world, path.ray)
            if hit.hit:
                ref record = hit.record
                var work = ShadeWork(path.copy(), record.copy())
                if record.surface.kind() == MAT_LAMBERTIAN:
                    lambertian_queue.append(work.copy())
                elif record.surface.kind() == MAT_METAL:
                    metal_queue.append(work.copy())
                elif record.surface.kind() == MAT_DIELECTRIC:
                    dielectric_queue.append(work.copy())
                else:
                    debug_assert["safe"](False, "unknown RT surface kind")
            else:
                pixels[Int(path.pixel_id)] += path.throughput * _sky_color(
                    path.ray
                )

        _shade_lambertian_queue(
            next_paths,
            lambertian_queue,
            world.surfaces,
            rng_states,
        )
        _shade_metal_queue(next_paths, metal_queue, world.surfaces, rng_states)
        _shade_dielectric_queue(
            next_paths,
            dielectric_queue,
            world.surfaces,
            rng_states,
        )

        active_paths = next_paths^

    var render_t1 = perf_counter_ns()

    for i in range(pixel_count):
        pixels[i] = pixels[i] * (1.0 / Float32(settings.samples_per_pixel))

    var total_t1 = perf_counter_ns()

    var timings = RenderTimings(
        Int(total_t1 - total_t0),
        Int(init_t1 - init_t0),
        Int(render_t1 - render_t0),
        pixel_count,
        pixel_count * settings.samples_per_pixel,
        MAX_DEPTH,
    )
    return RenderResult(pixels^, timings)


def render_depth_first[
    ALGORITHM: UInt32 = RENDER_PATH, MAX_DEPTH: Int = 8
](settings: RenderSettings, camera: Camera, world: World) -> RenderResult:
    comptime assert MAX_DEPTH >= 0, "max depth must be non-negative"

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
            pixels[pixel_idx] = _render_pixel[ALGORITHM, MAX_DEPTH](
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
        MAX_DEPTH,
    )
    return RenderResult(pixels^, timings)


def render[
    ALGORITHM: UInt32 = RENDER_PATH, MAX_DEPTH: Int = 8
](settings: RenderSettings, camera: Camera, world: World) -> RenderResult:
    return render_depth_first[ALGORITHM, MAX_DEPTH](settings, camera, world)


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
