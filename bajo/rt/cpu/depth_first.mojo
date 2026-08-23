"""Depth-first tiled CPU renderer."""

from max.algorithm import parallelize
from std.io.file_descriptor import FileDescriptor
from std.math import ceildiv, sqrt
from std.sys import num_logical_cores
from std.time import perf_counter_ns

from bajo.bvh.constants import f32_max
from bajo.core import Rayf32, normalize
from bajo.core.random import Rng, random_on_hemisphere
from bajo.bvh import Camera
from bajo.rt.types import (
    Color,
    Integrator,
    RenderResult,
    RenderSettings,
    RenderTimings,
)
from .scene import CpuScene
from bajo.rt.common import path_stage_rng, russian_roulette, sky_color


from .bsdf import sample_bsdf
from .common import (
    _init_pixel_rngs,
    _make_primary_ray,
    _shading_point,
)
from .lighting import (
    _emissive_hit_weight,
    emitted_radiance,
    sample_direct_lighting,
)
from bajo.rt.wavefront_contract import wavefront_rng_light_stage


comptime CPU_RENDER_TILE_WIDTH = 16
comptime CPU_RENDER_TILE_HEIGHT = 16


def _trace_path[
    integrator: Integrator,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    settings: RenderSettings,
    world: CpuScene[world_bvh_width, instance_bvh_width],
    ray: Rayf32[.WORLD],
    mut rng: Rng,
    path_id: UInt32,
) -> Color:
    comptime assert integrator in (Integrator.PATH, Integrator.NEE, Integrator.MIS)

    var cur_ray = ray
    var throughput = Color(1.0)
    var radiance = Color(0.0)
    var previous_delta = True
    var previous_bsdf_pdf = Float32(0.0)

    for _bounce in range(settings.max_depth):
        var hit = world.trace_surface(cur_ray)
        if hit.hit:
            var point = _shading_point(cur_ray, hit)
            var emission = emitted_radiance(
                hit.surface, world.scene_data().surfaces(), hit.front_face
            )
            if emission.x > 0.0 or emission.y > 0.0 or emission.z > 0.0:
                var emission_weight = _emissive_hit_weight[integrator](
                    world,
                    cur_ray,
                    hit,
                    _bounce,
                    previous_bsdf_pdf,
                    previous_delta,
                )
                radiance += throughput * emission * emission_weight
                return radiance
            comptime if integrator != .PATH:
                var light_rng = path_stage_rng(
                    settings.rng_seed,
                    path_id,
                    wavefront_rng_light_stage(UInt32(_bounce)),
                )
                var direct = sample_direct_lighting[integrator](
                    hit.surface, world, cur_ray, point, light_rng
                )
                radiance += throughput * direct
            var scattered = sample_bsdf(
                hit.surface,
                world.scene_data().surfaces(),
                cur_ray,
                point,
                rng,
            )
            if not scattered.ok:
                return radiance

            throughput *= scattered.weight
            previous_delta = scattered.delta
            previous_bsdf_pdf = scattered.pdf
            var roulette = russian_roulette(
                settings.rng_seed,
                path_id,
                UInt32(_bounce + 1),
                throughput,
            )
            if not roulette.survived:
                return radiance
            throughput = roulette.throughput
            cur_ray = Rayf32[.WORLD](
                point.p, scattered.direction, 0.001, f32_max
            )
        else:
            return radiance + throughput * sky_color(cur_ray.d)

    return radiance


def _trace_normals[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    world: CpuScene[world_bvh_width, instance_bvh_width],
    ray: Rayf32[.WORLD],
) -> Color:
    var hit = world.trace_surface(ray)
    if not hit.hit:
        return Color(0.0)

    return 0.5 * (hit.normal + Color(1.0))


def _trace_ao[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    world: CpuScene[world_bvh_width, instance_bvh_width],
    ray: Rayf32[.WORLD],
    mut rng: Rng,
) -> Color:
    var hit = world.trace_surface(ray)
    if not hit.hit:
        return sky_color(ray.d)

    var p = ray.o + hit.t * ray.d
    var ao_dir = random_on_hemisphere[.WORLD](rng, hit.normal)
    var ao_ray = Rayf32[.WORLD](p, normalize(ao_dir), 0.001, 4.0)
    if world.occluded(ao_ray):
        return Color(0.08)

    return Color(1.0)


def _trace_integrator[
    integrator: Integrator,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    settings: RenderSettings,
    world: CpuScene[world_bvh_width, instance_bvh_width],
    ray: Rayf32[.WORLD],
    mut rng: Rng,
    path_id: UInt32,
) -> Color:
    comptime if integrator == .PATH:
        return _trace_path[.PATH, world_bvh_width, instance_bvh_width](
            settings, world, ray, rng, path_id
        )
    elif integrator == .NORMALS:
        return _trace_normals(world, ray)
    elif integrator == .AO:
        return _trace_ao(world, ray, rng)
    elif integrator == .NEE:
        return _trace_path[.NEE, world_bvh_width, instance_bvh_width](
            settings, world, ray, rng, path_id
        )
    elif integrator == .MIS:
        return _trace_path[.MIS, world_bvh_width, instance_bvh_width](
            settings, world, ray, rng, path_id
        )
    else:
        comptime assert False, "unknown RT integrator"


def _render_pixel[
    integrator: Integrator,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    settings: RenderSettings,
    camera: Camera,
    world: CpuScene[world_bvh_width, instance_bvh_width],
    px: Int,
    py: Int,
    mut rng: Rng,
) -> Color:
    var pixel_color = Color(0.0)

    var pixel_idx = py * settings.image_width + px
    for sample_idx in range(settings.samples_per_pixel):
        var path_id = UInt32(
            pixel_idx * settings.samples_per_pixel + sample_idx
        )
        var ray = _make_primary_ray(settings, camera, px, py, rng)
        pixel_color += _trace_integrator[
            integrator, world_bvh_width, instance_bvh_width
        ](settings, world, ray, rng, path_id)

    return pixel_color * (1.0 / Float32(settings.samples_per_pixel))


def render_depth_first[
    integrator: Integrator = .PATH,
    TILE_WIDTH: Int = CPU_RENDER_TILE_WIDTH,
    TILE_HEIGHT: Int = CPU_RENDER_TILE_HEIGHT,
    SCHEDULER_MODE: Int = 2,
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
](
    settings: RenderSettings,
    camera: Camera,
    world: CpuScene[world_bvh_width, instance_bvh_width],
) -> RenderResult:
    """Render depth-first using compile-time tile and scheduling choices."""
    # Mode 0 uses the runtime default, 1 caps workers to logical cores, and 2
    # exposes one worker per tile.
    comptime assert TILE_WIDTH > 0, "tile width must be positive"
    comptime assert TILE_HEIGHT > 0, "tile height must be positive"
    comptime assert 0 <= SCHEDULER_MODE <= 2, "unknown scheduler mode"

    var total_t0 = perf_counter_ns()
    var pixel_count = settings.image_width * settings.image_height
    var init_t0 = perf_counter_ns()
    var rng_states = _init_pixel_rngs(settings)
    var pixels = List[Color](length=pixel_count, fill=Color(0.0))
    var init_t1 = perf_counter_ns()

    var tiles_x = ceildiv(settings.image_width, TILE_WIDTH)
    var tiles_y = ceildiv(settings.image_height, TILE_HEIGHT)
    var tile_count = tiles_x * tiles_y

    def worker(tile_idx: Int) {imm, mut pixels, mut rng_states}:
        var tile_x = tile_idx % tiles_x
        var tile_y = tile_idx / tiles_x
        var x0 = tile_x * TILE_WIDTH
        var y0 = tile_y * TILE_HEIGHT
        var x1 = min(x0 + TILE_WIDTH, settings.image_width)
        var y1 = min(y0 + TILE_HEIGHT, settings.image_height)
        for py in range(y0, y1):
            for px in range(x0, x1):
                var pixel_idx = py * settings.image_width + px
                ref rng = rng_states[pixel_idx]
                pixels[pixel_idx] = _render_pixel[
                    integrator,
                    world_bvh_width,
                    instance_bvh_width,
                ](
                    settings,
                    camera,
                    world,
                    px,
                    py,
                    rng,
                )

    var render_t0 = perf_counter_ns()
    comptime if SCHEDULER_MODE == 1:
        parallelize(worker, tile_count, min(num_logical_cores(), tile_count))
    elif SCHEDULER_MODE == 2:
        parallelize(worker, tile_count, tile_count)
    else:
        parallelize(worker, tile_count)
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
    pixels: ImmSpan[Color, _],
) raises:
    with open(path, "w") as f:
        var fd = FileDescriptor(f)
        fd.write(t"P6\n{width} {height}\n255\n")

        var bytes = List[UInt8](length=width * height * 3, fill=0)
        var out_i = 0

        for p in pixels:
            bytes[out_i + 0] = color_to_byte(p.x)
            bytes[out_i + 1] = color_to_byte(p.y)
            bytes[out_i + 2] = color_to_byte(p.z)
            out_i += 3

        fd.write_bytes(bytes)
