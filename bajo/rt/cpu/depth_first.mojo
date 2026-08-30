"""Depth-first tiled CPU renderer."""

from max.algorithm import parallelize
from std.io.file_descriptor import FileDescriptor
from std.math import ceildiv, sqrt
from std.sys import num_logical_cores
from std.time import perf_counter_ns

from bajo.core import Rayf32
from bajo.core.random import Rng
from bajo.bvh import Camera
from bajo.rt.types import (
    Color,
    Integrator,
    RenderResult,
    RenderSettings,
    RenderTimings,
    SamplingConfig,
    ShadingPoint,
)
from .scene import CpuScene
from .scheduler_mode import CpuSchedulerMode
from bajo.rt.common import path_stage_rng, russian_roulette, sky_color
from bajo.rt.rays import make_ao_ray, spawn_surface_ray


from .bsdf import sample_bsdf
from .common import (
    _make_primary_ray,
)
from .lighting import (
    _emissive_hit_weight,
    sample_direct_lighting,
)
from bajo.rt.wavefront_contract import (
    wavefront_rng_light_stage,
    wavefront_rng_stage,
)


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
    path_id: UInt32,
) -> Color where Integrator.is_path_tracing[integrator]:
    var cur_ray = ray
    var throughput = Color(1.0)
    var radiance = Color(0.0)
    var previous_delta = True
    var previous_bsdf_pdf = Float32(0.0)
    var sampling = SamplingConfig.from_settings(settings)

    for _bounce in range(settings.max_depth):
        var hit = world.trace_surface(cur_ray)
        if hit.hit:
            var point = ShadingPoint.from_hit(cur_ray, hit)
            var emission = (
                world.scene_data()
                .surfaces()
                .emitted_radiance(hit.surface, hit.front_face)
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
            comptime if Integrator.uses_direct_lighting[integrator]:
                var light_rng = path_stage_rng(
                    sampling,
                    path_id,
                    wavefront_rng_light_stage(UInt32(_bounce)),
                )
                var direct = sample_direct_lighting[integrator](
                    hit.surface, world, cur_ray, point, light_rng
                )
                radiance += throughput * direct
            var bsdf_rng = path_stage_rng(
                sampling,
                path_id,
                wavefront_rng_stage(UInt32(_bounce)),
            )
            var scattered = sample_bsdf(
                hit.surface,
                world.scene_data().surfaces(),
                cur_ray,
                point,
                bsdf_rng,
            )
            if not scattered.ok:
                return radiance

            throughput *= scattered.weight
            previous_delta = scattered.delta
            previous_bsdf_pdf = scattered.pdf
            var roulette = russian_roulette(
                sampling,
                path_id,
                UInt32(_bounce + 1),
                throughput,
            )
            if not roulette.survived:
                return radiance
            throughput = roulette.throughput
            cur_ray = spawn_surface_ray(point.p, scattered.direction)
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

    var ao_ray = make_ao_ray(ray.at(hit.t), hit.normal, rng)
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
    path_id: UInt32,
) -> Color:
    comptime if integrator == .PATH:
        return _trace_path[.PATH, world_bvh_width, instance_bvh_width](
            settings, world, ray, path_id
        )
    elif integrator == .NORMALS:
        return _trace_normals(world, ray)
    elif integrator == .AO:
        var ao_rng = path_stage_rng(
            SamplingConfig.from_settings(settings), path_id, UInt32(1)
        )
        return _trace_ao(world, ray, ao_rng)
    elif integrator == .NEE:
        return _trace_path[.NEE, world_bvh_width, instance_bvh_width](
            settings, world, ray, path_id
        )
    elif integrator == .MIS:
        return _trace_path[.MIS, world_bvh_width, instance_bvh_width](
            settings, world, ray, path_id
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
) -> Color:
    var pixel_color = Color(0.0)

    var pixel_idx = py * settings.image_width + px
    for sample_idx in range(settings.samples_per_pixel):
        var path_id = UInt32(
            pixel_idx * settings.samples_per_pixel + sample_idx
        )
        var rng = path_stage_rng(
            SamplingConfig.from_settings(settings), path_id, 0
        )
        var ray = _make_primary_ray(settings, camera, px, py, rng)
        pixel_color += _trace_integrator[
            integrator, world_bvh_width, instance_bvh_width
        ](settings, world, ray, path_id)

    return pixel_color * (1.0 / Float32(settings.samples_per_pixel))


def render_depth_first[
    integrator: Integrator = .PATH,
    TILE_WIDTH: Int = CPU_RENDER_TILE_WIDTH,
    TILE_HEIGHT: Int = CPU_RENDER_TILE_HEIGHT,
    scheduler_mode: CpuSchedulerMode = .TASK_PARTITIONS,
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
](
    settings: RenderSettings,
    camera: Camera,
    world: CpuScene[world_bvh_width, instance_bvh_width],
) -> RenderResult where (
    TILE_WIDTH > 0,
    "tile width must be positive",
) where (
    TILE_HEIGHT > 0,
    "tile height must be positive",
):
    """Render depth-first using compile-time tile and scheduling choices."""
    comptime assert CpuSchedulerMode.is_valid[
        scheduler_mode
    ], "unknown scheduler mode"

    var total_t0 = perf_counter_ns()
    var pixel_count = settings.image_width * settings.image_height
    var init_t0 = perf_counter_ns()
    var pixels = List[Color](length=pixel_count, fill=Color(0.0))
    var init_t1 = perf_counter_ns()

    var tiles_x = ceildiv(settings.image_width, TILE_WIDTH)
    var tiles_y = ceildiv(settings.image_height, TILE_HEIGHT)
    var tile_count = tiles_x * tiles_y

    def worker(tile_idx: Int) {imm, mut pixels}:
        var tile_x = tile_idx % tiles_x
        var tile_y = tile_idx / tiles_x
        var x0 = tile_x * TILE_WIDTH
        var y0 = tile_y * TILE_HEIGHT
        var x1 = min(x0 + TILE_WIDTH, settings.image_width)
        var y1 = min(y0 + TILE_HEIGHT, settings.image_height)
        for py in range(y0, y1):
            for px in range(x0, x1):
                var pixel_idx = py * settings.image_width + px
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
                )

    var render_t0 = perf_counter_ns()
    comptime if scheduler_mode == .LOGICAL_CORES:
        parallelize(worker, tile_count, min(num_logical_cores(), tile_count))
    elif scheduler_mode == .TASK_PARTITIONS:
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
