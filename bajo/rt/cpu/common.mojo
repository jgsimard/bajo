"""Shared CPU path state, Philox stream selection, and ray helpers."""

from bajo.core import Frame, Rayf32, normalize
from bajo.core.random import Rng, random_in_unit_disk
from bajo.bvh.camera import Camera
from bajo.rt.types import (
    Color,
    RenderSettings,
    ShadingPoint,
    SurfaceHit,
)
from bajo.rt.wavefront_contract import (
    wavefront_rng_roulette_stage,
    wavefront_rng_subsequence,
)


comptime RUSSIAN_ROULETTE_START_DEPTH = UInt32(5)
comptime RUSSIAN_ROULETTE_MIN_SURVIVAL = Float32(0.05)
comptime RUSSIAN_ROULETTE_MAX_SURVIVAL = Float32(0.95)


@fieldwise_init
struct RussianRouletteResult(Copyable, Writable):
    var survived: Bool
    var throughput: Color


def _sky_color(ray: Rayf32[Frame.WORLD]) -> Color:
    var unit_direction = normalize(ray.d)
    var a = 0.5 * (unit_direction.y + 1.0)
    return (1.0 - a) * Color(1.0) + a * Color(0.5, 0.7, 1.0)


def _shading_point(ray: Rayf32[Frame.WORLD], hit: SurfaceHit) -> ShadingPoint:
    return ShadingPoint(
        ray.o + hit.t * ray.d,
        hit.normal,
        hit.front_face,
    )


def _init_pixel_rngs(settings: RenderSettings) -> List[Rng]:
    return [
        Rng(seed=settings.rng_seed, id=UInt64(i))
        for i in range(settings.image_width * settings.image_height)
    ]


def _path_stage_rng(
    settings: RenderSettings, path_id: UInt32, stage: UInt32
) -> Rng:
    """Create a deterministic Philox stream for one path and render stage."""
    return Rng(
        seed=settings.rng_seed,
        id=wavefront_rng_subsequence(path_id, stage),
    )


@always_inline
def _russian_roulette(
    settings: RenderSettings,
    path_id: UInt32,
    depth: UInt32,
    throughput: Color,
) -> RussianRouletteResult:
    """Unbiased continuation using a Philox domain separate from the BSDF."""
    if depth < RUSSIAN_ROULETTE_START_DEPTH:
        return RussianRouletteResult(True, throughput)

    var maximum = max(throughput.x, max(throughput.y, throughput.z))
    if maximum <= 0.0:
        return RussianRouletteResult(False, throughput)
    var survival = min(
        max(maximum, RUSSIAN_ROULETTE_MIN_SURVIVAL),
        RUSSIAN_ROULETTE_MAX_SURVIVAL,
    )
    var rng = _path_stage_rng(
        settings,
        path_id,
        wavefront_rng_roulette_stage(depth - UInt32(1)),
    )
    if rng.f32() >= survival:
        return RussianRouletteResult(False, throughput)
    return RussianRouletteResult(True, throughput / survival)


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
