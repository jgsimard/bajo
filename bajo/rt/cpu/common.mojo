"""Shared CPU path state, Philox stream selection, and ray helpers."""

from bajo.core import Rayf32
from bajo.core.random import Rng, random_in_unit_disk
from bajo.bvh import Camera
from bajo.rt.types import RenderSettings, ShadingPoint, SurfaceHit


def _shading_point(
    ray: Rayf32[.WORLD], hit: SurfaceHit[1]
) -> ShadingPoint[1]:
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


def _make_primary_ray(
    settings: RenderSettings,
    camera: Camera,
    px: Int,
    py: Int,
    mut rng: Rng,
) -> Rayf32[.WORLD]:
    var lens = random_in_unit_disk[.WORLD](rng)
    return camera.make_ray_sampled(
        Float32(px),
        Float32(py),
        Float32(settings.image_width),
        Float32(settings.image_height),
        rng.f32(),
        rng.f32(),
        lens.x,
        lens.y,
        0.001,
    )
