"""Shared CPU path state, Philox stream selection, and ray helpers."""

from bajo.core import Rayf32
from bajo.core.random import Rng, random_in_unit_disk
from bajo.bvh import Camera
from bajo.rt.rays import make_camera_ray_from_samples
from bajo.rt.types import RenderSettings


def _make_primary_ray(
    settings: RenderSettings,
    camera: Camera,
    px: Int,
    py: Int,
    mut rng: Rng,
) -> Rayf32[.WORLD]:
    var lens = random_in_unit_disk[.WORLD](rng)
    return make_camera_ray_from_samples(
        camera,
        Float32(px),
        Float32(py),
        Float32(settings.image_width),
        Float32(settings.image_height),
        rng.f32(),
        rng.f32(),
        lens.x,
        lens.y,
    )
