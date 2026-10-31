"""Shared CPU/GPU primary sample dimensions and reconstruction-filter sampling."""

from bajo.core.random import random_in_unit_disk
from bajo.rt.common import path_stage_rng
from bajo.rt.render_types import SamplingConfig


@fieldwise_init
struct CameraSamples[width: SIMDLength](TrivialRegisterPassable):
    var pixel_u: SIMD[.float32, Self.width]
    var pixel_v: SIMD[.float32, Self.width]
    var lens_u: SIMD[.float32, Self.width]
    var lens_v: SIMD[.float32, Self.width]
    var filter_weight: SIMD[.float32, Self.width]


@always_inline
def sample_camera[
    width: SIMDLength
](sampling: SamplingConfig, ids: SIMD[.uint32, width]) -> CameraSamples[width]:
    var rng = path_stage_rng(sampling, ids, UInt32(0))
    var lens = random_in_unit_disk[.WORLD, width](rng)
    var filter = sampling.filter.sample(rng.f32(), rng.f32())
    return CameraSamples[width](
        filter.pixel_u, filter.pixel_v, lens.x, lens.y, filter.weight
    )
