"""Host/device-safe path tracing helpers shared by CPU and GPU integrators."""

from bajo.core import Vec3, normalize
from bajo.core.random import Rng
from bajo.rt.types import Color
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


@always_inline
def sky_color[
    length: SIMDLength
](direction: Vec3[.float32, .WORLD, length]) -> Vec3[.float32, .WORLD, length]:
    var unit_direction = normalize(direction)
    var a = 0.5 * (unit_direction.y + 1.0)
    return (1.0 - a) * Vec3[.float32, .WORLD, length](1.0) + a * Vec3[
        .float32, .WORLD, length
    ](0.5, 0.7, 1.0)


@always_inline
def path_stage_rng(seed: UInt64, path_id: UInt32, stage: UInt32) -> Rng:
    """Create the deterministic Philox stream owned by one path stage."""
    return Rng(seed=seed, id=wavefront_rng_subsequence(path_id, stage))


@always_inline
def russian_roulette(
    seed: UInt64,
    path_id: UInt32,
    depth: UInt32,
    throughput: Color,
) -> RussianRouletteResult:
    """Unbiased continuation using a stream separate from BSDF sampling."""
    if depth < RUSSIAN_ROULETTE_START_DEPTH:
        return RussianRouletteResult(True, throughput)

    var maximum = max(throughput.x, max(throughput.y, throughput.z))
    if maximum <= 0.0:
        return RussianRouletteResult(False, throughput)
    var survival = min(
        max(maximum, RUSSIAN_ROULETTE_MIN_SURVIVAL),
        RUSSIAN_ROULETTE_MAX_SURVIVAL,
    )
    var rng = path_stage_rng(
        seed,
        path_id,
        wavefront_rng_roulette_stage(depth - UInt32(1)),
    )
    if rng.f32() >= survival:
        return RussianRouletteResult(False, throughput)
    return RussianRouletteResult(True, throughput / survival)
