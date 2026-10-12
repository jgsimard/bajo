"""Shared surface continuation math; queue reservation stays in each backend."""

from bajo.core import Point3, Vec3, dot
from bajo.rt.common import (
    path_stage_rng,
    russian_roulette,
    RUSSIAN_ROULETTE_START_DEPTH,
    RUSSIAN_ROULETTE_MIN_SURVIVAL,
    RUSSIAN_ROULETTE_MAX_SURVIVAL,
)
from bajo.rt.render_types import SamplingConfig, Color
from bajo.rt.shading_types import BsdfSample
from bajo.rt.path_values import PathContinuation
from bajo.rt.wavefront_contract import wavefront_rng_roulette_stage


@fieldwise_init
struct ContinuationBatch[length: SIMDLength](TrivialRegisterPassable):
    var values: PathContinuation[Self.length]
    var active: SIMD[.bool, Self.length]


@always_inline
def continuation_medium[
    length: SIMDLength
](
    previous: SIMD[.uint32, length],
    boundary: SIMD[.uint32, length],
    front_face: SIMD[.bool, length],
    direction: Vec3[.float32, .WORLD, length],
    geometric: Vec3[.float32, .WORLD, length],
) -> SIMD[.uint32, length]:
    var transmitted = front_face.select(
        boundary & UInt32(65535), boundary >> UInt32(16)
    )
    var crossing = boundary.ne(0) & dot(direction, geometric).lt(0.0)
    return crossing.select(transmitted, previous)


@always_inline
def finish_continuation[
    length: SIMDLength
](
    path_ids: SIMD[.uint32, length],
    origin: Point3[.float32, .WORLD, length],
    incident: Vec3[.float32, .WORLD, length],
    hit_t: SIMD[.float32, length],
    throughput: Vec3[.float32, .WORLD, length],
    medium_ids: SIMD[.uint32, length],
    sampled: BsdfSample[length],
    active: SIMD[.bool, length],
    sampling: SamplingConfig,
    depth: UInt32,
) -> ContinuationBatch[length]:
    var weight = throughput * sampled.weight
    var survived = active & sampled.ok
    comptime if length == 1:
        if survived[0]:
            var roulette = russian_roulette(
                sampling,
                path_ids[0],
                depth,
                Color(weight.x[0], weight.y[0], weight.z[0]),
            )
            survived = roulette.survived
            weight = Vec3[.float32, .WORLD, length](
                roulette.throughput.x[0],
                roulette.throughput.y[0],
                roulette.throughput.z[0],
            )
    else:
        if depth >= RUSSIAN_ROULETTE_START_DEPTH:
            var maximum = max(weight.x, max(weight.y, weight.z))
            var survival = min(
                max(maximum, RUSSIAN_ROULETTE_MIN_SURVIVAL),
                RUSSIAN_ROULETTE_MAX_SURVIVAL,
            )
            var eligible = survived & maximum.gt(0.0)
            var random = SIMD[.float32, length](1.0)
            for lane in range(Int(length)):
                if eligible[lane]:
                    var rng = path_stage_rng(
                        sampling,
                        path_ids[lane],
                        wavefront_rng_roulette_stage(depth - UInt32(1)),
                    )
                    random[lane] = rng.f32()
            survived = eligible & random.lt(survival)
            var divisor = survived.select(survival, 1.0)
            weight = weight / divisor
    return ContinuationBatch[length](
        PathContinuation[length](
            path_ids,
            origin + hit_t * incident,
            sampled.direction,
            weight,
            sampled.pdf,
            sampled.delta,
            medium_ids,
        ),
        survived,
    )
