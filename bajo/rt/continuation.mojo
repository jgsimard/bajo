"""Shared surface continuation math; queue reservation stays in each backend."""

from bajo.core import Point3, Vec3, dot
from bajo.rt.common import (
    russian_roulette,
)
from bajo.rt.render_types import SamplingConfig
from bajo.rt.shading_types import BsdfSample
from bajo.rt.path_values import PathContinuation


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
    if survived.reduce_or():
        var roulette = russian_roulette(sampling, path_ids, depth, weight)
        survived &= roulette.survived
        weight = roulette.throughput
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
