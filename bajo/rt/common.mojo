"""Host/device-safe path tracing helpers shared by CPU and GPU integrators."""

from std.math import abs, cos, pi, sin, sqrt

from bajo.core import Frame, Vec3, normalize
from bajo.core.random import Rng, Sampler, _mix32
from bajo.rt.render_types import (
    Color,
    SamplingConfig,
)
from bajo.rt.material_types import (
    Environment,
    EnvironmentKind,
    SurfaceStore,
)
from bajo.rt.lighting_types import _light_importance
from bajo.rt.wavefront_contract import (
    wavefront_rng_roulette_stage,
    wavefront_rng_subsequence,
)


comptime RUSSIAN_ROULETTE_START_DEPTH = UInt32(5)
comptime RUSSIAN_ROULETTE_MIN_SURVIVAL = Float32(0.05)
comptime RUSSIAN_ROULETTE_MAX_SURVIVAL = Float32(0.95)


@fieldwise_init
struct RussianRouletteResult[length: SIMDLength = 1](Copyable, Writable):
    var survived: SIMD[.bool, Self.length]
    var throughput: Vec3[.float32, .WORLD, Self.length]


@always_inline
def sky_color[
    length: SIMDLength
](direction: Vec3[.float32, .WORLD, length]) -> Vec3[.float32, .WORLD, length]:
    var unit_direction = normalize(direction)
    var a = 0.5 * (unit_direction.y + 1.0)
    return (1.0 - a) * Vec3[.float32, .WORLD, length](1.0) + a * Vec3[
        .float32, .WORLD, length
    ](0.5, 0.7, 1.0)


@fieldwise_init
struct EnvironmentUV[length: SIMDLength = 1](
    Copyable, TrivialRegisterPassable, Writable
):
    var u: SIMD[.float32, Self.length]
    var v: SIMD[.float32, Self.length]


@always_inline
def equal_area_square_to_sphere[
    frame: Frame, length: SIMDLength
](u01: SIMD[.float32, length], v01: SIMD[.float32, length]) -> Vec3[
    .float32, frame, length
]:
    """PBRT v4's equal-area unit-square to unit-sphere mapping."""
    var u = 2.0 * u01 - 1.0
    var v = 2.0 * v01 - 1.0
    var up = abs(u)
    var vp = abs(v)
    var signed_distance = 1.0 - (up + vp)
    var r = 1.0 - abs(signed_distance)
    var nonzero_r = r.ne(0.0)
    var phi = (
        nonzero_r.select((vp - up) / nonzero_r.select(r, 1.0), 0.0) + 1.0
    ) * (pi / 4.0)
    var z = 1.0 - r * r
    z = signed_distance.lt(0.0).select(-z, z)
    var cos_phi = cos(phi)
    var sin_phi = sin(phi)
    cos_phi = u.lt(0.0).select(-cos_phi, cos_phi)
    sin_phi = v.lt(0.0).select(-sin_phi, sin_phi)
    var radial = r * sqrt(max(2.0 - r * r, 0.0))
    return Vec3[.float32, frame, length](cos_phi * radial, sin_phi * radial, z)


@always_inline
def equal_area_sphere_to_square[
    frame: Frame, length: SIMDLength
](direction: Vec3[.float32, frame, length]) -> EnvironmentUV[length]:
    """PBRT v4's equal-area unit-sphere to unit-square mapping."""
    var d = normalize(direction)
    var x = abs(d.x)
    var y = abs(d.y)
    var z = abs(d.z)
    var r = sqrt(max(Float32(1.0) - z, Float32(0.0)))
    var a = max(x, y)
    var nonzero_a = a.ne(0.0)
    var b = min(x, y) / nonzero_a.select(a, Float32(1.0))

    # PBRT's 6th-degree minimax approximation of atan(b) * 2 / pi.
    var phi = SIMD[.float32, length](
        Float32(-0.251390972343483509333252996350e-1)
    )
    phi = Float32(0.419038818029165735901852432784e-1) + b * phi
    phi = Float32(0.881770664775316294736387951347e-1) + b * phi
    phi = Float32(-0.247333733281268944196501420480) + b * phi
    phi = Float32(0.61572017898280213493197203466e-2) + b * phi
    phi = Float32(0.636226545274016134946890922156) + b * phi
    phi = Float32(0.406758566246788489601959989e-5) + b * phi
    phi = x.lt(y).select(1.0 - phi, phi)

    var v = phi * r
    var u = r - v
    var original_u = u
    var south = d.z.lt(0.0)
    u = south.select(1.0 - v, u)
    v = south.select(1.0 - original_u, v)
    u = d.x.lt(0.0).select(-u, u)
    v = d.y.lt(0.0).select(-v, v)
    return EnvironmentUV[length](0.5 * (u + 1.0), 0.5 * (v + 1.0))


@always_inline
def environment_radiance[
    length: SIMDLength
](
    environment: Environment,
    surfaces: SurfaceStore,
    direction: Vec3[.float32, .WORLD, length],
) -> Vec3[.float32, .WORLD, length]:
    """Evaluate scene-owned miss radiance on the CPU."""
    match environment.kind:
        case .BLACK:
            return Vec3[.float32, .WORLD, length](0.0)
        case .PROCEDURAL:
            return sky_color(direction)
        case .UNIFORM:
            return Vec3[.float32, .WORLD, length](
                environment.scale.x[0],
                environment.scale.y[0],
                environment.scale.z[0],
            )
        case .IMAGE:
            pass
    var transform = environment.world_to_light.copy()
    var light_direction = Vec3[.float32, .LOCAL, length](
        transform.m00[0] * direction.x
        + transform.m01[0] * direction.y
        + transform.m02[0] * direction.z,
        transform.m10[0] * direction.x
        + transform.m11[0] * direction.y
        + transform.m12[0] * direction.z,
        transform.m20[0] * direction.x
        + transform.m21[0] * direction.y
        + transform.m22[0] * direction.z,
    )
    var uv = equal_area_sphere_to_square(light_direction)
    var sampled = surfaces.image_textures[
        Int(environment.texture_index)
    ].sample_environment(uv.u, uv.v)
    return sampled * Vec3[.float32, .WORLD, length](
        environment.scale.x, environment.scale.y, environment.scale.z
    )


@always_inline
def environment_light_pdf[
    length: SIMDLength
](
    environment: Environment,
    surfaces: SurfaceStore,
    environment_weight: Float32,
    total_light_weight: Float32,
    direction: Vec3[.float32, .WORLD, length],
) -> SIMD[.float32, length]:
    """Environment sampling density in world-space solid-angle measure."""
    if environment_weight <= 0.0 or total_light_weight <= 0.0:
        return 0.0
    match environment.kind:
        case .BLACK | .PROCEDURAL | .UNIFORM:
            return environment_weight / (4.0 * pi * total_light_weight)
        case .IMAGE:
            pass

    return environment_pdf_from_radiance(
        environment,
        environment_weight,
        total_light_weight,
        environment_radiance(environment, surfaces, direction),
    )


@always_inline
def environment_pdf_from_radiance[
    length: SIMDLength
](
    environment: Environment,
    environment_weight: Float32,
    total_light_weight: Float32,
    emission: Vec3[.float32, .WORLD, length],
) -> SIMD[.float32, length]:
    """Reuse evaluated emission when computing the image-light density."""
    if environment_weight <= 0.0 or total_light_weight <= 0.0:
        return 0.0
    if environment.kind != .IMAGE:
        return environment_weight / (4.0 * pi * total_light_weight)
    return _light_importance(emission) / total_light_weight


@always_inline
def path_stage_rng(seed: UInt64, path_id: UInt32, stage: UInt32) -> Rng[1]:
    """Create the deterministic Philox stream owned by one path stage."""
    return path_stage_rng(
        SamplingConfig(seed, Sampler.INDEPENDENT.value, 1, 0, 1, 1),
        path_id,
        stage,
    )


@always_inline
def path_stage_rng[
    length: SIMDLength = 1
](
    sampling: SamplingConfig, path_id: SIMD[.uint32, length], stage: UInt32
) -> Rng[length]:
    """Create a batch-invariant stream for one pixel sample and path stage."""
    var batch_spp = sampling.samples_per_pixel
    var pixel_id = path_id / batch_spp
    var sample_index = sampling.sample_offset + path_id % batch_spp
    var sampler = Sampler(unsafe_from_value=sampling.sampler_value)
    match sampler:
        case .INDEPENDENT:
            var canonical_path_id = (
                pixel_id * sampling.sequence_length + sample_index
            )
            return Rng[length](
                seed=sampling.seed,
                id=wavefront_rng_subsequence(canonical_path_id, stage),
            )
        case .HALTON | .R2 | .OWEN_SOBOL | .SZ | .STBN:
            pass
    return Rng[length](
        seed=sampling.seed,
        id=wavefront_rng_subsequence(pixel_id, stage),
        sampler=sampler,
        sample_index=sample_index.cast[.uint64](),
        pixel_id=pixel_id,
        image_width=sampling.image_width,
        stage=stage,
    )


@always_inline
def path_stage_seed[
    length: SIMDLength = 1
](
    sampling: SamplingConfig, path_id: SIMD[.uint32, length], stage: UInt32
) -> SIMD[.uint32, length]:
    """Hash an integer pixel/sample/stage identity for internal estimators.

    Absolute sample indices preserve sample-batch and queue-chunk invariance.
    Floating-point directions must not choose estimator random streams: small
    CPU/GPU rounding differences would otherwise change every internal draw.
    """
    var pixel_id = path_id / sampling.samples_per_pixel
    var sample_index = (
        sampling.sample_offset + path_id % sampling.samples_per_pixel
    )
    var seed = _mix32(UInt32(sampling.seed))
    seed = _mix32(seed ^ UInt32(sampling.seed >> UInt64(32)))
    var path_seed = _mix32(seed ^ pixel_id)
    path_seed = _mix32(path_seed ^ sample_index)
    return _mix32(path_seed ^ stage)


@always_inline
def russian_roulette(
    seed: UInt64,
    path_id: UInt32,
    depth: UInt32,
    throughput: Color,
) -> RussianRouletteResult[1]:
    """Unbiased continuation using a stream separate from BSDF sampling."""
    return russian_roulette(
        SamplingConfig(seed, Sampler.INDEPENDENT.value, 1, 0, 1, 1),
        path_id,
        depth,
        throughput,
    )


@always_inline
def russian_roulette[
    length: SIMDLength = 1
](
    sampling: SamplingConfig,
    path_id: SIMD[.uint32, length],
    depth: UInt32,
    throughput: Vec3[.float32, .WORLD, length],
) -> RussianRouletteResult[length]:
    """Batch-invariant continuation using the configured sample sequence."""
    if depth < RUSSIAN_ROULETTE_START_DEPTH:
        return RussianRouletteResult[length](
            SIMD[.bool, length](fill=True), throughput
        )
    var maximum = max(throughput.x, max(throughput.y, throughput.z))
    var eligible = maximum.gt(0.0)
    var survival = min(
        max(maximum, RUSSIAN_ROULETTE_MIN_SURVIVAL),
        RUSSIAN_ROULETTE_MAX_SURVIVAL,
    )
    if not eligible.reduce_or():
        return RussianRouletteResult[length](eligible, throughput)
    var rng = path_stage_rng(
        sampling, path_id, wavefront_rng_roulette_stage(depth - UInt32(1))
    )
    var survived = eligible & rng.f32().lt(survival)
    return RussianRouletteResult[length](
        survived, throughput / survived.select(survival, 1.0)
    )
