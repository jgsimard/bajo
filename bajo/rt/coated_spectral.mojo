"""Width-generic coating math and layered Monte Carlo estimators.

Scalar CPU/GPU shading specializes the same SIMD functions with length=1.
Lanes retain independent sample counts, depths, seeds and termination masks.
"""

from bajo.core.optics import fresnel_dielectric, fresnel_conductor

from std.math import abs, cos, exp, log, pi, sin, sqrt
from std.memory import bitcast
from bajo.core import (
    Vec3,
    cross,
    dot,
    length2,
    normalize,
    orthonormal_basis,
    reflect,
)
from bajo.core.random import _mix32, _u32_to_unit_float
from bajo.rt.spectral_shading_types import (
    SpectralCoatedParameters as CoatedParameters,
)
from bajo.core.sampled_spectrum import SampledSpectrum
from bajo.core.sampling import power_heuristic
from bajo.rt.spectral_shading_types import (
    SpectralBsdfEvaluation as BsdfEvaluation,
    SpectralBsdfSample as BsdfSample,
)

comptime _INV_PI = 1.0 / pi
comptime _INV_4_PI = 1.0 / (4.0 * pi)


from bajo.rt.coated_diffuse import (
    _random,
    _sample_seed,
    _ggx_lambda,
    _ggx_d,
    _ggx_g,
    _ggx_visible_pdf,
    _sample_ggx_visible_normal,
    _dielectric_geometry,
    _dielectric_f,
    _dielectric_pdf,
    _sample_dielectric,
    _transmittance,
    _phase_hg,
    _sample_phase_hg,
    _to_local,
    _from_local,
    _DielectricSample,
)


@always_inline
def _is_diffuse[
    length: SIMDLength
](p: CoatedParameters[length]) -> SIMD[.bool, length]:
    return p.conductor_eta.max_component().eq(
        0.0
    ) & p.conductor_k.max_component().eq(0.0)


@always_inline
def _substrate_f[
    length: SIMDLength
](
    wo: Vec3[.float32, .WORLD, length],
    wi: Vec3[.float32, .WORLD, length],
    albedo: SampledSpectrum[length],
    p: CoatedParameters[length],
) -> SampledSpectrum[length]:
    var same_side = (wo.z * wi.z).gt(0.0)
    var diffuse = _is_diffuse(p)
    var result = SampledSpectrum[length].select(
        same_side & diffuse,
        albedo * _INV_PI,
        SampledSpectrum[length](0.0),
    )
    var valid = (
        same_side
        & ~diffuse
        & p.conductor_roughness.ge(1.0e-3)
        & length2(wo + wi).ne(0.0)
    )
    if valid.reduce_or():
        var wm = normalize(wo + wi)
        var alpha = max(p.conductor_roughness, SIMD[.float32, length](1.0e-4))
        var value = fresnel_conductor(
            dot(wo, wm), p.conductor_eta, p.conductor_k
        ) * (
            _ggx_d(wm, alpha)
            * _ggx_g(wo, wi, alpha)
            / valid.select(abs(4.0 * wo.z * wi.z), 1.0)
        )
        result = SampledSpectrum[length].select(valid, value, result)
    return result


@always_inline
def _substrate_pdf[
    length: SIMDLength
](
    wo: Vec3[.float32, .WORLD, length],
    wi: Vec3[.float32, .WORLD, length],
    p: CoatedParameters[length],
) -> SIMD[.float32, length]:
    var same_side = (wo.z * wi.z).gt(0.0)
    var diffuse = _is_diffuse(p)
    var result = (same_side & diffuse).select(abs(wi.z) * _INV_PI, 0.0)
    var valid = (
        same_side
        & ~diffuse
        & p.conductor_roughness.ge(1.0e-3)
        & length2(wo + wi).ne(0.0)
    )
    if valid.reduce_or():
        var wm = normalize(wo + wi)
        wm = Vec3[.float32, .WORLD, length].select(wm.z.lt(0.0), -wm, wm)
        var value = _ggx_visible_pdf(
            wo, wm, max(p.conductor_roughness, SIMD[.float32, length](1.0e-4))
        ) / max(SIMD[.float32, length](1.0e-20), 4.0 * abs(dot(wo, wm)))
        result = valid.select(value, result)
    return result


@always_inline
def _sample_substrate[
    length: SIMDLength
](
    wo: Vec3[.float32, .WORLD, length],
    albedo: SampledSpectrum[length],
    p: CoatedParameters[length],
    u: SIMD[.float32, length],
    v: SIMD[.float32, length],
) -> BsdfSample[length]:
    var conductor = ~_is_diffuse(p)
    var result = BsdfSample[length](
        Vec3[.float32, .WORLD, length](0.0),
        albedo,
        SIMD[.float32, length](0.0),
        SIMD[.bool, length](fill=False),
        SIMD[.bool, length](fill=True),
    )
    if (~conductor).reduce_or():
        var radius = sqrt(u)
        var phi = Float32(2.0 * pi) * v
        var z = sqrt(max(SIMD[.float32, length](0.0), 1.0 - u))
        z = wo.z.lt(0.0).select(-z, z)
        result.direction = Vec3[.float32, .WORLD, length](
            radius * cos(phi), radius * sin(phi), z
        )
        result.pdf = abs(z) * _INV_PI
    if not conductor.reduce_or():
        return result^
    var smooth = p.conductor_roughness.lt(1.0e-3)
    var wm = Vec3[.float32, .WORLD, length](0.0, 0.0, 1.0)
    if (conductor & ~smooth).reduce_or():
        wm = Vec3[.float32, .WORLD, length].select(
            smooth,
            wm,
            _sample_ggx_visible_normal(wo, p.conductor_roughness, u, v),
        )
    var wi = -reflect(wo, wm)
    var valid = (wo.z * wi.z).gt(0.0)
    var weight = fresnel_conductor(dot(wo, wm), p.conductor_eta, p.conductor_k)
    var pdf = SIMD[.float32, length](1.0)
    if (conductor & valid & ~smooth).reduce_or():
        var rough_pdf = _substrate_pdf(wo, wi, p)
        var rough_weight = _substrate_f(wo, wi, albedo, p) * (
            abs(wi.z) / max(rough_pdf, SIMD[.float32, length](1.0e-20))
        )
        pdf = smooth.select(pdf, rough_pdf)
        weight = SampledSpectrum[length].select(smooth, weight, rough_weight)
    result.direction = Vec3[.float32, .WORLD, length].select(
        conductor, wi, result.direction
    )
    result.weight = SampledSpectrum[length].select(
        conductor,
        SampledSpectrum[length].select(
            valid, weight, SampledSpectrum[length](0.0)
        ),
        result.weight,
    )
    result.pdf = conductor.select(valid.select(pdf, 0.0), result.pdf)
    result.delta = conductor & smooth
    result.ok = ~conductor | (valid & pdf.gt(0.0))
    return result^


@no_inline
def _layered_pdf[
    length: SIMDLength
](
    wo: Vec3[.float32, .WORLD, length],
    wi: Vec3[.float32, .WORLD, length],
    p: CoatedParameters[length],
    seed: SIMD[.uint32, length],
    active: SIMD[.bool, length],
) -> SIMD[.float32, length]:
    var valid = active & (wo.z * wi.z).gt(0.0)
    if not valid.reduce_or():
        return SIMD[.float32, length](0.0)
    var counts = valid.select(p.n_samples, 0.0).cast[.int64]()
    var denominator = max(counts, SIMD[.int64, length](1)).cast[.float32]()
    var sum = denominator * _dielectric_pdf(wo, wi, p.roughness, p.eta, 1)
    var sample = 0
    while (valid & counts.gt(SIMD[.int64, length](Int64(sample)))).reduce_or():
        var sampling = valid & counts.gt(SIMD[.int64, length](Int64(sample)))
        var dimension = SIMD[.uint32, length](UInt32(sample) * 8)
        var wos = _sample_dielectric(
            wo,
            p.roughness,
            p.eta,
            _random(seed, dimension),
            _random(seed, dimension + 1),
            _random(seed, dimension + 2),
            True,
            2,
        )
        var wis = _sample_dielectric(
            wi,
            p.roughness,
            p.eta,
            _random(seed, dimension + 3),
            _random(seed, dimension + 4),
            _random(seed, dimension + 5),
            False,
            2,
        )
        var ok = sampling & wos.ok & wis.ok
        var substrate_pdf = _substrate_pdf(-wos.direction, -wis.direction, p)
        var smooth = p.roughness.lt(1.0e-3) | p.eta.eq(1.0)
        sum += (ok & smooth).select(substrate_pdf, 0.0)
        var rough = ok & ~smooth
        if rough.reduce_or():
            var reflected = _sample_substrate(
                -wos.direction,
                SampledSpectrum[length](1.0),
                p,
                _random(seed, dimension + 6),
                _random(seed, dimension + 7),
            )
            var exit_pdf = _dielectric_pdf(
                -reflected.direction, wi, p.roughness, p.eta, 2
            )
            var value = reflected.delta.select(
                exit_pdf,
                power_heuristic(wis.pdf, substrate_pdf) * substrate_pdf
                + power_heuristic(reflected.pdf, exit_pdf) * exit_pdf,
            )
            sum += (rough & reflected.ok).select(value, 0.0)
        sample += 1
    return valid.select(0.1 * _INV_4_PI + 0.9 * sum / denominator, 0.0)


@no_inline
def _evaluate_layered[
    length: SIMDLength
](
    wo: Vec3[.float32, .WORLD, length],
    wi: Vec3[.float32, .WORLD, length],
    albedo: SampledSpectrum[length],
    p: CoatedParameters[length],
    seed: SIMD[.uint32, length],
    active: SIMD[.bool, length],
) -> SampledSpectrum[length]:
    var valid = active & wo.z.gt(0.0) & wi.z.gt(0.0)
    if not valid.reduce_or():
        return SampledSpectrum[length](0.0)
    var direct = _dielectric_f(wo, wi, p.roughness, p.eta, True)
    var result = SampledSpectrum[length](direct)
    var indirect = SampledSpectrum[length](0.0)
    var counts = valid.select(p.n_samples, 0.0).cast[.int64]()
    var depths = valid.select(p.max_depth, 0.0).cast[.int64]()
    var medium = p.layer_albedo.max_component().gt(0.0)
    var sample = 0
    while (valid & counts.gt(SIMD[.int64, length](Int64(sample)))).reduce_or():
        # The scalar estimator's stride depends on this lane's own max depth.
        var base = 16 + SIMD[.uint32, length](UInt32(sample)) * (
            depths.cast[.uint32]() * 8 + 16
        )
        var wos = _sample_dielectric(
            wo,
            p.roughness,
            p.eta,
            _random(seed, base),
            _random(seed, base + 1),
            _random(seed, base + 2),
            True,
            2,
        )
        var wis = _sample_dielectric(
            wi,
            p.roughness,
            p.eta,
            _random(seed, base + 3),
            _random(seed, base + 4),
            _random(seed, base + 5),
            False,
            2,
        )
        var running = (
            valid
            & counts.gt(SIMD[.int64, length](Int64(sample)))
            & wos.ok
            & wis.ok
            & depths.gt(0)
        )
        var initial_beta = (
            wos.value * abs(wos.direction.z) / wos.ok.select(wos.pdf, 1.0)
        )
        var beta = SampledSpectrum[length](initial_beta)
        var direction = wos.direction
        var z = p.thickness
        var depth = 0
        while running.reduce_or():
            var dimension = base + UInt32(6 + depth * 8)
            if depth > 3:
                var rr = running & beta.max_component().lt(0.25)
                var q = max(
                    SIMD[.float32, length](0.0), 1.0 - beta.max_component()
                )
                running &= ~(rr & _random(seed, dimension).lt(q))
                beta = SampledSpectrum[length].select(
                    rr & running, beta / (1.0 - q), beta
                )
            if not running.reduce_or():
                break
            var scattered = SIMD[.bool, length](fill=False)
            if (running & medium).reduce_or():
                var distance = -log(
                    max(
                        SIMD[.float32, length](1.0e-7),
                        1.0 - _random(seed, dimension + 1),
                    )
                ) * abs(direction.z)
                var zp = direction.z.gt(0.0).select(z + distance, z - distance)
                scattered = running & medium & zp.gt(0.0) & zp.lt(p.thickness)
                if scattered.reduce_or():
                    var target = -wis.direction
                    var phase = _phase_hg(dot(-direction, target), p.g)
                    var phase_weight = power_heuristic(wis.pdf, phase)
                    var contribution = (
                        beta
                        * p.layer_albedo
                        * phase
                        * phase_weight
                        * _transmittance(zp - p.thickness, wis.direction)
                        * (wis.value / wis.ok.select(wis.pdf, 1.0))
                    )
                    indirect += SampledSpectrum[length].select(
                        scattered,
                        contribution,
                        SampledSpectrum[length](0.0),
                    )
                    var phase_direction = _sample_phase_hg(
                        -direction,
                        p.g,
                        _random(seed, dimension + 2),
                        _random(seed, dimension + 3),
                    )
                    beta = SampledSpectrum[length].select(
                        scattered, beta * p.layer_albedo, beta
                    )
                    direction = Vec3[.float32, .WORLD, length].select(
                        scattered, phase_direction, direction
                    )
                    z = scattered.select(zp, z)
                    var exit_value = _dielectric_f(
                        -direction, wi, p.roughness, p.eta, True
                    )
                    var exit_pdf = _dielectric_pdf(
                        -direction, wi, p.roughness, p.eta, 2
                    )
                    contribution = (
                        beta
                        * _transmittance(z - p.thickness, direction)
                        * exit_value
                        * power_heuristic(phase, exit_pdf)
                    )
                    indirect += SampledSpectrum[length].select(
                        scattered & direction.z.gt(0.0) & exit_value.gt(0.0),
                        contribution,
                        SampledSpectrum[length](0.0),
                    )
                z = (running & medium & ~scattered).select(
                    zp.ge(p.thickness).select(p.thickness, 0.0), z
                )
            var absorbing = running & ~medium
            if absorbing.reduce_or():
                z = absorbing.select(
                    z.eq(p.thickness).select(
                        SIMD[.float32, length](0.0), p.thickness
                    ),
                    z,
                )
                beta = SampledSpectrum[length].select(
                    absorbing,
                    beta * _transmittance(p.thickness, direction),
                    beta,
                )
            var boundary = running & ~scattered
            var top = boundary & z.eq(p.thickness)
            var bottom = boundary & ~top
            if top.reduce_or():
                var reflected = _sample_dielectric(
                    -direction,
                    p.roughness,
                    p.eta,
                    _random(seed, dimension + 2),
                    _random(seed, dimension + 3),
                    _random(seed, dimension + 4),
                    True,
                    1,
                )
                running &= ~(top & ~reflected.ok)
                beta = SampledSpectrum[length].select(
                    top & reflected.ok,
                    beta
                    * (
                        reflected.value
                        * abs(reflected.direction.z)
                        / reflected.ok.select(reflected.pdf, 1.0)
                    ),
                    beta,
                )
                direction = Vec3[.float32, .WORLD, length].select(
                    top & reflected.ok, reflected.direction, direction
                )
            if bottom.reduce_or():
                var target = -wis.direction
                var substrate_pdf = _substrate_pdf(-direction, target, p)
                var nee_weight = (
                    p.roughness.ge(1.0e-3) & p.eta.ne(1.0)
                ).select(power_heuristic(wis.pdf, substrate_pdf), 1.0)
                var contribution = (
                    beta
                    * _substrate_f(-direction, target, albedo, p)
                    * abs(wis.direction.z)
                    * nee_weight
                    * _transmittance(p.thickness, wis.direction)
                    * (wis.value / wis.ok.select(wis.pdf, 1.0))
                )
                indirect += SampledSpectrum[length].select(
                    bottom, contribution, SampledSpectrum[length](0.0)
                )
                var reflected = _sample_substrate(
                    -direction,
                    albedo,
                    p,
                    _random(seed, dimension + 5),
                    _random(seed, dimension + 6),
                )
                running &= ~(bottom & ~reflected.ok)
                var reflected_ok = bottom & reflected.ok
                beta = SampledSpectrum[length].select(
                    reflected_ok, beta * reflected.weight, beta
                )
                direction = Vec3[.float32, .WORLD, length].select(
                    reflected_ok, reflected.direction, direction
                )
                var exit_value = _dielectric_f(
                    -direction, wi, p.roughness, p.eta, True
                )
                var exit_pdf = _dielectric_pdf(
                    -direction, wi, p.roughness, p.eta, 2
                )
                var exit_weight = (
                    p.roughness.ge(1.0e-3) & p.eta.ne(1.0) & ~reflected.delta
                ).select(power_heuristic(reflected.pdf, exit_pdf), 1.0)
                contribution = (
                    beta
                    * _transmittance(p.thickness, direction)
                    * exit_value
                    * exit_weight
                )
                indirect += SampledSpectrum[length].select(
                    reflected_ok & exit_value.gt(0.0),
                    contribution,
                    SampledSpectrum[length](0.0),
                )
            depth += 1
            running &= depths.gt(SIMD[.int64, length](Int64(depth)))
        sample += 1
    return SampledSpectrum[length].select(
        valid,
        result
        + indirect / max(counts, SIMD[.int64, length](1)).cast[.float32](),
        SampledSpectrum[length](0.0),
    )


@always_inline
def _safe_parameters[
    length: SIMDLength
](p: CoatedParameters[length], active: SIMD[.bool, length]) -> CoatedParameters[
    length
]:
    # Inactive storage may be uninitialized/NaN. In particular, mask counts
    # before integer conversion and keep inactive reciprocals well-defined.
    return CoatedParameters[length](
        active.select(p.roughness, 0.0),
        active.select(p.eta, 1.5),
        active.select(p.thickness, 0.01),
        SampledSpectrum[length].select(
            active, p.layer_albedo, SampledSpectrum[length](0.0)
        ),
        active.select(p.g, 0.0),
        active.select(p.max_depth, 0.0),
        active.select(p.n_samples, 0.0),
        SampledSpectrum[length].select(
            active, p.conductor_eta, SampledSpectrum[length](0.0)
        ),
        SampledSpectrum[length].select(
            active, p.conductor_k, SampledSpectrum[length](0.0)
        ),
        active.select(p.conductor_roughness, 0.0),
    )


@always_inline
def _bare_conductor[
    length: SIMDLength
](p: CoatedParameters[length]) -> SIMD[.bool, length]:
    return (
        (
            p.conductor_eta.max_component().gt(0.0)
            | p.conductor_k.max_component().gt(0.0)
        )
        & p.eta.eq(1.0)
        & p.thickness.eq(0.0)
        & p.layer_albedo.max_component().eq(0.0)
    )


@always_inline
def _evaluate_coated_diffuse[
    length: SIMDLength
](
    ray_direction: Vec3[.float32, .WORLD, length],
    normal: Vec3[.float32, .WORLD, length],
    albedo: SampledSpectrum[length],
    coated: CoatedParameters[length],
    out_direction: Vec3[.float32, .WORLD, length],
    active: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    estimator_seed: SIMD[.uint32, length] = 0,
) -> BsdfEvaluation[length]:
    var result = BsdfEvaluation[length](
        SampledSpectrum[length](0.0),
        SIMD[.float32, length](0.0),
        SIMD[.bool, length](fill=False),
    )
    if not active.reduce_or():
        return result^
    var p = _safe_parameters(coated, active)
    var n = Vec3[.float32, .WORLD, length].select(
        active, normal, Vec3[.float32, .WORLD, length](0.0, 0.0, 1.0)
    )
    var tangent, bitangent = orthonormal_basis(n)
    var wo = _to_local(
        -normalize(
            Vec3[.float32, .WORLD, length].select(active, ray_direction, -n)
        ),
        tangent,
        bitangent,
        n,
    )
    var wi = _to_local(
        normalize(
            Vec3[.float32, .WORLD, length].select(active, out_direction, n)
        ),
        tangent,
        bitangent,
        n,
    )
    var bare = active & _bare_conductor(p)
    if bare.reduce_or():
        result.value = SampledSpectrum[length].select(
            bare, _substrate_f(wo, wi, albedo, p), result.value
        )
        result.pdf = bare.select(_substrate_pdf(wo, wi, p), result.pdf)
        result.delta = bare & p.conductor_roughness.lt(1.0e-3)
    var layered = active & ~bare
    if layered.reduce_or():
        p.thickness = max(p.thickness, SIMD[.float32, length](1.0e-20))
        result.value += _evaluate_layered(
            wo, wi, albedo, p, estimator_seed, layered
        )
        result.pdf += _layered_pdf(wo, wi, p, estimator_seed, layered)
    return result^


@no_inline
def _sample_coated_diffuse[
    length: SIMDLength
](
    ray_direction: Vec3[.float32, .WORLD, length],
    normal: Vec3[.float32, .WORLD, length],
    albedo: SampledSpectrum[length],
    coated: CoatedParameters[length],
    random_u: SIMD[.float32, length],
    random_v: SIMD[.float32, length],
    active: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    estimator_seed: SIMD[.uint32, length] = 0,
) -> BsdfSample[length]:
    var result = BsdfSample[length](
        normal,
        SampledSpectrum[length].select(
            active, albedo, SampledSpectrum[length](0.0)
        ),
        SIMD[.float32, length](0.0),
        SIMD[.bool, length](fill=False),
        SIMD[.bool, length](fill=False),
    )
    if not active.reduce_or():
        return result^
    var p = _safe_parameters(coated, active)
    var n = Vec3[.float32, .WORLD, length].select(
        active, normal, Vec3[.float32, .WORLD, length](0.0, 0.0, 1.0)
    )
    var tangent, bitangent = orthonormal_basis(n)
    var wo = _to_local(
        -normalize(
            Vec3[.float32, .WORLD, length].select(active, ray_direction, -n)
        ),
        tangent,
        bitangent,
        n,
    )
    var valid = active & wo.z.gt(0.0)
    var bare = valid & _bare_conductor(p)
    if bare.reduce_or():
        var sampled = _sample_substrate(wo, albedo, p, random_u, random_v)
        result.direction = Vec3[.float32, .WORLD, length].select(
            bare,
            normalize(_from_local(sampled.direction, tangent, bitangent, n)),
            result.direction,
        )
        result.weight = SampledSpectrum[length].select(
            bare, sampled.weight, result.weight
        )
        result.pdf = bare.select(sampled.pdf, result.pdf)
        result.delta = bare & sampled.delta
        result.ok = bare & sampled.ok
    var layered = valid & ~bare
    if not layered.reduce_or():
        return result^
    p.thickness = max(p.thickness, SIMD[.float32, length](1.0e-20))
    var seed = _sample_seed(random_u, random_v)
    var interface_sample = _sample_dielectric(
        wo,
        p.roughness,
        p.eta,
        random_u,
        random_v,
        _random(seed, SIMD[.uint32, length](0)),
        True,
    )
    var initial_weight = (
        interface_sample.value
        * abs(interface_sample.direction.z)
        / interface_sample.ok.select(interface_sample.pdf, 1.0)
    )
    var weight = SampledSpectrum[length](initial_weight)
    var reflected = layered & interface_sample.ok & interface_sample.reflection
    var out_local = interface_sample.direction
    if reflected.reduce_or():
        result.direction = Vec3[.float32, .WORLD, length].select(
            reflected,
            normalize(_from_local(out_local, tangent, bitangent, n)),
            result.direction,
        )
        result.weight = SampledSpectrum[length].select(
            reflected, weight, result.weight
        )
        result.pdf = reflected.select(interface_sample.pdf, result.pdf)
        result.delta |= reflected & interface_sample.delta
        result.ok |= reflected
    var direction = interface_sample.direction
    var z = p.thickness
    var medium = p.layer_albedo.max_component().gt(0.0)
    var specular = interface_sample.delta
    var depths = layered.select(p.max_depth, 0.0).cast[.int64]()
    var running = (
        layered
        & interface_sample.ok
        & ~interface_sample.reflection
        & depths.gt(0)
    )
    var depth = 0
    while running.reduce_or():
        var dimension = SIMD[.uint32, length](UInt32(1 + depth * 8))
        if depth > 3:
            var rr = running & weight.max_component().lt(0.25)
            var q = max(
                SIMD[.float32, length](0.0), 1.0 - weight.max_component()
            )
            running &= ~(rr & _random(seed, dimension).lt(q))
            weight = SampledSpectrum[length].select(
                rr & running, weight / (1.0 - q), weight
            )
        if not running.reduce_or():
            break
        var scattered = SIMD[.bool, length](fill=False)
        if (running & medium).reduce_or():
            var distance = -log(
                max(
                    SIMD[.float32, length](1.0e-7),
                    1.0 - _random(seed, dimension + 1),
                )
            ) * abs(direction.z)
            var zp = direction.z.gt(0.0).select(z + distance, z - distance)
            scattered = running & medium & zp.gt(0.0) & zp.lt(p.thickness)
            if scattered.reduce_or():
                weight = SampledSpectrum[length].select(
                    scattered, weight * p.layer_albedo, weight
                )
                direction = Vec3[.float32, .WORLD, length].select(
                    scattered,
                    _sample_phase_hg(
                        -direction,
                        p.g,
                        _random(seed, dimension + 2),
                        _random(seed, dimension + 3),
                    ),
                    direction,
                )
                z = scattered.select(zp, z)
                specular &= ~scattered
            z = (running & medium & ~scattered).select(
                zp.ge(p.thickness).select(p.thickness, 0.0), z
            )
        var absorbing = running & ~medium
        if absorbing.reduce_or():
            z = absorbing.select(
                z.eq(p.thickness).select(
                    SIMD[.float32, length](0.0), p.thickness
                ),
                z,
            )
            weight = SampledSpectrum[length].select(
                absorbing,
                weight * _transmittance(p.thickness, direction),
                weight,
            )
        var boundary = running & ~scattered
        var top = boundary & z.eq(p.thickness)
        var bottom = boundary & ~top
        if top.reduce_or():
            var sampled = _sample_dielectric(
                -direction,
                p.roughness,
                p.eta,
                _random(seed, dimension + 4),
                _random(seed, dimension + 5),
                _random(seed, dimension + 6),
                True,
            )
            running &= ~(top & ~sampled.ok)
            var ok = top & sampled.ok
            weight = SampledSpectrum[length].select(
                ok,
                weight
                * (
                    sampled.value
                    * abs(sampled.direction.z)
                    / sampled.ok.select(sampled.pdf, 1.0)
                ),
                weight,
            )
            specular = ok.select(specular & sampled.delta, specular)
            direction = Vec3[.float32, .WORLD, length].select(
                ok, sampled.direction, direction
            )
            var exiting = ok & ~sampled.reflection
            if exiting.reduce_or():
                out_local = Vec3[.float32, .WORLD, length].select(
                    exiting, direction, out_local
                )
                result.direction = Vec3[.float32, .WORLD, length].select(
                    exiting,
                    normalize(_from_local(direction, tangent, bitangent, n)),
                    result.direction,
                )
                result.weight = SampledSpectrum[length].select(
                    exiting, weight, result.weight
                )
                result.pdf = exiting.select(sampled.pdf, result.pdf)
                result.delta |= exiting & specular
                result.ok |= exiting
                running &= ~exiting
        if bottom.reduce_or():
            var sampled = _sample_substrate(
                -direction,
                albedo,
                p,
                _random(seed, dimension + 4),
                _random(seed, dimension + 5),
            )
            var ok = bottom & sampled.ok
            running &= ~(bottom & ~sampled.ok)
            weight = SampledSpectrum[length].select(
                ok, weight * sampled.weight, weight
            )
            specular = ok.select(specular & sampled.delta, specular)
            direction = Vec3[.float32, .WORLD, length].select(
                ok, sampled.direction, direction
            )
        depth += 1
        running &= depths.gt(SIMD[.int64, length](Int64(depth)))
    # Completed lanes keep their result while other lanes walk on. Evaluate all
    # non-delta exit PDFs together; the stateless estimator seed is unchanged.
    var need_pdf = result.ok & ~result.delta & layered
    if need_pdf.reduce_or():
        result.pdf = need_pdf.select(
            _layered_pdf(wo, out_local, p, estimator_seed, need_pdf), result.pdf
        )
    return result^
