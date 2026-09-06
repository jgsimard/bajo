"""PBRT-style GGX dielectric coating over a diffuse substrate.

The coating is sampled as a real dielectric interface.  Paths transmitted
through it bounce between the Lambertian substrate and the coat until they
leave the layer or Russian roulette terminates them.  Evaluation and PDF use
the corresponding Monte Carlo estimators from PBRT's LayeredBxDF.
"""

from std.math import abs, cos, exp, log, pi, sin, sqrt
from std.memory import bitcast

from bajo.core import cross, dot, length2, normalize, Vec3
from bajo.rt.types import BsdfEvaluation, BsdfSample


comptime _INV_PI = Float32(0.3183098861837907)
comptime _INV_4_PI = Float32(0.07957747154594767)


@fieldwise_init
struct _DielectricSample(Copyable, Writable):
    var direction: Vec3[.float32, .WORLD]
    var value: Float32
    var pdf: Float32
    var reflection: Bool
    var delta: Bool
    var ok: Bool


@always_inline
def _empty_dielectric_sample() -> _DielectricSample:
    return _DielectricSample(
        Vec3[.float32, .WORLD](0.0), 0.0, 0.0, False, False, False
    )


@always_inline
def _max_component(value: Vec3[.float32, .WORLD]) -> Float32:
    return max(value.x[0], max(value.y[0], value.z[0]))


@always_inline
def _power_heuristic(a: Float32, b: Float32) -> Float32:
    var aa = a * a
    var bb = b * b
    if aa + bb == 0.0:
        return 0.0
    return aa / (aa + bb)


@always_inline
def _mix_bits(value: UInt32) -> UInt32:
    var bits = value
    bits ^= bits >> UInt32(16)
    bits *= UInt32(0x7FEB352D)
    bits ^= bits >> UInt32(15)
    bits *= UInt32(0x846CA68B)
    bits ^= bits >> UInt32(16)
    return bits


@always_inline
def _random(seed: UInt32, dimension: Int) -> Float32:
    var bits = _mix_bits(seed + UInt32(dimension) * UInt32(0x9E3779B9))
    return Float32(bits >> UInt32(8)) * Float32(5.960464477539063e-8)


@always_inline
def _sample_seed(u: Float32, v: Float32) -> UInt32:
    return _mix_bits(
        bitcast[.uint32](u)
        ^ (bitcast[.uint32](v) * UInt32(0x85EBCA6B))
        ^ UInt32(0xC2B2AE35)
    )


@always_inline
def _direction_seed(
    wo: Vec3[.float32, .WORLD], wi: Vec3[.float32, .WORLD]
) -> UInt32:
    var seed = bitcast[.uint32](wo.x[0])
    seed = _mix_bits(seed ^ bitcast[.uint32](wo.y[0]))
    seed = _mix_bits(seed ^ bitcast[.uint32](wo.z[0]))
    seed = _mix_bits(seed ^ bitcast[.uint32](wi.x[0]))
    seed = _mix_bits(seed ^ bitcast[.uint32](wi.y[0]))
    return _mix_bits(seed ^ bitcast[.uint32](wi.z[0]))


@always_inline
def _fresnel_dielectric(cos_theta_i: Float32, eta: Float32) -> Float32:
    var cosine = cos_theta_i.clamp(-1.0, 1.0)
    var relative_eta = eta
    if cosine < 0.0:
        relative_eta = 1.0 / relative_eta
        cosine = -cosine
    var sin2_i = max(Float32(0.0), Float32(1.0) - cosine * cosine)
    var sin2_t = sin2_i / (relative_eta * relative_eta)
    if sin2_t >= 1.0:
        return 1.0
    var cos_t = sqrt(max(Float32(0.0), Float32(1.0) - sin2_t))
    var parallel = (relative_eta * cosine - cos_t) / (
        relative_eta * cosine + cos_t
    )
    var perpendicular = (cosine - relative_eta * cos_t) / (
        cosine + relative_eta * cos_t
    )
    return Float32(0.5) * (parallel * parallel + perpendicular * perpendicular)


@always_inline
def _ggx_lambda(w: Vec3[.float32, .WORLD], alpha: Float32) -> Float32:
    var z2 = w.z[0] * w.z[0]
    if z2 <= 1.0e-20:
        return 0.0
    var tan2_theta = (w.x[0] * w.x[0] + w.y[0] * w.y[0]) / z2
    return Float32(0.5) * (sqrt(1.0 + alpha * alpha * tan2_theta) - 1.0)


@always_inline
def _ggx_d(wm: Vec3[.float32, .WORLD], alpha: Float32) -> Float32:
    var cos2_theta = wm.z[0] * wm.z[0]
    if cos2_theta <= 1.0e-16:
        return 0.0
    var tan2_theta = (wm.x[0] * wm.x[0] + wm.y[0] * wm.y[0]) / cos2_theta
    var denominator = Float32(pi) * alpha * alpha * cos2_theta * cos2_theta
    var e = tan2_theta / (alpha * alpha)
    return 1.0 / (denominator * (1.0 + e) * (1.0 + e))


@always_inline
def _ggx_g1(w: Vec3[.float32, .WORLD], alpha: Float32) -> Float32:
    return 1.0 / (1.0 + _ggx_lambda(w, alpha))


@always_inline
def _ggx_g(
    wo: Vec3[.float32, .WORLD],
    wi: Vec3[.float32, .WORLD],
    alpha: Float32,
) -> Float32:
    return 1.0 / (1.0 + _ggx_lambda(wo, alpha) + _ggx_lambda(wi, alpha))


@always_inline
def _ggx_visible_pdf(
    w: Vec3[.float32, .WORLD],
    wm: Vec3[.float32, .WORLD],
    alpha: Float32,
) -> Float32:
    var cosine = abs(w.z[0])
    if cosine <= 1.0e-20:
        return 0.0
    return _ggx_g1(w, alpha) * _ggx_d(wm, alpha) * abs(dot(w, wm)[0]) / cosine


@always_inline
def _sample_ggx_visible_normal(
    w: Vec3[.float32, .WORLD],
    alpha_in: Float32,
    u1: Float32,
    u2: Float32,
) -> Vec3[.float32, .WORLD]:
    var alpha = max(alpha_in, Float32(1.0e-4))
    var wh = normalize(Vec3[.float32, .WORLD](alpha * w.x, alpha * w.y, w.z))
    if wh.z[0] < 0.0:
        wh = -wh
    var t1 = Vec3[.float32, .WORLD](1.0, 0.0, 0.0)
    if wh.z[0] < 0.99999:
        t1 = normalize(cross(Vec3[.float32, .WORLD](0.0, 0.0, 1.0), wh))
    var t2 = cross(wh, t1)
    var radius = sqrt(u1)
    var phi = Float32(2.0 * pi) * u2
    var px = radius * cos(phi)
    var py = radius * sin(phi)
    var h = sqrt(max(Float32(0.0), Float32(1.0) - px * px))
    var interpolation = Float32(0.5) * (1.0 + wh.z[0])
    py = (1.0 - interpolation) * h + interpolation * py
    var pz = sqrt(max(Float32(0.0), Float32(1.0) - px * px - py * py))
    var nh = t1 * px + t2 * py + wh * pz
    return normalize(
        Vec3[.float32, .WORLD](
            alpha * nh.x, alpha * nh.y, max(Float32(1.0e-6), nh.z[0])
        )
    )


@always_inline
def _reflect_about(
    wo: Vec3[.float32, .WORLD], wm: Vec3[.float32, .WORLD]
) -> Vec3[.float32, .WORLD]:
    return -wo + Float32(2.0) * dot(wo, wm) * wm


@always_inline
def _refract_from(
    wo: Vec3[.float32, .WORLD],
    wm_in: Vec3[.float32, .WORLD],
    eta_in: Float32,
) -> _DielectricSample:
    var wm = wm_in
    var eta = eta_in
    var cosine = dot(wm, wo)[0]
    if cosine < 0.0:
        eta = 1.0 / eta
        cosine = -cosine
        wm = -wm
    var sin2_i = max(Float32(0.0), Float32(1.0) - cosine * cosine)
    var sin2_t = sin2_i / (eta * eta)
    if sin2_t >= 1.0:
        return _empty_dielectric_sample()
    var cos_t = sqrt(max(Float32(0.0), Float32(1.0) - sin2_t))
    var wi = normalize(-wo / eta + (cosine / eta - cos_t) * wm)
    return _DielectricSample(wi, eta, 0.0, False, False, True)


@always_inline
def _dielectric_f(
    wo: Vec3[.float32, .WORLD],
    wi: Vec3[.float32, .WORLD],
    alpha_in: Float32,
    eta: Float32,
    radiance: Bool,
) -> Float32:
    if alpha_in < 1.0e-3:
        return 0.0
    var cos_o = wo.z[0]
    var cos_i = wi.z[0]
    var reflection = cos_i * cos_o > 0.0
    var etap = Float32(1.0)
    if not reflection:
        etap = eta if cos_o > 0.0 else 1.0 / eta
    var wm = wi * etap + wo
    if cos_i == 0.0 or cos_o == 0.0 or length2(wm)[0] == 0.0:
        return 0.0
    wm = normalize(wm)
    if wm.z[0] < 0.0:
        wm = -wm
    if dot(wm, wi)[0] * cos_i < 0.0 or dot(wm, wo)[0] * cos_o < 0.0:
        return 0.0
    var alpha = max(alpha_in, Float32(1.0e-4))
    var fresnel = _fresnel_dielectric(dot(wo, wm)[0], eta)
    if reflection:
        return (
            _ggx_d(wm, alpha)
            * _ggx_g(wo, wi, alpha)
            * fresnel
            / abs(Float32(4.0) * cos_i * cos_o)
        )
    var sum = dot(wi, wm)[0] + dot(wo, wm)[0] / etap
    var denominator = sum * sum * cos_i * cos_o
    if denominator == 0.0:
        return 0.0
    var value = (
        _ggx_d(wm, alpha)
        * (1.0 - fresnel)
        * _ggx_g(wo, wi, alpha)
        * abs(dot(wi, wm)[0] * dot(wo, wm)[0] / denominator)
    )
    if radiance:
        value /= etap * etap
    return value


@always_inline
def _dielectric_pdf(
    wo: Vec3[.float32, .WORLD],
    wi: Vec3[.float32, .WORLD],
    alpha_in: Float32,
    eta: Float32,
    flags: Int,
) -> Float32:
    # flags: 0 = all, 1 = reflection only, 2 = transmission only.
    if alpha_in < 1.0e-3:
        return 0.0
    var cos_o = wo.z[0]
    var cos_i = wi.z[0]
    var reflection = cos_i * cos_o > 0.0
    if (reflection and flags == 2) or (not reflection and flags == 1):
        return 0.0
    var etap = Float32(1.0)
    if not reflection:
        etap = eta if cos_o > 0.0 else 1.0 / eta
    var wm = wi * etap + wo
    if cos_i == 0.0 or cos_o == 0.0 or length2(wm)[0] == 0.0:
        return 0.0
    wm = normalize(wm)
    if wm.z[0] < 0.0:
        wm = -wm
    if dot(wm, wi)[0] * cos_i < 0.0 or dot(wm, wo)[0] * cos_o < 0.0:
        return 0.0
    var alpha = max(alpha_in, Float32(1.0e-4))
    var fresnel = _fresnel_dielectric(dot(wo, wm)[0], eta)
    var pr = fresnel
    var pt = 1.0 - fresnel
    if flags == 1:
        pt = 0.0
    elif flags == 2:
        pr = 0.0
    if pr + pt == 0.0:
        return 0.0
    if reflection:
        return (
            _ggx_visible_pdf(wo, wm, alpha)
            / (Float32(4.0) * abs(dot(wo, wm)[0]))
            * pr
            / (pr + pt)
        )
    var sum = dot(wi, wm)[0] + dot(wo, wm)[0] / etap
    if sum == 0.0:
        return 0.0
    var dwm_dwi = abs(dot(wi, wm)[0]) / (sum * sum)
    return _ggx_visible_pdf(wo, wm, alpha) * dwm_dwi * pt / (pr + pt)


@always_inline
def _sample_dielectric(
    wo: Vec3[.float32, .WORLD],
    alpha_in: Float32,
    eta: Float32,
    uc: Float32,
    u1: Float32,
    u2: Float32,
    radiance: Bool,
    flags: Int = 0,
) -> _DielectricSample:
    var smooth = alpha_in < 1.0e-3
    var wm = Vec3[.float32, .WORLD](0.0, 0.0, 1.0)
    if not smooth:
        wm = _sample_ggx_visible_normal(wo, alpha_in, u1, u2)
    var fresnel = _fresnel_dielectric(dot(wo, wm)[0], eta)
    var pr = fresnel
    var pt = 1.0 - fresnel
    if flags == 1:
        pt = 0.0
    elif flags == 2:
        pr = 0.0
    if pr + pt == 0.0:
        return _empty_dielectric_sample()
    var choose_reflection = uc < pr / (pr + pt)
    if choose_reflection:
        var wi = _reflect_about(wo, wm)
        if wi.z[0] * wo.z[0] <= 0.0 or wi.z[0] == 0.0:
            return _empty_dielectric_sample()
        var pdf = pr / (pr + pt)
        var value = fresnel / abs(wi.z[0])
        if not smooth:
            pdf = (
                _ggx_visible_pdf(wo, wm, max(alpha_in, Float32(1.0e-4)))
                / (Float32(4.0) * abs(dot(wo, wm)[0]))
                * pr
                / (pr + pt)
            )
            value = _dielectric_f(wo, wi, alpha_in, eta, radiance)
        return _DielectricSample(wi, value, pdf, True, smooth, pdf > 0.0)

    var refracted = _refract_from(wo, wm, eta)
    if not refracted.ok or refracted.direction.z[0] * wo.z[0] >= 0.0:
        return _empty_dielectric_sample()
    var etap = refracted.value
    var wi = refracted.direction
    var pdf = pt / (pr + pt)
    var value = (1.0 - fresnel) / abs(wi.z[0])
    if radiance:
        value /= etap * etap
    if not smooth:
        pdf = _dielectric_pdf(wo, wi, alpha_in, eta, flags)
        value = _dielectric_f(wo, wi, alpha_in, eta, radiance)
    return _DielectricSample(wi, value, pdf, False, smooth, pdf > 0.0)


@always_inline
def _sample_diffuse(
    wo: Vec3[.float32, .WORLD],
    albedo: Vec3[.float32, .WORLD],
    u1: Float32,
    u2: Float32,
) -> BsdfSample[1]:
    var radius = sqrt(u1)
    var phi = Float32(2.0 * pi) * u2
    var z = sqrt(max(Float32(0.0), Float32(1.0) - u1))
    if wo.z[0] < 0.0:
        z = -z
    var wi = Vec3[.float32, .WORLD](radius * cos(phi), radius * sin(phi), z)
    return BsdfSample[1](wi, albedo, abs(z) * _INV_PI, False, True)


@always_inline
def _transmittance(distance: Float32, w: Vec3[.float32, .WORLD]) -> Float32:
    if abs(w.z[0]) <= 1.0e-20:
        return 0.0
    return exp(-abs(distance / w.z[0]))


@always_inline
def _phase_hg(cosine: Float32, g_in: Float32) -> Float32:
    var g = g_in.clamp(-0.99, 0.99)
    var denominator = 1.0 + g * g + 2.0 * g * cosine
    return (
        _INV_4_PI
        * (1.0 - g * g)
        / (denominator * sqrt(max(denominator, Float32(1.0e-20))))
    )


@always_inline
def _sample_phase_hg(
    wo: Vec3[.float32, .WORLD],
    g_in: Float32,
    u1: Float32,
    u2: Float32,
) -> Vec3[.float32, .WORLD]:
    var g = g_in.clamp(-0.99, 0.99)
    var cosine = 1.0 - 2.0 * u1
    if abs(g) >= 1.0e-3:
        var term = (1.0 - g * g) / (1.0 + g - 2.0 * g * u1)
        cosine = -(1.0 + g * g - term * term) / (2.0 * g)
        cosine = cosine.clamp(-1.0, 1.0)
    var sine = sqrt(max(Float32(0.0), Float32(1.0) - cosine * cosine))
    var axis = -wo
    var helper = Vec3[.float32, .WORLD](0.0, 1.0, 0.0)
    if abs(axis.y[0]) > 0.99:
        helper = Vec3[.float32, .WORLD](1.0, 0.0, 0.0)
    var tangent = normalize(cross(helper, axis))
    var bitangent = cross(axis, tangent)
    var phi = Float32(2.0 * pi) * u2
    return normalize(
        tangent * (sine * cos(phi))
        + bitangent * (sine * sin(phi))
        + axis * cosine
    )


@no_inline
def _layered_pdf(
    wo: Vec3[.float32, .WORLD],
    wi: Vec3[.float32, .WORLD],
    alpha: Float32,
    eta: Float32,
    n_samples: Int,
    seed: UInt32,
) -> Float32:
    if wo.z[0] * wi.z[0] <= 0.0:
        return 0.0
    var pdf_sum = Float32(max(n_samples, 1)) * _dielectric_pdf(
        wo, wi, alpha, eta, 1
    )
    for sample_idx in range(n_samples):
        var dimension = sample_idx * 8
        var wos = _sample_dielectric(
            wo,
            alpha,
            eta,
            _random(seed, dimension),
            _random(seed, dimension + 1),
            _random(seed, dimension + 2),
            True,
            2,
        )
        var wis = _sample_dielectric(
            wi,
            alpha,
            eta,
            _random(seed, dimension + 3),
            _random(seed, dimension + 4),
            _random(seed, dimension + 5),
            False,
            2,
        )
        if wos.ok and wis.ok:
            var diffuse_pdf = abs(wis.direction.z[0]) * _INV_PI
            if alpha < 1.0e-3:
                pdf_sum += diffuse_pdf
            else:
                var reflected = _sample_diffuse(
                    -wos.direction,
                    Vec3[.float32, .WORLD](1.0),
                    _random(seed, dimension + 6),
                    _random(seed, dimension + 7),
                )
                var reflected_direction = reflected.direction
                var exit_pdf = _dielectric_pdf(
                    -reflected_direction, wi, alpha, eta, 2
                )
                pdf_sum += (
                    _power_heuristic(wis.pdf, diffuse_pdf) * diffuse_pdf
                    + _power_heuristic(reflected.pdf[0], exit_pdf) * exit_pdf
                )
    var estimate = pdf_sum / Float32(max(n_samples, 1))
    return Float32(0.1) * _INV_4_PI + Float32(0.9) * estimate


@no_inline
def _evaluate_layered(
    wo: Vec3[.float32, .WORLD],
    wi: Vec3[.float32, .WORLD],
    albedo: Vec3[.float32, .WORLD],
    alpha: Float32,
    eta: Float32,
    thickness: Float32,
    layer_albedo: Vec3[.float32, .WORLD],
    g: Float32,
    max_depth: Int,
    n_samples: Int,
    seed: UInt32,
) -> Vec3[.float32, .WORLD]:
    if wo.z[0] <= 0.0 or wi.z[0] <= 0.0:
        return Vec3[.float32, .WORLD](0.0)
    var result = Vec3[.float32, .WORLD](_dielectric_f(wo, wi, alpha, eta, True))
    var indirect = Vec3[.float32, .WORLD](0.0)
    var medium_active = _max_component(layer_albedo) > 0.0
    for sample_idx in range(n_samples):
        var base = 16 + sample_idx * (max_depth * 8 + 16)
        var wos = _sample_dielectric(
            wo,
            alpha,
            eta,
            _random(seed, base),
            _random(seed, base + 1),
            _random(seed, base + 2),
            True,
            2,
        )
        var wis = _sample_dielectric(
            wi,
            alpha,
            eta,
            _random(seed, base + 3),
            _random(seed, base + 4),
            _random(seed, base + 5),
            False,
            2,
        )
        if not wos.ok or not wis.ok:
            continue
        var beta = Vec3[.float32, .WORLD](
            wos.value * abs(wos.direction.z[0]) / wos.pdf
        )
        var w = wos.direction
        var z = thickness
        for depth in range(max_depth):
            var dimension = base + 6 + depth * 8
            if depth > 3 and _max_component(beta) < 0.25:
                var q = max(Float32(0.0), Float32(1.0) - _max_component(beta))
                if _random(seed, dimension) < q:
                    break
                beta = beta / (1.0 - q)
            if medium_active:
                var distance = -log(
                    max(
                        Float32(1.0e-7),
                        Float32(1.0) - _random(seed, dimension + 1),
                    )
                ) * abs(w.z[0])
                var zp = z + distance if w.z[0] > 0.0 else z - distance
                if zp > 0.0 and zp < thickness:
                    var target = -wis.direction
                    var phase = _phase_hg(dot(-w, target)[0], g)
                    var phase_weight = _power_heuristic(wis.pdf, phase)
                    indirect += (
                        beta
                        * layer_albedo
                        * phase
                        * phase_weight
                        * _transmittance(zp - thickness, wis.direction)
                        * (wis.value / wis.pdf)
                    )
                    var phase_direction = _sample_phase_hg(
                        -w,
                        g,
                        _random(seed, dimension + 2),
                        _random(seed, dimension + 3),
                    )
                    beta *= layer_albedo
                    w = phase_direction
                    z = zp
                    if w.z[0] > 0.0:
                        var exit_value = _dielectric_f(-w, wi, alpha, eta, True)
                        if exit_value > 0.0:
                            var exit_pdf = _dielectric_pdf(
                                -w, wi, alpha, eta, 2
                            )
                            indirect += (
                                beta
                                * _transmittance(z - thickness, w)
                                * exit_value
                                * _power_heuristic(phase, exit_pdf)
                            )
                    continue
                z = thickness if zp >= thickness else Float32(0.0)
            else:
                z = Float32(0.0) if z == thickness else thickness
                beta = beta * _transmittance(thickness, w)

            if z == thickness:
                var reflected = _sample_dielectric(
                    -w,
                    alpha,
                    eta,
                    _random(seed, dimension + 2),
                    _random(seed, dimension + 3),
                    _random(seed, dimension + 4),
                    True,
                    1,
                )
                if not reflected.ok:
                    break
                beta = beta * (
                    reflected.value
                    * abs(reflected.direction.z[0])
                    / reflected.pdf
                )
                w = reflected.direction
                continue

            var target = -wis.direction
            var diffuse_pdf = abs(target.z[0]) * _INV_PI
            var nee_weight = Float32(1.0)
            if alpha >= 1.0e-3:
                nee_weight = _power_heuristic(wis.pdf, diffuse_pdf)
            indirect += (
                beta
                * albedo
                * _INV_PI
                * abs(wis.direction.z[0])
                * nee_weight
                * _transmittance(thickness, wis.direction)
                * (wis.value / wis.pdf)
            )
            var reflected = _sample_diffuse(
                -w,
                albedo,
                _random(seed, dimension + 5),
                _random(seed, dimension + 6),
            )
            beta *= reflected.weight
            w = reflected.direction
            var exit_value = _dielectric_f(-w, wi, alpha, eta, True)
            if exit_value > 0.0:
                var exit_pdf = _dielectric_pdf(-w, wi, alpha, eta, 2)
                var exit_weight = Float32(1.0)
                if alpha >= 1.0e-3:
                    exit_weight = _power_heuristic(reflected.pdf[0], exit_pdf)
                indirect += (
                    beta
                    * _transmittance(thickness, w)
                    * exit_value
                    * exit_weight
                )
    return result + indirect / Float32(max(n_samples, 1))


@always_inline
def _to_local(
    value: Vec3[.float32, .WORLD],
    tangent: Vec3[.float32, .WORLD],
    bitangent: Vec3[.float32, .WORLD],
    normal: Vec3[.float32, .WORLD],
) -> Vec3[.float32, .WORLD]:
    return Vec3[.float32, .WORLD](
        dot(value, tangent), dot(value, bitangent), dot(value, normal)
    )


@always_inline
def _from_local(
    value: Vec3[.float32, .WORLD],
    tangent: Vec3[.float32, .WORLD],
    bitangent: Vec3[.float32, .WORLD],
    normal: Vec3[.float32, .WORLD],
) -> Vec3[.float32, .WORLD]:
    return tangent * value.x + bitangent * value.y + normal * value.z


@always_inline
def _basis(
    normal: Vec3[.float32, .WORLD]
) -> Tuple[Vec3[.float32, .WORLD], Vec3[.float32, .WORLD]]:
    var helper = Vec3[.float32, .WORLD](0.0, 1.0, 0.0)
    if abs(normal.y[0]) > 0.99:
        helper = Vec3[.float32, .WORLD](1.0, 0.0, 0.0)
    var tangent = normalize(cross(helper, normal))
    return tangent, cross(normal, tangent)


@always_inline
def _evaluate_coated_diffuse_scalar(
    ray_direction: Vec3[.float32, .WORLD],
    normal: Vec3[.float32, .WORLD],
    albedo: Vec3[.float32, .WORLD],
    roughness: Float32,
    eta: Float32,
    thickness: Float32,
    layer_albedo: Vec3[.float32, .WORLD],
    g: Float32,
    max_depth: Int,
    n_samples: Int,
    out_direction: Vec3[.float32, .WORLD],
) -> BsdfEvaluation[1]:
    var tangent, bitangent = _basis(normal)
    var wo = _to_local(-normalize(ray_direction), tangent, bitangent, normal)
    var wi = _to_local(normalize(out_direction), tangent, bitangent, normal)
    var seed = _direction_seed(wo, wi)
    var safe_thickness = max(thickness, Float32(1.0e-20))
    var value = _evaluate_layered(
        wo,
        wi,
        albedo,
        roughness,
        eta,
        safe_thickness,
        layer_albedo,
        g,
        max_depth,
        n_samples,
        seed,
    )
    var pdf = _layered_pdf(wo, wi, roughness, eta, n_samples, seed)
    return BsdfEvaluation[1](value, pdf, False)


@no_inline
def _sample_coated_diffuse_scalar(
    ray_direction: Vec3[.float32, .WORLD],
    normal: Vec3[.float32, .WORLD],
    albedo: Vec3[.float32, .WORLD],
    roughness: Float32,
    eta: Float32,
    thickness: Float32,
    layer_albedo: Vec3[.float32, .WORLD],
    g: Float32,
    max_depth: Int,
    n_samples: Int,
    random_u: Float32,
    random_v: Float32,
) -> BsdfSample[1]:
    var tangent, bitangent = _basis(normal)
    var wo = _to_local(-normalize(ray_direction), tangent, bitangent, normal)
    if wo.z[0] <= 0.0:
        return BsdfSample[1](normal, albedo, 0.0, False, False)
    var seed = _sample_seed(random_u, random_v)
    var safe_thickness = max(thickness, Float32(1.0e-20))
    var interface_sample = _sample_dielectric(
        wo,
        roughness,
        eta,
        random_u,
        random_v,
        _random(seed, 0),
        True,
    )
    if not interface_sample.ok:
        return BsdfSample[1](normal, albedo, 0.0, False, False)
    var weight = Vec3[.float32, .WORLD](
        interface_sample.value
        * abs(interface_sample.direction.z[0])
        / interface_sample.pdf
    )
    if interface_sample.reflection:
        var world_direction = normalize(
            _from_local(interface_sample.direction, tangent, bitangent, normal)
        )
        var pdf = interface_sample.pdf
        if not interface_sample.delta:
            pdf = _layered_pdf(
                wo,
                interface_sample.direction,
                roughness,
                eta,
                n_samples,
                _direction_seed(wo, interface_sample.direction),
            )
        return BsdfSample[1](
            world_direction, weight, pdf, interface_sample.delta, True
        )

    var w = interface_sample.direction
    var z = safe_thickness
    var medium_active = _max_component(layer_albedo) > 0.0
    var specular_path = interface_sample.delta
    for depth in range(max_depth):
        var dimension = 1 + depth * 8
        if depth > 3 and _max_component(weight) < 0.25:
            var q = max(Float32(0.0), Float32(1.0) - _max_component(weight))
            if _random(seed, dimension) < q:
                return BsdfSample[1](normal, albedo, 0.0, False, False)
            weight = weight / (1.0 - q)
        if medium_active:
            var distance = -log(
                max(
                    Float32(1.0e-7),
                    Float32(1.0) - _random(seed, dimension + 1),
                )
            ) * abs(w.z[0])
            var zp = z + distance if w.z[0] > 0.0 else z - distance
            if zp > 0.0 and zp < safe_thickness:
                weight *= layer_albedo
                w = _sample_phase_hg(
                    -w,
                    g,
                    _random(seed, dimension + 2),
                    _random(seed, dimension + 3),
                )
                z = zp
                specular_path = False
                continue
            z = safe_thickness if zp >= safe_thickness else Float32(0.0)
        else:
            z = Float32(0.0) if z == safe_thickness else safe_thickness
            weight = weight * _transmittance(safe_thickness, w)
        if z == safe_thickness:
            var sampled = _sample_dielectric(
                -w,
                roughness,
                eta,
                _random(seed, dimension + 4),
                _random(seed, dimension + 5),
                _random(seed, dimension + 6),
                True,
            )
            if not sampled.ok:
                return BsdfSample[1](normal, albedo, 0.0, False, False)
            weight = weight * (
                sampled.value * abs(sampled.direction.z[0]) / sampled.pdf
            )
            specular_path = specular_path and sampled.delta
            w = sampled.direction
            if not sampled.reflection:
                var world_direction = normalize(
                    _from_local(w, tangent, bitangent, normal)
                )
                var pdf = sampled.pdf
                if not specular_path:
                    pdf = _layered_pdf(
                        wo,
                        w,
                        roughness,
                        eta,
                        n_samples,
                        _direction_seed(wo, w),
                    )
                return BsdfSample[1](
                    world_direction, weight, pdf, specular_path, True
                )
            continue

        var sampled = _sample_diffuse(
            -w,
            albedo,
            _random(seed, dimension + 4),
            _random(seed, dimension + 5),
        )
        weight *= sampled.weight
        specular_path = False
        w = sampled.direction
    return BsdfSample[1](normal, albedo, 0.0, False, False)


@always_inline
def _evaluate_coated_diffuse[
    length: SIMDLength
](
    ray_direction: Vec3[.float32, .WORLD, length],
    normal: Vec3[.float32, .WORLD, length],
    albedo: Vec3[.float32, .WORLD, length],
    roughness: SIMD[.float32, length],
    eta: SIMD[.float32, length],
    thickness: SIMD[.float32, length],
    layer_albedo: Vec3[.float32, .WORLD, length],
    g: SIMD[.float32, length],
    max_depth: SIMD[.float32, length],
    n_samples: SIMD[.float32, length],
    out_direction: Vec3[.float32, .WORLD, length],
) -> BsdfEvaluation[length]:
    var result = BsdfEvaluation[length](
        Vec3[.float32, .WORLD, length](0.0),
        SIMD[.float32, length](0.0),
        SIMD[.bool, length](fill=False),
    )
    for lane in range(Int(length)):
        var evaluated = _evaluate_coated_diffuse_scalar(
            Vec3[.float32, .WORLD](
                ray_direction.x[lane],
                ray_direction.y[lane],
                ray_direction.z[lane],
            ),
            Vec3[.float32, .WORLD](
                normal.x[lane], normal.y[lane], normal.z[lane]
            ),
            Vec3[.float32, .WORLD](
                albedo.x[lane], albedo.y[lane], albedo.z[lane]
            ),
            roughness[lane],
            eta[lane],
            thickness[lane],
            Vec3[.float32, .WORLD](
                layer_albedo.x[lane],
                layer_albedo.y[lane],
                layer_albedo.z[lane],
            ),
            g[lane],
            Int(max_depth[lane]),
            Int(n_samples[lane]),
            Vec3[.float32, .WORLD](
                out_direction.x[lane],
                out_direction.y[lane],
                out_direction.z[lane],
            ),
        )
        result.value.x[lane] = evaluated.value.x[0]
        result.value.y[lane] = evaluated.value.y[0]
        result.value.z[lane] = evaluated.value.z[0]
        result.pdf[lane] = evaluated.pdf[0]
    return result^


@always_inline
def _sample_coated_diffuse[
    length: SIMDLength
](
    ray_direction: Vec3[.float32, .WORLD, length],
    normal: Vec3[.float32, .WORLD, length],
    albedo: Vec3[.float32, .WORLD, length],
    roughness: SIMD[.float32, length],
    eta: SIMD[.float32, length],
    thickness: SIMD[.float32, length],
    layer_albedo: Vec3[.float32, .WORLD, length],
    g: SIMD[.float32, length],
    max_depth: SIMD[.float32, length],
    n_samples: SIMD[.float32, length],
    random_u: SIMD[.float32, length],
    random_v: SIMD[.float32, length],
) -> BsdfSample[length]:
    var result = BsdfSample[length](
        normal,
        Vec3[.float32, .WORLD, length](0.0),
        SIMD[.float32, length](0.0),
        SIMD[.bool, length](fill=False),
        SIMD[.bool, length](fill=False),
    )
    for lane in range(Int(length)):
        var sampled = _sample_coated_diffuse_scalar(
            Vec3[.float32, .WORLD](
                ray_direction.x[lane],
                ray_direction.y[lane],
                ray_direction.z[lane],
            ),
            Vec3[.float32, .WORLD](
                normal.x[lane], normal.y[lane], normal.z[lane]
            ),
            Vec3[.float32, .WORLD](
                albedo.x[lane], albedo.y[lane], albedo.z[lane]
            ),
            roughness[lane],
            eta[lane],
            thickness[lane],
            Vec3[.float32, .WORLD](
                layer_albedo.x[lane],
                layer_albedo.y[lane],
                layer_albedo.z[lane],
            ),
            g[lane],
            Int(max_depth[lane]),
            Int(n_samples[lane]),
            random_u[lane],
            random_v[lane],
        )
        result.direction.x[lane] = sampled.direction.x[0]
        result.direction.y[lane] = sampled.direction.y[0]
        result.direction.z[lane] = sampled.direction.z[0]
        result.weight.x[lane] = sampled.weight.x[0]
        result.weight.y[lane] = sampled.weight.y[0]
        result.weight.z[lane] = sampled.weight.z[0]
        result.pdf[lane] = sampled.pdf[0]
        result.delta[lane] = sampled.delta[0]
        result.ok[lane] = sampled.ok[0]
    return result^
