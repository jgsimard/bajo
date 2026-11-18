"""Scalar/SIMD port of Falcor SVGFCommon, FilterMoments and Atrous (BSD-3-Clause).

Copyright (c) 2015-2023 NVIDIA CORPORATION. See FALCOR_LICENSE.txt.
Only the spatial stages are used. The progressive renderer supplies sample
moments instead of SVGF's reprojected exponential temporal history.
"""
from std.math import abs, exp, pow, sqrt
from bajo.core import Vec3, dot
from bajo.core.color import LUMINANCE_WEIGHTS, luminance
from .types import GuideLayout
from .statistics import CascadeLayout, reconstruct_component


def _reconstruct_pixel[
    length: SIMDLength = 1
](
    bins: ImmPointer[Float64, _],
    meta: ImmPointer[Float64, _],
    pixel: SIMD[.int, length],
    width: Int,
    height: Int,
    n_bins: Int,
    samples: Int,
    albedo: Vec3[.float32, .WORLD, length],
    firefly: Bool,
) -> Tuple[
    Vec3[.float32, .WORLD, length],
    Vec3[.float32, .WORLD, length],
    Vec3[.float32, .WORLD, length],
    SIMD[.float32, length],
]:
    var diffuse, variance = reconstruct_component(
        bins,
        meta,
        pixel,
        width,
        height,
        0,
        n_bins,
        samples,
        LUMINANCE_WEIGHTS[.float32].broadcast[length]() / albedo,
        firefly,
    )
    var glossy, _ = reconstruct_component(
        bins,
        meta,
        pixel,
        width,
        height,
        1,
        n_bins,
        samples,
        Vec3[.float32, .WORLD, length](0),
        firefly,
    )
    var first = pixel * CascadeLayout.META
    var normalization = meta.unsafe_gather(first)
    var valid = normalization.gt(0)
    var denom = valid.select(normalization, 1)
    var passthrough = Vec3[.float32, .WORLD, length](0)
    comptime for channel in range(3):
        passthrough[channel] = valid.select(
            (
                meta.unsafe_gather(first + CascadeLayout.PASSTHROUGH + channel)
                / denom
            ).cast[.float32](),
            0,
        )
    return diffuse, glossy, passthrough, variance


def _filter_pixel[
    length: SIMDLength = 1
](
    source: ImmPointer[Float32, _],
    variance: ImmPointer[Float32, _],
    fields: ImmPointer[Float32, _],
    surfaces: ImmPointer[UInt32, _],
    width: Int,
    height: Int,
    pixel: SIMD[.int, length],
    iteration: Int,
    samples: Int,
    phi_color: Float32,
    phi_normal: Float32,
) -> Tuple[Vec3[.float32, .WORLD, length], SIMD[.float32, length]]:
    if iteration < 0:
        return _bootstrap_pixel(
            source,
            variance,
            fields,
            surfaces,
            width,
            height,
            pixel,
            samples,
            phi_color,
            phi_normal,
        )
    return _atrous_pixel(
        source,
        fields,
        surfaces,
        variance,
        width,
        height,
        pixel,
        1 << iteration,
        phi_color,
        phi_normal,
    )


def _color[
    length: SIMDLength = 1
](p: ImmPointer[Float32, _], index: SIMD[.int, length]) -> Vec3[
    .float32, .WORLD, length
]:
    return Vec3[.float32, .WORLD, length](
        p.unsafe_gather(3 * index),
        p.unsafe_gather(3 * index + 1),
        p.unsafe_gather(3 * index + 2),
    )


def _guide_vec[
    length: SIMDLength = 1
](p: ImmPointer[Float32, _], index: SIMD[.int, length], field: Int) -> Vec3[
    .float32, .WORLD, length
]:
    var offset = index * GuideLayout.STRIDE + field
    return Vec3[.float32, .WORLD, length](
        p.unsafe_gather(offset),
        p.unsafe_gather(offset + 1),
        p.unsafe_gather(offset + 2),
    )


def _svgf_weight[
    length: SIMDLength = 1
](
    fields: ImmPointer[Float32, _],
    p: SIMD[.int, length],
    q: SIMD[.int, length],
    distance: Float32,
    step: Float32,
    lp: SIMD[.float32, length],
    lq: SIMD[.float32, length],
    phi_color: SIMD[.float32, length],
    phi_normal: Float32,
) -> SIMD[.float32, length]:
    var z = fields.unsafe_gather(p * GuideLayout.STRIDE + GuideLayout.DEPTH)
    var qz = fields.unsafe_gather(q * GuideLayout.STRIDE + GuideLayout.DEPTH)
    var derivative = fields.unsafe_gather(
        p * GuideLayout.STRIDE + GuideLayout.DEPTH_FWIDTH
    )
    var phi_depth = max(derivative, Float32(1e-8)) * step * distance
    var n = _guide_vec(fields, p, GuideLayout.NORMAL)
    var qn = _guide_vec(fields, q, GuideLayout.NORMAL)
    return pow(dot(n, qn).clamp(0, 1), phi_normal) * exp(
        -abs(z - qz) / max(phi_depth, Float32(1e-8))
        - abs(lp - lq) / max(phi_color, Float32(1e-10))
    )


def _bootstrap_pixel[
    length: SIMDLength = 1
](
    source: ImmPointer[Float32, _],
    variance: ImmPointer[Float32, _],
    fields: ImmPointer[Float32, _],
    surfaces: ImmPointer[UInt32, _],
    width: Int,
    height: Int,
    p: SIMD[.int, length],
    samples: Int,
    phi_color: Float32,
    phi_normal: Float32,
) -> Tuple[Vec3[.float32, .WORLD, length], SIMD[.float32, length]]:
    var center = _color(source, p)
    var center_variance = variance.unsafe_gather(p)
    var filtering = surfaces.unsafe_gather(p).ne(
        GuideLayout.MISS
    ) & fields.unsafe_gather(p * GuideLayout.STRIDE + GuideLayout.DIFFUSE).ne(0)
    if samples >= 4 or not filtering.reduce_or():
        return center, center_variance
    var lp = luminance(center)
    var sum_color = Vec3[.float32, .WORLD, length](0)
    var sum_w = SIMD[.float32, length](0)
    var first = SIMD[.float32, length](0)
    var second = SIMD[.float32, length](0)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            var x = p % width + dx
            var y = p // width + dy
            var inside = (
                filtering & x.ge(0) & y.ge(0) & x.lt(width) & y.lt(height)
            )
            var q = inside.select(y * width + x, 0)
            var valid = (
                inside
                & surfaces.unsafe_gather(q).ne(GuideLayout.MISS)
                & fields.unsafe_gather(
                    q * GuideLayout.STRIDE + GuideLayout.DIFFUSE
                ).ne(0)
            )
            if not valid.reduce_or():
                continue
            var color = Vec3.select(
                valid, _color(source, q), Vec3[.float32, .WORLD, length](0)
            )
            var lq = luminance(color)
            var w = valid.select(
                _svgf_weight(
                    fields,
                    p,
                    q,
                    sqrt(Float32(dx * dx + dy * dy)),
                    3,
                    lp,
                    lq,
                    SIMD[.float32, length](phi_color),
                    phi_normal,
                ),
                0,
            )
            sum_w += w
            sum_color += w * color
            first += w * lq
            second += w * (valid.select(variance.unsafe_gather(q), 0) + lq * lq)
    sum_w = max(sum_w, Float32(1e-6))
    var mean = first / sum_w
    var v = max(second / sum_w - mean * mean, Float32(0)) * (
        4 / Float32(samples)
    )
    return Vec3.select(filtering, sum_color / sum_w, center), filtering.select(
        v, center_variance
    )


def _atrous_pixel[
    length: SIMDLength = 1
](
    source: ImmPointer[Float32, _],
    fields: ImmPointer[Float32, _],
    surfaces: ImmPointer[UInt32, _],
    variance: ImmPointer[Float32, _],
    width: Int,
    height: Int,
    p: SIMD[.int, length],
    step: Int,
    phi_color: Float32,
    phi_normal: Float32,
) -> Tuple[Vec3[.float32, .WORLD, length], SIMD[.float32, length]]:
    var center = _color(source, p)
    var center_variance = variance.unsafe_gather(p)
    var filtering = surfaces.unsafe_gather(p).ne(
        GuideLayout.MISS
    ) & fields.unsafe_gather(p * GuideLayout.STRIDE + GuideLayout.DIFFUSE).ne(0)
    if not filtering.reduce_or():
        return center, center_variance
    var blurred = SIMD[.float32, length](0)
    var norm = SIMD[.float32, length](0)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            var x = p % width + dx
            var y = p // width + dy
            var inside = x.ge(0) & y.ge(0) & x.lt(width) & y.lt(height)
            var q = inside.select(y * width + x, 0)
            var w = Float32((2 if dx == 0 else 1) * (2 if dy == 0 else 1)) / 16
            blurred += w * inside.select(variance.unsafe_gather(q), 0)
            norm += inside.select(w, 0)
    var phi_l = phi_color * sqrt(max(blurred / norm, Float32(0)) + 1e-10)
    var lp = luminance(center)
    var sum_w = SIMD[.float32, length](1)
    var result = center
    var v = center_variance
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            var x = p % width + dx * step
            var y = p // width + dy * step
            var inside = (
                filtering & x.ge(0) & y.ge(0) & x.lt(width) & y.lt(height)
            )
            var q = inside.select(y * width + x, 0)
            var valid = (
                inside
                & surfaces.unsafe_gather(q).ne(GuideLayout.MISS)
                & fields.unsafe_gather(
                    q * GuideLayout.STRIDE + GuideLayout.DIFFUSE
                ).ne(0)
            )
            if not valid.reduce_or():
                continue
            var wx = Float32(
                1 if dx == 0 else (2.0 / 3.0 if abs(dx) == 1 else 1.0 / 6.0)
            )
            var wy = Float32(
                1 if dy == 0 else (2.0 / 3.0 if abs(dy) == 1 else 1.0 / 6.0)
            )
            var color = Vec3.select(
                valid, _color(source, q), Vec3[.float32, .WORLD, length](0)
            )
            var w = valid.select(
                wx
                * wy
                * _svgf_weight(
                    fields,
                    p,
                    q,
                    sqrt(Float32(dx * dx + dy * dy)),
                    Float32(step),
                    lp,
                    luminance(color),
                    phi_l,
                    phi_normal,
                ),
                0,
            )
            result += w * color
            v += w * w * valid.select(variance.unsafe_gather(q), 0)
            sum_w += w
    return Vec3.select(filtering, result / sum_w, center), filtering.select(
        v / (sum_w * sum_w), center_variance
    )


def _albedo[
    length: SIMDLength = 1
](fields: ImmPointer[Float32, _], i: SIMD[.int, length]) -> Vec3[
    .float32, .WORLD, length
]:
    var a = _guide_vec(fields, i, GuideLayout.ALBEDO)
    return Vec3[.float32, .WORLD, length](
        max(a.x, Float32(0.001)),
        max(a.y, Float32(0.001)),
        max(a.z, Float32(0.001)),
    )
