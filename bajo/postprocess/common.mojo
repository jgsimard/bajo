"""Scalar port of Falcor SVGFCommon, FilterMoments and Atrous (BSD-3-Clause).

Copyright (c) 2015-2023 NVIDIA CORPORATION. See FALCOR_LICENSE.txt.
Only the spatial stages are used. The progressive renderer supplies sample
moments instead of SVGF's reprojected exponential temporal history.
"""
from std.math import abs, exp, pow, sqrt
from bajo.core import dot
from bajo.core.color import Color, luminance, LUMINANCE_WEIGHTS
from .types import GuideLayout
from .statistics import CascadeLayout, reconstruct_component


def _reconstruct_pixel(
    bins: ImmPointer[Float64, _],
    meta: ImmPointer[Float64, _],
    pixel: Int,
    width: Int,
    height: Int,
    n_bins: Int,
    samples: Int,
    albedo: Color,
    firefly: Bool,
) -> Tuple[Color, Color, Color, Float32]:
    var diffuse, variance = reconstruct_component(
        bins,
        meta,
        pixel,
        width,
        height,
        0,
        n_bins,
        samples,
        LUMINANCE_WEIGHTS[.float32] / albedo,
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
        Color(0),
        firefly,
    )
    var m = pixel * CascadeLayout.META
    var normalization = meta[unsafe_offset=m]
    var passthrough = Color(0)
    if normalization > 0:
        comptime for c in range(3):
            passthrough[c] = Float32(
                meta[unsafe_offset=m + CascadeLayout.PASSTHROUGH + c]
                / normalization
            )
    return diffuse, glossy, passthrough, variance


def _filter_pixel(
    source: ImmPointer[Float32, _],
    variance: ImmPointer[Float32, _],
    fields: ImmPointer[Float32, _],
    surfaces: ImmPointer[UInt32, _],
    width: Int,
    height: Int,
    pixel: Int,
    iteration: Int,
    samples: Int,
    phi_color: Float32,
    phi_normal: Float32,
) -> Tuple[Color, Float32]:
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


def _color(p: ImmPointer[Float32, _], index: Int) -> Color:
    return Color(p.unsafe_offset(3 * index))


def _guide_vec(p: ImmPointer[Float32, _], index: Int, field: Int) -> Color:
    return Color(p.unsafe_offset(GuideLayout.STRIDE * index + field))


def _svgf_weight(
    fields: ImmPointer[Float32, _],
    p: Int,
    q: Int,
    distance: Float32,
    step: Float32,
    lp: Float32,
    lq: Float32,
    phi_color: Float32,
    phi_normal: Float32,
) -> Float32:
    var z = fields[unsafe_offset=p * GuideLayout.STRIDE + GuideLayout.DEPTH]
    var qz = fields[unsafe_offset=q * GuideLayout.STRIDE + GuideLayout.DEPTH]
    var derivative = fields[
        unsafe_offset=p * GuideLayout.STRIDE + GuideLayout.DEPTH_FWIDTH
    ]
    var phi_depth = max(derivative, Float32(1e-8)) * step * distance
    var n = _guide_vec(fields, p, GuideLayout.NORMAL)
    var qn = _guide_vec(fields, q, GuideLayout.NORMAL)
    return pow(dot(n, qn).clamp(0, 1), phi_normal) * exp(
        -abs(z - qz) / max(phi_depth, Float32(1e-8))
        - abs(lp - lq) / max(phi_color, Float32(1e-10))
    )


def _bootstrap_pixel(
    source: ImmPointer[Float32, _],
    variance: ImmPointer[Float32, _],
    fields: ImmPointer[Float32, _],
    surfaces: ImmPointer[UInt32, _],
    width: Int,
    height: Int,
    p: Int,
    samples: Int,
    phi_color: Float32,
    phi_normal: Float32,
) -> Tuple[Color, Float32]:
    var center = _color(source, p)
    if (
        samples >= 4
        or surfaces[unsafe_offset=p] == GuideLayout.MISS
        or fields[unsafe_offset=p * GuideLayout.STRIDE + GuideLayout.DIFFUSE]
        == 0
    ):
        return center, variance[unsafe_offset=p]
    var lp = luminance(center)
    var sum_color = Color(0)
    var sum_w = Float32(0)
    var first = Float32(0)
    var second = Float32(0)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            var x = p % width + dx
            var y = p // width + dy
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            var q = y * width + x
            if (
                surfaces[unsafe_offset=q] == GuideLayout.MISS
                or fields[
                    unsafe_offset=q * GuideLayout.STRIDE + GuideLayout.DIFFUSE
                ]
                == 0
            ):
                continue
            var color = _color(source, q)
            var lq = luminance(color)
            var w = _svgf_weight(
                fields,
                p,
                q,
                sqrt(Float32(dx * dx + dy * dy)),
                3,
                lp,
                lq,
                phi_color,
                phi_normal,
            )
            sum_w += w
            sum_color += w * color
            first += w * lq
            second += w * (variance[unsafe_offset=q] + lq * lq)
    sum_w = max(sum_w, Float32(1e-6))
    var mean = first / sum_w
    return sum_color / sum_w, max(second / sum_w - mean * mean, Float32(0)) * (
        4 / Float32(samples)
    )


def _atrous_pixel(
    source: ImmPointer[Float32, _],
    fields: ImmPointer[Float32, _],
    surfaces: ImmPointer[UInt32, _],
    variance: ImmPointer[Float32, _],
    width: Int,
    height: Int,
    p: Int,
    step: Int,
    phi_color: Float32,
    phi_normal: Float32,
) -> Tuple[Color, Float32]:
    var center = _color(source, p)
    if (
        surfaces[unsafe_offset=p] == GuideLayout.MISS
        or fields[unsafe_offset=p * GuideLayout.STRIDE + GuideLayout.DIFFUSE]
        == 0
    ):
        return center, variance[unsafe_offset=p]
    var blurred = Float32(0)
    var norm = Float32(0)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            var x = p % width + dx
            var y = p // width + dy
            if x >= 0 and y >= 0 and x < width and y < height:
                var w = (
                    Float32((2 if dx == 0 else 1) * (2 if dy == 0 else 1)) / 16
                )
                blurred += w * variance[unsafe_offset=y * width + x]
                norm += w
    var phi_l = phi_color * sqrt(max(blurred / norm, Float32(0)) + 1e-10)
    var lp = luminance(center)
    var sum_w = Float32(1)
    var result = center
    var v = variance[unsafe_offset=p]
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            var x = p % width + dx * step
            var y = p // width + dy * step
            if (
                (dx == 0 and dy == 0)
                or x < 0
                or y < 0
                or x >= width
                or y >= height
            ):
                continue
            var q = y * width + x
            if (
                surfaces[unsafe_offset=q] == GuideLayout.MISS
                or fields[
                    unsafe_offset=q * GuideLayout.STRIDE + GuideLayout.DIFFUSE
                ]
                == 0
            ):
                continue
            var wx = Float32(
                1 if dx == 0 else (2.0 / 3.0 if abs(dx) == 1 else 1.0 / 6.0)
            )
            var wy = Float32(
                1 if dy == 0 else (2.0 / 3.0 if abs(dy) == 1 else 1.0 / 6.0)
            )
            var color = _color(source, q)
            var w = (
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
                )
            )
            result += w * color
            v += w * w * variance[unsafe_offset=q]
            sum_w += w
    return result / sum_w, v / (sum_w * sum_w)


def _albedo(fields: ImmPointer[Float32, _], i: Int) -> Color:
    var a = _guide_vec(fields, i, GuideLayout.ALBEDO)
    return Color(
        max(a.x, Float32(0.001)),
        max(a.y, Float32(0.001)),
        max(a.z, Float32(0.001)),
    )
