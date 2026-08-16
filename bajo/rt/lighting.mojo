"""Host/device-safe direct-lighting weighting primitives."""

from bajo.rt.types import RENDER


@always_inline
def power_heuristic[
    length: SIMDLength = 1
](
    pdf_a: SIMD[DType.float32, length],
    pdf_b: SIMD[DType.float32, length],
) -> SIMD[DType.float32, length]:
    var a2 = pdf_a * pdf_a
    var b2 = pdf_b * pdf_b
    var denominator = a2 + b2
    var nonzero = denominator.gt(0.0)
    return nonzero.select(
        a2 / nonzero.select(denominator, Float32(1.0)), Float32(0.0)
    )


@always_inline
def _direct_light_scale[
    ALGORITHM: RENDER, length: SIMDLength
](
    surface_cosine: SIMD[DType.float32, length],
    light_pdf: SIMD[DType.float32, length],
    bsdf_pdf: SIMD[DType.float32, length],
    valid: SIMD[DType.bool, length],
) -> SIMD[DType.float32, length]:
    comptime assert ALGORITHM in (RENDER.NEE, RENDER.MIS)
    var ok = valid & light_pdf.gt(0.0) & bsdf_pdf.gt(0.0)
    var safe_light_pdf = ok.select(light_pdf, Float32(1.0))
    var estimator_weight = SIMD[DType.float32, length](1.0)
    comptime if ALGORITHM == RENDER.MIS:
        estimator_weight = power_heuristic(light_pdf, bsdf_pdf)
    return ok.select(
        surface_cosine * estimator_weight / safe_light_pdf, Float32(0.0)
    )
