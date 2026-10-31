"""Geometry-owned opacity and deterministic stochastic hit acceptance."""

from std.memory import bitcast
from std.utils.numerics import isfinite
from bajo.core import Rayf32, Point3f32, Vec3
from std.math import acos, atan2, pi
from bajo.core.random import _mix32, _u32_to_unit_float
from bajo.rt.material_types import NO_TEXTURE, SurfaceStore


@fieldwise_init
struct Opacity(Copyable, Writable):
    var scale: Float32
    var texture_index: UInt32
    var u_scale: Float32
    var v_scale: Float32

    def validate(self, image_count: Int) raises:
        if (
            not isfinite(self.scale)
            or self.scale < 0.0
            or not isfinite(self.u_scale)
            or not isfinite(self.v_scale)
        ):
            raise Error(
                "opacity parameters must be finite and scale non-negative"
            )
        if self.texture_index != NO_TEXTURE and self.texture_index >= UInt32(
            image_count
        ):
            raise Error("opacity image index is out of range")

    @always_inline
    def evaluate(
        self, surfaces: SurfaceStore, u: Float32, v: Float32
    ) -> Float32:
        var value = self.scale
        if self.texture_index != NO_TEXTURE:
            value *= surfaces.image_textures[
                Int(self.texture_index)
            ].sample_scalar_bilinear(u * self.u_scale, v * self.v_scale)
        return value.clamp(0.0, 1.0)


struct SceneOpacity:
    var records: Dict[UInt32, Opacity]

    def __init__(out self):
        self.records = Dict[UInt32, Opacity]()


@always_inline
def opacity_ray_sample(ray: Rayf32[.WORLD]) -> Float32:
    """A deterministic primary/shadow decision independent of BVH order."""
    var seed = UInt32(0xA511E9B3)
    for value in [ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z]:
        seed = _mix32(seed ^ bitcast[.uint32](value))
    return _u32_to_unit_float(seed)


@always_inline
def opacity_point_sample(point: Point3f32[.WORLD]) -> Float32:
    """Position-stable coverage for area-light emission, independent of RNG."""
    var seed = UInt32(0x728E5B29)
    for value in [point.x, point.y, point.z]:
        seed = _mix32(seed ^ bitcast[.uint32](value))
    return _u32_to_unit_float(seed)


@always_inline
def opacity_sphere_uv[
    length: SIMDLength = 1
](normal: Vec3[.float32, .WORLD, length]) -> Tuple[
    SIMD[.float32, length], SIMD[.float32, length]
]:
    var phi = atan2(normal.y, normal.x)
    phi = phi.lt(0.0).select(phi + Float32(2.0 * pi), phi)
    return (
        phi / Float32(2.0 * pi),
        acos(normal.z.clamp(-1.0, 1.0)) / Float32(pi),
    )


@always_inline
def opacity_accepts(alpha: Float32, sample: Float32) -> Bool:
    return alpha >= 1.0 or (alpha > 0.0 and sample <= alpha)
