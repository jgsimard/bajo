"""Runtime absorption media and geometry-owned medium boundaries."""

from std.math import exp
from std.sys.info import size_of
from std.utils.numerics import max_finite
from bajo.core import Vec3
from bajo.core.tensor_layout import record_tile, gather_record_vec3
from bajo.rt.render_types import Color


@always_inline
def medium_transmittance[
    length: SIMDLength = 1
](
    sigma_a: Vec3[.float32, .WORLD, length],
    distance: SIMD[.float32, length],
) -> Vec3[.float32, .WORLD, length]:
    # Mask the distance before multiplication: zero absorption is transparent
    # even for infinite segments, without forming 0 * infinity.
    var result = Vec3[.float32, .WORLD, length](0.0)
    comptime for axis in range(3):
        var depth = sigma_a[axis] * sigma_a[axis].ne(0.0).select(distance, 0.0)
        # SIMD exp may clamp its argument; preserve exact zero for an
        # infinitely absorbing segment (including overflowed optical depth).
        result[axis] = depth.gt(max_finite[.float32]()).select(
            Float32(0.0), exp(-depth)
        )
    return result


@always_inline
def transmitted_medium(boundary: UInt32, front_face: Bool) -> UInt32:
    return (boundary & UInt32(65535)) if front_face else (boundary >> 16)


struct SceneMedia:
    # ID zero denotes vacuum. Boundary words pack inside/outside UInt16 IDs.
    # Empty sidecars mean vacuum; allocate prefixes only when boundaries exist.
    var absorption: List[Color]
    var spheres: List[UInt32]
    var triangles: List[UInt32]
    var instances: List[UInt32]

    def __init__(out self):
        self.absorption = [Color(0.0)]
        self.spheres = List[UInt32]()
        self.triangles = List[UInt32]()
        self.instances = List[UInt32]()

    @always_inline
    def transmittance[
        length: SIMDLength
    ](
        self,
        ids: SIMD[.uint32, length],
        distance: SIMD[.float32, length],
        active: SIMD[.bool, length],
    ) -> Vec3[.float32, .WORLD, length]:
        var absorbing = active & ids.ne(0)
        if not absorbing.reduce_or():
            return Vec3[.float32, .WORLD, length](1.0)
        # Color is three adjacent scalar channels. Inactive IDs may be invalid;
        # vacuum row zero is always available in a validated scene.
        comptime assert size_of[Color]() == 3 * size_of[Float32]()
        var fields = record_tile[.float32, 3](
            self.absorption.unsafe_ptr().unsafe_bitcast[Float32](),
            len(self.absorption),
        )
        var sigma = gather_record_vec3(fields, absorbing.select(ids, 0))
        return medium_transmittance(sigma, absorbing.select(distance, 0.0))

    def add(mut self, sigma_a: Color) raises -> UInt32:
        if (
            not sigma_a.is_finite()[0]
            or sigma_a.x < 0.0
            or sigma_a.y < 0.0
            or sigma_a.z < 0.0
        ):
            raise Error("medium absorption must be finite and non-negative")
        if len(self.absorption) >= 65536:
            raise Error("too many media")
        var index = UInt32(len(self.absorption))
        self.absorption.append(sigma_a)
        return index

    @staticmethod
    def set_boundary(mut values: List[UInt32], index: Int, boundary: UInt32):
        if boundary == 0 and index >= len(values):
            return
        if index >= len(values):
            values.resize(index + 1, fill=UInt32(0))
        values[index] = boundary

    @staticmethod
    @always_inline
    def boundary(values: ImmSpan[UInt32, _], index: Int) -> UInt32:
        return values[index] if index < len(values) else UInt32(0)

    def validate(
        self, sphere_count: Int, triangle_count: Int, instance_count: Int
    ) raises:
        if (
            len(self.spheres) > sphere_count
            or len(self.triangles) > triangle_count
            or len(self.instances) > instance_count
        ):
            raise Error("medium boundary sidecar exceeds geometry count")
        if len(self.absorption) == 0 or (
            self.absorption[0].x != 0.0
            or self.absorption[0].y != 0.0
            or self.absorption[0].z != 0.0
        ):
            raise Error("medium zero must be vacuum")
        self._validate_boundaries(self.spheres)
        self._validate_boundaries(self.triangles)
        self._validate_boundaries(self.instances)
        for sigma_a in self.absorption:
            if (
                not sigma_a.is_finite()[0]
                or sigma_a.x < 0.0
                or sigma_a.y < 0.0
                or sigma_a.z < 0.0
            ):
                raise Error("medium absorption must be finite and non-negative")

    def _validate_boundaries(self, values: ImmSpan[UInt32, _]) raises:
        for boundary in values:
            if Int(boundary & UInt32(65535)) >= len(self.absorption) or Int(
                boundary >> 16
            ) >= len(self.absorption):
                raise Error("medium boundary reference is out of range")
