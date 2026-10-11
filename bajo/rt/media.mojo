"""Runtime absorption media and geometry-owned medium boundaries."""

from std.math import exp
from bajo.rt.render_types import Color


@always_inline
def medium_transmittance(sigma_a: Color, distance: Float32) -> Color:
    # Zero absorption remains transparent even for an infinite segment.
    return Color(
        1.0 if sigma_a.x == 0.0 else exp(-sigma_a.x * distance),
        1.0 if sigma_a.y == 0.0 else exp(-sigma_a.y * distance),
        1.0 if sigma_a.z == 0.0 else exp(-sigma_a.z * distance),
    )


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
        while len(values) <= index:
            values.append(UInt32(0))
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
