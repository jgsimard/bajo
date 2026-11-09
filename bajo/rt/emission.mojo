"""Geometry-owned emission, independent of the surface scattering material."""

from bajo.core.color import Color
from bajo.rt.spectrum_sources import NO_SPECTRUM
from bajo.rt.material_types import (
    Emissive,
    PrimitiveId,
)


struct AreaEmitter(Copyable, Writable):
    var radiance: Color
    var two_sided: Bool
    var spectrum: UInt32

    def __init__(
        out self,
        radiance: Color = Color(0.0),
        two_sided: Bool = False,
        spectrum: UInt32 = NO_SPECTRUM,
    ):
        self.radiance = radiance
        self.two_sided = two_sided
        self.spectrum = spectrum

    def validate(self) raises:
        Emissive(self.radiance).validate()


struct SceneEmission:
    var records: Dict[UInt32, AreaEmitter]

    def __init__(out self):
        self.records = Dict[UInt32, AreaEmitter]()

    def validate(self, spheres: Int, triangles: Int, instances: Int) raises:
        for entry in self.records.items():
            var primitive = PrimitiveId(entry.key)
            var count = 0
            match primitive.kind():
                case .SPHERE:
                    count = spheres
                case .TRIANGLE:
                    count = triangles
                case .TRIANGLE_INSTANCE:
                    count = instances
                case .UNKNOWN:
                    pass
            if primitive.index() >= UInt32(count):
                raise Error("emission primitive index is out of range")
            entry.value.validate()
