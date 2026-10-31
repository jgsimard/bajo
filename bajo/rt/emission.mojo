"""Geometry-owned emission, independent of the surface scattering material."""

from bajo.core.color import Color
from bajo.bvh.constants import PrimitiveKind
from bajo.rt.material_types import (
    Emissive,
    PRIMITIVE_INDEX_BITS,
    PRIMITIVE_INDEX_MASK,
)


struct AreaEmitter(Copyable, Writable):
    var radiance: Color
    var two_sided: Bool

    def __init__(
        out self, radiance: Color = Color(0.0), two_sided: Bool = False
    ):
        self.radiance = radiance
        self.two_sided = two_sided

    def validate(self) raises:
        Emissive(self.radiance).validate()


struct SceneEmission:
    var records: Dict[UInt32, AreaEmitter]

    def __init__(out self):
        self.records = Dict[UInt32, AreaEmitter]()

    def validate(self, spheres: Int, triangles: Int, instances: Int) raises:
        for entry in self.records.items():
            var kind = entry.key >> PRIMITIVE_INDEX_BITS
            var count = (
                spheres if kind
                == PrimitiveKind.SPHERE.value else triangles if kind
                == PrimitiveKind.TRIANGLE.value else instances if kind
                == PrimitiveKind.TRIANGLE_INSTANCE.value else 0
            )
            if (entry.key & PRIMITIVE_INDEX_MASK) >= UInt32(count):
                raise Error("emission primitive index is out of range")
            entry.value.validate()
