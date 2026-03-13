from bajo.core.vec import Vec3, vmin, vmax, cross, length
from bajo.core.aabb import AxisAlignedBoundingBox


comptime Trianglef32 = Triangle[DType.float32]
comptime Trianglef64 = Triangle[DType.float64]


@fieldwise_init
struct Triangle[dtype: DType](Copyable):
    var v0: Vec3[Self.dtype]
    var v1: Vec3[Self.dtype]
    var v2: Vec3[Self.dtype]

    def centroid(self) -> Vec3[Self.dtype]:
        return 0.5 * (self.v0 + self.v1 + self.v2)

    def bounds(self) -> AxisAlignedBoundingBox[Self.dtype]:
        return AxisAlignedBoundingBox(self.v0, self.v1, self.v2)

    def normal(self) -> Vec3[Self.dtype]:
        edge0 = self.v1 - self.v0
        edge1 = self.v2 - self.v0
        return cross(edge0, edge1)

    def area(self) -> Scalar[Self.dtype]:
        normal = self.normal()
        return 0.5 * length(normal)
