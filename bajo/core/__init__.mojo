from .aabb import AxisAlignedBoundingBox
from .quat import Quaternion
from .transform import Affine3
from .vec import (
    Vec2,
    Vec3,
    normalize,
    length,
    length2,
    vmin,
    vmax,
    cross,
    assert_vec_equal,
    dot,
    longest_axis,
)
from .mat import Mat


comptime AABB = AxisAlignedBoundingBox[DType.float32]
comptime Quat = Quaternion[DType.float32]
comptime Affine3f32 = Affine3[DType.float32]
comptime Vec2f32 = Vec2[DType.float32]
comptime Vec3f32 = Vec3[DType.float32]

comptime Mat22 = Mat[_, 2, 2]
comptime Mat33 = Mat[_, 3, 3]
comptime Mat44 = Mat[_, 4, 4]
comptime Mat33f32 = Mat[DType.float32, 3, 3]
