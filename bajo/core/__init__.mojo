from .aabb import AxisAlignedBoundingBox
from .quat import Quaternion
from .transform import Affine3
from .vec import (
    Vec2,
    Vec3,
    normalize,
    length,
    vmin,
    vmax,
    cross,
    assert_vec_equal,
    dot,
    longest_axis,
)


comptime AABB = AxisAlignedBoundingBox[DType.float32]
comptime Quat = Quaternion[DType.float32]
comptime Affine3f32 = Affine3[DType.float32]
comptime Vec2f32 = Vec2[DType.float32]
comptime Vec3f32 = Vec3[DType.float32]
