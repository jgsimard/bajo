from .aabb import AxisAlignedBoundingBox
from .quat import Quaternion
from .transform import Affine3
from .vec import (
    GeoKind,
    Vec2,
    Vec3,
    Point3,
    Normal3,
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
from .frame import Frame
from .ray import Ray


comptime AABB[frame: Frame = Frame.WORLD] = AxisAlignedBoundingBox[
    DType.float32, frame
]
comptime Quat = Quaternion[DType.float32, _]
comptime Affine3f32[From: Frame, To: Frame] = Affine3[DType.float32, From, To]
comptime Vec2f32 = Vec2[DType.float32, _]
comptime Vec3f32[frame: Frame] = Vec3[DType.float32, frame]
comptime Point3f32[frame: Frame] = Point3[DType.float32, frame]
comptime Normal3f32[frame: Frame] = Normal3[DType.float32, frame]
comptime Rayf32[frame: Frame] = Ray[DType.float32, frame]

comptime Vec3W = Vec3[DType.float32, Frame.WORLD]
comptime Point3W = Point3[DType.float32, Frame.WORLD]


comptime Mat22 = Mat[_, 2, 2, _]
comptime Mat33 = Mat[_, 3, 3, _]
comptime Mat44 = Mat[_, 4, 4, _]
comptime Mat33f32 = Mat[DType.float32, 3, 3, _]
