from std.math import sqrt

from bajo.core.quat import Quaternion
from bajo.core.vec import Vec, Vec3, dot, cross, lerp as vlerp
from bajo.core.mat import Mat, Mat33


comptime transformf32 = Transform[DType.float32]
comptime transformf64 = Transform[DType.float64]

comptime spatial_vectorf32 = SpatialVector[DType.float32]
comptime spatial_vectorf64 = SpatialVector[DType.float64]


comptime SpatialVector[dtype: DType] = Vec[dtype, 6]
"""A 6D spatial vector (twist or wrench)."""


comptime SpatialMatrix[dtype: DType] = Mat[dtype, 6, 6]


@always_inline
def w_vec[dtype: DType](a: SpatialVector[dtype]) -> Vec[dtype, 3]:
    """Returns the angular (top) portion of the spatial vector."""
    return a.xyz()


@always_inline
def v_vec[dtype: DType](a: SpatialVector[dtype]) -> Vec[dtype, 3]:
    """Returns the linear (bottom) portion of the spatial vector."""
    return Vec[dtype, 3](a[3], a[4], a[5])


def spatial_dot[
    dtype: DType
](a: SpatialVector[dtype], b: SpatialVector[dtype]) -> Scalar[dtype]:
    return dot(a, b)


def spatial_cross[
    dtype: DType
](a: SpatialVector[dtype], b: SpatialVector[dtype]) -> SpatialVector[dtype]:
    w = cross(w_vec(a), w_vec(b))
    v = cross(v_vec(a), w_vec(b)) + cross(w_vec(a), v_vec(b))
    return SpatialVector[dtype](
        v=w, w=v
    )  # Using the Vec6(v3, v3) constructor you defined


def spatial_cross_dual[
    dtype: DType
](a: SpatialVector[dtype], b: SpatialVector[dtype]) -> SpatialVector[dtype]:
    w = cross(w_vec(a), w_vec(b)) + cross(v_vec(a), v_vec(b))
    v = cross(w_vec(a), v_vec(b))
    return SpatialVector[dtype](v=w, w=v)


def spatial_top[dtype: DType](a: SpatialVector[dtype]) -> Vec[dtype, 3]:
    return w_vec(a)


def spatial_bottom[dtype: DType](a: SpatialVector[dtype]) -> Vec[dtype, 3]:
    return v_vec(a)


@fieldwise_init
struct Transform[dtype: DType where dtype.is_floating_point()](
    Copyable, Equatable, Writable
):
    comptime P = Vec3[Self.dtype]
    comptime Q = Quaternion[Self.dtype]

    var p: Vec3[Self.dtype]
    var q: Quaternion[Self.dtype]

    def __init__(out self):
        self = Self.identity()

    @staticmethod
    def identity() -> Self:
        return Self(Self.P(0), Self.Q.identity())

    def __neg__(self) -> Self:
        return Self(-self.p, -self.q)

    def get_translation(ref self) -> ref[self.p] Self.P:
        return self.p

    def get_rotation(ref self) -> ref[self.q] Self.Q:
        return self.q

    def __mul__(self, rhs: Self) -> Self:
        return Self(self.q.rotate(rhs.p) + self.p, self.q * rhs.q)

    def inverse(self) -> Self:
        q_inv = self.q.inverse()
        return Self(-(q_inv.rotate(self.p)), q_inv)

    def transform_vector(self, x: Self.P) -> Self.P:
        """Applies only rotation to a vector."""
        return self.q.rotate(x)

    def transform_point(self, x: Self.P) -> Self.P:
        """Applies full rigid transformation to a point."""
        return self.p + self.q.rotate(x)


def spatial_adjoint[
    dtype: DType
](R: Mat33[dtype], S: Mat33[dtype]) -> SpatialMatrix[dtype]:
    """
    Builds a 6x6 spatial adjoint matrix:
    [ R  0 ]
    [ S  R ].
    """
    out = SpatialMatrix[dtype](uninitialized=True)

    comptime for i in range(3):
        comptime for j in range(3):
            out[i][j] = R[i][j]  # 11
            out[i][j + 3] = 0  # 12
            out[i + 3][j] = S[i][j]  # 21
            out[i + 3][j + 3] = R[i][j]  # 22
    return out^


def row_index(stride: Int, i: Int, j: Int) -> Int:
    return i * stride + j


# def spatial_jacobian[
#     dtype: DType
# ](
#     S: UnsafePointer[SpatialVector[dtype], ImmutAnyOrigin],
#     joint_parents: UnsafePointer[Int, ImmutAnyOrigin],
#     joint_qd_start: UnsafePointer[Int, ImmutAnyOrigin],
#     joint_start: Int, # offset of the first joint for the articulation
#     joint_count: Int,
#     J_start: Int,
#     J: UnsafePointer[Scalar[dtype]],
# ):
#     """builds spatial Jacobian J which is an (joint_count*6)x(dof_count) matrix"""

#     articulation_dof_start = joint_qd_start[joint_start]
#     articulation_dof_end = joint_qd_start[joint_start + joint_count]
#     articulation_dof_count = articulation_dof_end - articulation_dof_start

#     for i in range(joint_count):
#         row_start = i * 6
#         j = joint_start + i

#         while j != -1:
#             joint_dof_start = joint_qd_start[j]
#             joint_dof_end = joint_qd_start[j + 1]
#             joint_dof_count = joint_dof_end - joint_dof_start

#             joint_dof_start = joint_qd_start[j]
#             joint_dof_count = joint_qd_start[j + 1] - joint_dof_start

#             # fill out each row of the Jacobian walking up the tree
#             for dof in range(joint_dof_count):
#                 col = (joint_dof_start - articulation_dof_start) + dof

#                 # Offset pointers logic
#                 s_val = S[articulation_dof_start + col]
#                 w = v_vec(S[col]
#                 J[row_index(articulation_dof_count, row_start + 0, col)] = S[col][0]
#                 J[row_index(articulation_dof_count, row_start + 1, col)] = S[col][1]
#                 J[row_index(articulation_dof_count, row_start + 2, col)] = S[col][2]
#                 J[row_index(articulation_dof_count, row_start + 3, col)] = S[col][3]
#                 J[row_index(articulation_dof_count, row_start + 4, col)] = S[col][4]
#                 J[row_index(articulation_dof_count, row_start + 5, col)] = S[col][5]

#             j = joint_parents[j]


# def spatial_mass[
#     dtype: DType
# ](
#     I_s: UnsafePointer[SpatialMatrix[dtype]],
#     joint_start: Int,
#     joint_count: Int,
#     M_start: Int,
#     M: UnsafePointer[Scalar[dtype]],
# ):
#     stride = joint_count * 6
#     for l in range(joint_count):
#         ref mat = I_s[joint_start + l]
#         for i in range(6):
#             for j in range(6):
#                 M[M_start + (l * 6 + i) * stride + (l * 6 + j)] = mat[i, j]


def main():
    print("core.spatial")
