from math import sqrt
from bajo.core.quat import Quaternion
from bajo.core.vec import Vec, Vec3, dot, cross, lerp as vlerp
from bajo.core.mat2 import Mat


comptime transformf32 = Transform[DType.float32]
comptime transformf64 = Transform[DType.float64]

comptime spatial_vectorf32 = SpatialVector[DType.float32]
comptime spatial_vectorf64 = SpatialVector[DType.float64]


comptime SpatialVector[dtype: DType] = Vec[dtype, 6]
"""A 6D spatial vector (twist or wrench)."""


comptime SpatialMatrix[dtype: DType] = Mat[dtype, 6, 6]


@always_inline
fn w_vec[dtype: DType](a: SpatialVector[dtype]) -> Vec[dtype, 3]:
    """Returns the angular (top) portion of the spatial vector."""
    return a.xyz()


@always_inline
fn v_vec[dtype: DType](a: SpatialVector[dtype]) -> Vec[dtype, 3]:
    """Returns the linear (bottom) portion of the spatial vector."""
    return Vec[dtype, 3](a[3], a[4], a[5])


fn spatial_dot[
    dtype: DType
](a: SpatialVector[dtype], b: SpatialVector[dtype]) -> Scalar[dtype]:
    return dot(a, b)


fn spatial_cross[
    dtype: DType
](a: SpatialVector[dtype], b: SpatialVector[dtype]) -> SpatialVector[dtype]:
    w = cross(w_vec(a), w_vec(b))
    v = cross(v_vec(a), w_vec(b)) + cross(w_vec(a), v_vec(b))
    return SpatialVector[dtype](
        v=w, w=v
    )  # Using the Vec6(v3, v3) constructor you defined


fn spatial_cross_dual[
    dtype: DType
](a: SpatialVector[dtype], b: SpatialVector[dtype]) -> SpatialVector[dtype]:
    w = cross(w_vec(a), w_vec(b)) + cross(v_vec(a), v_vec(b))
    v = cross(w_vec(a), v_vec(b))
    return SpatialVector[dtype](v=w, w=v)


fn spatial_top[dtype: DType](a: SpatialVector[dtype]) -> Vec[dtype, 3]:
    return w_vec(a)


fn spatial_bottom[dtype: DType](a: SpatialVector[dtype]) -> Vec[dtype, 3]:
    return v_vec(a)


# --- Transform Struct ---


@fieldwise_init
struct Transform[dtype: DType where dtype.is_floating_point()](
    Copyable, Equatable
):
    var p: Vec3[Self.dtype]
    var q: Quaternion[Self.dtype]

    fn __init__(out self):
        self = Self.identity()

    @staticmethod
    fn identity() -> Self:
        return Self(Vec[dtype, 3](0), Quaternion[dtype].identity())

    fn __neg__(self) -> Self:
        return Self(-self.p, -self.q)

    fn __getitem__(self, idx: Int) -> Scalar[dtype]:
        if idx < 3:
            return self.p[idx]
        return self.q[idx - 3]

    fn __setitem__(mut self, idx: Int, val: Scalar[dtype]):
        if idx < 3:
            self.p[idx] = val
        else:
            self.q[idx - 3] = val

    fn get_translation(ref self) -> ref[self.p] Vec[dtype, 3]:
        return t.p

    fn get_rotation(ref self) -> ref[self.q] Quaternion[dtype]:
        return t.q


fn transform_multiply[
    dtype: DType where dtype.is_floating_point()
](a: Transform[dtype], b: Transform[dtype]) -> Transform[dtype]:
    # a.q * b.p rotates the second translation into the first's frame, then adds the offset
    return Transform[dtype](a.q.rotate(b.p) + a.p, a.q * b.q)


fn transform_inverse[
    dtype: DType where dtype.is_floating_point()
](t: Transform[dtype]) -> Transform[dtype]:
    q_inv = t.q.inverse()
    return Transform[dtype](-(q_inv.rotate(t.p)), q_inv)


fn transform_vector[
    dtype: DType where dtype.is_floating_point()
](t: Transform[dtype], x: Vec[dtype, 3]) -> Vec[dtype, 3]:
    """Applies only rotation to a vector."""
    return t.q.rotate(x)


fn transform_point[
    dtype: DType where dtype.is_floating_point()
](t: Transform[dtype], x: Vec[dtype, 3]) -> Vec[dtype, 3]:
    """Applies full rigid transformation to a point."""
    return t.p + t.q.rotate(x)


fn lerp[
    dtype: DType where dtype.is_floating_point()
](a: Transform[dtype], b: Transform[dtype], t: Scalar[dtype]) -> Transform[
    dtype
]:
    return Transform[dtype](
        vlerp(a.p, b.p, t),
        # Note: Strictly speaking, Warp uses linear addition for quats in lerp.
        # For better results, one might use Slerp, but following Warp's logic:
        a.q * (1.0 - t) + b.q * t,
    )


# --- Spatial Dynamics ---


fn spatial_adjoint[
    dtype: DType
](R: Mat[dtype, 3, 3], S: Mat[dtype, 3, 3]) -> SpatialMatrix[dtype]:
    """
    Builds a 6x6 spatial adjoint matrix:
    [ R  0 ]
    [ S  R ].
    """
    out = SpatialMatrix[dtype](uninitialized=True)

    comptime for i in range(3):
        comptime for j in range(3):
            # Diagonal blocks
            out[i, j] = R[i, j]
            out[i + 3, j + 3] = R[i, j]
            # Lower off-diagonal
            out[i + 3, j] = S[i, j]
            # Upper off-diagonal
            out[i, j + 3] = 0

    return out


fn spatial_jacobian[
    dtype: DType
](
    S: UnsafePointer[SpatialVector[dtype]],
    joint_parents: UnsafePointer[Int],
    joint_qd_start: UnsafePointer[Int],
    joint_start: Int,
    joint_count: Int,
    J_start: Int,
    J: UnsafePointer[Scalar[dtype]],
):
    articulation_dof_start = joint_qd_start[joint_start]
    articulation_dof_end = joint_qd_start[joint_start + joint_count]
    articulation_dof_count = articulation_dof_end - articulation_dof_start

    for i in range(joint_count):
        row_start = i * 6
        var j = joint_start + i

        while j != -1:
            joint_dof_start = joint_qd_start[j]
            joint_dof_count = joint_qd_start[j + 1] - joint_dof_start

            for dof in range(joint_dof_count):
                col = (joint_dof_start - articulation_dof_start) + dof

                # Offset pointers logic
                s_val = S[articulation_dof_start + col]

                J[
                    J_start + (row_start + 0) * articulation_dof_count + col
                ] = s_val[0]
                J[
                    J_start + (row_start + 1) * articulation_dof_count + col
                ] = s_val[1]
                J[
                    J_start + (row_start + 2) * articulation_dof_count + col
                ] = s_val[2]
                J[
                    J_start + (row_start + 3) * articulation_dof_count + col
                ] = s_val[3]
                J[
                    J_start + (row_start + 4) * articulation_dof_count + col
                ] = s_val[4]
                J[
                    J_start + (row_start + 5) * articulation_dof_count + col
                ] = s_val[5]

            j = joint_parents[j]


fn spatial_mass[
    dtype: DType
](
    I_s: UnsafePointer[SpatialMatrix[dtype]],
    joint_start: Int,
    joint_count: Int,
    M_start: Int,
    M: UnsafePointer[Scalar[dtype]],
):
    stride = joint_count * 6
    for l in range(joint_count):
        mat = I_s[joint_start + l]
        for i in range(6):
            for j in range(6):
                M[M_start + (l * 6 + i) * stride + (l * 6 + j)] = mat[i, j]


fn main():
    print("core.spatial")
