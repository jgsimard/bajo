"""Pack all mesh result fields for cross-backend verification."""
from bajo.core import Point3f32, Vec3f32, Rayf32
from bajo.bvh.spatial.mesh_query import MeshQueryView

comptime FIELD_COUNT = 49


def mesh_result_fields[
    origin: ImmOrigin
](
    view: MeshQueryView[origin, .WORLD],
    lane: Int,
    root: UInt32,
    output: MutSpan[Float64, _],
):
    var p = Point3f32[.WORLD](
        Float32(lane % 7) * 0.19 - 0.1,
        Float32(lane % 5) * 0.23 + 0.017,
        Float32(lane % 3) * 0.37 + 0.043,
    )
    var closest = view.query_point_no_sign(p, root=root)
    var furthest = view.query_furthest_point_no_sign(p, root=root)
    var hit = view.query_ray(
        Rayf32[.WORLD](p, Vec3f32[.WORLD](0.13, -0.27, 1.0)), root
    )
    output[0] = Float64(closest.result)
    output[1] = Float64(closest.face)
    output[2] = Float64(closest.u)
    output[3] = Float64(closest.v)
    output[4] = closest.distance
    output[5] = Float64(closest.point.x)
    output[6] = Float64(closest.point.y)
    output[7] = Float64(closest.point.z)
    output[8] = Float64(closest.sign)
    output[9] = Float64(furthest.result)
    output[10] = Float64(furthest.face)
    output[11] = Float64(furthest.u)
    output[12] = Float64(furthest.v)
    output[13] = furthest.distance
    output[14] = Float64(furthest.point.x)
    output[15] = Float64(furthest.point.y)
    output[16] = Float64(furthest.point.z)
    output[17] = Float64(furthest.sign)
    output[18] = Float64(hit.result)
    output[19] = Float64(hit.face)
    output[20] = Float64(hit.u)
    output[21] = Float64(hit.v)
    output[22] = hit.t
    output[23] = Float64(hit.normal.x)
    output[24] = Float64(hit.normal.y)
    output[25] = Float64(hit.normal.z)
    output[26] = Float64(hit.front_face)
    var position = view.eval_position(0, 0.5, 0.25)
    var velocity = view.eval_velocity(0, 0.5, 0.25)
    var normal = view.eval_normal(0)
    output[27] = Float64(position.x)
    output[28] = Float64(position.y)
    output[29] = Float64(position.z)
    output[30] = Float64(velocity.x)
    output[31] = Float64(velocity.y)
    output[32] = Float64(velocity.z)
    output[33] = Float64(normal.x)
    output[34] = Float64(normal.y)
    output[35] = Float64(normal.z)
    output[36] = view.winding_number(p, root)
    output[37] = Float64(view.query_point_sign_normal(p, root=root).sign)
    output[38] = Float64(view.query_point_sign_parity(p, root=root).sign)
    output[39] = Float64(
        view.query_point_sign_winding_number(p, root=root).sign
    )
    var signed = view.query_point_sign_normal(p, root=root)
    output[40] = Float64(signed.u)
    output[41] = Float64(signed.v)
    output[42] = Float64(signed.face)
    var signed_position = Point3f32[.WORLD](0.0)
    var signed_velocity = Vec3f32[.WORLD](0.0)
    if signed.result:
        signed_position = view.eval_position(signed.face, signed.u, signed.v)
        signed_velocity = view.eval_velocity(signed.face, signed.u, signed.v)
    output[43] = Float64(signed_position.x)
    output[44] = Float64(signed_position.y)
    output[45] = Float64(signed_position.z)
    output[46] = Float64(signed_velocity.x)
    output[47] = Float64(signed_velocity.y)
    output[48] = Float64(signed_velocity.z)
