from std.gpu import DeviceContext
from std.testing import assert_true

from bajo.core.vec_simd import Vec3f32
from bajo.core.bvh.gpu.lbvh import GpuLBVH


def _append_tri(mut verts: List[Vec3f32], cx: Float32, z: Float32):
    verts.append(Vec3f32(cx - 1.0, -1.0, z))
    verts.append(Vec3f32(cx + 1.0, -1.0, z))
    verts.append(Vec3f32(cx, 1.0, z))


def _append_tri(
    mut verts: List[Vec3f32],
    cx: Float32,
    cy: Float32,
    z: Float32,
):
    verts.append(Vec3f32(cx - 1.0, cy - 1.0, z))
    verts.append(Vec3f32(cx + 1.0, cy - 1.0, z))
    verts.append(Vec3f32(cx, cy + 1.0, z))


def _make_strip(count: Int, z: Float32 = 2.0) -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=count * 3)

    for i in range(count):
        var cx = Float32(i * 4 - count * 2)
        _append_tri(verts, cx, z)

    return verts^


def _make_two_depth_triangles() -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=6)
    _append_tri(verts, 0.0, 0.0, 2.0)
    _append_tri(verts, 0.0, 0.0, 4.0)
    return verts^


def _build_gpu_blas(
    ctx: DeviceContext,
    mut gpu_blas: GpuLBVH,
    verts: List[Vec3f32],
) raises:
    var result = gpu_blas.build_from_triangles(ctx, verts)
    var validation = result[1]

    assert_true(validation.sorted_ok, "GPU BLAS keys sorted")
    assert_true(validation.values_ok, "GPU BLAS values valid")
    assert_true(validation.topology_ok, "GPU BLAS topology valid")
    assert_true(validation.bounds_ok, "GPU BLAS bounds valid")
