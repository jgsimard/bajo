from std.gpu import DeviceContext
from std.testing import assert_true

from bajo.core.vec import Vec3f32
from bajo.core.bvh.gpu.lbvh import GpuLBVH
from bajo.core.bvh.gpu.kernels import compute_centroid_bounds
from bajo.core.bvh import compute_bounds


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


@always_inline
def _safe_inv_extent_axis(a: Float32, b: Float32) -> Float32:
    var e = b - a
    if abs(e) <= 1.0e-20:
        return 0.0
    return Float32(1.0) / e


def _inv_extent(cmin: Vec3f32, cmax: Vec3f32) -> Vec3f32:
    return Vec3f32(
        _safe_inv_extent_axis(cmin.x(), cmax.x()),
        _safe_inv_extent_axis(cmin.y(), cmax.y()),
        _safe_inv_extent_axis(cmin.z(), cmax.z()),
    )


def _build_gpu_blas(
    ctx: DeviceContext,
    mut gpu_blas: GpuLBVH,
    verts: List[Vec3f32],
) raises:
    var bounds = compute_bounds(verts)
    var bmin = bounds[0].copy()
    var bmax = bounds[1].copy()
    var cbounds = compute_centroid_bounds(verts)
    var cmin = cbounds[0].copy()
    var cmax = cbounds[1].copy()
    var inv = _inv_extent(cmin, cmax)

    _ = gpu_blas.build(ctx, cmin, inv)
    var validation = gpu_blas.validate(bmin, bmax)
    assert_true(validation.sorted_ok, "GPU BLAS keys sorted")
    assert_true(validation.values_ok, "GPU BLAS values valid")
    assert_true(validation.topology_ok, "GPU BLAS topology valid")
    assert_true(validation.bounds_ok, "GPU BLAS bounds valid")
