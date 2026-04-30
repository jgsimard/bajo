from std.testing import TestSuite, assert_true
from std.sys import has_accelerator
from std.benchmark import keep

from bajo.core.vec import Vec3f32
from bajo.core.bvh.tinybvh import BVH
from bajo.core.bvh.gpu_lbvh import (
    compute_bounds,
    compute_centroid_bounds,
    generate_camera_params,
    generate_primary_rays,
    trace_bvh_primary,
    trace_bvh_shadow,
)


def _append_tri(mut verts: List[Vec3f32], cx: Float32, cy: Float32, z: Float32):
    verts.append(Vec3f32(cx - 0.5, cy - 0.5, z))
    verts.append(Vec3f32(cx + 0.5, cy - 0.5, z))
    verts.append(Vec3f32(cx, cy + 0.5, z))


def _make_small_scene() -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=8 * 3)
    _append_tri(verts, -1.0, -1.0, 2.0)
    _append_tri(verts, 1.0, -1.0, 2.0)
    _append_tri(verts, -1.0, 1.0, 2.0)
    _append_tri(verts, 1.0, 1.0, 2.0)
    _append_tri(verts, -1.0, -1.0, 4.0)
    _append_tri(verts, 1.0, -1.0, 4.0)
    _append_tri(verts, -1.0, 1.0, 4.0)
    _append_tri(verts, 1.0, 1.0, 4.0)
    return verts^


def test_gpu_lbvh_bounds_helpers() raises:
    var verts = _make_small_scene()
    var bounds = compute_bounds(verts)
    var bmin = bounds[0].copy()
    var bmax = bounds[1].copy()

    assert_true(bmin.x() < -1.4)
    assert_true(bmin.y() < -1.4)
    assert_true(bmin.z() == 2.0)
    assert_true(bmax.x() > 1.4)
    assert_true(bmax.y() > 1.4)
    assert_true(bmax.z() == 4.0)

    var cbounds = compute_centroid_bounds(verts)
    var cmin = cbounds[0].copy()
    var cmax = cbounds[1].copy()
    assert_true(cmin.z() == 2.0)
    assert_true(cmax.z() == 4.0)

    keep(len(verts))


# def test_gpu_lbvh_generated_primary_and_shadow_match_cpu() raises:
#     comptime if has_accelerator():
#         var verts = _make_small_scene()
#         var tri_count = UInt32(len(verts) // 3)

#         var bounds = compute_bounds(verts)
#         var bmin = bounds[0].copy()
#         var bmax = bounds[1].copy()
#         var cbounds = compute_centroid_bounds(verts)
#         var cmin = cbounds[0].copy()
#         var cmax = cbounds[1].copy()

#         var rays = generate_primary_rays(bmin, bmax, 640, 360, 3)
#         var camera_params = generate_camera_params(bmin, bmax, 3)

#         var ref_bvh = BVH(verts.unsafe_ptr(), tri_count)
#         ref_bvh.build["sah", False]()
#         var ref_checksum = trace_bvh_primary(ref_bvh, rays)
#         var ref_occluded = trace_bvh_shadow(ref_bvh, rays)

#         var gpu_result = run_gpu_lbvh_camera_reduce_and_shadow_benchmark(
#             verts,
#             cmin,
#             cmax,
#             bmin,
#             bmax,
#             camera_params,
#             len(rays),
#             ref_checksum,
#             ref_occluded,
#             1,
#         )

#         var primary_diff = gpu_result[10]
#         var shadow_diff = gpu_result[15]
#         var sorted_ok = gpu_result[16]
#         var topology_ok = gpu_result[17]
#         var bounds_ok = gpu_result[18]

#         assert_true(sorted_ok)
#         assert_true(topology_ok)
#         assert_true(bounds_ok)
#         assert_true(primary_diff <= 0.01)
#         assert_true(shadow_diff == 0)

#         keep(len(verts))
#         keep(len(rays))
#     else:
#         # Keep the test suite green on machines without an accelerator.
#         assert_true(False, "No GPU detected")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
