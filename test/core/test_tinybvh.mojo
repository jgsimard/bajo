from std.testing import (
    TestSuite,
    assert_true,
)
from std.math import abs

from bajo.core.vec import Vec3f32
from bajo.core.random import Rng
from bajo.core.intersect import intersect_ray_aabb
from bajo.core.bvh.tinybvh import (
    BVH,
    BVHGPU,
    BVHNode,
    Fragment,
    Ray,
    WideBVH,
    _partition_fragments,
    _partition_fragments_by_bin,
    _sah,
)


def _make_fragments(verts: List[Vec3f32]) -> List[Fragment]:
    var frags = List[Fragment](capacity=len(verts) // 3)

    for i in range(len(verts) // 3):
        frags.append(
            Fragment(
                UInt32(i),
                verts[i * 3 + 0],
                verts[i * 3 + 1],
                verts[i * 3 + 2],
            )
        )

    return frags^


def _append_tri(mut verts: List[Vec3f32], cx: Float32, z: Float32):
    verts.append(Vec3f32(cx - 1.0, -1.0, z))
    verts.append(Vec3f32(cx + 1.0, -1.0, z))
    verts.append(Vec3f32(cx, 1.0, z))


def _make_strip(count: Int, z: Float32 = 2.0) -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=count * 3)

    for i in range(count):
        var cx = Float32(i * 4 - count * 2)
        _append_tri(verts, cx, z)

    return verts^


def _assert_close(a: Float32, b: Float32, eps: Float32) raises:
    assert_true(abs(a - b) <= eps)


def test_bvh_ray_query_inside_outside() raises:
    # Regression for rays starting inside an AABB.
    # https://github.com/NVIDIA/warp/issues/288
    var lower = Vec3f32(0.5, -1.0, -1.0)
    var upper = Vec3f32(1.0, 1.0, 1.0)

    var query_dir = Vec3f32(1.0, 0.0, 0.0)
    var rcp_dir = 1.0 / query_dir
    var t = Float32(0.0)

    var query_start_outside = Vec3f32(0.0, 0.0, 0.0)
    var hit_outside = intersect_ray_aabb(
        query_start_outside, rcp_dir, lower, upper, t
    )
    assert_true(hit_outside, "Ray starting outside failed to hit")

    var query_start_inside = Vec3f32(0.75, 0.0, 0.0)
    var hit_inside = intersect_ray_aabb(
        query_start_inside, rcp_dir, lower, upper, t
    )
    assert_true(hit_inside, "Ray starting inside failed to hit")


def test_fragment_bounds_and_mapping() raises:
    var frag = Fragment(
        UInt32(42),
        Vec3f32(-1.0, 2.0, 3.0),
        Vec3f32(2.0, -4.0, 5.0),
        Vec3f32(0.0, 1.0, -6.0),
    )

    assert_true(frag.prim_idx == 42)
    assert_true(frag.bmin.x() == -1.0)
    assert_true(frag.bmin.y() == -4.0)
    assert_true(frag.bmin.z() == -6.0)
    assert_true(frag.bmax.x() == 2.0)
    assert_true(frag.bmax.y() == 2.0)
    assert_true(frag.bmax.z() == 5.0)
    _assert_close(frag.center_axis(0), 0.5, 1e-5)


def test_sah_clear_separation() raises:
    var verts: List[Vec3f32] = [
        Vec3f32(-11.0, -1.0, 0.0),
        Vec3f32(-9.0, -1.0, 0.0),
        Vec3f32(-10.0, 1.0, 0.0),  # Tri 0, centered near x=-10
        Vec3f32(9.0, -1.0, 0.0),
        Vec3f32(11.0, -1.0, 0.0),
        Vec3f32(10.0, 1.0, 0.0),  # Tri 1, centered near x=10
    ]
    var frags = _make_fragments(verts)
    var prims: List[UInt32] = [0, 1]

    var node = BVHNode()
    node.leftFirst = 0
    node.triCount = 2

    var split = _sah(
        node,
        prims.unsafe_ptr(),
        frags.unsafe_ptr(),
    )

    assert_true(split.axis == 0)
    assert_true(split.pos > -10.0 and split.pos < 10.0)
    assert_true(split.cost < 20.0)
    assert_true(split.bin >= 0)


def test_sah_degenerate() raises:
    var verts: List[Vec3f32] = [
        Vec3f32(0.0, 0.0, 0.0),
        Vec3f32(1.0, 0.0, 0.0),
        Vec3f32(0.0, 1.0, 0.0),
        Vec3f32(0.0, 0.0, 0.0),
        Vec3f32(1.0, 0.0, 0.0),
        Vec3f32(0.0, 1.0, 0.0),
    ]
    var frags = _make_fragments(verts)
    var prims: List[UInt32] = [0, 1]

    var node = BVHNode()
    node.leftFirst = 0
    node.triCount = 2

    var split = _sah(
        node,
        prims.unsafe_ptr(),
        frags.unsafe_ptr(),
    )

    assert_true(split.axis == -1)
    assert_true(not split.valid())


def test_sah_axis_preference() raises:
    var verts = List[Vec3f32]()

    for i in range(10):
        var z = Float32(i * 10 - 50)
        verts.append(Vec3f32(-1.0, -1.0, z))
        verts.append(Vec3f32(1.0, -1.0, z))
        verts.append(Vec3f32(0.0, 1.0, z))

    var frags = _make_fragments(verts)
    var prims = List[UInt32]()
    for i in range(10):
        prims.append(UInt32(i))

    var node = BVHNode()
    node.leftFirst = 0
    node.triCount = 10

    var split = _sah(
        node,
        prims.unsafe_ptr(),
        frags.unsafe_ptr(),
    )

    assert_true(split.axis == 2)


def test_partition_fragments_non_empty() raises:
    var verts: List[Vec3f32] = [
        Vec3f32(-11.0, -1.0, 0.0),
        Vec3f32(-9.0, -1.0, 0.0),
        Vec3f32(-10.0, 1.0, 0.0),
        Vec3f32(9.0, -1.0, 0.0),
        Vec3f32(11.0, -1.0, 0.0),
        Vec3f32(10.0, 1.0, 0.0),
    ]
    var frags = _make_fragments(verts)
    var prims: List[UInt32] = [0, 1]

    var split_idx = _partition_fragments(
        prims.unsafe_ptr(),
        frags.unsafe_ptr(),
        0,
        2,
        0,
        0.0,
    )

    assert_true(split_idx == 1)
    assert_true(prims[0] != prims[1])


def test_sah_partition_by_bin_non_empty() raises:
    var verts = _make_strip(10)
    var frags = _make_fragments(verts)

    var prims = List[UInt32]()
    for i in range(10):
        prims.append(UInt32(i))

    var node = BVHNode()
    node.leftFirst = 0
    node.triCount = 10

    var split = _sah(
        node,
        prims.unsafe_ptr(),
        frags.unsafe_ptr(),
    )
    assert_true(split.valid())

    var split_idx = _partition_fragments_by_bin(
        prims.unsafe_ptr(),
        frags.unsafe_ptr(),
        0,
        10,
        split.axis,
        split.bin,
        split.bin_min,
        split.bin_scale,
    )

    assert_true(split_idx > 0)
    assert_true(split_idx < 10)
    assert_true(split.left_bounds.surface_area() > 0.0)
    assert_true(split.right_bounds.surface_area() > 0.0)


def test_bvh_build_invariants_median() raises:
    var verts = _make_strip(12)
    var bvh = BVH(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    bvh.build["median", False]()

    assert_true(bvh.nodes_used > 1)
    assert_true(Int(bvh.nodes_used) <= len(bvh.bvh_nodes))

    var leaf_prim_total = UInt32(0)
    for i in range(Int(bvh.nodes_used)):
        ref node = bvh.bvh_nodes[i]

        if node.is_leaf():
            assert_true(node.triCount > 0)
            assert_true(
                Int(node.leftFirst) + Int(node.triCount)
                <= len(bvh.prim_indices)
            )
            leaf_prim_total += node.triCount
        else:
            assert_true(node.triCount == 0)
            assert_true(node.leftFirst + 1 < bvh.nodes_used)

    assert_true(leaf_prim_total == bvh.tri_count)


def test_bvh_traverse_returns_nearest_triangle() raises:
    var verts: List[Vec3f32] = [
        # Tri 0 at z=2.
        Vec3f32(-1.0, -1.0, 2.0),
        Vec3f32(1.0, -1.0, 2.0),
        Vec3f32(0.0, 1.0, 2.0),
        # Tri 1 at z=4, behind tri 0.
        Vec3f32(-1.0, -1.0, 4.0),
        Vec3f32(1.0, -1.0, 4.0),
        Vec3f32(0.0, 1.0, 4.0),
    ]

    var bvh = BVH(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    bvh.build["median", False]()

    var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    bvh.traverse(ray)

    assert_true(ray.hit.prim == 0)
    _assert_close(ray.hit.t, 2.0, 1e-4)


def test_bvh_traverse_reports_original_primitive_after_reorder() raises:
    var verts = _make_strip(8)
    var bvh = BVH(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    bvh.build["sah", False]()

    # Aim at the triangle with original primitive id 6.
    # _make_strip centers primitive i at x = i * 4 - count * 2.
    var ray = Ray(Vec3f32(8.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    bvh.traverse(ray)

    assert_true(ray.hit.prim == 6)
    _assert_close(ray.hit.t, 2.0, 1e-4)


def test_bvh_shadow_ray() raises:
    var verts = _make_strip(4)
    var bvh = BVH(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    bvh.build["median", False]()

    # Primitive 2 is centered at x=0.
    var ray_hit = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(bvh.is_occluded(ray_hit))

    var ray_miss = Ray(Vec3f32(100.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(not bvh.is_occluded(ray_miss))


def test_wide_bvh4_matches_binary_for_basic_hit() raises:
    var verts = _make_strip(8)

    var bvh = BVH(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    bvh.build["sah", False]()

    var wide = WideBVH[4](bvh)

    var ray_binary = Ray(Vec3f32(8.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var ray_wide = Ray(Vec3f32(8.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))

    bvh.traverse(ray_binary)
    wide.traverse(ray_wide)

    assert_true(ray_binary.hit.prim == ray_wide.hit.prim)
    _assert_close(ray_binary.hit.t, ray_wide.hit.t, 1e-4)


# -----------------------------------------------------------------------------
# Randomized / brute-force regression tests
# -----------------------------------------------------------------------------


@always_inline
def _rng_f32(mut rng: Rng, lo: Float32, hi: Float32) -> Float32:
    return lo + (hi - lo) * rng.f32()


@always_inline
def _brute_intersect_tri(
    verts: List[Vec3f32],
    O: Vec3f32,
    D: Vec3f32,
    prim_idx: Int,
) -> Tuple[Bool, Float32]:
    ref v0 = verts[prim_idx * 3 + 0]
    ref v1 = verts[prim_idx * 3 + 1]
    ref v2 = verts[prim_idx * 3 + 2]

    var e1x = v1.x() - v0.x()
    var e1y = v1.y() - v0.y()
    var e1z = v1.z() - v0.z()
    var e2x = v2.x() - v0.x()
    var e2y = v2.y() - v0.y()
    var e2z = v2.z() - v0.z()

    var px = D.y() * e2z - D.z() * e2y
    var py = D.z() * e2x - D.x() * e2z
    var pz = D.x() * e2y - D.y() * e2x
    var det = e1x * px + e1y * py + e1z * pz

    if det > -1e-12 and det < 1e-12:
        return (False, Float32(1e30))

    var inv_det = 1.0 / det
    var tx = O.x() - v0.x()
    var ty = O.y() - v0.y()
    var tz = O.z() - v0.z()

    var u = (tx * px + ty * py + tz * pz) * inv_det
    if u < 0.0 or u > 1.0:
        return (False, Float32(1e30))

    var qx = ty * e1z - tz * e1y
    var qy = tz * e1x - tx * e1z
    var qz = tx * e1y - ty * e1x

    var v = (D.x() * qx + D.y() * qy + D.z() * qz) * inv_det
    if v < 0.0 or u + v > 1.0:
        return (False, Float32(1e30))

    var t = (e2x * qx + e2y * qy + e2z * qz) * inv_det
    if t > 1e-4:
        return (True, t)

    return (False, Float32(1e30))


def _brute_trace(
    verts: List[Vec3f32],
    O: Vec3f32,
    D: Vec3f32,
) -> Tuple[Bool, UInt32, Float32]:
    var best_t = Float32(1e30)
    var best_prim = UInt32(0xFFFFFFFF)

    for i in range(len(verts) // 3):
        var res = _brute_intersect_tri(verts, O, D, i)
        var hit = res[0]
        var t = res[1]

        if hit and t < best_t:
            best_t = t
            best_prim = UInt32(i)

    return (best_prim != UInt32(0xFFFFFFFF), best_prim, best_t)


def _make_random_xy_triangles(count: Int, seed: UInt64) -> List[Vec3f32]:
    var rng = Rng(seed, UInt64(0))
    var verts = List[Vec3f32](capacity=count * 3)

    for _ in range(count):
        var cx = _rng_f32(rng, -8.0, 8.0)
        var cy = _rng_f32(rng, -8.0, 8.0)
        var z = _rng_f32(rng, 1.0, 30.0)
        var sx = _rng_f32(rng, 0.25, 1.25)
        var sy = _rng_f32(rng, 0.25, 1.25)

        # Flat XY triangle. This deliberately creates zero-thickness AABBs on Z,
        # which catches the tmin == tmax AABB case.
        verts.append(Vec3f32(cx - sx, cy - sy, z))
        verts.append(Vec3f32(cx + sx, cy - sy, z))
        verts.append(Vec3f32(cx, cy + sy, z))

    return verts^


@always_inline
def _triangle_center_xy(verts: List[Vec3f32], prim_idx: Int) -> Vec3f32:
    ref v0 = verts[prim_idx * 3 + 0]
    ref v1 = verts[prim_idx * 3 + 1]
    ref v2 = verts[prim_idx * 3 + 2]

    return Vec3f32(
        (v0.x() + v1.x() + v2.x()) / 3.0,
        (v0.y() + v1.y() + v2.y()) / 3.0,
        0.0,
    )


def _assert_bvh_matches_bruteforce(
    mut bvh: BVH,
    verts: List[Vec3f32],
    O: Vec3f32,
    D: Vec3f32,
) raises:
    var ray = Ray(O, D)
    bvh.traverse(ray)

    var brute = _brute_trace(verts, O, D)
    var brute_hit = brute[0]
    var brute_prim = brute[1]
    var brute_t = brute[2]

    if brute_hit:
        assert_true(ray.hit.t < Float32(1e20), "BVH missed a brute-force hit")
        assert_true(
            ray.hit.prim == brute_prim, "BVH returned the wrong primitive"
        )
        _assert_close(ray.hit.t, brute_t, 1e-4)
    else:
        assert_true(ray.hit.t > Float32(1e20), "BVH hit but brute force missed")


def _assert_wide_matches_binary(
    mut bvh: BVH,
    mut wide: WideBVH[4],
    O: Vec3f32,
    D: Vec3f32,
) raises:
    var ray_binary = Ray(O, D)
    var ray_wide = Ray(O, D)

    bvh.traverse(ray_binary)
    wide.traverse(ray_wide)

    var binary_hit = ray_binary.hit.t < Float32(1e20)
    var wide_hit = ray_wide.hit.t < Float32(1e20)

    assert_true(
        binary_hit == wide_hit, "WideBVH hit/miss differs from binary BVH"
    )

    if binary_hit:
        assert_true(ray_binary.hit.prim == ray_wide.hit.prim)
        _assert_close(ray_binary.hit.t, ray_wide.hit.t, 1e-4)


def test_bvh_median_matches_bruteforce_many_rays() raises:
    var verts = _make_random_xy_triangles(64, UInt64(12345))
    var bvh = BVH(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    bvh.build["median", False]()

    for i in range(64):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_bvh_matches_bruteforce(bvh, verts, O, D)

    for i in range(16):
        var O = Vec3f32(100.0 + Float32(i), 100.0, 0.0)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_bvh_matches_bruteforce(bvh, verts, O, D)


def test_bvh_sah_matches_bruteforce_many_rays() raises:
    var verts = _make_random_xy_triangles(96, UInt64(98765))
    var bvh = BVH(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    bvh.build["sah", False]()

    for i in range(96):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_bvh_matches_bruteforce(bvh, verts, O, D)

    for i in range(24):
        var O = Vec3f32(-100.0 - Float32(i), 100.0, 0.0)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_bvh_matches_bruteforce(bvh, verts, O, D)


def test_wide_bvh4_matches_binary_many_rays() raises:
    var verts = _make_random_xy_triangles(96, UInt64(24680))
    var bvh = BVH(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    bvh.build["sah", False]()

    var wide = WideBVH[4](bvh)

    for i in range(96):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_wide_matches_binary(bvh, wide, O, D)

    for i in range(24):
        var O = Vec3f32(100.0, -100.0 - Float32(i), 0.0)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_wide_matches_binary(bvh, wide, O, D)


def test_bvh_sah_mt_matches_bruteforce_many_rays() raises:
    var verts = _make_random_xy_triangles(256, UInt64(13579))
    var bvh = BVH(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    bvh.build["sah", True]()

    for i in range(128):
        var prim_idx = (i * 17) % 256
        var O = _triangle_center_xy(verts, prim_idx)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_bvh_matches_bruteforce(bvh, verts, O, D)


# -----------------------------------------------------------------------------
# BVHGPU layout validation tests
#
# Add `BVHGPU` to your existing tinybvh import list:
#
# from bajo.core.bvh.tinybvh import (
#     BVH,
#     BVHGPU,
#     ...
# )
#
# These tests assume the existing helpers from test_tinybvh_fragment.mojo:
#   _assert_close
#   _make_strip
#   _make_random_xy_triangles
#   _triangle_center_xy
# -----------------------------------------------------------------------------


def _assert_gpu_matches_binary(
    mut bvh: BVH,
    mut gpu: BVHGPU,
    O: Vec3f32,
    D: Vec3f32,
) raises:
    var ray_binary = Ray(O, D)
    var ray_gpu = Ray(O, D)

    bvh.traverse(ray_binary)
    gpu.traverse(ray_gpu)

    var binary_hit = ray_binary.hit.t < Float32(1e20)
    var gpu_hit = ray_gpu.hit.t < Float32(1e20)

    assert_true(
        binary_hit == gpu_hit, "BVHGPU hit/miss differs from binary BVH"
    )

    if binary_hit:
        assert_true(ray_binary.hit.prim == ray_gpu.hit.prim)
        _assert_close(ray_binary.hit.t, ray_gpu.hit.t, 1e-4)


def _assert_gpu_shadow_matches_binary(
    mut bvh: BVH,
    mut gpu: BVHGPU,
    O: Vec3f32,
    D: Vec3f32,
) raises:
    var ray_binary = Ray(O, D)
    var ray_gpu = Ray(O, D)

    var binary_hit = bvh.is_occluded(ray_binary)
    var gpu_hit = gpu.is_occluded(ray_gpu)

    assert_true(
        binary_hit == gpu_hit, "BVHGPU shadow result differs from binary BVH"
    )


def test_bvh_gpu_root_leaf_matches_binary() raises:
    # Root remains a leaf because MAX_LEAF_SIZE is 4.
    var verts: List[Vec3f32] = [
        Vec3f32(-1.0, -1.0, 2.0),
        Vec3f32(1.0, -1.0, 2.0),
        Vec3f32(0.0, 1.0, 2.0),
        Vec3f32(-1.0, -1.0, 4.0),
        Vec3f32(1.0, -1.0, 4.0),
        Vec3f32(0.0, 1.0, 4.0),
    ]

    var bvh = BVH(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    bvh.build["median", False]()
    var gpu = BVHGPU(bvh)

    assert_true(gpu.root_is_leaf)
    assert_true(len(gpu.nodes) == 0)
    assert_true(len(gpu.prim_indices) == 2)

    _assert_gpu_matches_binary(
        bvh,
        gpu,
        Vec3f32(0.0, 0.0, 0.0),
        Vec3f32(0.0, 0.0, 1.0),
    )

    # Keep backing vertex storage visibly alive until after traversal.
    assert_true(len(verts) == 6)


def test_bvh_gpu_internal_layout_basic() raises:
    var verts = _make_strip(12)
    var bvh = BVH(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    bvh.build["sah", False]()
    var gpu = BVHGPU(bvh)

    assert_true(not gpu.root_is_leaf)
    assert_true(len(gpu.nodes) > 0)
    assert_true(len(gpu.prim_indices) == len(bvh.prim_indices))

    var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    gpu.traverse(ray)

    assert_true(ray.hit.t < Float32(1e20))

    # Keep backing vertex storage visibly alive until after traversal.
    assert_true(len(verts) == 36)


def test_bvh_gpu_matches_binary_many_rays() raises:
    var verts = _make_random_xy_triangles(128, UInt64(424242))
    var bvh = BVH(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    bvh.build["sah", False]()
    var gpu = BVHGPU(bvh)

    for i in range(128):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_gpu_matches_binary(bvh, gpu, O, D)

    for i in range(32):
        var O = Vec3f32(100.0 + Float32(i), -100.0, 0.0)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_gpu_matches_binary(bvh, gpu, O, D)

    # Keep backing vertex storage visibly alive until after traversal.
    assert_true(len(verts) == 128 * 3)


def test_bvh_gpu_shadow_matches_binary_many_rays() raises:
    var verts = _make_random_xy_triangles(128, UInt64(515151))
    var bvh = BVH(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    bvh.build["sah", False]()
    var gpu = BVHGPU(bvh)

    for i in range(128):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_gpu_shadow_matches_binary(bvh, gpu, O, D)

    for i in range(32):
        var O = Vec3f32(-100.0, 100.0 + Float32(i), 0.0)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_gpu_shadow_matches_binary(bvh, gpu, O, D)

    # Keep backing vertex storage visibly alive until after traversal.
    assert_true(len(verts) == 128 * 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
