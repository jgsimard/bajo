from std.testing import TestSuite, assert_true, assert_almost_equal
from std.math import abs

from bajo.core.vec import Vec3f32
from bajo.core.random import Rng
from bajo.core.intersect import intersect_ray_aabb, intersect_ray_tri
from bajo.core.bvh.build import (
    _partition_fragments,
    _partition_fragments_by_bin,
    _sah,
)
from bajo.core.bvh.cpu.binary_bvh import BinaryBvh
from bajo.core.bvh.cpu.wide import WideBvh
from bajo.core.bvh.types import BvhNode, Fragment, Ray

from fixtures import _append_tri, _make_strip


def _make_fragments(verts: List[Vec3f32]) -> List[Fragment]:
    var frags = List[Fragment](capacity=len(verts) / 3)

    for i in range(len(verts) / 3):
        frags.append(
            Fragment(
                UInt32(i),
                verts[i * 3 + 0],
                verts[i * 3 + 1],
                verts[i * 3 + 2],
            )
        )

    return frags^


def test_bvh_ray_query_inside_outside() raises:
    # Regression for rays starting inside an AABB.
    # https://github.com/NVIDIA/warp/issues/288
    var lower = Vec3f32(0.5, -1.0, -1.0)
    var upper = Vec3f32(1.0, 1.0, 1.0)

    var query_dir = Vec3f32(1.0, 0.0, 0.0)
    var rcp_dir = 1.0 / query_dir

    var query_start_outside = Vec3f32(0.0, 0.0, 0.0)
    var hit_outside = intersect_ray_aabb(
        query_start_outside, rcp_dir, lower, upper, 1e30
    )
    assert_true(hit_outside.mask, "Ray starting outside failed to hit")

    var query_start_inside = Vec3f32(0.75, 0.0, 0.0)
    var hit_inside = intersect_ray_aabb(
        query_start_inside, rcp_dir, lower, upper, 1e30
    )
    assert_true(hit_inside.mask, "Ray starting inside failed to hit")


def test_fragment_bounds_and_mapping() raises:
    var frag = Fragment(
        UInt32(42),
        Vec3f32(-1.0, 2.0, 3.0),
        Vec3f32(2.0, -4.0, 5.0),
        Vec3f32(0.0, 1.0, -6.0),
    )

    assert_true(frag.prim_idx == 42)
    assert_true(frag.bmin.x == -1.0)
    assert_true(frag.bmin.y == -4.0)
    assert_true(frag.bmin.z == -6.0)
    assert_true(frag.bmax.x == 2.0)
    assert_true(frag.bmax.y == 2.0)
    assert_true(frag.bmax.z == 5.0)
    assert_almost_equal(frag.center_axis(0), 0.5)


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

    var node = BvhNode()
    node.left_first = 0
    node.item_count = 2

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

    var node = BvhNode()
    node.left_first = 0
    node.item_count = 2

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

    var node = BvhNode()
    node.left_first = 0
    node.item_count = 10

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

    var node = BvhNode()
    node.left_first = 0
    node.item_count = 10

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
    var bvh = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) / 3))
    bvh.build["median", False]()

    assert_true(bvh.nodes_used > 1)
    assert_true(Int(bvh.nodes_used) <= len(bvh.bvh_nodes))

    var leaf_prim_total = UInt32(0)
    for i in range(Int(bvh.nodes_used)):
        ref node = bvh.bvh_nodes[i]

        if node.is_leaf():
            assert_true(node.item_count > 0)
            assert_true(
                Int(node.left_first) + Int(node.item_count)
                <= len(bvh.prim_indices)
            )
            leaf_prim_total += node.item_count
        else:
            assert_true(node.item_count == 0)
            assert_true(node.left_first + 1 < bvh.nodes_used)

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

    var bvh = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) / 3))
    bvh.build["median", False]()

    var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    bvh.traverse(ray)

    assert_true(ray.hit.prim == 0)
    assert_almost_equal(ray.hit.t, 2.0)


def test_bvh_traverse_reports_original_primitive_after_reorder() raises:
    var verts = _make_strip(8)
    var bvh = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) / 3))
    bvh.build["sah", False]()

    # Aim at the triangle with original primitive id 6.
    # _make_strip centers primitive i at x = i * 4 - count * 2.
    var ray = Ray(Vec3f32(8.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    bvh.traverse(ray)

    assert_true(ray.hit.prim == 6)
    assert_almost_equal(ray.hit.t, 2.0)


def test_bvh_shadow_ray() raises:
    var verts = _make_strip(4)
    var bvh = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) / 3))
    bvh.build["median", False]()

    # Primitive 2 is centered at x=0.
    var ray_hit = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(bvh.is_occluded(ray_hit))

    var ray_miss = Ray(Vec3f32(100.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(not bvh.is_occluded(ray_miss))


def test_wide_bvh4_matches_binary_for_basic_hit() raises:
    var verts = _make_strip(8)

    var bvh = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) / 3))
    bvh.build["sah", False]()

    var wide = WideBvh[4](bvh)

    var ray_binary = Ray(Vec3f32(8.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var ray_wide = Ray(Vec3f32(8.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))

    bvh.traverse(ray_binary)
    wide.traverse(ray_wide)

    assert_true(ray_binary.hit.prim == ray_wide.hit.prim)
    assert_almost_equal(ray_binary.hit.t, ray_wide.hit.t)


# -----------------------------------------------------------------------------
# Randomized / brute-force regression tests
# -----------------------------------------------------------------------------


@always_inline
def _rng_f32(mut rng: Rng, lo: Float32, hi: Float32) -> Float32:
    return lo + (hi - lo) * rng.f32()


def _brute_trace(
    verts: List[Vec3f32],
    O: Vec3f32,
    D: Vec3f32,
) -> Tuple[Bool, UInt32, Float32]:
    var best_t = Float32(1e30)
    var best_prim = UInt32(0xFFFFFFFF)

    for i in range(len(verts) / 3):
        ref v0 = verts[i * 3 + 0]
        ref v1 = verts[i * 3 + 1]
        ref v2 = verts[i * 3 + 2]
        var res = intersect_ray_tri(
            O,
            D,
            v0,
            v1,
            v2,
            Float32.MAX,
        )
        var hit = res.mask
        var t = res.t

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
        (v0.x + v1.x + v2.x) / 3.0,
        (v0.y + v1.y + v2.y) / 3.0,
        0.0,
    )


def _assert_bvh_matches_bruteforce(
    mut bvh: BinaryBvh,
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
        assert_almost_equal(ray.hit.t, brute_t)
    else:
        assert_true(ray.hit.t > Float32(1e20), "BVH hit but brute force missed")


def _assert_wide_matches_binary(
    mut bvh: BinaryBvh,
    mut wide: WideBvh[4],
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
        binary_hit == wide_hit, "WideBvh hit/miss differs from binary BVH"
    )

    if binary_hit:
        assert_true(ray_binary.hit.prim == ray_wide.hit.prim)
        assert_almost_equal(ray_binary.hit.t, ray_wide.hit.t)


def test_bvh_median_matches_bruteforce_many_rays() raises:
    var verts = _make_random_xy_triangles(64, UInt64(12345))
    var bvh = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) / 3))
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
    var bvh = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) / 3))
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
    var bvh = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) / 3))
    bvh.build["sah", False]()

    var wide = WideBvh[4](bvh)

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
    var bvh = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) / 3))
    bvh.build["sah", True]()

    for i in range(128):
        var prim_idx = (i * 17) % 256
        var O = _triangle_center_xy(verts, prim_idx)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_bvh_matches_bruteforce(bvh, verts, O, D)


# -----------------------------------------------------------------------------
# CPU LBVH builder validation tests
# -----------------------------------------------------------------------------
def test_bvh_lbvh_build_invariants() raises:
    var verts = _make_random_xy_triangles(96, UInt64(606060))
    var bvh = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) / 3))
    bvh.build["lbvh", False]()

    assert_true(bvh.nodes_used > 1)
    assert_true(Int(bvh.nodes_used) <= len(bvh.bvh_nodes))

    var leaf_prim_total = UInt32(0)
    for i in range(Int(bvh.nodes_used)):
        ref node = bvh.bvh_nodes[i]
        if node.is_leaf():
            assert_true(node.item_count > 0)
            assert_true(
                Int(node.left_first) + Int(node.item_count)
                <= len(bvh.prim_indices)
            )
            leaf_prim_total += node.item_count
        else:
            assert_true(node.item_count == 0)
            assert_true(node.left_first + 1 < bvh.nodes_used)

    assert_true(leaf_prim_total == bvh.tri_count)
    assert_true(len(verts) == 96 * 3)


def test_bvh_lbvh_matches_bruteforce_many_rays() raises:
    var verts = _make_random_xy_triangles(128, UInt64(707070))
    var bvh = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) / 3))
    bvh.build["lbvh", False]()

    for i in range(128):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_bvh_matches_bruteforce(bvh, verts, O, D)

    for i in range(32):
        var O = Vec3f32(100.0 + Float32(i), 100.0, 0.0)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_bvh_matches_bruteforce(bvh, verts, O, D)

    assert_true(len(verts) == 128 * 3)


def test_wide_bvh4_from_lbvh_matches_binary() raises:
    var verts = _make_random_xy_triangles(128, UInt64(808080))
    var bvh = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) / 3))
    bvh.build["lbvh", False]()
    var wide = WideBvh[4](bvh)

    for i in range(128):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_wide_matches_binary(bvh, wide, O, D)

    assert_true(len(verts) == 128 * 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
