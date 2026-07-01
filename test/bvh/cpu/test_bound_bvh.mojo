from std.testing import TestSuite, assert_true, assert_almost_equal

from bajo.core import AABB, Vec3f32
from bajo.core.intersect import intersect_ray_aabb
from bajo.bvh.types import Ray, Sphere
from bajo.core.random import Rng
from bajo.bvh.constants import EMPTY_LANE, TRACE, f32_max
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvhBuilder,
    BoundsItem,
    BoundsBvh,
)
from bajo.bvh.cpu.builder.builder import _partition_items_by_median_center
from bajo.bvh.cpu.builder.sah import _find_sah_split
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.host_utils import triangle_bounds

from test.bvh.fixtures import _brute_triangle_trace, _brute_sphere_trace


def _rng_f32(mut rng: Rng, lo: Float32, hi: Float32) -> Float32:
    return lo + (hi - lo) * rng.f32()


def _z_ray(origin: Vec3f32) -> Ray:
    return Ray(origin, Vec3f32(0.0, 0.0, 1.0))


def _make_random_xy_triangles(count: Int, seed: UInt64) -> List[Vec3f32]:
    var rng = Rng(seed, 0)
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


def _make_strip(count: Int) -> List[Vec3f32]:
    """Create `count` separated triangles at z = 2.

    Primitive i is centered at x = i * 4 - count * 2.
    """
    var verts = List[Vec3f32](capacity=count * 3)

    for i in range(count):
        var cx = Float32(i * 4 - count * 2)
        verts.append(Vec3f32(cx - 1.0, -1.0, 2.0))
        verts.append(Vec3f32(cx + 1.0, -1.0, 2.0))
        verts.append(Vec3f32(cx, 1.0, 2.0))

    return verts^


def _make_depth_pair() -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=6)

    # Primitive 0 at z = 2.
    verts.append(Vec3f32(-1.0, -1.0, 2.0))
    verts.append(Vec3f32(1.0, -1.0, 2.0))
    verts.append(Vec3f32(0.0, 1.0, 2.0))

    # Primitive 1 at z = 4, behind primitive 0.
    verts.append(Vec3f32(-1.0, -1.0, 4.0))
    verts.append(Vec3f32(1.0, -1.0, 4.0))
    verts.append(Vec3f32(0.0, 1.0, 4.0))

    return verts^


def _make_bounds_items(verts: List[Vec3f32]) -> List[BoundsItem]:
    var tri_count = len(verts) / 3
    var items = List[BoundsItem](capacity=tri_count)

    for i in range(tri_count):
        ref v0 = verts[i * 3 + 0]
        ref v1 = verts[i * 3 + 1]
        ref v2 = verts[i * 3 + 2]

        items.append(BoundsItem(triangle_bounds(v0, v1, v2), UInt32(i)))

    return items^


def _make_spheres() -> List[Sphere]:
    return [
        Sphere(Vec3f32(0.0, 0.0, 2.0), 1.0),
        Sphere(Vec3f32(4.0, 0.0, 4.0), 1.0),
        Sphere(Vec3f32(-4.0, 0.0, 6.0), 1.0),
        Sphere(Vec3f32(0.0, 4.0, 8.0), 1.0),
    ]


def _triangle_center_xy(verts: List[Vec3f32], prim_idx: Int) -> Vec3f32:
    ref v0 = verts[prim_idx * 3 + 0]
    ref v1 = verts[prim_idx * 3 + 1]
    ref v2 = verts[prim_idx * 3 + 2]

    var out = (v0 + v1 + v2) / 3.0
    out.z = 0.0
    return out


def _assert_builder_leaf_sizes_at_most(
    builder: BoundsBvhBuilder,
    max_leaf_size: UInt32,
) raises:
    var leaf_item_total = UInt32(0)

    for i in range(Int(builder.nodes_used)):
        ref node = builder.nodes[i]

        if node.is_leaf():
            assert_true(node.item_count > 0)
            assert_true(
                node.item_count <= max_leaf_size,
                String(
                    t"leaf_size invariant violated,"
                    t" {node.item_count} {max_leaf_size}"
                ),
            )
            assert_true(
                Int(node.first_item()) + Int(node.item_count)
                <= len(builder.item_indices)
            )
            leaf_item_total += node.item_count
        else:
            assert_true(node.item_count == 0)
            assert_true(node.left_child() + 1 < builder.nodes_used)

    assert_true(leaf_item_total == builder.item_count)


def _assert_wide_leaf_counts_at_most_width[
    width: SIMDSize
](wide: BoundsBvh[width]) raises:
    for ref node in wide.nodes:
        for lane in range(width):
            var count = node.counts[lane]

            if count == EMPTY_LANE:
                continue

            if count == 0:
                assert_true(node.data[lane] < UInt32(len(wide.nodes)))
            else:
                assert_true(count <= UInt32(width))
                assert_true(Int(node.data[lane]) < len(wide.item_indices))
                assert_true(
                    Int(node.data[lane]) + Int(count) <= len(wide.item_indices)
                )


def _assert_triangle_bvh_matches_bruteforce[
    width: SIMDSize
](mut bvh: TriangleBvh[width], verts: List[Vec3f32], origin: Vec3f32) raises:
    var ray = _z_ray(origin)
    var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)

    var brute = _brute_triangle_trace(
        verts,
        origin,
        Vec3f32(0.0, 0.0, 1.0),
    )
    var brute_hit = brute.is_hit()

    var bvh_hit = hit.is_hit()

    assert_true(
        bvh_hit == brute_hit,
        "TriangleBvh hit/miss differs from brute force",
    )

    if brute_hit:
        assert_true(
            hit.prim == brute.prim,
            "TriangleBvh returned the wrong primitive",
        )
        assert_almost_equal(hit.t, brute.t)


def _assert_sphere_bvh_matches_bruteforce[
    width: SIMDSize
](mut bvh: SphereBvh[width], spheres: List[Sphere], origin: Vec3f32) raises:
    var ray = _z_ray(origin)
    var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)

    var brute = _brute_sphere_trace(
        spheres,
        origin,
        Vec3f32(0.0, 0.0, 1.0),
    )
    var brute_hit = brute.is_hit()

    var bvh_hit = hit.is_hit()

    assert_true(
        bvh_hit == brute_hit,
        "SphereBvh hit/miss differs from brute force",
    )

    if brute_hit:
        assert_true(
            hit.prim == brute.prim,
            "SphereBvh returned the wrong primitive",
        )
        assert_almost_equal(hit.t, brute.t)


def _test_bounds_bvh_leaf_invariant[
    width: SIMDSize,
    mode: String,
]() raises:
    comptime assert mode in ["median", "sah", "lbvh"]

    var verts = _make_random_xy_triangles(24 * width, UInt64(606060 + width))
    var items = _make_bounds_items(verts)

    var builder = BoundsBvhBuilder[width](items)
    builder.build[mode]()

    assert_true(builder.nodes_used > 0)
    assert_true(Int(builder.nodes_used) <= len(builder.nodes))

    _assert_builder_leaf_sizes_at_most(builder, UInt32(width))

    var wide = BoundsBvh[width](builder)
    _assert_wide_leaf_counts_at_most_width[width](wide)


def test_bounds_bvh_leaf_invariants() raises:
    comptime for w in [2, 4, 8]:
        comptime for mode in ["median", "sah", "lbvh"]:
            _test_bounds_bvh_leaf_invariant[w, mode]()


def test_wide_bounds_root_bounds_is_valid() raises:
    var verts = _make_strip(4)
    var items = _make_bounds_items(verts)

    var builder = BoundsBvhBuilder[4](items)
    builder.build["median"]()

    var wide = BoundsBvh[4](builder)
    var bounds = wide.root_bounds()

    assert_true(bounds._min.x <= -9.0)
    assert_true(bounds._max.x >= 5.0)
    assert_true(bounds._min.z <= 2.0)
    assert_true(bounds._max.z >= 2.0)


def test_bounds_ray_query_inside_outside_regression() raises:
    var lower = Vec3f32(0.5, -1.0, -1.0)
    var upper = Vec3f32(1.0, 1.0, 1.0)

    var query_ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(1.0, 0.0, 0.0))
    var rcp_dir = query_ray.rcp_direction[1]()

    var hit_outside = intersect_ray_aabb(
        Vec3f32(0.0, 0.0, 0.0),
        rcp_dir,
        lower,
        upper,
        f32_max,
    )
    assert_true(hit_outside.mask, "Ray starting outside failed to hit")

    var hit_inside = intersect_ray_aabb(
        Vec3f32(0.75, 0.0, 0.0),
        rcp_dir,
        lower,
        upper,
        f32_max,
    )
    assert_true(hit_inside.mask, "Ray starting inside failed to hit")


def test_ray_rcp_direction_uses_finite_parallel_axes() raises:
    var ray = Ray(Vec3f32(0.0), Vec3f32(2.0, 0.0, -4.0))
    var rcp_dir = ray.rcp_direction[4]()

    assert_almost_equal(rcp_dir.x, 0.5)
    assert_almost_equal(rcp_dir.y, 1.0e9)
    assert_almost_equal(rcp_dir.z, -0.25)


def test_bounds_item_bounds_and_payload_mapping() raises:
    var bounds = triangle_bounds(
        Vec3f32(-1.0, 2.0, 3.0),
        Vec3f32(2.0, -4.0, 5.0),
        Vec3f32(0.0, 1.0, -6.0),
    )
    var item = BoundsItem(bounds, UInt32(42))

    assert_true(item.payload == 42)
    assert_almost_equal(item.bounds._min.x, -1.0)
    assert_almost_equal(item.bounds._min.y, -4.0)
    assert_almost_equal(item.bounds._min.z, -6.0)
    assert_almost_equal(item.bounds._max.x, 2.0)
    assert_almost_equal(item.bounds._max.y, 2.0)
    assert_almost_equal(item.bounds._max.z, 5.0)
    assert_almost_equal(item.center_axis(0), 0.5)


def test_bounds_sah_clear_separation() raises:
    var verts: List[Vec3f32] = [
        Vec3f32(-11.0, -1.0, 0.0),
        Vec3f32(-9.0, -1.0, 0.0),
        Vec3f32(-10.0, 1.0, 0.0),  # Tri 0, centered near x=-10
        Vec3f32(9.0, -1.0, 0.0),
        Vec3f32(11.0, -1.0, 0.0),
        Vec3f32(10.0, 1.0, 0.0),  # Tri 1, centered near x=10
    ]
    var items = _make_bounds_items(verts)
    var builder = BoundsBvhBuilder[2](items)
    builder.build["sah"]()

    var split = _find_sah_split(
        builder.nodes[0],
        builder.item_indices.unsafe_ptr(),
        builder.items.unsafe_ptr(),
    )

    assert_true(split.axis == 0)
    assert_true(split.pos > -10.0 and split.pos < 10.0)
    assert_true(split.cost < 20.0)
    assert_true(split.bin >= 0)


def test_bounds_sah_degenerate() raises:
    var verts: List[Vec3f32] = [
        Vec3f32(0.0, 0.0, 0.0),
        Vec3f32(1.0, 0.0, 0.0),
        Vec3f32(0.0, 1.0, 0.0),
        Vec3f32(0.0, 0.0, 0.0),
        Vec3f32(1.0, 0.0, 0.0),
        Vec3f32(0.0, 1.0, 0.0),
    ]
    var items = _make_bounds_items(verts)
    var builder = BoundsBvhBuilder[2](items)
    builder.build["sah"]()

    var split = _find_sah_split(
        builder.nodes[0],
        builder.item_indices.unsafe_ptr(),
        builder.items.unsafe_ptr(),
    )

    assert_true(split.axis == -1)
    assert_true(not split.valid())


def test_bounds_partition_items_non_empty() raises:
    var verts: List[Vec3f32] = [
        Vec3f32(-11.0, -1.0, 0.0),
        Vec3f32(-9.0, -1.0, 0.0),
        Vec3f32(-10.0, 1.0, 0.0),
        Vec3f32(9.0, -1.0, 0.0),
        Vec3f32(11.0, -1.0, 0.0),
        Vec3f32(10.0, 1.0, 0.0),
    ]
    var items = _make_bounds_items(verts)
    var builder = BoundsBvhBuilder[2](items)
    builder.build["sah"]()

    var split_idx = _partition_items_by_median_center(
        Span(builder.item_indices),
        builder.items.unsafe_ptr(),
        0,
        2,
        0,
    )

    assert_true(split_idx == 1)
    assert_true(builder.item_indices[0] != builder.item_indices[1])


def test_triangle_bvh2_leaf_size_equals_width_returns_nearest_triangle() raises:
    var verts = _make_depth_pair()
    var bvh = TriangleBvh[2].__init__["median"](verts^)

    var hit = bvh.trace[TRACE.CLOSEST_HIT](_z_ray(Vec3f32(0.0, 0.0, 0.0)))

    assert_true(hit.is_hit())
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 2.0)


def _test_triangle_bvh_matches_bruteforce[
    width: SIMDSize,
    split_mode: String,
]() raises:
    var n = {2: 24, 4: 32, 8: 40}[width]
    var verts = _make_strip(n)
    var bvh = TriangleBvh[width].__init__[split_mode](verts.copy())

    for i in range(n):
        _assert_triangle_bvh_matches_bruteforce[width](
            bvh,
            verts,
            _triangle_center_xy(verts, i),
        )

    for i in range(8):
        _assert_triangle_bvh_matches_bruteforce[width](
            bvh,
            verts,
            Vec3f32(100.0 + Float32(i), 100.0, 0.0),
        )


def test_triangle_bvh_matches_bruteforce() raises:
    comptime for w in [2, 4, 8]:
        comptime for mode in ["median", "sah", "lbvh"]:
            _test_triangle_bvh_matches_bruteforce[w, mode]()


def _test_triangle_bvh_shadow_hit_and_miss[
    width: SIMDSize,
    mode: String,
]() raises:
    var verts = _make_strip(2 * width)
    var bvh = TriangleBvh[width].__init__[mode](verts^)

    assert_true(
        bvh.trace[TRACE.ANY_HIT](_z_ray(Vec3f32(0.0, 0.0, 0.0))).is_occluded()
    )

    assert_true(
        not bvh.trace[TRACE.ANY_HIT](
            _z_ray(Vec3f32(100.0, 100.0, 0.0))
        ).is_occluded()
    )


def test_triangle_bvh_shadow_hit_and_miss() raises:
    comptime for w in [2, 4, 8]:
        comptime for mode in ["median", "sah", "lbvh"]:
            _test_triangle_bvh_shadow_hit_and_miss[w, mode]()


def test_sphere_bounds() raises:
    var s = Sphere(Vec3f32(1.0, 2.0, 3.0), 2.0)
    var b = s.bounds()

    assert_almost_equal(b._min.x, -1.0)
    assert_almost_equal(b._min.y, 0.0)
    assert_almost_equal(b._min.z, 1.0)

    assert_almost_equal(b._max.x, 3.0)
    assert_almost_equal(b._max.y, 4.0)
    assert_almost_equal(b._max.z, 5.0)


def test_sphere_bvh4_single_leaf_layout_and_hit() raises:
    var spheres = _make_spheres()
    var bvh = SphereBvh[4](spheres^)

    assert_true(len(bvh.tree.nodes) == 1)
    assert_true(bvh.tree.nodes[0].counts[0] == 4)
    assert_true(bvh.tree.nodes[0].data[0] == 0)

    var hit = bvh.trace[TRACE.CLOSEST_HIT](_z_ray(Vec3f32(0.0, 0.0, 0.0)))

    assert_true(hit.is_hit())
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 1.0)


def _test_sphere_bvh_matches_bruteforce[
    width: SIMDSize,
    mode: String,
]() raises:
    var spheres = _make_spheres()
    var bvh = SphereBvh[width].__init__[mode](spheres.copy())

    _assert_sphere_bvh_matches_bruteforce[width](
        bvh,
        spheres,
        Vec3f32(0.0, 0.0, 0.0),
    )
    _assert_sphere_bvh_matches_bruteforce[width](
        bvh,
        spheres,
        Vec3f32(4.0, 0.0, 0.0),
    )
    _assert_sphere_bvh_matches_bruteforce[width](
        bvh,
        spheres,
        Vec3f32(-4.0, 0.0, 0.0),
    )
    _assert_sphere_bvh_matches_bruteforce[width](
        bvh,
        spheres,
        Vec3f32(100.0, 0.0, 0.0),
    )


def test_sphere_bvh_matches_bruteforce() raises:
    comptime for w in [2, 4, 8]:
        comptime for mode in ["median", "sah", "lbvh"]:
            _test_sphere_bvh_matches_bruteforce[w, mode]()


def _test_sphere_bvh_shadow_hit_and_miss[
    width: SIMDSize,
    mode: String,
]() raises:
    var spheres = _make_spheres()
    var bvh = SphereBvh[width].__init__[mode](spheres^)

    assert_true(
        bvh.trace[TRACE.ANY_HIT](_z_ray(Vec3f32(0.0, 0.0, 0.0))).is_occluded()
    )

    assert_true(
        not bvh.trace[TRACE.ANY_HIT](
            _z_ray(Vec3f32(100.0, 0.0, 0.0))
        ).is_occluded()
    )


def test_sphere_bvh_shadow_hit_and_miss() raises:
    comptime for w in [2, 4, 8]:
        comptime for mode in ["median", "sah", "lbvh"]:
            _test_sphere_bvh_shadow_hit_and_miss[w, mode]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
