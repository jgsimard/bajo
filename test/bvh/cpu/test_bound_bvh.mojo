from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
)

from bajo.core import (
    AABB,
    Vec3f32,
    Point3f32,
    Normal3f32,
    Frame,
    Vec3W,
    Point3W,
    Point3,
    Ray,
    Rayf32,
    Vec3,
)
from bajo.core.intersect import intersect_ray_aabb, intersect_ray_aabb_rcp
from bajo.bvh.types import Hit, Sphere
from bajo.core.random import Rng
from bajo.bvh.constants import EMPTY_LANE, TRACE, f32_max
from bajo.bvh.cpu.bounds_bvh import (
    BinaryBoundsBvh,
    BoundsItem,
    BoundsBvh,
)
from bajo.bvh.cpu.packet import (
    _coherent_packet_frustum,
    _intersect_coherent_packet_frustum,
)
from bajo.bvh.tagged_ref import (
    decode_ref_index,
    encode_leaf_ref,
    is_leaf_ref,
)
from bajo.bvh.cpu.builder.builder import _partition_items_by_median_center
from bajo.bvh.cpu.builder.lbvh import (
    MortonItem,
    _radix_sort_morton_pairs,
    _radix_sort_morton_pairs_parallel,
)
from bajo.bvh.cpu.builder.sah import _find_sah_split, _partition_items_by_bin
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.cpu.trace import (
    CpuBvhTraversalStats,
    _extract_f32_lane,
    _extract_u32_lane,
)
from bajo.bvh.host_utils import triangle_bounds

from test.bvh.fixtures import _brute_triangle_trace, _brute_sphere_trace


def test_shadow_hit_sentinel_distinguishes_bounded_miss() raises:
    assert_true(Hit[Frame.WORLD].shadow_hit().is_occluded())
    assert_true(not Hit[Frame.WORLD].miss(4.0).is_occluded())


def test_hit_load_store_span_with_nonzero_index() raises:
    var data = List[Float32](length=2 * Hit.STRIDE, fill=-1.0)
    var expected = Hit[Frame.WORLD](
        0.25,
        0.5,
        UInt32(7),
        UInt32(3),
        Normal3f32[Frame.WORLD](1.0, 2.0, 3.0),
        4.0,
    )

    expected.store(Span(data), 1)
    var actual = Hit[Frame.WORLD].load(Span(data), 1)

    assert_almost_equal(actual.u, expected.u)
    assert_almost_equal(actual.v, expected.v)
    assert_true(actual.prim == expected.prim)
    assert_true(actual.inst == expected.inst)
    assert_almost_equal(actual.normal.x, expected.normal.x)
    assert_almost_equal(actual.normal.y, expected.normal.y)
    assert_almost_equal(actual.normal.z, expected.normal.z)
    assert_almost_equal(actual.t, expected.t)


def _test_extract_lane[width: SIMDLength]() raises:
    var u32_values = SIMD[DType.uint32, width](0)
    var f32_values = SIMD[DType.float32, width](0.0)
    comptime for lane in range(width):
        u32_values[lane] = UInt32(100 + lane)
        f32_values[lane] = Float32(lane) + 0.25

    for lane in range(Int(width)):
        assert_true(_extract_u32_lane(u32_values, lane) == UInt32(100 + lane))
        assert_almost_equal(
            _extract_f32_lane(f32_values, lane), Float32(lane) + 0.25
        )


def test_extract_lane_all_cpu_bvh_widths() raises:
    comptime for width in [2, 4, 8, 16]:
        _test_extract_lane[width]()


def _rng_f32(mut rng: Rng, lo: Float32, hi: Float32) -> Float32:
    return lo + (hi - lo) * rng.f32()


def _z_ray[frame: Frame](origin: Point3f32[frame]) -> Rayf32[frame]:
    return Rayf32[frame](origin, Vec3f32[frame](0.0, 0.0, 1.0))


def _make_random_xy_triangles[
    frame: Frame
](count: Int, seed: UInt64) -> List[Point3f32[frame]]:
    var rng = Rng(seed, 0)
    var verts = List[Point3f32[frame]](capacity=count * 3)

    for _ in range(count):
        var cx = _rng_f32(rng, -8.0, 8.0)
        var cy = _rng_f32(rng, -8.0, 8.0)
        var z = _rng_f32(rng, 1.0, 30.0)
        var sx = _rng_f32(rng, 0.25, 1.25)
        var sy = _rng_f32(rng, 0.25, 1.25)

        # Flat XY triangle. This deliberately creates zero-thickness AABBs on Z,
        # which catches the tmin == tmax AABB case.
        verts.append(Point3f32[frame](cx - sx, cy - sy, z))
        verts.append(Point3f32[frame](cx + sx, cy - sy, z))
        verts.append(Point3f32[frame](cx, cy + sy, z))

    return verts^


def _make_strip[frame: Frame](count: Int) -> List[Point3f32[frame]]:
    """Create `count` separated triangles at z = 2.

    Primitive i is centered at x = i * 4 - count * 2.
    """
    var verts = List[Point3f32[frame]](capacity=count * 3)

    for i in range(count):
        var cx = Float32(i * 4 - count * 2)
        verts.append(Point3f32[frame](cx - 1.0, -1.0, 2.0))
        verts.append(Point3f32[frame](cx + 1.0, -1.0, 2.0))
        verts.append(Point3f32[frame](cx, 1.0, 2.0))

    return verts^


def _make_depth_pair[frame: Frame]() -> List[Point3f32[frame]]:
    var verts = List[Point3f32[frame]](capacity=6)

    # Primitive 0 at z = 2.
    verts.append(Point3f32[frame](-1.0, -1.0, 2.0))
    verts.append(Point3f32[frame](1.0, -1.0, 2.0))
    verts.append(Point3f32[frame](0.0, 1.0, 2.0))

    # Primitive 1 at z = 4, behind primitive 0.
    verts.append(Point3f32[frame](-1.0, -1.0, 4.0))
    verts.append(Point3f32[frame](1.0, -1.0, 4.0))
    verts.append(Point3f32[frame](0.0, 1.0, 4.0))

    return verts^


def _make_depth_stack[frame: Frame](count: Int) -> List[Point3f32[frame]]:
    var verts = List[Point3f32[frame]](capacity=count * 3)

    for i in range(count):
        var z = 2.0 + Float32(i)
        verts.append(Point3f32[frame](-1.0, -1.0, z))
        verts.append(Point3f32[frame](1.0, -1.0, z))
        verts.append(Point3f32[frame](0.0, 1.0, z))

    return verts^


def _make_bounds_items[
    frame: Frame
](verts: List[Point3f32[frame]]) -> List[BoundsItem[frame]]:
    var tri_count = len(verts) / 3
    var items = List[BoundsItem[frame]](capacity=tri_count)

    for i in range(tri_count):
        ref v0 = verts[i * 3 + 0]
        ref v1 = verts[i * 3 + 1]
        ref v2 = verts[i * 3 + 2]

        items.append(BoundsItem(triangle_bounds(v0, v1, v2), UInt32(i)))

    return items^


def _make_spheres[frame: Frame]() -> List[Sphere[frame]]:
    return [
        Sphere(Point3f32[frame](0.0, 0.0, 2.0), 1.0),
        Sphere(Point3f32[frame](4.0, 0.0, 4.0), 1.0),
        Sphere(Point3f32[frame](-4.0, 0.0, 6.0), 1.0),
        Sphere(Point3f32[frame](0.0, 4.0, 8.0), 1.0),
    ]


def _triangle_center_xy[
    frame: Frame
](verts: List[Point3f32[frame]], prim_idx: Int) -> Point3f32[frame]:
    ref v0 = verts[prim_idx * 3 + 0]
    ref v1 = verts[prim_idx * 3 + 1]
    ref v2 = verts[prim_idx * 3 + 2]

    var out = v0.unsafe_add(v1).unsafe_add(v2) / 3.0
    out.z = 0.0
    return out


def _assert_builder_leaf_sizes_at_most(
    builder: BinaryBoundsBvh,
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


def _assert_wide_leaf_ranges_at_most_width[
    frame: Frame, width: SIMDLength
](wide: BoundsBvh[frame, width]) raises:
    assert_true(len(wide.child_masks) == len(wide.nodes))
    for node_idx in range(len(wide.nodes)):
        ref node = wide.nodes[node_idx]
        var expected_child_mask = UInt32(0)
        for lane in range(width):
            var child_ref = node.data[lane]

            if child_ref == EMPTY_LANE:
                continue

            expected_child_mask |= UInt32(1) << UInt32(lane)

            if is_leaf_ref(child_ref):
                var leaf_range_idx = decode_ref_index(child_ref)
                assert_true(Int(leaf_range_idx) < len(wide.leaf_ranges))

                ref leaf_range = wide.leaf_ranges[Int(leaf_range_idx)]
                assert_true(leaf_range.item_count > 0)
                assert_true(leaf_range.item_count <= UInt32(width))
                assert_true(
                    Int(leaf_range.first_item) + Int(leaf_range.item_count)
                    <= len(wide.item_indices)
                )
            else:
                assert_true(child_ref < UInt32(len(wide.nodes)))

        assert_true(wide.child_masks[node_idx] == expected_child_mask)


def _assert_triangle_bvh_matches_bruteforce[
    frame: Frame,
    bounds_width: SIMDLength,
    leaf_width: SIMDLength = bounds_width,
](
    mut bvh: TriangleBvh[frame, bounds_width, leaf_width],
    verts: List[Point3f32[frame]],
    origin: Point3f32[frame],
) raises:
    var ray = _z_ray(origin)
    var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)

    var brute = _brute_triangle_trace(
        verts,
        origin,
        Vec3f32[frame](0.0, 0.0, 1.0),
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
        assert_almost_equal(
            hit.normal.x * hit.normal.x
            + hit.normal.y * hit.normal.y
            + hit.normal.z * hit.normal.z,
            1.0,
        )


def _assert_sphere_bvh_matches_bruteforce[
    frame: Frame, width: SIMDLength
](
    mut bvh: SphereBvh[frame, width],
    spheres: List[Sphere[frame]],
    origin: Point3f32[frame],
) raises:
    var ray = _z_ray(origin)
    var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)

    var brute = _brute_sphere_trace(
        spheres,
        origin,
        Vec3f32[frame](0.0, 0.0, 1.0),
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
    frame: Frame,
    width: SIMDLength,
    mode: String,
]() raises:
    comptime assert mode in ["median", "sah", "lbvh", "hploc"]

    var verts = _make_random_xy_triangles[frame](
        24 * width, UInt64(606060 + width)
    )
    var items = _make_bounds_items(verts)

    var builder = BinaryBoundsBvh[frame, width, mode](items^)

    assert_true(builder.nodes_used > 0)
    assert_true(Int(builder.nodes_used) == len(builder.nodes))

    _assert_builder_leaf_sizes_at_most(builder, UInt32(width))

    var wide = BoundsBvh[frame, width](builder)
    _assert_wide_leaf_ranges_at_most_width[frame, width](wide)


def test_bounds_bvh_leaf_invariants() raises:
    comptime for w in [2, 4, 8]:
        comptime for mode in ["median", "sah", "lbvh", "hploc"]:
            _test_bounds_bvh_leaf_invariant[Frame.WORLD, w, mode]()


def test_parallel_sah_builder_leaf_invariants() raises:
    # Cross the parallel-build threshold and validate both the compacted
    # binary storage and the final wide leaf ranges.
    var verts = _make_random_xy_triangles[Frame.WORLD](5000, UInt64(909090))
    var items = _make_bounds_items(verts)
    var builder = BinaryBoundsBvh[Frame.WORLD, 16, "sah"](items^)

    assert_true(Int(builder.nodes_used) == len(builder.nodes))
    _assert_builder_leaf_sizes_at_most(builder, UInt32(16))

    var wide = BoundsBvh[Frame.WORLD, 16](builder)
    _assert_wide_leaf_ranges_at_most_width[Frame.WORLD, 16](wide)


def test_parallel_median_builder_leaf_invariants() raises:
    var verts = _make_random_xy_triangles[Frame.WORLD](1500, UInt64(919191))
    var items = _make_bounds_items(verts)
    var builder = BinaryBoundsBvh[Frame.WORLD, 16, "median"](items^)

    assert_true(Int(builder.nodes_used) == len(builder.nodes))
    _assert_builder_leaf_sizes_at_most(builder, UInt32(16))

    var wide = BoundsBvh[Frame.WORLD, 16](builder)
    _assert_wide_leaf_ranges_at_most_width[Frame.WORLD, 16](wide)


def test_parallel_radix_lbvh_builder_leaf_invariants() raises:
    var verts = _make_random_xy_triangles[Frame.WORLD](17000, UInt64(929292))
    var items = _make_bounds_items(verts)
    var builder = BinaryBoundsBvh[Frame.WORLD, 16, "lbvh"](items^)

    assert_true(Int(builder.nodes_used) == len(builder.nodes))
    _assert_builder_leaf_sizes_at_most(builder, UInt32(16))

    var wide = BoundsBvh[Frame.WORLD, 16](builder)
    _assert_wide_leaf_ranges_at_most_width[Frame.WORLD, 16](wide)


def test_cpu_lbvh_radix_sort_orders_all_bytes() raises:
    var pairs = List[MortonItem](capacity=7)
    pairs.append(MortonItem(UInt32(0xFF000000), UInt32(0)))
    pairs.append(MortonItem(UInt32(0x000000FF), UInt32(1)))
    pairs.append(MortonItem(UInt32(0x00FF0000), UInt32(2)))
    pairs.append(MortonItem(UInt32(0x0000FF00), UInt32(3)))
    pairs.append(MortonItem(UInt32(0x00000000), UInt32(4)))
    pairs.append(MortonItem(UInt32(0xFFFFFFFF), UInt32(5)))
    pairs.append(MortonItem(UInt32(0x000000FF), UInt32(6)))
    var expected_codes = [
        UInt32(0x00000000),
        UInt32(0x000000FF),
        UInt32(0x000000FF),
        UInt32(0x0000FF00),
        UInt32(0x00FF0000),
        UInt32(0xFF000000),
        UInt32(0xFFFFFFFF),
    ]
    var expected_indices = [
        UInt32(4),
        UInt32(1),
        UInt32(6),
        UInt32(3),
        UInt32(2),
        UInt32(0),
        UInt32(5),
    ]

    var serial = pairs.copy()
    var parallel = pairs.copy()
    _radix_sort_morton_pairs(serial)
    _radix_sort_morton_pairs_parallel(parallel, 3)

    for i in range(len(pairs)):
        assert_equal(serial[i].code, expected_codes[i])
        assert_equal(serial[i].item_idx, expected_indices[i])
        assert_equal(parallel[i].code, expected_codes[i])
        assert_equal(parallel[i].item_idx, expected_indices[i])


def test_wide_bounds_root_bounds_is_valid() raises:
    var verts = _make_strip[Frame.WORLD](4)
    var items = _make_bounds_items(verts)

    var builder = BinaryBoundsBvh[Frame.WORLD, 4, "median"](items^)

    var wide = BoundsBvh[Frame.WORLD, 4](builder)
    var bounds = wide.root_bounds()

    assert_true(bounds._min.x <= -9.0)
    assert_true(bounds._max.x >= 5.0)
    assert_true(bounds._min.z <= 2.0)
    assert_true(bounds._max.z >= 2.0)


def test_bounds_ray_query_inside_outside_regression() raises:
    var lower = Point3W(0.5, -1.0, -1.0)
    var upper = Point3W(1.0, 1.0, 1.0)

    var query_ray = Rayf32[Frame.WORLD](
        Point3W(0.0, 0.0, 0.0), Vec3W(1.0, 0.0, 0.0)
    )
    var rcp_dir = query_ray.rcp_direction[1]()

    var hit_outside = intersect_ray_aabb(
        Point3W(0.0, 0.0, 0.0),
        rcp_dir,
        lower,
        upper,
        f32_max,
    )
    assert_true(hit_outside.mask, "Rayf32 starting outside failed to hit")

    var hit_inside = intersect_ray_aabb(
        Point3W(0.75, 0.0, 0.0),
        rcp_dir,
        lower,
        upper,
        f32_max,
    )
    assert_true(hit_inside.mask, "Rayf32 starting inside failed to hit")


def test_ray_rcp_direction_uses_finite_parallel_axes() raises:
    var ray = Rayf32[Frame.WORLD](Point3W(0.0), Vec3W(2.0, 0.0, -4.0))
    var rcp_dir = ray.rcp_direction[4]()

    assert_almost_equal(rcp_dir.x, 0.5)
    assert_almost_equal(rcp_dir.y, 1.0e9)
    assert_almost_equal(rcp_dir.z, -0.25)


def test_bounds_item_bounds_and_payload_mapping() raises:
    var bounds = triangle_bounds(
        Point3W(-1.0, 2.0, 3.0),
        Point3W(2.0, -4.0, 5.0),
        Point3W(0.0, 1.0, -6.0),
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
    var verts: List[Point3f32[Frame.WORLD]] = [
        Point3W(-11.0, -1.0, 0.0),
        Point3W(-9.0, -1.0, 0.0),
        Point3W(-10.0, 1.0, 0.0),  # Tri 0, centered near x=-10
        Point3W(9.0, -1.0, 0.0),
        Point3W(11.0, -1.0, 0.0),
        Point3W(10.0, 1.0, 0.0),  # Tri 1, centered near x=10
    ]
    var items = _make_bounds_items(verts)
    var builder = BinaryBoundsBvh[Frame.WORLD, 2, "sah"](items^)
    var centroid_bounds = builder.update_node_bounds_and_centroid_bounds(0)

    var split = _find_sah_split[Frame.WORLD, 16](
        builder.nodes[0],
        centroid_bounds,
        Span(builder.item_indices),
        Span(builder.items),
    )

    assert_true(split.axis == 0)
    assert_true(split.pos > -10.0 and split.pos < 10.0)
    assert_true(split.cost < 20.0)
    assert_true(split.bin >= 0)

    var partition = _partition_items_by_bin[Frame.WORLD, 16](
        Span(builder.item_indices),
        Span(builder.items),
        0,
        2,
        split.axis,
        split.bin,
        split.bin_min,
        split.bin_scale,
    )
    assert_true(partition.split_idx == 1)
    assert_almost_equal(partition.left_bounds._min.x, -11.0)
    assert_almost_equal(partition.left_bounds._max.x, -9.0)
    assert_almost_equal(partition.right_bounds._min.x, 9.0)
    assert_almost_equal(partition.right_bounds._max.x, 11.0)
    assert_almost_equal(partition.left_centroid_bounds._min.x, -10.0)
    assert_almost_equal(partition.right_centroid_bounds._min.x, 10.0)


def test_bounds_sah_degenerate() raises:
    var verts: List[Point3f32[Frame.WORLD]] = [
        Point3W(0.0, 0.0, 0.0),
        Point3W(1.0, 0.0, 0.0),
        Point3W(0.0, 1.0, 0.0),
        Point3W(0.0, 0.0, 0.0),
        Point3W(1.0, 0.0, 0.0),
        Point3W(0.0, 1.0, 0.0),
    ]
    var items = _make_bounds_items(verts)
    var builder = BinaryBoundsBvh[Frame.WORLD, 2, "sah"](items^)
    var centroid_bounds = builder.update_node_bounds_and_centroid_bounds(0)

    var split = _find_sah_split[Frame.WORLD, 16](
        builder.nodes[0],
        centroid_bounds,
        Span(builder.item_indices),
        Span(builder.items),
    )

    assert_true(split.axis == -1)
    assert_true(not split.valid())


def test_bounds_partition_items_non_empty() raises:
    var verts: List[Point3f32[Frame.WORLD]] = [
        Point3W(-11.0, -1.0, 0.0),
        Point3W(-9.0, -1.0, 0.0),
        Point3W(-10.0, 1.0, 0.0),
        Point3W(9.0, -1.0, 0.0),
        Point3W(11.0, -1.0, 0.0),
        Point3W(10.0, 1.0, 0.0),
    ]
    var items = _make_bounds_items(verts)
    var builder = BinaryBoundsBvh[Frame.WORLD, 2, "sah"](items^)

    var split_idx = _partition_items_by_median_center(
        Span(builder.item_indices),
        Span(builder.items),
        0,
        2,
        0,
    )

    assert_true(split_idx == 1)
    assert_true(builder.item_indices[0] != builder.item_indices[1])


def test_triangle_bvh2_leaf_size_equals_width_returns_nearest_triangle() raises:
    var verts = _make_depth_pair[Frame.WORLD]()
    var bvh = TriangleBvh[Frame.WORLD, 2].__init__["median"](verts)

    var hit = bvh.trace[TRACE.CLOSEST_HIT](_z_ray(Point3W(0.0, 0.0, 0.0)))

    assert_true(hit.is_hit())
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 2.0)


def _test_triangle_bvh_matches_bruteforce[
    width: SIMDLength,
    split_mode: String,
]() raises:
    var n = {2: 24, 4: 32, 8: 40}[width]
    var verts = _make_strip[Frame.WORLD](n)
    var bvh = TriangleBvh[Frame.WORLD, width].__init__[split_mode](verts)

    for i in range(n):
        _assert_triangle_bvh_matches_bruteforce[Frame.WORLD, width](
            bvh,
            verts,
            _triangle_center_xy(verts, i),
        )

    for i in range(8):
        _assert_triangle_bvh_matches_bruteforce[Frame.WORLD, width](
            bvh,
            verts,
            Point3W(100.0 + Float32(i), 100.0, 0.0),
        )


def test_triangle_bvh_matches_bruteforce() raises:
    comptime for w in [2, 4, 8]:
        comptime for mode in ["median", "sah", "lbvh", "hploc"]:
            _test_triangle_bvh_matches_bruteforce[w, mode]()


def _test_triangle_bvh16_leaf_width[
    leaf_width: SIMDLength,
    mode: String,
]() raises:
    var n = 48
    var verts = _make_strip[Frame.WORLD](n)
    var bvh = TriangleBvh[Frame.WORLD, 16, leaf_width].__init__[mode](verts)

    assert_true(len(bvh.tree.leaf_ranges) == 0)
    assert_true(len(bvh.tree.item_indices) == 0)
    assert_true(len(bvh.tree.item_payloads) == 0)

    var packed_primitive_count = 0
    for ref block in bvh.leaf_blocks:
        comptime for lane in range(leaf_width):
            if block.prim_indices[lane] != EMPTY_LANE:
                packed_primitive_count += 1
    assert_true(packed_primitive_count == n)

    for ref node in bvh.tree.nodes:
        comptime for lane in range(16):
            var child_ref = node.data[lane]
            if child_ref != EMPTY_LANE and is_leaf_ref(child_ref):
                assert_true(
                    Int(decode_ref_index(child_ref)) < len(bvh.leaf_blocks)
                )

    for i in range(n):
        _assert_triangle_bvh_matches_bruteforce[Frame.WORLD, 16, leaf_width](
            bvh,
            verts,
            _triangle_center_xy(verts, i),
        )

    assert_true(
        bvh.trace[TRACE.ANY_HIT](
            _z_ray(_triangle_center_xy(verts, 0))
        ).is_occluded()
    )
    assert_true(
        not bvh.trace[TRACE.ANY_HIT](
            _z_ray(Point3W(100.0, 100.0, 0.0))
        ).is_occluded()
    )


def test_triangle_bvh16_decoupled_leaf_widths() raises:
    comptime for leaf_width in [2, 4, 8, 16]:
        comptime for mode in ["median", "sah", "lbvh", "hploc"]:
            _test_triangle_bvh16_leaf_width[leaf_width, mode]()


def _assert_packet_hits_equal[
    length: SIMDLength
](actual: Hit[Frame.WORLD, length], expected: Hit[Frame.WORLD, length],) raises:
    comptime for lane in range(length):
        assert_true(actual.prim[lane] == expected.prim[lane])
        assert_true(actual.inst[lane] == expected.inst[lane])
        assert_true(actual.t[lane] == expected.t[lane])
        assert_true(actual.u[lane] == expected.u[lane])
        assert_true(actual.v[lane] == expected.v[lane])
        assert_true(actual.normal.x[lane] == expected.normal.x[lane])
        assert_true(actual.normal.y[lane] == expected.normal.y[lane])
        assert_true(actual.normal.z[lane] == expected.normal.z[lane])


def _test_triangle_packet_paths_match[length: SIMDLength]() raises:
    var verts = _make_strip[Frame.WORLD](64)
    var bvh = TriangleBvh[Frame.WORLD, 16, 16].__init__["sah"](verts)
    var ox = SIMD[DType.float32, length](0.0)
    var oy = SIMD[DType.float32, length](0.0)
    comptime for lane in range(length):
        if lane % 4 == 0:
            ox[lane] = 1000.0 + Float32(lane)
            oy[lane] = 1000.0
        else:
            var center = _triangle_center_xy(verts, (7 * lane) % 64)
            ox[lane] = center.x
            oy[lane] = center.y

    var packet = Ray[DType.float32, Frame.WORLD, length](
        Point3[DType.float32, Frame.WORLD, length](ox, oy, 0.0),
        Vec3[DType.float32, Frame.WORLD, length](0.0, 0.0, 1.0),
    )
    var valid = SIMD[DType.bool, length](fill=True)
    var production = bvh.trace[TRACE.CLOSEST_HIT](packet, valid)
    var common_octant = bvh.trace_packet_common_octant(packet, valid)
    _assert_packet_hits_equal(common_octant, production)


def test_triangle_packet_paths_match() raises:
    _test_triangle_packet_paths_match[4]()
    _test_triangle_packet_paths_match[8]()
    _test_triangle_packet_paths_match[16]()


def _test_coherent_packet_frustum_is_conservative[
    positive_x: Bool,
    positive_y: Bool,
    positive_z: Bool,
]() raises:
    comptime length = 8
    var sign_x = Float32(1.0)
    var sign_y = Float32(1.0)
    var sign_z = Float32(1.0)
    comptime if not positive_x:
        sign_x = -1.0
    comptime if not positive_y:
        sign_y = -1.0
    comptime if not positive_z:
        sign_z = -1.0

    var ox = SIMD[DType.float32, length](0.0)
    var oy = SIMD[DType.float32, length](0.0)
    var oz = SIMD[DType.float32, length](0.0)
    var dx = SIMD[DType.float32, length](0.0)
    var dy = SIMD[DType.float32, length](0.0)
    var dz = SIMD[DType.float32, length](0.0)
    comptime for lane in range(length):
        ox[lane] = -0.7 + 0.2 * Float32(lane)
        oy[lane] = 0.5 - 0.15 * Float32(lane)
        oz[lane] = -0.3 + 0.1 * Float32(lane)
        dx[lane] = sign_x * (0.3 + 0.04 * Float32(lane))
        dy[lane] = sign_y * (0.2 + 0.03 * Float32(lane))
        dz[lane] = sign_z * (0.8 + 0.05 * Float32(lane))

    var rays = Ray[DType.float32, Frame.WORLD, length](
        Point3[DType.float32, Frame.WORLD, length](ox, oy, oz),
        Vec3[DType.float32, Frame.WORLD, length](dx, dy, dz),
    )
    var valid = SIMD[DType.bool, length](fill=True)
    valid[length - 1] = False
    var reciprocal_direction = rays.reciprocal_direction()
    var frustum = _coherent_packet_frustum[
        Frame.WORLD, length, positive_x, positive_y, positive_z
    ](rays, valid, reciprocal_direction)

    var bounds_min = Point3[DType.float32, Frame.WORLD, length](0.0)
    var bounds_max = Point3[DType.float32, Frame.WORLD, length](0.0)
    comptime for child in range(length):
        var distance = 2.0 + Float32(child)
        var center_x = sign_x * distance * 0.4
        var center_y = sign_y * distance * 0.3
        var center_z = sign_z * distance
        if child >= 6:
            center_x += sign_x * 20.0
        bounds_min.x[child] = center_x - 1.0
        bounds_min.y[child] = center_y - 1.0
        bounds_min.z[child] = center_z - 1.0
        bounds_max.x[child] = center_x + 1.0
        bounds_max.y[child] = center_y + 1.0
        bounds_max.z[child] = center_z + 1.0

    var frustum_mask = _intersect_coherent_packet_frustum[
        Frame.WORLD, length, positive_x, positive_y, positive_z
    ](bounds_min, bounds_max, frustum, Float32(f32_max))
    var exact_child_hits = 0
    comptime for child in range(length):
        var exact = intersect_ray_aabb_rcp(
            rays.o,
            reciprocal_direction,
            Point3[DType.float32, Frame.WORLD, length](
                bounds_min.x[child],
                bounds_min.y[child],
                bounds_min.z[child],
            ),
            Point3[DType.float32, Frame.WORLD, length](
                bounds_max.x[child],
                bounds_max.y[child],
                bounds_max.z[child],
            ),
            rays.t_max,
        )
        if (valid & exact.mask).reduce_or():
            exact_child_hits += 1
            assert_true(frustum_mask[child])
    assert_true(exact_child_hits > 0)


def test_coherent_packet_frustum_is_conservative() raises:
    _test_coherent_packet_frustum_is_conservative[True, True, True]()
    _test_coherent_packet_frustum_is_conservative[True, True, False]()
    _test_coherent_packet_frustum_is_conservative[True, False, True]()
    _test_coherent_packet_frustum_is_conservative[True, False, False]()
    _test_coherent_packet_frustum_is_conservative[False, True, True]()
    _test_coherent_packet_frustum_is_conservative[False, True, False]()
    _test_coherent_packet_frustum_is_conservative[False, False, True]()
    _test_coherent_packet_frustum_is_conservative[False, False, False]()


def _test_triangle_bvh_shadow_hit_and_miss[
    width: SIMDLength,
    mode: String,
]() raises:
    var verts = _make_strip[Frame.WORLD](2 * width)
    var bvh = TriangleBvh[Frame.WORLD, width].__init__[mode](verts)

    assert_true(
        bvh.trace[TRACE.ANY_HIT](_z_ray(Point3W(0.0, 0.0, 0.0))).is_occluded()
    )

    assert_true(
        not bvh.trace[TRACE.ANY_HIT](
            _z_ray(Point3W(100.0, 100.0, 0.0))
        ).is_occluded()
    )


def test_triangle_bvh_shadow_hit_and_miss() raises:
    comptime for w in [2, 4, 8]:
        comptime for mode in ["median", "sah", "lbvh", "hploc"]:
            _test_triangle_bvh_shadow_hit_and_miss[w, mode]()


def test_triangle_bvh_measured_trace_matches_normal_and_counts_work() raises:
    var verts = _make_depth_stack[Frame.WORLD](64)
    var bvh = TriangleBvh[Frame.WORLD, 16, 4].__init__["sah"](verts)
    var hit_ray = _z_ray(Point3W(0.0, 0.0, 0.0))

    var normal_hit = bvh.trace[TRACE.CLOSEST_HIT](hit_ray)
    var stats = CpuBvhTraversalStats()
    var measured_hit = bvh.trace_with_stats[TRACE.CLOSEST_HIT](hit_ray, stats)

    assert_true(normal_hit.is_hit() == measured_hit.is_hit())
    assert_true(normal_hit.prim == measured_hit.prim)
    assert_almost_equal(normal_hit.t, measured_hit.t)
    assert_almost_equal(normal_hit.u, measured_hit.u)
    assert_almost_equal(normal_hit.v, measured_hit.v)

    assert_true(stats.rays == 1)
    assert_true(stats.internal_nodes > 0)
    assert_true(stats.nodes_with_hits > 0)
    assert_true(stats.aabb_packet_lanes == 16 * stats.internal_nodes)
    assert_true(stats.active_child_lanes >= stats.aabb_hit_lanes)
    assert_true(stats.leaf_blocks > 0)
    assert_true(stats.primitive_packet_lanes == 4 * stats.leaf_blocks)
    assert_true(stats.valid_primitives > 0)
    assert_true(stats.valid_primitives <= stats.primitive_packet_lanes)
    assert_true(stats.primitive_hit_candidates > 0)
    assert_true(stats.closer_hit_updates > 0)
    assert_true(stats.stack_pushes >= stats.stack_pops)
    assert_true(stats.max_stack_depth <= stats.stack_pushes)

    var miss_ray = _z_ray(Point3W(1000.0, 1000.0, 0.0))
    var normal_miss = bvh.trace[TRACE.CLOSEST_HIT](miss_ray)
    var measured_miss = bvh.trace_with_stats[TRACE.CLOSEST_HIT](miss_ray, stats)
    assert_true(normal_miss.is_hit() == measured_miss.is_hit())
    assert_true(stats.rays == 2)

    var any_stats = CpuBvhTraversalStats()
    var normal_any = bvh.trace[TRACE.ANY_HIT](hit_ray)
    var measured_any = bvh.trace_with_stats[TRACE.ANY_HIT](hit_ray, any_stats)
    assert_true(normal_any.is_occluded() == measured_any.is_occluded())
    assert_true(any_stats.rays == 1)
    assert_true(any_stats.primitive_hit_candidates > 0)
    assert_true(any_stats.any_hit_early_exits == 1)


def test_sphere_bounds() raises:
    var s = Sphere(Point3f32[Frame.WORLD](1.0, 2.0, 3.0), 2.0)
    var b = s.bounds()

    assert_almost_equal(b._min.x, -1.0)
    assert_almost_equal(b._min.y, 0.0)
    assert_almost_equal(b._min.z, 1.0)

    assert_almost_equal(b._max.x, 3.0)
    assert_almost_equal(b._max.y, 4.0)
    assert_almost_equal(b._max.z, 5.0)


def test_sphere_bvh4_single_leaf_layout_and_hit() raises:
    var spheres = _make_spheres[Frame.WORLD]()
    var bvh = SphereBvh[Frame.WORLD, 4](spheres^)

    assert_true(len(bvh.tree.nodes) == 1)
    assert_true(len(bvh.tree.leaf_ranges) == 0)
    assert_true(len(bvh.tree.item_indices) == 0)
    assert_true(len(bvh.tree.item_payloads) == 0)
    assert_true(bvh.tree.nodes[0].data[0] == encode_leaf_ref(0))
    assert_true(len(bvh.leaf_blocks) == 1)
    assert_true(bvh.leaf_blocks[0].prim_indices[0] == 0)
    assert_true(bvh.leaf_blocks[0].prim_indices[1] == 1)
    assert_true(bvh.leaf_blocks[0].prim_indices[2] == 2)
    assert_true(bvh.leaf_blocks[0].prim_indices[3] == 3)

    var hit = bvh.trace[TRACE.CLOSEST_HIT](_z_ray(Point3W(0.0, 0.0, 0.0)))

    assert_true(hit.is_hit())
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 1.0)


def _test_sphere_bvh_matches_bruteforce[
    width: SIMDLength,
    mode: String,
]() raises:
    var spheres = _make_spheres[Frame.WORLD]()
    var bvh = SphereBvh[Frame.WORLD, width].__init__[mode](spheres.copy())

    _assert_sphere_bvh_matches_bruteforce[Frame.WORLD, width](
        bvh,
        spheres,
        Point3W(0.0, 0.0, 0.0),
    )
    _assert_sphere_bvh_matches_bruteforce[Frame.WORLD, width](
        bvh,
        spheres,
        Point3W(4.0, 0.0, 0.0),
    )
    _assert_sphere_bvh_matches_bruteforce[Frame.WORLD, width](
        bvh,
        spheres,
        Point3W(-4.0, 0.0, 0.0),
    )
    _assert_sphere_bvh_matches_bruteforce[Frame.WORLD, width](
        bvh,
        spheres,
        Point3W(100.0, 0.0, 0.0),
    )


def test_sphere_bvh_matches_bruteforce() raises:
    comptime for w in [2, 4, 8]:
        comptime for mode in ["median", "sah", "lbvh", "hploc"]:
            _test_sphere_bvh_matches_bruteforce[w, mode]()


def _test_sphere_bvh_shadow_hit_and_miss[
    width: SIMDLength,
    mode: String,
]() raises:
    var spheres = _make_spheres[Frame.WORLD]()
    var bvh = SphereBvh[Frame.WORLD, width].__init__[mode](spheres^)

    assert_true(
        bvh.trace[TRACE.ANY_HIT](_z_ray(Point3W(0.0, 0.0, 0.0))).is_occluded()
    )

    assert_true(
        not bvh.trace[TRACE.ANY_HIT](
            _z_ray(Point3W(100.0, 0.0, 0.0))
        ).is_occluded()
    )


def test_sphere_bvh_shadow_hit_and_miss() raises:
    comptime for w in [2, 4, 8]:
        comptime for mode in ["median", "sah", "lbvh", "hploc"]:
            _test_sphere_bvh_shadow_hit_and_miss[w, mode]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
