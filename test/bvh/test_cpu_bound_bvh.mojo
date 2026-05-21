from std.testing import TestSuite, assert_true, assert_almost_equal
from std.math import sqrt
from std.utils.numerics import max_finite

from bajo.core.aabb import AABB
from bajo.core.vec import Vec3f32, Vec3
from bajo.core.intersect import intersect_ray_tri, intersect_ray_aabb
from bajo.bvh.types import Ray, Sphere, Instance, Hit
from bajo.core.random import Rng
from bajo.bvh.constants import EMPTY_LANE, TRACE_CLOSEST_HIT, TRACE_ANY_HIT
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvhBuilder,
    BoundsItem,
    BoundsBvh,
)
from bajo.bvh.cpu.builder.builder import _partition_items_by_center
from bajo.bvh.cpu.builder.sah import _sah_items, _partition_items_by_bin
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.cpu.tlas import Tlas


def _rng_f32(mut rng: Rng, lo: Float32, hi: Float32) -> Float32:
    return lo + (hi - lo) * rng.f32()


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


def _make_tri_bounds(
    v0: Vec3f32,
    v1: Vec3f32,
    v2: Vec3f32,
) -> AABB:
    var bounds = AABB.invalid()
    bounds.grow(v0)
    bounds.grow(v1)
    bounds.grow(v2)
    return bounds


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

        items.append(BoundsItem(_make_tri_bounds(v0, v1, v2), UInt32(i)))

    return items^


def _make_spheres() -> List[Sphere]:
    return [
        Sphere(Vec3f32(0.0, 0.0, 2.0), 1.0),
        Sphere(Vec3f32(4.0, 0.0, 4.0), 1.0),
        Sphere(Vec3f32(-4.0, 0.0, 6.0), 1.0),
        Sphere(Vec3f32(0.0, 4.0, 8.0), 1.0),
    ]


def _brute_trace(
    verts: List[Vec3f32],
    O: Vec3f32,
    D: Vec3f32,
) -> Hit:
    var best_hit = Hit.miss()

    for i in range(len(verts) / 3):
        ref v0 = verts[i * 3 + 0]
        ref v1 = verts[i * 3 + 1]
        ref v2 = verts[i * 3 + 2]

        var h = intersect_ray_tri(
            O,
            D,
            v0,
            v1,
            v2,
            best_hit.t,
        )

        if h.mask and h.t < best_hit.t:
            best_hit.t = h.t
            best_hit.u = h.u
            best_hit.v = h.v
            best_hit.prim = UInt32(i)
            best_hit.inst = EMPTY_LANE
            best_hit.occluded = UInt32(0)

    return best_hit


def _brute_sphere_trace(
    spheres: List[Sphere],
    O: Vec3f32,
    D: Vec3f32,
) -> Hit:
    var best_hit = Hit.miss()

    for i in range(len(spheres)):
        ref s = spheres[i]

        var oc = O - s.center
        var a = D.x * D.x + D.y * D.y + D.z * D.z
        var b = 2.0 * (oc.x * D.x + oc.y * D.y + oc.z * D.z)
        var c = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - s.radius * s.radius

        var det = b * b - 4.0 * a * c
        if det < 0.0:
            continue

        # These test rays are axis-aligned and use simple sphere positions.
        # The direct formula is fine for the brute-force oracle.
        var sqrt_det = sqrt(det)
        var t0 = (-b - sqrt_det) / (2.0 * a)
        var t1 = (-b + sqrt_det) / (2.0 * a)

        var t = t0
        if t <= 0.0:
            t = t1

        if t > 0.0 and t < best_hit.t:
            best_hit.t = t
            best_hit.u = 0.0
            best_hit.v = 0.0
            best_hit.prim = UInt32(i)
            best_hit.inst = EMPTY_LANE
            best_hit.occluded = 0

    return best_hit


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
                "leaf_size invariant violated",
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
    width: Int
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
    width: Int
](
    mut bvh: TriangleBvh[width],
    verts: List[Vec3f32],
    O: Vec3f32,
    D: Vec3f32,
) raises:
    var ray = Ray(O, D)
    var hit = bvh.trace[TRACE_CLOSEST_HIT](ray)

    var brute = _brute_trace(verts, O, D)
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
    width: Int
](
    mut bvh: SphereBvh[width],
    spheres: List[Sphere],
    O: Vec3f32,
    D: Vec3f32,
) raises:
    var ray = Ray(O, D)
    var hit = bvh.trace[TRACE_CLOSEST_HIT](ray)

    var brute = _brute_sphere_trace(spheres, O, D)
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


def _make_identity_triangle_instance[
    blas_width: Int
](blas_idx: UInt32, blas: TriangleBvh[blas_width]) -> Instance:
    var inst = Instance()
    inst.blas_idx = blas_idx
    inst.bounds = blas.bounds()
    return inst^


def _make_identity_sphere_instance[
    blas_width: Int
](blas_idx: UInt32, blas: SphereBvh[blas_width]) -> Instance:
    var inst = Instance()
    inst.blas_idx = blas_idx
    inst.bounds = blas.bounds()
    return inst^


def test_bounds_bvh_empty_input() raises:
    var items = List[BoundsItem]()
    var builder = BoundsBvhBuilder[4](items)
    builder.build["median"]()

    assert_true(builder.item_count == 0)
    assert_true(builder.nodes_used == 0)

    var wide = BoundsBvh[4](builder)
    assert_true(len(wide.nodes) == 0)


def test_bounds_bvh_leaf_size_2_invariant() raises:
    var verts = _make_strip(12)
    var items = _make_bounds_items(verts)

    var builder = BoundsBvhBuilder[2](items)
    builder.build["median"]()

    assert_true(builder.nodes_used > 0)
    _assert_builder_leaf_sizes_at_most(builder, UInt32(2))

    var wide = BoundsBvh[2](builder)
    _assert_wide_leaf_counts_at_most_width[2](wide)


def test_bounds_bvh_leaf_size_4_invariant() raises:
    var verts = _make_strip(24)
    var items = _make_bounds_items(verts)

    var builder = BoundsBvhBuilder[4](items)
    builder.build["median"]()

    assert_true(builder.nodes_used > 0)
    _assert_builder_leaf_sizes_at_most(builder, UInt32(4))

    var wide = BoundsBvh[4](builder)
    _assert_wide_leaf_counts_at_most_width[4](wide)


def test_bounds_bvh_leaf_size_8_invariant() raises:
    var verts = _make_strip(40)
    var items = _make_bounds_items(verts)

    var builder = BoundsBvhBuilder[8](items)
    builder.build["median"]()

    assert_true(builder.nodes_used > 0)
    _assert_builder_leaf_sizes_at_most(builder, UInt32(8))

    var wide = BoundsBvh[8](builder)
    _assert_wide_leaf_counts_at_most_width[8](wide)


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


def test_triangle_bvh2_leaf_size_equals_width_returns_nearest_triangle() raises:
    var verts = _make_depth_pair()
    var bvh = TriangleBvh[2].__init__["median"](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = bvh.trace[TRACE_CLOSEST_HIT](ray)

    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 2.0)


def test_triangle_bvh2_leaf_size_equals_width_matches_bruteforce() raises:
    var verts = _make_strip(24)
    var bvh = TriangleBvh[2].__init__["median"](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    for i in range(24):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[2](bvh, verts, O, D)

    for i in range(8):
        var O = Vec3f32(100.0 + Float32(i), 100.0, 0.0)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[2](bvh, verts, O, D)


def test_triangle_bvh4_leaf_size_equals_width_matches_bruteforce() raises:
    var verts = _make_strip(32)
    var bvh = TriangleBvh[4].__init__["median"](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    for i in range(32):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[4](bvh, verts, O, D)

    for i in range(8):
        var O = Vec3f32(-100.0 - Float32(i), 100.0, 0.0)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[4](bvh, verts, O, D)


def test_triangle_bvh8_leaf_size_equals_width_matches_bruteforce() raises:
    var verts = _make_strip(40)
    var bvh = TriangleBvh[8].__init__["median"](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    for i in range(40):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[8](bvh, verts, O, D)

    for i in range(8):
        var O = Vec3f32(100.0, -100.0 - Float32(i), 0.0)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[8](bvh, verts, O, D)


def test_triangle_bvh4_shadow_hit_and_miss() raises:
    var verts = _make_strip(8)
    var bvh = TriangleBvh[4].__init__["median"](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    # Primitive 4 is centered at x = 0.
    var ray_hit = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(bvh.trace[TRACE_ANY_HIT](ray_hit).is_occluded())

    var ray_miss = Ray(Vec3f32(100.0, 100.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(not bvh.trace[TRACE_ANY_HIT](ray_miss).is_occluded())


def test_triangle_bvh4_sah_leaf_size_equals_width_matches_bruteforce() raises:
    var verts = _make_strip(48)
    var bvh = TriangleBvh[4].__init__["sah"](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    for i in range(48):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[4](bvh, verts, O, D)


def test_sphere_bounds() raises:
    var s = Sphere(Vec3f32(1.0, 2.0, 3.0), 2.0)
    var b = s.bounds()

    assert_almost_equal(b._min.x, -1.0)
    assert_almost_equal(b._min.y, 0.0)
    assert_almost_equal(b._min.z, 1.0)

    assert_almost_equal(b._max.x, 3.0)
    assert_almost_equal(b._max.y, 4.0)
    assert_almost_equal(b._max.z, 5.0)


def test_sphere_bvh4_returns_nearest_sphere() raises:
    var spheres = _make_spheres()
    var bvh = SphereBvh[4](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )

    assert_true(len(bvh.tree.nodes) == 1)
    assert_true(bvh.tree.nodes[0].counts[0] == 4)
    assert_true(bvh.tree.nodes[0].data[0] == 0)

    var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = bvh.trace[TRACE_CLOSEST_HIT](ray)

    assert_true(hit.t < Float32(1e20), "SphereBvh did not hit")
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 1.0)


def test_sphere_bvh2_matches_bruteforce() raises:
    var spheres = _make_spheres()
    var bvh = SphereBvh[2](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )

    _assert_sphere_bvh_matches_bruteforce[2](
        bvh,
        spheres,
        Vec3f32(0.0, 0.0, 0.0),
        Vec3f32(0.0, 0.0, 1.0),
    )
    _assert_sphere_bvh_matches_bruteforce[2](
        bvh,
        spheres,
        Vec3f32(4.0, 0.0, 0.0),
        Vec3f32(0.0, 0.0, 1.0),
    )
    _assert_sphere_bvh_matches_bruteforce[2](
        bvh,
        spheres,
        Vec3f32(100.0, 0.0, 0.0),
        Vec3f32(0.0, 0.0, 1.0),
    )


def test_sphere_bvh4_shadow_hit_and_miss() raises:
    var spheres = _make_spheres()
    var bvh = SphereBvh[4](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )

    var ray_hit = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    hit = bvh.trace[TRACE_ANY_HIT](ray_hit)
    assert_true(hit.is_occluded())

    var ray_miss = Ray(Vec3f32(100.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    hit = bvh.trace[TRACE_ANY_HIT](ray_miss)
    assert_true(not hit.is_occluded())


def test_sphere_bvh4_sah_matches_bruteforce() raises:
    var spheres = _make_spheres()
    var bvh = SphereBvh[4].__init__["sah"](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )

    _assert_sphere_bvh_matches_bruteforce[4](
        bvh,
        spheres,
        Vec3f32(-4.0, 0.0, 0.0),
        Vec3f32(0.0, 0.0, 1.0),
    )
    _assert_sphere_bvh_matches_bruteforce[4](
        bvh,
        spheres,
        Vec3f32(0.0, 4.0, 0.0),
        Vec3f32(0.0, 0.0, 1.0),
    )


def test_tlas_triangle_single_instance_matches_blas() raises:
    var verts = _make_strip(8)

    var blas = TriangleBvh[4].__init__["median"](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    var blases = List[TriangleBvh[4]](capacity=1)
    blases.append(blas.copy())

    var instances = List[Instance](capacity=1)
    instances.append(_make_identity_triangle_instance[4](0, blas))

    var tlas = Tlas[4](instances)

    var ray_blas = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var ray_tlas = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))

    var hit_blas = blas.trace[TRACE_CLOSEST_HIT](ray_blas)
    var hit_tlas = tlas.trace_triangles[TRACE_CLOSEST_HIT, 4](
        ray_tlas, blases.unsafe_ptr()
    )

    assert_true(hit_tlas.prim == hit_blas.prim)
    assert_true(hit_tlas.inst == 0)
    assert_almost_equal(hit_tlas.t, hit_blas.t)


def test_tlas_triangle_two_instances_returns_nearest_instance() raises:
    var near_verts = _make_depth_pair()

    var far_verts = List[Vec3f32](capacity=6)
    far_verts.append(Vec3f32(-1.0, -1.0, 8.0))
    far_verts.append(Vec3f32(1.0, -1.0, 8.0))
    far_verts.append(Vec3f32(0.0, 1.0, 8.0))
    far_verts.append(Vec3f32(-1.0, -1.0, 10.0))
    far_verts.append(Vec3f32(1.0, -1.0, 10.0))
    far_verts.append(Vec3f32(0.0, 1.0, 10.0))

    var near_blas = TriangleBvh[4].__init__["median"](
        near_verts.unsafe_ptr(),
        UInt32(len(near_verts) / 3),
    )
    var far_blas = TriangleBvh[4].__init__["median"](
        far_verts.unsafe_ptr(),
        UInt32(len(far_verts) / 3),
    )

    var blases = List[TriangleBvh[4]](capacity=2)
    blases.append(near_blas.copy())
    blases.append(far_blas.copy())

    var instances = List[Instance](capacity=2)
    instances.append(_make_identity_triangle_instance[4](0, near_blas))
    instances.append(_make_identity_triangle_instance[4](1, far_blas))

    var tlas = Tlas[4](instances)

    var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = tlas.trace_triangles[TRACE_CLOSEST_HIT, 4](
        ray, blases.unsafe_ptr()
    )

    assert_true(hit.inst == 0)
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 2.0)


def test_tlas_triangle_shadow_hit_and_miss() raises:
    var verts = _make_strip(8)

    var blas = TriangleBvh[4].__init__["median"](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    var blases = List[TriangleBvh[4]](capacity=1)
    blases.append(blas.copy())

    var instances = List[Instance](capacity=1)
    instances.append(_make_identity_triangle_instance[4](0, blas))

    var tlas = Tlas[4](instances)

    var ray_hit = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(
        tlas.trace_triangles[TRACE_ANY_HIT, 4](
            ray_hit, blases.unsafe_ptr()
        ).is_occluded()
    )

    var ray_miss = Ray(Vec3f32(100.0, 100.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(
        not tlas.trace_triangles[TRACE_ANY_HIT, 4](
            ray_miss, blases.unsafe_ptr()
        ).is_occluded()
    )


def test_tlas_sphere_single_instance_matches_blas() raises:
    var spheres = _make_spheres()

    var blas = SphereBvh[4](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )

    var blases = List[SphereBvh[4]](capacity=1)
    blases.append(blas.copy())

    var instances = List[Instance](capacity=1)
    instances.append(_make_identity_sphere_instance[4](0, blas))

    var tlas = Tlas[4](instances)

    var ray_blas = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var ray_tlas = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))

    var hit_blas = blas.trace[TRACE_CLOSEST_HIT](ray_blas)
    var hit_tlas = tlas.trace_spheres[TRACE_CLOSEST_HIT, 4](
        ray_tlas, blases.unsafe_ptr()
    )

    assert_true(hit_tlas.prim == hit_blas.prim)
    assert_true(hit_tlas.inst == 0)
    assert_almost_equal(hit_tlas.t, hit_blas.t)


def test_tlas_sphere_two_instances_returns_nearest_instance() raises:
    var near_spheres = [Sphere(Vec3f32(0.0, 0.0, 2.0), 1.0)]
    var far_spheres = [Sphere(Vec3f32(0.0, 0.0, 8.0), 1.0)]

    var near_blas = SphereBvh[4](
        near_spheres.unsafe_ptr(),
        UInt32(len(near_spheres)),
    )
    var far_blas = SphereBvh[4](
        far_spheres.unsafe_ptr(),
        UInt32(len(far_spheres)),
    )

    var blases = List[SphereBvh[4]](capacity=2)
    blases.append(near_blas.copy())
    blases.append(far_blas.copy())

    var instances = List[Instance](capacity=2)
    instances.append(_make_identity_sphere_instance[4](0, near_blas))
    instances.append(_make_identity_sphere_instance[4](1, far_blas))

    var tlas = Tlas[4](instances)

    var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = tlas.trace_spheres[TRACE_CLOSEST_HIT, 4](ray, blases.unsafe_ptr())

    assert_true(hit.inst == 0)
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 1.0)


def test_tlas_sphere_shadow_hit_and_miss() raises:
    var spheres = _make_spheres()

    var blas = SphereBvh[4](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )

    var blases = List[SphereBvh[4]](capacity=1)
    blases.append(blas.copy())

    var instances = List[Instance](capacity=1)
    instances.append(_make_identity_sphere_instance[4](0, blas))

    var tlas = Tlas[4](instances)

    var ray_hit = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(
        tlas.trace_spheres[TRACE_ANY_HIT, 4](
            ray_hit, blases.unsafe_ptr()
        ).is_occluded()
    )

    var ray_miss = Ray(Vec3f32(100.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(
        not tlas.trace_spheres[TRACE_ANY_HIT, 4](
            ray_miss, blases.unsafe_ptr()
        ).is_occluded()
    )


def test_bounds_ray_query_inside_outside_regression() raises:
    var lower = Vec3f32(0.5, -1.0, -1.0)
    var upper = Vec3f32(1.0, 1.0, 1.0)

    var query_dir = Vec3f32(1.0, 0.0, 0.0)
    var rcp_dir = 1.0 / query_dir

    var query_start_outside = Vec3f32(0.0, 0.0, 0.0)
    var hit_outside = intersect_ray_aabb(
        query_start_outside,
        rcp_dir,
        lower,
        upper,
        Float32(1e30),
    )
    assert_true(hit_outside.mask, "Ray starting outside failed to hit")

    var query_start_inside = Vec3f32(0.75, 0.0, 0.0)
    var hit_inside = intersect_ray_aabb(
        query_start_inside,
        rcp_dir,
        lower,
        upper,
        Float32(1e30),
    )
    assert_true(hit_inside.mask, "Ray starting inside failed to hit")


def test_bounds_item_bounds_and_payload_mapping() raises:
    var bounds = _make_tri_bounds(
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

    var split = _sah_items(
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

    var split = _sah_items(
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

    var split_idx = _partition_items_by_center(
        builder.item_indices.unsafe_ptr(),
        builder.items.unsafe_ptr(),
        0,
        2,
        0,
        0.0,
    )

    assert_true(split_idx == 1)
    assert_true(builder.item_indices[0] != builder.item_indices[1])


def test_triangle_bvh4_median_matches_bruteforce_many_random_rays() raises:
    var verts = _make_random_xy_triangles(64, UInt64(12345))
    var bvh = TriangleBvh[4].__init__["median"](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    for i in range(64):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[4](bvh, verts, O, D)

    for i in range(16):
        var O = Vec3f32(100.0 + Float32(i), 100.0, 0.0)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[4](bvh, verts, O, D)


def test_triangle_bvh4_sah_matches_bruteforce_many_random_rays() raises:
    var verts = _make_random_xy_triangles(96, UInt64(98765))
    var bvh = TriangleBvh[4].__init__["sah"](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    for i in range(96):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[4](bvh, verts, O, D)

    for i in range(24):
        var O = Vec3f32(-100.0 - Float32(i), 100.0, 0.0)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[4](bvh, verts, O, D)


def test_triangle_bvh4_sah_reports_original_primitive_after_reorder() raises:
    var verts = _make_strip(8)
    var bvh = TriangleBvh[4].__init__["sah"](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    # Aim at the triangle with original primitive id 6.
    # _make_strip centers primitive i at x = i * 4 - count * 2.
    var ray = Ray(Vec3f32(8.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = bvh.trace[TRACE_CLOSEST_HIT](ray)

    assert_true(hit.prim == 6)
    assert_almost_equal(hit.t, 2.0)


def test_bounds_bvh_lbvh_leaf_size_4_invariant() raises:
    var verts = _make_random_xy_triangles(96, UInt64(606060))
    var items = _make_bounds_items(verts)

    var builder = BoundsBvhBuilder[4](items)
    builder.build["lbvh"]()

    assert_true(builder.nodes_used > 1)
    assert_true(Int(builder.nodes_used) <= len(builder.nodes))
    _assert_builder_leaf_sizes_at_most(builder, UInt32(4))

    var wide = BoundsBvh[4](builder)
    _assert_wide_leaf_counts_at_most_width[4](wide)

    assert_true(len(verts) == 96 * 3)


def test_triangle_bvh4_lbvh_matches_bruteforce_many_random_rays() raises:
    var verts = _make_random_xy_triangles(128, UInt64(707070))
    var bvh = TriangleBvh[4].__init__["lbvh"](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    for i in range(128):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[4](bvh, verts, O, D)

    for i in range(32):
        var O = Vec3f32(100.0 + Float32(i), 100.0, 0.0)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[4](bvh, verts, O, D)

    assert_true(len(verts) == 128 * 3)


def test_triangle_bvh8_lbvh_matches_bruteforce_many_random_rays() raises:
    var verts = _make_random_xy_triangles(128, UInt64(808080))
    var bvh = TriangleBvh[8].__init__["lbvh"](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    for i in range(128):
        var O = _triangle_center_xy(verts, i)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[8](bvh, verts, O, D)

    for i in range(32):
        var O = Vec3f32(100.0, -100.0 - Float32(i), 0.0)
        var D = Vec3f32(0.0, 0.0, 1.0)
        _assert_triangle_bvh_matches_bruteforce[8](bvh, verts, O, D)

    assert_true(len(verts) == 128 * 3)


def test_sphere_bvh4_lbvh_matches_bruteforce() raises:
    var spheres = _make_spheres()
    var bvh = SphereBvh[4].__init__["lbvh"](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )

    _assert_sphere_bvh_matches_bruteforce[4](
        bvh,
        spheres,
        Vec3f32(0.0, 0.0, 0.0),
        Vec3f32(0.0, 0.0, 1.0),
    )
    _assert_sphere_bvh_matches_bruteforce[4](
        bvh,
        spheres,
        Vec3f32(4.0, 0.0, 0.0),
        Vec3f32(0.0, 0.0, 1.0),
    )
    _assert_sphere_bvh_matches_bruteforce[4](
        bvh,
        spheres,
        Vec3f32(-4.0, 0.0, 0.0),
        Vec3f32(0.0, 0.0, 1.0),
    )
    _assert_sphere_bvh_matches_bruteforce[4](
        bvh,
        spheres,
        Vec3f32(100.0, 0.0, 0.0),
        Vec3f32(0.0, 0.0, 1.0),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
