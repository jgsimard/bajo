"""Exact closed-component pruning against unpruned winding accumulation."""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
)
from std.memory import bitcast
from bajo.core import Point3f32, Vec3f32, AABB
from bajo.bvh.spatial import SpatialMesh, MeshQueryView
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.spatial.winding import WindingLayout


def append_tetra(
    mut points: List[Point3f32[.WORLD]],
    mut indices: List[UInt32],
    origin: Point3f32[.WORLD],
    scale: Float32,
    reverse: Bool = False,
    bad_orientation: Bool = False,
    duplicate_face: Bool = False,
):
    var base = UInt32(len(points))
    points.extend(
        [
            origin,
            origin + Vec3f32[.WORLD](scale, 0.0, 0.0),
            origin + Vec3f32[.WORLD](0.0, scale, 0.0),
            origin + Vec3f32[.WORLD](0.0, 0.0, scale),
        ]
    )
    var faces: List[UInt32] = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
    for face in range(4):
        var b = faces[face * 3 + 1]
        var c = faces[face * 3 + 2]
        if reverse or (bad_orientation and face == 0):
            var tmp = b
            b = c
            c = tmp
        indices.extend([base + faces[face * 3], base + b, base + c])
    if duplicate_face:
        indices.extend([base, base + 2, base + 1])


def check_closed_components[width: SIMDLength]() raises:
    var points = List[Point3f32[.WORLD]]()
    var indices = List[UInt32]()
    append_tetra(points, indices, Point3f32[.WORLD](0.0), 4.0)
    append_tetra(points, indices, Point3f32[.WORLD](0.5), 1.0, reverse=True)
    append_tetra(points, indices, Point3f32[.WORLD](8.0, 0.0, 0.0), 1.0)
    append_tetra(
        points,
        indices,
        Point3f32[.WORLD](30.0, 0.0, 0.0),
        1.0,
        bad_orientation=True,
    )
    append_tetra(
        points,
        indices,
        Point3f32[.WORLD](40.0, 0.0, 0.0),
        1.0,
        duplicate_face=True,
    )
    var base = UInt32(len(points))
    points.extend(
        [
            Point3f32[.WORLD](20.0, 0.0, 2.0),
            Point3f32[.WORLD](21.0, 0.0, 2.0),
            Point3f32[.WORLD](20.0, 1.0, 2.0),
        ]
    )
    indices.extend([base, base + 1, base + 2])
    indices.extend([base, base, base + 1])
    var groups = List[UInt32]()
    for face in range(len(indices) // 3):
        groups.append(UInt32(face % 3))
    var mesh = SpatialMesh[.WORLD, width](points, indices, groups)
    assert_equal(len(mesh.winding.bvh.item_bounds), 3)
    var cache_address = Int(mesh.packed_triangles.unsafe_ptr())
    var bounds_address = Int(mesh.winding.bvh.bounds.unsafe_ptr())
    var nodes_address = Int(mesh.winding.bvh.nodes.unsafe_ptr())
    var offsets_address = Int(mesh.winding.offsets.unsafe_ptr())
    for phase in range(3):
        if phase > 0:
            for vertex in range(len(points)):
                points[vertex] = points[vertex] + Vec3f32[.WORLD](
                    0.07,
                    Float32(vertex % 4) * 0.013,
                    -Float32(vertex % 3) * 0.031,
                )
            if phase == 1:
                mesh.refit(points)
            else:
                mesh.rebuild(points)
        assert_equal(Int(mesh.packed_triangles.unsafe_ptr()), cache_address)
        assert_equal(Int(mesh.winding.bvh.bounds.unsafe_ptr()), bounds_address)
        assert_equal(Int(mesh.winding.bvh.nodes.unsafe_ptr()), nodes_address)
        assert_equal(Int(mesh.winding.offsets.unsafe_ptr()), offsets_address)
        for component in range(3):
            var box = AABB[.WORLD].invalid()
            for face in range(len(indices) // 3):
                if mesh.winding.face_components[face] == UInt32(component):
                    box = AABB[.WORLD].merge(box, mesh.bvh.item_bounds[face])
            assert_true(
                box._min == mesh.winding.bvh.item_bounds[component]._min
            )
            assert_true(
                box._max == mesh.winding.bvh.item_bounds[component]._max
            )
        var roots: List[UInt32] = [
            0,
            mesh.group_root(0),
            mesh.group_root(1),
            mesh.group_root(2),
            mesh.group_root(99),
        ]
        for root in roots:
            var view = mesh.view()
            var reference = MeshQueryView[origin_of(mesh), .WORLD, width](
                view.points,
                view.indices,
                view.velocities,
                view.bvh.copy(),
            )
            for q in range(128):
                var p = Point3f32[.WORLD](
                    Float32(q % 23) * 2.1 - 2.0,
                    Float32(q % 7) * 0.17 - 0.1,
                    Float32(q % 5) * 0.19 + 0.037,
                )
                assert_almost_equal(
                    view.winding_number(p, root),
                    reference.winding_number[1](p, root),
                    atol=1e-12,
                    rtol=1e-12,
                )
            var inside = Point3f32[.WORLD](0.6 + Float32(phase) * 0.07)
            assert_almost_equal(
                view.winding_number(inside, root),
                reference.winding_number[1](inside, root),
                atol=1e-12,
                rtol=1e-12,
            )
    var previous = mesh.winding.bvh.bounds.copy()
    var cache = mesh.packed_triangles.copy()
    points[0].x = bitcast[.float32](UInt32(0x7FC00000))
    var rejected = False
    try:
        mesh.refit(points)
    except:
        rejected = True
    assert_true(rejected)
    for i in range(len(previous)):
        assert_equal(mesh.winding.bvh.bounds[i], previous[i])
    for i in range(len(cache)):
        assert_equal(mesh.packed_triangles[i], cache[i])


def test_closed_components_open_orientation_nonmanifold_groups_updates_and_widths() raises:
    check_closed_components[1]()
    check_closed_components[2]()
    check_closed_components[4]()
    check_closed_components[8]()
    check_closed_components[16]()


def test_opposite_closed_components_cancel_and_outside_bounds_contribute_zero() raises:
    var points = List[Point3f32[.WORLD]]()
    var indices = List[UInt32]()
    append_tetra(points, indices, Point3f32[.WORLD](0.0), 4.0)
    append_tetra(points, indices, Point3f32[.WORLD](0.5), 1.0, reverse=True)
    var mesh = SpatialMesh(points, indices)
    assert_almost_equal(
        mesh.view().winding_number(Point3f32[.WORLD](0.6)),
        Float64(0.0),
        atol=1e-12,
    )
    assert_almost_equal(
        mesh.view().winding_number(Point3f32[.WORLD](0.1)),
        Float64(1.0),
        atol=1e-12,
    )
    assert_equal(
        mesh.view().winding_number(Point3f32[.WORLD](5.0)), Float64(0.0)
    )
    # Winding preserves signed multiplicity, unlike crossing parity.
    for face in range(4, 8):
        var tmp = indices[face * 3 + 1]
        indices[face * 3 + 1] = indices[face * 3 + 2]
        indices[face * 3 + 2] = tmp
    var overlapping = SpatialMesh(points, indices)
    assert_almost_equal(
        overlapping.view().winding_number(Point3f32[.WORLD](0.6)),
        Float64(2.0),
        atol=1e-12,
    )


def test_components_touching_only_at_one_vertex_remain_independent() raises:
    var points = List[Point3f32[.WORLD]]()
    var indices = List[UInt32]()
    append_tetra(points, indices, Point3f32[.WORLD](0.0), 1.0)
    append_tetra(points, indices, Point3f32[.WORLD](0.0), -1.0)
    for i in range(12, len(indices)):
        if indices[i] == 4:
            indices[i] = 0
    var mesh = SpatialMesh(points, indices)
    assert_equal(len(mesh.winding.bvh.item_bounds), 2)
    assert_almost_equal(
        mesh.view().winding_number(Point3f32[.WORLD](0.1)),
        Float64(1.0),
        atol=1e-12,
    )
    assert_almost_equal(
        mesh.view().winding_number(Point3f32[.WORLD](-0.1)),
        Float64(-1.0),
        atol=1e-12,
    )


def test_uncached_winding_queries_and_position_updates() raises:
    var points = List[Point3f32[.WORLD]]()
    var indices = List[UInt32]()
    append_tetra(points, indices, Point3f32[.WORLD](0.0), 1.0)
    append_tetra(points, indices, Point3f32[.WORLD](2.0), 1.0, reverse=True)
    append_tetra(
        points, indices, Point3f32[.WORLD](4.0), 1.0, bad_orientation=True
    )
    var groups = List[UInt32]()
    for face in range(len(indices) // 3):
        groups.append(UInt32(face % 3))
    var cached = SpatialMesh(points, indices, groups)
    var uncached = SpatialMesh(points, indices, groups, cache_winding=False)
    assert_equal(len(uncached.packed_triangles), 0)
    assert_equal(len(uncached.winding.face_components), 0)
    for phase in range(3):
        if phase > 0:
            for i in range(len(points)):
                points[i] = points[i] + Vec3f32[.WORLD](
                    0.11, Float32(i % 3) * 0.02, -0.07
                )
            if phase == 1:
                cached.refit(points)
                uncached.refit(points)
            else:
                cached.rebuild(points)
                uncached.rebuild(points)
        for i in range(len(cached.bvh.bounds)):
            assert_equal(uncached.bvh.bounds[i], cached.bvh.bounds[i])
        for scope in range(4):
            var root = UInt32(0) if scope == 0 else uncached.group_root(
                UInt32(scope - 1)
            )
            var expected_root = UInt32(0) if scope == 0 else cached.group_root(
                UInt32(scope - 1)
            )
            for q in range(64):
                var point = Point3f32[.WORLD](
                    Float32(q % 13) * 0.51 - 0.17,
                    Float32(q % 7) * 0.12 + 0.03,
                    Float32(q % 5) * 0.19 - 0.02,
                )
                assert_almost_equal(
                    uncached.view().winding_number(point, root),
                    cached.view().winding_number(point, expected_root),
                    atol=1e-12,
                    rtol=1e-12,
                )
                var actual = uncached.view().query_point_no_sign(
                    point, 100.0, root
                )
                var expected = cached.view().query_point_no_sign(
                    point, 100.0, expected_root
                )
                assert_equal(actual.result, expected.result)
                assert_equal(actual.face, expected.face)
                assert_almost_equal(
                    actual.distance, expected.distance, atol=1e-12, rtol=1e-12
                )
        assert_equal(len(uncached.packed_triangles), 0)


def test_winding_extreme_edge_ids_degenerate_and_nonmanifold_certificates() raises:
    # Topological labels are independent of face bounds in WindingLayout.
    var labels: List[UInt32] = [0, 0xFFFFFFFF, 0xFFFFFFFE, 0x80000000]
    var tetra: List[UInt32] = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
    for mode in range(3):
        var indices = List[UInt32]()
        for id in tetra:
            indices.append(labels[Int(id)])
        if mode == 1:
            # A third incidence on a tetrahedron edge invalidates its chain.
            indices.extend([labels[0], labels[1], UInt32(42)])
        elif mode == 2:
            # The all-ones edge key remains a valid table key, but a self-edge
            # cannot certify the disconnected degenerate face as closed.
            indices.extend([labels[1], labels[1], labels[1]])
        var boxes = List[AABB[.WORLD]]()
        for face in range(len(indices) // 3):
            var point = Point3f32[.WORLD](Float32(face))
            boxes.append(AABB[.WORLD](point, point))
        var layout = WindingLayout[.WORLD, 8](indices, boxes)
        assert_equal(len(layout.bvh.item_bounds), 0 if mode == 1 else 1)
        for face in range(len(indices) // 3):
            var expected = EMPTY_LANE if mode == 1 or face == 4 else UInt32(0)
            assert_equal(layout.face_components[face], expected)
        assert_equal(layout.offsets[0], 0)
        assert_equal(
            layout.offsets[len(layout.offsets) - 1], 2 if mode == 2 else 1
        )


def check_connected_periodic_chain[width: SIMDLength]() raises:
    # Wrapping both grid axes gives a closed, connected oriented torus chain.
    # A missing or individually reversed face must disable its certificate.
    for mode in range(4):
        var indices = List[UInt32]()
        var boxes = List[AABB[.WORLD]]()
        for cell in range(128):
            var x = cell % 16
            var y = cell // 16
            var a = UInt32(y * 16 + x)
            var b = UInt32(y * 16 + (x + 1) % 16)
            var c = UInt32(((y + 1) % 8) * 16 + x)
            var d = UInt32(((y + 1) % 8) * 16 + (x + 1) % 16)
            for triangle in range(2):
                if mode == 1 and cell == 0 and triangle == 0:
                    continue
                var first = a if triangle == 0 else b
                var second = b if triangle == 0 else d
                var third = c
                if mode == 3 or (mode == 2 and cell == 0 and triangle == 0):
                    var temporary = second
                    second = third
                    third = temporary
                indices.extend([first, second, third])
                var point = Point3f32[.WORLD](Float32(cell * 2 + triangle))
                boxes.append(AABB[.WORLD](point, point))
        var layout = WindingLayout[.WORLD, width](indices, boxes)
        var closed = mode == 0 or mode == 3
        assert_equal(len(layout.bvh.item_bounds), 1 if closed else 0)
        assert_equal(len(layout.offsets), 3 if closed else 2)
        var blocks = (len(boxes) + width - 1) // width
        assert_equal(layout.offsets[0], 0)
        assert_equal(layout.offsets[1], blocks)
        assert_equal(layout.offsets[len(layout.offsets) - 1], blocks)
        for face in range(len(boxes)):
            assert_equal(
                layout.face_components[face],
                UInt32(0) if closed else EMPTY_LANE,
            )
            assert_equal(layout.face_slots[face], face)
        if closed:
            assert_equal(layout.bvh.item_bounds[0]._min.x[0], Float32(0.0))
            assert_equal(layout.bvh.item_bounds[0]._max.x[0], Float32(255.0))


def test_winding_connected_chain_missing_face_and_orientation_certificates() raises:
    check_connected_periodic_chain[1]()
    check_connected_periodic_chain[8]()
    check_connected_periodic_chain[16]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
