"""Large transactional mesh refits compared with the independent CPU forest."""
from std.testing import TestSuite, assert_equal
from std.math import nan
from max.gpu.host import DeviceContext
from bajo.core import Point3f32
from bajo.bvh.spatial import SpatialMesh
from bajo.bvh.spatial.gpu_mesh import GpuSpatialMesh
from test.bvh.cpu.test_spatial_construction import check_forest
from bajo.bvh.gpu.utils import upload_list


def _check[
    block_size: Int, face_count: Int = 4097, rebuild: Bool = False
]() raises:
    var points = List[Point3f32[.WORLD]]()
    var indices = List[UInt32]()
    var groups = List[UInt32]()
    for face in range(face_count):
        var x = Float32(face % 19)
        var y = Float32(face // 19)
        points.extend(
            [
                Point3f32[.WORLD](x, y, 0.0),
                Point3f32[.WORLD](x + 0.5, y, 1.0),
                Point3f32[.WORLD](x, y + 0.5, -1.0),
            ]
        )
        indices.extend(
            [UInt32(face * 3), UInt32(face * 3 + 1), UInt32(face * 3 + 2)]
        )
        groups.append(UInt32(face % 3))
    # An unreferenced vertex must still be validated and copied.
    points.append(Point3f32[.WORLD](1000.0))
    var cpu = SpatialMesh(points, indices, groups)
    with DeviceContext() as ctx:
        var gpu = GpuSpatialMesh(ctx, cpu.copy())
        var flat = List[Float32](length=len(points) * 3, fill=0.0)
        var input = upload_list(ctx, flat)
        var bounds_address = Int(gpu.bounds.unsafe_ptr())
        var points_address = Int(gpu.points.unsafe_ptr())
        var graph = gpu.create_update_graph[rebuild, block_size](ctx, input)
        for phase in range(4):
            for vertex in range(len(points)):
                points[vertex] = Point3f32[.WORLD](
                    Float32((vertex * 17 + phase) % 41),
                    Float32(vertex % 11) * -0.125,
                    Float32(phase) + Float32(vertex % 7),
                )
                comptime for axis in range(3):
                    flat[vertex * 3 + axis] = points[vertex][axis]
            with input.map_to_host() as data:
                for i in range(len(flat)):
                    data[i] = flat[i]
            if phase == 2:
                # Rebuild changes leaf order while retaining the balanced shape.
                cpu.rebuild(points)
                gpu.rebuild(ctx, points)
            else:
                cpu.refit(points)
            if phase == 1:
                graph.replay()
            else:
                gpu.enqueue_update[rebuild, block_size](ctx, input)
            gpu.check_update(ctx)
            comptime if rebuild:
                with gpu.nodes.map_to_host() as data:
                    for i in range(len(cpu.bvh.nodes)):
                        cpu.bvh.nodes[i] = data[i]
                cpu.refit(points)
                check_forest(cpu.bvh, cpu.bvh.item_bounds)
            with gpu.bounds.map_to_host() as data:
                for i in range(len(cpu.bvh.bounds)):
                    assert_equal(data[i], cpu.bvh.bounds[i])
            with gpu.points.map_to_host() as data:
                for i in range(len(flat)):
                    assert_equal(data[i], flat[i])
            assert_equal(Int(gpu.bounds.unsafe_ptr()), bounds_address)
            assert_equal(Int(gpu.points.unsafe_ptr()), points_address)
        var original_nodes = List[UInt32]()
        with gpu.nodes.map_to_host() as data:
            for i in range(len(data)):
                original_nodes.append(data[i])
        with input.map_to_host() as data:
            data[len(flat) - 1] = nan[.float32]()
        graph.replay()
        ctx.synchronize()
        with gpu.update_status.map_to_host() as status:
            assert_equal(status[0], UInt32(1))
        with gpu.nodes.map_to_host() as data:
            for i in range(len(original_nodes)):
                assert_equal(data[i], original_nodes[i])
        with gpu.bounds.map_to_host() as data:
            for i in range(len(cpu.bvh.bounds)):
                assert_equal(data[i], cpu.bvh.bounds[i])
        with gpu.points.map_to_host() as data:
            for i in range(len(flat)):
                assert_equal(data[i], flat[i])


def test_parallel_mesh_refit_uneven_groups_and_rejected_unused_vertex() raises:
    _check[256]()
    _check[32, 1024]()


def test_parallel_mesh_rebuild_uneven_groups_and_rejected_unused_vertex() raises:
    _check[256, 4097, True]()


def run_tests[_discovery: Bool = True]() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
