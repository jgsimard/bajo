"""Parallel mesh sorting preserves exact centroid/ID order for uneven forests."""
from std.testing import TestSuite, assert_equal, assert_true
from max.gpu.host import DeviceContext
from bajo.core import AABB, Point3f32
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.spatial import SpatialMesh
from bajo.bvh.spatial.gpu_mesh import GpuSpatialMesh
from bajo.bvh.gpu.utils import upload_list
from test.bvh.cpu.test_spatial_construction import check_forest


def _check[count: Int, block_size: Int]() raises:
    var points = List[Point3f32[.WORLD]]()
    var indices = List[UInt32]()
    var groups = List[UInt32]()
    for i in range(count):
        points.append(Point3f32[.WORLD](Float32(i), 0.0, 0.0))
        indices.extend([UInt32(i), UInt32(i), UInt32(i)])
        groups.append(UInt32(i % 7))
    var values: List[Float32] = [
        -3e38,
        -1e35,
        -1.0,
        -1e-38,
        -1e-45,
        -0.0,
        0.0,
        1e-45,
        1e-38,
        1.0,
        1e35,
        3e38,
        0.0,
        -0.0,
        1.0,
        -1.0,
        3e38,
        -3e38,
    ]
    with DeviceContext() as ctx:
        var gpu = GpuSpatialMesh(
            ctx, SpatialMesh(points, indices, groups, cache_winding=False)
        )
        var flat = List[Float32](length=count * 3, fill=0.0)
        var input = upload_list(ctx, flat)
        var graph = gpu.create_update_graph[True, block_size](ctx, input)
        var bp = Int(gpu.bounds.unsafe_ptr())
        var np = Int(gpu.nodes.unsafe_ptr())
        var pp = Int(gpu.points.unsafe_ptr())
        for phase in range(2):
            var boxes = List[AABB[.WORLD]]()
            for i in range(count):
                var x = values[
                    (i * (17 if phase == 0 else 13) + phase) % len(values)
                ]
                points[i] = Point3f32[.WORLD](x, 0.0, 0.0)
                flat[i * 3] = x
                boxes.append(AABB(points[i], points[i]))
            with input.map_to_host() as data:
                for i in range(len(flat)):
                    data[i] = flat[i]
            if phase == 0:
                gpu.enqueue_update[True, block_size](ctx, input)
            else:
                graph.replay()
            gpu.check_update(ctx)
            assert_equal(Int(gpu.bounds.unsafe_ptr()), bp)
            assert_equal(Int(gpu.nodes.unsafe_ptr()), np)
            assert_equal(Int(gpu.points.unsafe_ptr()), pp)
            var published = gpu.host.bvh.copy()
            with gpu.nodes.map_to_host() as data:
                for i in range(len(data)):
                    published.nodes[i] = data[i]
            with gpu.bounds.map_to_host() as data:
                for i in range(len(data)):
                    published.bounds[i] = data[i]
            with gpu.points.map_to_host() as data:
                for i in range(len(flat)):
                    assert_equal(data[i], flat[i])
            check_forest(published, boxes)
            var seen = List[UInt32](length=count, fill=0)
            var root = 0
            while root < len(published.nodes) // 2:
                var end = Int(published.nodes[root * 2])
                var previous = EMPTY_LANE
                for node in range(root, end):
                    var item = published.nodes[node * 2 + 1]
                    if item == EMPTY_LANE:
                        continue
                    seen[Int(item)] += 1
                    if previous != EMPTY_LANE:
                        var a = Float64(points[Int(previous)].x) * 2.0
                        var b = Float64(points[Int(item)].x) * 2.0
                        assert_true(a < b or (a == b and previous < item))
                    previous = item
                root = end
            for occurrences in seen:
                assert_equal(occurrences, UInt32(2))


def test_exact_parallel_mesh_order_extremes_ties_and_uneven_roots() raises:
    _check[256, 32]()
    _check[257, 256]()
    _check[1024, 32]()
    _check[1025, 256]()
    _check[4096, 32]()
    _check[4097, 256]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
