"""Multi-block bounds updates: CPU oracle, graph replay and atomic rejection."""

from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from std.memory import bitcast
from max.gpu.host import DeviceContext
from bajo.core import AABB, Point3f32
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.spatial import SpatialBoundsBvh
from bajo.bvh.spatial.gpu import GpuSpatialBoundsBvh
from bajo.gpu.buffers import upload_list
from test.bvh.cpu.test_spatial_subnormal_bounds import check_bits


def _check[
    block_size: Int, rebuild: Bool = False
](count: Int, grouped: Bool) raises:
    var boxes = List[AABB[.WORLD]](capacity=count)
    var groups = List[UInt32](capacity=count if grouped else 0)
    var flat = List[Float32](length=count * 6, fill=0.0)
    var invalid_bits: List[UInt32] = [0x7FC00000, 0x7F800000, 0xFF800000]
    for i in range(count):
        var x = Float32(i % 31)
        boxes.append(AABB(Point3f32[.WORLD](x), Point3f32[.WORLD](x + 0.5)))
        if grouped:
            # Uneven sparse groups, including a singleton and UInt32.MAX.
            groups.append(
                UInt32.MAX if i == count - 1 else UInt32(i % 17) * 65537
            )
    with DeviceContext() as ctx:
        var gpu = GpuSpatialBoundsBvh(ctx, SpatialBoundsBvh(boxes, groups))
        var input = upload_list(ctx, flat)
        var graph = gpu.create_update_graph[rebuild, block_size](ctx, input)
        var addresses: List[Int] = [
            Int(gpu.bounds.unsafe_ptr()),
            Int(gpu.nodes.unsafe_ptr()),
            Int(gpu.update_items.unsafe_ptr()),
            Int(gpu.update_order.unsafe_ptr()),
            Int(gpu.update_parents.unsafe_ptr()),
        ]
        for phase in range(6):
            for i in range(count):
                # Duplicate centroids, changing order, extreme finite endpoints,
                # degenerate boxes and both signs of nonzero subnormals.
                var x = Float32((i * 17 + phase * 13) % 31 - 15)
                var z = bitcast[.float32](
                    UInt32(1 + (i + phase) % 71)
                    | (UInt32((i + phase) % 2) << 31)
                )
                boxes[i] = AABB(
                    Point3f32[.WORLD](x, -1.0, z),
                    Point3f32[.WORLD](x + 0.5, 1.0, z),
                )
                if i % 23 == 0:
                    var tiny = bitcast[.float32](
                        UInt32(1 + (i + phase) % 71)
                        | (UInt32((i + phase) % 2) << 31)
                    )
                    boxes[i]._min.x = tiny
                    boxes[i]._max.x = tiny
                if i % 29 == 0:
                    boxes[i]._min.x = -0.0 if (i + phase) % 2 else 0.0
                    boxes[i]._max.x = boxes[i]._min.x
                if i == count // 2:
                    boxes[i]._min.x = 2.0e38
                    boxes[i]._max.x = 3.0e38
                comptime for axis in range(3):
                    flat[i * 6 + axis] = boxes[i]._min[axis]
                    flat[i * 6 + axis + 3] = boxes[i]._max[axis]
            with input.map_to_host() as data:
                for i in range(len(flat)):
                    data[i] = flat[i]
            if phase == 2:
                # Replaying the original graph must see refreshed parent links.
                gpu.rebuild(ctx, boxes)
            elif phase == 3:
                # A refit after a device reorder must use current leaf IDs.
                gpu.rebuild_device(ctx, input)
            if phase == 4:
                # Host-input refit aliases the retained input scratch.
                gpu.refit(ctx, boxes)
            elif phase % 2:
                graph.replay()
                gpu.check_update(ctx)
            else:
                gpu.enqueue_update[rebuild, block_size](ctx, input)
                gpu.check_update(ctx)
            var snapshot = gpu.host.copy()
            with gpu.nodes.map_to_host() as nodes:
                for i in range(len(snapshot.nodes)):
                    snapshot.nodes[i] = nodes[i]
            with gpu.bounds.map_to_host() as bounds:
                for i in range(len(snapshot.bounds)):
                    snapshot.bounds[i] = bounds[i]
            check_bits(snapshot, boxes)
            comptime if rebuild:
                if phase != 4:
                    # Each independent tree retains exact centroid/ID order.
                    var root = 0
                    while root < len(snapshot.nodes) // 2:
                        var end = Int(snapshot.nodes[root * 2])
                        var previous = EMPTY_LANE
                        for node in range(root, end):
                            var item = snapshot.nodes[node * 2 + 1]
                            if item == EMPTY_LANE:
                                continue
                            if previous != EMPTY_LANE:
                                var a = Float64(
                                    boxes[Int(previous)]._min.x
                                ) + Float64(boxes[Int(previous)]._max.x)
                                var b = Float64(
                                    boxes[Int(item)]._min.x
                                ) + Float64(boxes[Int(item)]._max.x)
                                assert_true(
                                    a < b or (a == b and previous < item)
                                )
                            previous = item
                        root = end
            # Invalid data in early, middle and final blocks must not publish.
            if phase < 5:
                var item = (
                    0 if phase == 0 else count // 2 if phase == 1 else count - 1
                )
                with input.map_to_host() as data:
                    if phase < 3:
                        data[item * 6 + 5] = bitcast[.float32](
                            invalid_bits[phase]
                        )
                    elif phase == 3:
                        data[item * 6] = data[item * 6 + 3] + 1.0
                    else:
                        data[item * 6 + 2] = bitcast[.float32](UInt32(17))
                        data[item * 6 + 5] = bitcast[.float32](
                            UInt32(0x80000011)
                        )
                if phase % 2:
                    graph.replay()
                else:
                    gpu.enqueue_update[rebuild, block_size](ctx, input)
                with assert_raises():
                    gpu.check_update(ctx)
                with gpu.nodes.map_to_host() as nodes:
                    for i in range(len(snapshot.nodes)):
                        assert_equal(nodes[i], snapshot.nodes[i])
                with gpu.bounds.map_to_host() as bounds:
                    for i in range(len(snapshot.bounds)):
                        assert_equal(
                            bitcast[.uint32](bounds[i]),
                            bitcast[.uint32](snapshot.bounds[i]),
                        )
            assert_equal(Int(gpu.bounds.unsafe_ptr()), addresses[0])
            assert_equal(Int(gpu.nodes.unsafe_ptr()), addresses[1])
            assert_equal(Int(gpu.update_items.unsafe_ptr()), addresses[2])
            assert_equal(Int(gpu.update_order.unsafe_ptr()), addresses[3])
            assert_equal(Int(gpu.update_parents.unsafe_ptr()), addresses[4])


def test_bounds_parallel_refit_thresholds_groups_graphs_and_rejection() raises:
    _check[256](1023, False)
    _check[256](1025, True)


def test_bounds_parallel_rebuild_groups_graphs_and_rejection() raises:
    _check[32, True](4097, True)
    _check[256, True](65537, False)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
