"""Raw subnormal bounds must survive publication and GPU update/graph paths."""
from std.testing import TestSuite, assert_equal, assert_true
from std.memory import bitcast
from max.gpu.host import DeviceContext
from bajo.core import AABB, Point3f32
from bajo.bvh.spatial import SpatialBoundsBvh
from bajo.bvh.spatial.gpu import GpuSpatialBoundsBvh
from bajo.bvh.gpu.utils import upload_list
from test.bvh.cpu.test_spatial_subnormal_bounds import check_bits


def test_gpu_subnormal_bounds_publication_refit_and_graph() raises:
    var boxes = List[AABB[.WORLD]]()
    var groups = List[UInt32]()
    var flat = List[Float32]()
    for i in range(257):
        var x = Float32(i % 17) * 0.25
        var y = Float32(i // 17) * 0.125
        var z = bitcast[.float32](UInt32(71362) | (UInt32(i % 2) << 31))
        boxes.append(
            AABB(
                Point3f32[.WORLD](x, y, z),
                Point3f32[.WORLD](x + 0.5, y + 0.25, z),
            )
        )
        groups.append(UInt32(i % 3))
    with DeviceContext() as ctx:
        var gpu = GpuSpatialBoundsBvh(ctx, SpatialBoundsBvh(boxes, groups))
        with gpu.bounds.map_to_host() as bounds:
            for i in range(len(gpu.host.bounds)):
                assert_equal(
                    bitcast[.uint32](bounds[i]),
                    bitcast[.uint32](gpu.host.bounds[i]),
                )
        for i in range(len(boxes)):
            var z = bitcast[.float32](UInt32(17) | (UInt32((i + 1) % 2) << 31))
            boxes[i]._min.z = z
            boxes[i]._max.z = z
            comptime for axis in range(3):
                flat.append(boxes[i]._min[axis])
            comptime for axis in range(3):
                flat.append(boxes[i]._max[axis])
        var input = upload_list(ctx, flat)
        var graph = gpu.create_update_graph[False](ctx, input)
        var rebuild_graph = gpu.create_update_graph[True](ctx, input)
        for operation in range(5):
            for i in range(len(boxes)):
                var z = bitcast[.float32](
                    UInt32(17 + operation * 16)
                    | (UInt32((i + operation + 1) % 2) << 31)
                )
                boxes[i]._min.z = z
                boxes[i]._max.z = z
                flat[i * 6 + 2] = z
                flat[i * 6 + 5] = z
            with input.map_to_host() as data:
                for i in range(len(flat)):
                    data[i] = flat[i]
            if operation == 0:
                gpu.refit(ctx, boxes)
            elif operation == 1:
                gpu.refit_device(ctx, input)
            elif operation == 2:
                graph.replay()
                gpu.check_update(ctx)
            elif operation == 3:
                gpu.rebuild_device(ctx, input)
            else:
                rebuild_graph.replay()
                gpu.check_update(ctx)
            var snapshot = gpu.host.copy()
            with gpu.bounds.map_to_host() as bounds:
                for i in range(len(snapshot.bounds)):
                    snapshot.bounds[i] = bounds[i]
            with gpu.nodes.map_to_host() as nodes:
                for i in range(len(snapshot.nodes)):
                    snapshot.nodes[i] = nodes[i]
            check_bits(snapshot, boxes)


def test_gpu_inverted_subnormal_device_and_graph_rejection() raises:
    var z = bitcast[.float32](UInt32(71362))
    var nz = bitcast[.float32](UInt32(0x800116C2))
    var boxes: List[AABB[.WORLD]] = [
        AABB(Point3f32[.WORLD](z), Point3f32[.WORLD](z))
    ]
    var flat: List[Float32] = [z, z, z, nz, nz, nz]
    with DeviceContext() as ctx:
        var gpu = GpuSpatialBoundsBvh(ctx, SpatialBoundsBvh(boxes))
        var input = upload_list(ctx, flat)
        var graph = gpu.create_update_graph[False](ctx, input)
        for operation in range(2):
            var rejected = False
            try:
                if operation == 0:
                    gpu.refit_device(ctx, input)
                else:
                    graph.replay()
                    gpu.check_update(ctx)
            except:
                rejected = True
            assert_true(rejected)
            with gpu.bounds.map_to_host() as actual:
                for i in range(len(gpu.host.bounds)):
                    assert_equal(
                        bitcast[.uint32](actual[i]),
                        bitcast[.uint32](gpu.host.bounds[i]),
                    )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
