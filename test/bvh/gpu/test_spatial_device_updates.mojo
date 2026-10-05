from std.testing import (
    TestSuite,
    assert_true,
    assert_equal,
    assert_almost_equal,
)
from std.memory import bitcast
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder
from bajo.core import AABB, Point3f32, Vec3f32, Rayf32
from bajo.bvh.spatial import SpatialBoundsBvh, SpatialMesh
from bajo.bvh.spatial.gpu import GpuSpatialBoundsBvh
from bajo.bvh.spatial.gpu_mesh import GpuSpatialMesh
from bajo.bvh.gpu.utils import _device_span
from bajo.bvh.spatial.device_update import bounds_update_kernel
from test.bvh.gpu.test_spatial import _query_kernel
from test.bvh.gpu.test_spatial_mesh import _mesh_kernel
from test.bvh.spatial_fixtures import tetrahedron


def _bounds_input[
    count: Int
](boxes: MutSpan[Float32, MutAnyOrigin], phase: Int32):
    var i = global_idx.x
    if i >= count:
        return
    boxes[i * 6] = Float32(count - 1 - i) * 0.13 + Float32(phase) * 0.03
    boxes[i * 6 + 1] = Float32(i % 3) * 0.2
    boxes[i * 6 + 2] = Float32(i % 5) * 0.2
    for axis in range(3):
        boxes[i * 6 + axis + 3] = boxes[i * 6 + axis] + 0.25


def _invalid_input[count: Int](input: MutSpan[Float32, MutAnyOrigin]):
    if global_idx.x < count:
        input[global_idx.x] = bitcast[.float32](UInt32(0x7FC00000))


def _mesh_input[
    count: Int
](points: MutSpan[Float32, MutAnyOrigin], phase: Int32):
    var i = global_idx.x
    if i >= count:
        return
    var vertex = i // 3
    var axis = i % 3
    points[i] = Float32(phase) * 0.1 + Float32(
        1.0 if vertex == axis + 1 else 0.0
    )


def test_device_bounds_refit_rebuild_groups_and_atomicity() raises:
    var boxes = List[AABB[.WORLD]]()
    var groups = List[UInt32]()
    for i in range(17):
        boxes.append(
            AABB(
                Point3f32[.WORLD](Float32(i) * 0.1),
                Point3f32[.WORLD](Float32(i) * 0.1 + 0.25),
            )
        )
        groups.append(UInt32(i % 3))
    var cpu = SpatialBoundsBvh(boxes, groups)
    with DeviceContext() as ctx:
        var gpu = GpuSpatialBoundsBvh(ctx, cpu.copy())
        var bp = Int(gpu.bounds.unsafe_ptr())
        var np = Int(gpu.nodes.unsafe_ptr())
        var input = ctx.enqueue_create_buffer[.float32](17 * 6)
        var output = ctx.enqueue_create_buffer[.uint32](17 * 4)
        for phase in range(1, 5):
            ctx.enqueue_function[_bounds_input[17]](
                _device_span[mut=True](input),
                Int32(phase),
                grid_dim=1,
                block_dim=32,
            )
            for i in range(17):
                var p = Point3f32[.WORLD](
                    Float32(16 - i) * 0.13 + Float32(phase) * 0.03,
                    Float32(i % 3) * 0.2,
                    Float32(i % 5) * 0.2,
                )
                boxes[i] = AABB(p, p + Vec3f32[.WORLD](0.25))
            cpu.refit(boxes)
            if phase == 2:
                gpu.rebuild_device(ctx, input)
            elif phase == 4:
                gpu.refit(
                    ctx, boxes
                )  # Host update following device reordering.
            else:
                gpu.refit_device(ctx, input)
            assert_equal(Int(gpu.bounds.unsafe_ptr()), bp)
            assert_equal(Int(gpu.nodes.unsafe_ptr()), np)
            var roots: List[UInt32] = [
                0,
                cpu.group_root(1),
                cpu.group_root(999),
            ]
            for root in roots:
                ctx.enqueue_function[_query_kernel[4]](
                    _device_span[mut=False](gpu.bounds),
                    _device_span[mut=False](gpu.nodes),
                    _device_span[mut=True](output),
                    root,
                    Int32(17),
                    grid_dim=1,
                    block_dim=32,
                )
                ctx.synchronize()
                with output.map_to_host() as actual:
                    for lane in range(4):
                        var expected = List[UInt32](length=17, fill=0)
                        var view = cpu.view()
                        var query = view.query_aabb(
                            AABB(
                                Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0)
                            ),
                            root,
                        )
                        if lane == 1:
                            query = view.query_ray(
                                Rayf32[.WORLD](
                                    Point3f32[.WORLD](0.5, 0.5, -1.0),
                                    Vec3f32[.WORLD](0.0, 0.0, 1.0),
                                ),
                                root,
                            )
                        elif lane == 2:
                            query = view.query_sphere(
                                Point3f32[.WORLD](1.0), 0.25, root
                            )
                        elif lane == 3:
                            query = view.query_capsule(
                                Point3f32[.WORLD](0.5, 0.5, -1.0),
                                Vec3f32[.WORLD](0.0, 0.0, 4.0),
                                0.25,
                                root,
                            )
                        var id = UInt32(0)
                        while query.next(id):
                            expected[Int(id)] += 1
                        for i in range(17):
                            assert_equal(actual[lane * 17 + i], expected[i])
        var before_bounds = List[Float32]()
        var before_nodes = List[UInt32]()
        with gpu.bounds.map_to_host() as data:
            for i in range(len(data)):
                before_bounds.append(data[i])
        with gpu.nodes.map_to_host() as data:
            for i in range(len(data)):
                before_nodes.append(data[i])
        ctx.enqueue_function[_invalid_input[1]](
            _device_span[mut=True](input), grid_dim=1, block_dim=1
        )
        var rejected = False
        try:
            gpu.rebuild_device(ctx, input)
        except:
            rejected = True
        assert_true(rejected)
        with gpu.bounds.map_to_host() as data:
            for i in range(len(data)):
                assert_equal(data[i], before_bounds[i])
        with gpu.nodes.map_to_host() as data:
            for i in range(len(data)):
                assert_equal(data[i], before_nodes[i])


def test_device_mesh_refit_rebuild_velocities_and_atomicity() raises:
    with DeviceContext() as ctx:
        var gpu = GpuSpatialMesh(ctx, tetrahedron())
        var pp = Int(gpu.points.unsafe_ptr())
        var bp = Int(gpu.bounds.unsafe_ptr())
        var np = Int(gpu.nodes.unsafe_ptr())
        var input = ctx.enqueue_create_buffer[.float32](12)
        var output = ctx.enqueue_create_buffer[.float64](17)
        for phase in range(1, 5):
            ctx.enqueue_function[_mesh_input[12]](
                _device_span[mut=True](input),
                Int32(phase),
                grid_dim=1,
                block_dim=32,
            )
            if phase == 2:
                gpu.rebuild_device(ctx, input)
            elif phase == 4:
                var points: List[Point3f32[.WORLD]] = [
                    Point3f32[.WORLD](0.4),
                    Point3f32[.WORLD](1.4, 0.4, 0.4),
                    Point3f32[.WORLD](0.4, 1.4, 0.4),
                    Point3f32[.WORLD](0.4, 0.4, 1.4),
                ]
                gpu.refit(ctx, points)
            else:
                gpu.refit_device(ctx, input)
            assert_equal(Int(gpu.points.unsafe_ptr()), pp)
            assert_equal(Int(gpu.bounds.unsafe_ptr()), bp)
            assert_equal(Int(gpu.nodes.unsafe_ptr()), np)
            ctx.enqueue_function[_mesh_kernel[1]](
                _device_span[mut=False](gpu.points),
                _device_span[mut=False](gpu.indices),
                _device_span[mut=False](gpu.velocities),
                _device_span[mut=False](gpu.bounds),
                _device_span[mut=False](gpu.nodes),
                _device_span[mut=True](output),
                UInt32(0),
                grid_dim=1,
                block_dim=32,
            )
            ctx.synchronize()
            var cpu = tetrahedron()
            var offset = Float32(phase) * 0.1
            var points: List[Point3f32[.WORLD]] = [
                Point3f32[.WORLD](offset),
                Point3f32[.WORLD](1.0 + offset, offset, offset),
                Point3f32[.WORLD](offset, 1.0 + offset, offset),
                Point3f32[.WORLD](offset, offset, 1.0 + offset),
            ]
            cpu.refit(points)
            var closest = cpu.view().query_point_no_sign(
                Point3f32[.WORLD](-0.1, 0.017, 0.043)
            )
            with output.map_to_host() as actual:
                assert_equal(actual[0], Float64(1.0))
                assert_equal(actual[1], Float64(closest.face))
                assert_almost_equal(actual[2], closest.distance, atol=1e-6)
        gpu.set_velocities_device(ctx, input)
        with gpu.velocities.map_to_host() as data:
            assert_equal(data[3], Float32(1.4))
        ctx.enqueue_function[_invalid_input[1]](
            _device_span[mut=True](input), grid_dim=1, block_dim=1
        )
        var rejected = False
        try:
            gpu.refit_device(ctx, input)
        except:
            rejected = True
        assert_true(rejected)
        with gpu.points.map_to_host() as data:
            assert_equal(data[0], Float32(0.4))
        rejected = False
        try:
            gpu.set_velocities_device(ctx, input)
        except:
            rejected = True
        assert_true(rejected)
        with gpu.velocities.map_to_host() as data:
            assert_equal(data[0], Float32(0.4))
        var empty = ctx.enqueue_create_buffer[.float32](0)
        gpu.set_velocities_device(ctx, empty)
        with gpu.velocities.map_to_host() as data:
            for i in range(len(data)):
                assert_equal(data[i], Float32(0.0))


def test_empty_device_bounds_updates() raises:
    with DeviceContext() as ctx:
        var gpu = GpuSpatialBoundsBvh(
            ctx, SpatialBoundsBvh[.WORLD](List[AABB[.WORLD]]())
        )
        var input = ctx.enqueue_create_buffer[.float32](0)
        gpu.refit_device(ctx, input)
        gpu.rebuild_device(ctx, input)
        assert_equal(len(gpu.bounds), 0)
        assert_equal(len(gpu.nodes), 0)


def test_graph_bounds_rebuild_and_query_replay() raises:
    var boxes: List[AABB[.WORLD]] = [
        AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0)),
        AABB(Point3f32[.WORLD](2.0), Point3f32[.WORLD](3.0)),
    ]
    with DeviceContext() as ctx:
        var gpu = GpuSpatialBoundsBvh(ctx, SpatialBoundsBvh(boxes))
        var input = ctx.enqueue_create_buffer[.float32](12)
        var output = ctx.enqueue_create_buffer[.uint32](8)
        var function = ctx.compile_function[bounds_update_kernel[True]]()
        var query_function = ctx.compile_function[_query_kernel[4]]()
        var arg0 = _device_span[mut=False](input)
        var arg1 = _device_span[mut=True](gpu.update_items)
        var arg2 = _device_span[mut=True](gpu.bounds)
        var arg3 = _device_span[mut=True](gpu.nodes)
        var arg4 = _device_span[mut=True](gpu.update_order)
        var arg5 = _device_span[mut=True](gpu.update_status)
        var query_bounds = _device_span[mut=False](gpu.bounds)
        var query_nodes = _device_span[mut=False](gpu.nodes)
        var query_output = _device_span[mut=True](output)

        def build(mut builder: DeviceGraphBuilder) raises {imm}:
            var update = builder.add_function(
                function,
                arg0,
                arg1,
                arg2,
                arg3,
                arg4,
                arg5,
                grid_dim=1,
                block_dim=1,
                dependencies=[],
            )
            _ = builder.add_function(
                query_function,
                query_bounds,
                query_nodes,
                query_output,
                UInt32(0),
                Int32(2),
                grid_dim=1,
                block_dim=32,
                dependencies=[update],
            )

        var graph = DeviceGraph.create(ctx, build)
        var bp = Int(gpu.bounds.unsafe_ptr())
        var np = Int(gpu.nodes.unsafe_ptr())
        for phase in range(3):
            with input.map_to_host() as data:
                for item in range(2):
                    var offset = Float32(phase) * 2.0 + Float32(1 - item) * 0.1
                    for axis in range(3):
                        data[item * 6 + axis] = offset
                        data[item * 6 + axis + 3] = offset + 0.25
            graph.replay()
            gpu.check_update(ctx)
            assert_equal(Int(gpu.bounds.unsafe_ptr()), bp)
            assert_equal(Int(gpu.nodes.unsafe_ptr()), np)
            with output.map_to_host() as actual:
                assert_equal(actual[0], UInt32(1 if phase == 0 else 0))
                assert_equal(actual[1], UInt32(1 if phase == 0 else 0))
        with input.map_to_host() as data:
            data[0] = bitcast[.float32](UInt32(0x7FC00000))
        graph.replay()
        var failed = False
        try:
            gpu.check_update(ctx)
        except:
            failed = True
        assert_true(failed)
        with gpu.bounds.map_to_host() as actual:
            assert_equal(actual[0], Float32(4.0))
        # Also instantiate the owner convenience API with the same retained
        # buffers, repair input and replay after a failed captured update.
        var refit_graph = gpu.create_update_graph[False](ctx, input)
        with input.map_to_host() as data:
            data[0] = Float32(4.1)
        refit_graph.replay()
        gpu.check_update(ctx)


def test_graph_mesh_update_replay() raises:
    with DeviceContext() as ctx:
        var gpu = GpuSpatialMesh(ctx, tetrahedron())
        var input = ctx.enqueue_create_buffer[.float32](12)
        var graph = gpu.create_update_graph[True](ctx, input)
        var pp = Int(gpu.points.unsafe_ptr())
        var bp = Int(gpu.bounds.unsafe_ptr())
        var np = Int(gpu.nodes.unsafe_ptr())
        for phase in range(3):
            ctx.enqueue_function[_mesh_input[12]](
                _device_span[mut=True](input),
                Int32(phase),
                grid_dim=1,
                block_dim=32,
            )
            graph.replay()
            gpu.check_update(ctx)
            assert_equal(Int(gpu.points.unsafe_ptr()), pp)
            assert_equal(Int(gpu.bounds.unsafe_ptr()), bp)
            assert_equal(Int(gpu.nodes.unsafe_ptr()), np)
            with gpu.bounds.map_to_host() as data:
                assert_almost_equal(data[0], Float32(phase) * 0.1)
                assert_almost_equal(data[3], Float32(phase) * 0.1 + 1.0)


def test_device_update_invalid_counts_order_and_empty_mesh() raises:
    with DeviceContext() as ctx:
        var boxes: List[AABB[.WORLD]] = [
            AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0))
        ]
        var gpu = GpuSpatialBoundsBvh(ctx, SpatialBoundsBvh(boxes))
        var empty = ctx.enqueue_create_buffer[.float32](0)
        var rejected = False
        try:
            gpu.refit_device(ctx, empty)
        except:
            rejected = True
        assert_true(rejected)
        var input = ctx.enqueue_create_buffer[.float32](6)
        with input.map_to_host() as data:
            for axis in range(3):
                data[axis] = 2.0
                data[axis + 3] = 1.0
        rejected = False
        try:
            gpu.rebuild_device(ctx, input)
        except:
            rejected = True
        assert_true(rejected)
        with gpu.bounds.map_to_host() as data:
            assert_equal(data[0], Float32(0.0))
            assert_equal(data[3], Float32(1.0))
        var mesh = GpuSpatialMesh(ctx, tetrahedron())
        rejected = False
        try:
            mesh.refit_device(ctx, input)
        except:
            rejected = True
        assert_true(rejected)
        var empty_mesh = GpuSpatialMesh(
            ctx, SpatialMesh[.WORLD](List[Point3f32[.WORLD]](), List[UInt32]())
        )
        empty_mesh.refit_device(ctx, empty)
        empty_mesh.rebuild_device(ctx, empty)
        assert_equal(len(empty_mesh.bounds), 0)
        assert_equal(len(empty_mesh.points), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
