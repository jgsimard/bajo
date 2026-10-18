from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
    assert_raises,
)
from std.memory import bitcast
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import AABB, Point3f32, Vec3f32, Rayf32
from bajo.bvh.spatial.mesh import SpatialMesh
from bajo.bvh.spatial.mesh_query import MeshQueryView
from bajo.bvh.spatial.query import BoundsQueryView
from bajo.bvh.spatial.gpu_mesh import GpuSpatialMesh
from bajo.bvh.gpu.utils import _device_span
from test.bvh.spatial_fixtures import tetrahedron
from test.bvh.gpu.test_spatial_block import _block_kernel


def _mesh_kernel[
    count: Int
](
    points: ImmSpan[Float32, ImmutAnyOrigin],
    indices: ImmSpan[UInt32, ImmutAnyOrigin],
    velocities: ImmSpan[Float32, ImmutAnyOrigin],
    bounds: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[Float64, MutAnyOrigin],
    root: UInt32,
    adjacency: ImmSpan[UInt32, ImmutAnyOrigin],
):
    var lane = global_idx.x
    if lane >= count:
        return
    var view = MeshQueryView[ImmutAnyOrigin, .WORLD](
        points,
        indices,
        velocities,
        BoundsQueryView[ImmutAnyOrigin, .WORLD](bounds, nodes),
    )
    view.adjacency = adjacency
    var p = Point3f32[.WORLD](
        Float32(lane % 7) * 0.19 - 0.1,
        Float32(lane % 5) * 0.23 + 0.017,
        Float32(lane % 3) * 0.37 + 0.043,
    )
    var closest = view.query_point_no_sign(p, root=root)
    var furthest = view.query_furthest_point_no_sign(p, root=root)
    var ray = Rayf32[.WORLD](p, Vec3f32[.WORLD](0.13, -0.27, 1.0))
    var hit = view.query_ray(ray, root)
    var base = lane * 17
    output[base] = Float64(closest.result)
    output[base + 1] = Float64(closest.face)
    output[base + 2] = closest.distance
    output[base + 3] = Float64(closest.u)
    output[base + 4] = Float64(closest.v)
    output[base + 5] = Float64(furthest.face)
    output[base + 6] = furthest.distance
    output[base + 7] = Float64(view.query_point_sign_normal(p, root=root).sign)
    output[base + 8] = Float64(view.query_point_sign_parity(p, root=root).sign)
    output[base + 9] = Float64(
        view.query_point_sign_winding_number(p, root=root).sign
    )
    output[base + 10] = Float64(hit.result)
    output[base + 11] = hit.t
    output[base + 12] = Float64(view.query_ray_count_intersections(ray, root))
    output[base + 13] = Float64(view.query_ray_anyhit(ray, root))
    var sphere = view.query_sphere(p, 0.2, root)
    var face = UInt32(0)
    var mask = UInt32(0)
    while sphere.next(face):
        mask |= UInt32(1) << face
    output[base + 14] = Float64(mask)
    output[base + 15] = (
        Float64(view.eval_velocity(0, 0.5, 0.25).x) if len(indices) > 0 else 0.0
    )

    var aabb = view.query_aabb(
        AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0)), root
    )
    mask = 0
    while aabb.next(face):
        mask |= UInt32(1) << face
    output[base + 16] = Float64(mask)


def test_gpu_mesh_queries_groups_and_updates_match_cpu() raises:
    var cpu = tetrahedron()
    with DeviceContext() as ctx:
        var gpu = GpuSpatialMesh(ctx, cpu.copy())
        var pp = Int(gpu.points.unsafe_ptr())
        var bp = Int(gpu.bounds.unsafe_ptr())
        var np = Int(gpu.nodes.unsafe_ptr())
        var output = ctx.enqueue_create_buffer[.float64](105 * 17)
        var velocities: List[Vec3f32[.WORLD]] = [
            Vec3f32[.WORLD](1.0),
            Vec3f32[.WORLD](2.0),
            Vec3f32[.WORLD](3.0),
            Vec3f32[.WORLD](4.0),
        ]
        cpu.set_velocities(velocities)
        gpu.set_velocities(ctx, velocities)
        for phase in range(3):
            if phase > 0:
                var offset = Float32(phase) * 0.1
                var points: List[Point3f32[.WORLD]] = [
                    Point3f32[.WORLD](offset),
                    Point3f32[.WORLD](1.0 + offset, offset, offset),
                    Point3f32[.WORLD](offset, 1.0 + offset, offset),
                    Point3f32[.WORLD](offset, offset, 1.0 + offset),
                ]
                if phase == 1:
                    cpu.refit(points)
                    gpu.refit(ctx, points)
                else:
                    cpu.rebuild(points)
                    gpu.rebuild(ctx, points)
            assert_equal(Int(gpu.points.unsafe_ptr()), pp)
            assert_equal(Int(gpu.bounds.unsafe_ptr()), bp)
            assert_equal(Int(gpu.nodes.unsafe_ptr()), np)
            var roots: List[UInt32] = [
                0,
                cpu.group_root(7),
                cpu.group_root(999),
            ]
            for root in roots:
                ctx.enqueue_function[_mesh_kernel[105]](
                    _device_span[mut=False](gpu.points),
                    _device_span[mut=False](gpu.indices),
                    _device_span[mut=False](gpu.velocities),
                    _device_span[mut=False](gpu.bounds),
                    _device_span[mut=False](gpu.nodes),
                    _device_span[mut=True](output),
                    root,
                    _device_span[mut=False](gpu.adjacency),
                    grid_dim=1,
                    block_dim=128,
                )
                ctx.synchronize()
                var view = cpu.view()
                with output.map_to_host() as actual:
                    for lane in range(105):
                        var p = Point3f32[.WORLD](
                            Float32(lane % 7) * 0.19 - 0.1,
                            Float32(lane % 5) * 0.23 + 0.017,
                            Float32(lane % 3) * 0.37 + 0.043,
                        )
                        var closest = view.query_point_no_sign(p, root=root)
                        var furthest = view.query_furthest_point_no_sign(
                            p, root=root
                        )
                        var ray = Rayf32[.WORLD](
                            p, Vec3f32[.WORLD](0.13, -0.27, 1.0)
                        )
                        var hit = view.query_ray(ray, root)
                        var sphere = view.query_sphere(p, 0.2, root)
                        var face = UInt32(0)
                        var mask = UInt32(0)
                        while sphere.next(face):
                            mask |= UInt32(1) << face
                        var expected: List[Float64] = [
                            Float64(closest.result),
                            Float64(closest.face),
                            closest.distance,
                            Float64(closest.u),
                            Float64(closest.v),
                            Float64(furthest.face),
                            furthest.distance,
                            Float64(
                                view.query_point_sign_normal(p, root=root).sign
                            ),
                            Float64(
                                view.query_point_sign_parity(p, root=root).sign
                            ),
                            Float64(
                                view.query_point_sign_winding_number(
                                    p, root=root
                                ).sign
                            ),
                            Float64(hit.result),
                            hit.t,
                            Float64(
                                view.query_ray_count_intersections(ray, root)
                            ),
                            Float64(view.query_ray_anyhit(ray, root)),
                            Float64(mask),
                            Float64(view.eval_velocity(0, 0.5, 0.25).x),
                        ]
                        var aabb = view.query_aabb(
                            AABB(
                                Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0)
                            ),
                            root,
                        )
                        mask = 0
                        while aabb.next(face):
                            mask |= UInt32(1) << face
                        expected.append(Float64(mask))
                        for slot in range(17):
                            assert_almost_equal(
                                actual[lane * 17 + slot],
                                expected[slot],
                                atol=1e-6,
                                rtol=1e-6,
                            )


def test_gpu_empty_mesh_and_failed_update_atomicity() raises:
    with DeviceContext() as ctx:
        var empty = GpuSpatialMesh(
            ctx, SpatialMesh[.WORLD](List[Point3f32[.WORLD]](), List[UInt32]())
        )
        var output = ctx.enqueue_create_buffer[.float64](17)
        ctx.enqueue_function[_mesh_kernel[1]](
            _device_span[mut=False](empty.points),
            _device_span[mut=False](empty.indices),
            _device_span[mut=False](empty.velocities),
            _device_span[mut=False](empty.bounds),
            _device_span[mut=False](empty.nodes),
            _device_span[mut=True](output),
            UInt32(0),
            _device_span[mut=False](empty.adjacency),
            grid_dim=1,
            block_dim=32,
        )
        ctx.synchronize()
        with output.map_to_host() as actual:
            assert_equal(actual[0], Float64(0.0))
            assert_equal(actual[10], Float64(0.0))
            assert_equal(actual[12], Float64(0.0))
            assert_equal(actual[14], Float64(0.0))
            assert_equal(actual[16], Float64(0.0))
        var gpu = GpuSpatialMesh(ctx, tetrahedron())
        var bad: List[Point3f32[.WORLD]] = [
            Point3f32[.WORLD](bitcast[.float32](UInt32(0x7FC00000))),
            Point3f32[.WORLD](0.0),
            Point3f32[.WORLD](0.0),
            Point3f32[.WORLD](0.0),
        ]
        var failed = False
        try:
            gpu.refit(ctx, bad)
        except:
            failed = True
        assert_true(failed)
        with gpu.points.map_to_host() as actual:
            assert_equal(actual[0], Float32(0.0))
            assert_equal(actual[3], Float32(1.0))


def _edge_kernel[
    count: Int
](
    points: ImmSpan[Float32, ImmutAnyOrigin],
    indices: ImmSpan[UInt32, ImmutAnyOrigin],
    velocities: ImmSpan[Float32, ImmutAnyOrigin],
    bounds: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[Float64, MutAnyOrigin],
    scale: Float32,
):
    if global_idx.x >= count:
        return
    var view = MeshQueryView[ImmutAnyOrigin, .WORLD](
        points,
        indices,
        velocities,
        BoundsQueryView[ImmutAnyOrigin, .WORLD](bounds, nodes),
    )
    var p = Point3f32[.WORLD](scale * 0.25, scale * 0.25, scale)
    var projection = view.query_point_no_sign(p)
    output[0] = projection.distance / Float64(scale)
    output[1] = view.query_ray(
        Rayf32[.WORLD](p, Vec3f32[.WORLD](0.0, 0.0, -scale))
    ).t
    var nan = bitcast[.float32](UInt32(0x7FC00000))
    output[2] = Float64(view.query_point_no_sign(Point3f32[.WORLD](nan)).result)
    output[3] = Float64(
        view.query_furthest_point_no_sign(Point3f32[.WORLD](nan)).result
    )
    output[4] = Float64(
        view.query_ray_anyhit(Rayf32[.WORLD](p, Vec3f32[.WORLD](nan)))
    )
    output[5] = Float64(view.query_point_sign_normal(p, epsilon=nan).result)
    output[6] = Float64(
        view.query_point_sign_winding_number(p, threshold=-1.0).result
    )
    var sphere = view.query_sphere(p, -1.0)
    var face = UInt32(99)
    output[7] = Float64(sphere.next(face))
    output[8] = Float64(face)
    var evaluated = view.eval_position(0, 0.5, 0.25)
    output[9] = Float64(evaluated.x / scale)
    output[10] = Float64(evaluated.y / scale)
    output[11] = Float64(view.eval_normal(0).z)


def test_gpu_mesh_extreme_scales_invalid_queries_and_evaluation() raises:
    with DeviceContext() as ctx:
        var output = ctx.enqueue_create_buffer[.float64](12)
        for scale in List[Float32](
            [Float32(1e-20), Float32(1.0), Float32(1e20)]
        ):
            var points: List[Point3f32[.WORLD]] = [
                Point3f32[.WORLD](0.0),
                Point3f32[.WORLD](scale, 0.0, 0.0),
                Point3f32[.WORLD](0.0, scale, 0.0),
            ]
            var indices: List[UInt32] = [0, 1, 2]
            var gpu = GpuSpatialMesh(ctx, SpatialMesh(points, indices))
            ctx.enqueue_function[_edge_kernel[1]](
                _device_span[mut=False](gpu.points),
                _device_span[mut=False](gpu.indices),
                _device_span[mut=False](gpu.velocities),
                _device_span[mut=False](gpu.bounds),
                _device_span[mut=False](gpu.nodes),
                _device_span[mut=True](output),
                scale,
                grid_dim=1,
                block_dim=32,
            )
            ctx.synchronize()
            with output.map_to_host() as actual:
                assert_almost_equal(actual[0], Float64(1.0), atol=1e-6)
                assert_almost_equal(actual[1], Float64(1.0), atol=1e-6)
                for slot in range(2, 8):
                    assert_equal(actual[slot], Float64(0.0))
                assert_equal(actual[8], Float64(99.0))
                assert_almost_equal(actual[9], Float64(0.25), atol=1e-6)
                assert_almost_equal(actual[10], Float64(0.25), atol=1e-6)
                assert_almost_equal(actual[11], Float64(1.0), atol=1e-6)


def test_mesh_aabb_cooperative_original_faces_groups_and_updates() raises:
    var cpu = tetrahedron()
    with DeviceContext() as ctx:
        var gpu = GpuSpatialMesh(ctx, cpu.copy())
        var output = ctx.enqueue_create_buffer[.uint32](16)
        var stats = ctx.enqueue_create_buffer[.uint32](4 * 32)
        for phase in range(3):
            if phase > 0:
                var offset = Float32(phase) * 0.1
                var points: List[Point3f32[.WORLD]] = [
                    Point3f32[.WORLD](offset),
                    Point3f32[.WORLD](1.0 + offset, offset, offset),
                    Point3f32[.WORLD](offset, 1.0 + offset, offset),
                    Point3f32[.WORLD](offset, offset, 1.0 + offset),
                ]
                if phase == 1:
                    gpu.refit(ctx, points)
                else:
                    gpu.rebuild(ctx, points)
            ctx.enqueue_function[_block_kernel[32, 0]](
                _device_span[mut=False](gpu.bounds),
                _device_span[mut=False](gpu.nodes),
                _device_span[mut=True](output),
                _device_span[mut=True](stats),
                gpu.group_root(7),
                Int32(4),
                grid_dim=4,
                block_dim=32,
            )
            ctx.synchronize()
            with output.map_to_host() as counts:
                for block in range(4):
                    for face in range(4):
                        assert_equal(
                            counts[block * 4 + face],
                            UInt32(
                                1 if block == 0
                                or (block == 1 and face < 2) else 0
                            ),
                        )
            with stats.map_to_host() as flags:
                for i in range(4 * 32):
                    assert_equal(flags[i] & UInt32(11), UInt32(0))


def _adjacent_normal_kernel[
    count: Int
](
    view: MeshQueryView[ImmutAnyOrigin, .WORLD],
    output: MutSpan[Float32, MutAnyOrigin],
    root: UInt32,
):
    var i = Int(global_idx.x)
    if i >= len(output):
        return
    var point = Point3f32[.WORLD](Float32(i * 3) - 0.1, -0.1, -0.1)
    output[i] = view.query_point_sign_normal(point, root=root).sign


def test_gpu_adjacency_view_survives_parallel_device_rebuild_and_graph() raises:
    var points = List[Point3f32[.WORLD]]()
    var indices = List[UInt32]()
    var groups = List[UInt32]()
    for t in range(257):
        var x = Float32(t * 3)
        points.extend(
            [
                Point3f32[.WORLD](x, 0.0, 0.0),
                Point3f32[.WORLD](x + 1.0, 0.0, 0.0),
                Point3f32[.WORLD](x, 1.0, 0.0),
                Point3f32[.WORLD](x, 0.0, 1.0),
            ]
        )
        for offset in [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]:
            indices.append(UInt32(t * 4 + offset))
        for _ in range(4):
            groups.append(UInt32(t % 3))
    var cpu = SpatialMesh(points, indices, groups, cache_winding=False)
    with DeviceContext() as ctx:
        var gpu = GpuSpatialMesh(ctx, cpu.copy())
        var input = ctx.enqueue_create_buffer[.float32](len(cpu.points))
        var graph = gpu.create_update_graph[True](ctx, input)
        var address = Int(gpu.adjacency.unsafe_ptr())
        var output = ctx.enqueue_create_buffer[.float32](17)
        for phase in range(3):
            if phase > 0:
                for i in range(len(points)):
                    points[i].x = -points[i].x + 768.0
                cpu.rebuild(points)
                with input.map_to_host() as target:
                    for i in range(len(cpu.points)):
                        target[i] = cpu.points[i]
                if phase == 1:
                    gpu.enqueue_update[True](ctx, input)
                else:
                    graph.replay()
                gpu.check_update(ctx)
            assert_equal(Int(gpu.adjacency.unsafe_ptr()), address)
            for root in [UInt32(0), cpu.group_root(0), cpu.group_root(2)]:
                ctx.enqueue_function[_adjacent_normal_kernel[17]](
                    gpu.view(),
                    _device_span[mut=True](output),
                    root,
                    grid_dim=1,
                    block_dim=32,
                )
                ctx.synchronize()
                var reference = cpu.view()
                reference.adjacency = ImmSpan[UInt32, origin_of(cpu)]()
                with output.map_to_host() as actual:
                    for i in range(17):
                        var point = Point3f32[.WORLD](
                            Float32(i * 3) - 0.1, -0.1, -0.1
                        )
                        assert_equal(
                            actual[i],
                            reference.query_point_sign_normal(
                                point, root=root
                            ).sign,
                        )

        var adjacency = List[UInt32]()
        with gpu.adjacency.map_to_host() as values:
            for i in range(len(values)):
                adjacency.append(values[i])
        with input.map_to_host() as values:
            values[0] = bitcast[.float32](UInt32(0x7FC00000))
        graph.replay()
        with assert_raises():
            gpu.check_update(ctx)
        with gpu.adjacency.map_to_host() as values:
            for i in range(len(values)):
                assert_equal(values[i], adjacency[i])


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
