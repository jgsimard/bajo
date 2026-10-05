from std.testing import TestSuite, assert_equal, assert_true
from std.memory import bitcast
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import AABB, Point3f32, Vec3f32, Rayf32
from bajo.bvh.spatial import SpatialBoundsBvh, BoundsQueryView
from bajo.bvh.spatial.gpu import GpuSpatialBoundsBvh
from bajo.bvh.gpu.utils import _device_span


def _query_kernel[
    query_count: Int
](
    bounds: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[UInt32, MutAnyOrigin],
    root: UInt32,
    count: Int32,
):
    var lane = global_idx.x
    if lane >= query_count:
        return
    var view = BoundsQueryView[ImmutAnyOrigin, .WORLD](bounds, nodes)
    var box = AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0))
    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](0.5, 0.5, -1.0), Vec3f32[.WORLD](0.0, 0.0, 1.0)
    )
    var query = view.query_aabb(box, root)
    if lane == 1:
        query = view.query_ray(ray, root)
    elif lane == 2:
        query = view.query_sphere(Point3f32[.WORLD](1.0), 0.25, root)
    elif lane == 3:
        query = view.query_capsule(
            Point3f32[.WORLD](0.5, 0.5, -1.0),
            Vec3f32[.WORLD](0.0, 0.0, 4.0),
            0.25,
            root,
        )
    for i in range(Int(count)):
        output[lane * Int(count) + i] = 0
    var index = UInt32(0)
    while query.next(index):
        output[lane * Int(count) + Int(index)] += 1


def test_gpu_bounds_queries_groups_and_updates_match_cpu() raises:
    var boxes = List[AABB[.WORLD]]()
    var groups = List[UInt32]()
    for i in range(31):
        var p = Point3f32[.WORLD](
            Float32(i % 3) * 0.5, Float32(i % 5) * 0.5, Float32(i % 7) * 0.5
        )
        boxes.append(AABB(p, p + Vec3f32[.WORLD](0.25)))
        groups.append(UInt32(i % 2))
    var cpu = SpatialBoundsBvh(boxes, groups)
    with DeviceContext() as ctx:
        var gpu = GpuSpatialBoundsBvh(ctx, cpu.copy())
        var db = Int(gpu.bounds.unsafe_ptr())
        var dn = Int(gpu.nodes.unsafe_ptr())
        var output = ctx.enqueue_create_buffer[.uint32](4 * len(boxes))
        for phase in range(3):
            if phase > 0:
                boxes[0] = AABB(
                    Point3f32[.WORLD](Float32(phase)),
                    Point3f32[.WORLD](Float32(phase) + 0.5),
                )
                if phase == 1:
                    cpu.refit(boxes)
                    gpu.refit(ctx, boxes)
                else:
                    cpu.rebuild(boxes)
                    gpu.rebuild(ctx, boxes)
            assert_equal(Int(gpu.bounds.unsafe_ptr()), db)
            assert_equal(Int(gpu.nodes.unsafe_ptr()), dn)
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
                    Int32(len(boxes)),
                    grid_dim=1,
                    block_dim=32,
                )
                ctx.synchronize()
                var view = cpu.view()
                with output.map_to_host() as actual:
                    for lane in range(4):
                        var expected = List[UInt32](length=len(boxes), fill=0)
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
                        var index = UInt32(0)
                        while query.next(index):
                            expected[Int(index)] += 1
                        for i in range(len(boxes)):
                            assert_equal(
                                actual[lane * len(boxes) + i], expected[i]
                            )


def _invalid_kernel[
    case_count: Int
](
    bounds: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[UInt32, MutAnyOrigin],
):
    var lane = global_idx.x
    if lane >= case_count:
        return
    var view = BoundsQueryView[ImmutAnyOrigin, .WORLD](bounds, nodes)
    var query = view.query_aabb(AABB[.WORLD].invalid())
    var nan = bitcast[.float32](UInt32(0x7FC00000))
    if lane == 1:
        query = view.query_ray(
            Rayf32[.WORLD](Point3f32[.WORLD](nan), Vec3f32[.WORLD](1.0))
        )
    elif lane == 2:
        query = view.query_sphere(Point3f32[.WORLD](0.0), -1.0)
    elif lane == 3:
        query = view.query_sphere(Point3f32[.WORLD](0.0), nan)
    elif lane == 4:
        query = view.query_capsule(
            Point3f32[.WORLD](0.0), Vec3f32[.WORLD](1.0), -1.0
        )
    var index = UInt32(99)
    output[lane * 2] = UInt32(query.next(index))
    output[lane * 2 + 1] = index


def test_gpu_empty_invalid_queries_and_failed_updates() raises:
    with DeviceContext() as ctx:
        var empty = GpuSpatialBoundsBvh(
            ctx, SpatialBoundsBvh[.WORLD](List[AABB[.WORLD]]())
        )
        var output = ctx.enqueue_create_buffer[.uint32](10)
        ctx.enqueue_function[_invalid_kernel[5]](
            _device_span[mut=False](empty.bounds),
            _device_span[mut=False](empty.nodes),
            _device_span[mut=True](output),
            grid_dim=1,
            block_dim=32,
        )
        ctx.synchronize()
        with output.map_to_host() as data:
            for i in range(5):
                assert_equal(data[i * 2], UInt32(0))
                assert_equal(data[i * 2 + 1], UInt32(99))
        var boxes: List[AABB[.WORLD]] = [
            AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0))
        ]
        var tree = GpuSpatialBoundsBvh(ctx, SpatialBoundsBvh(boxes))
        ctx.enqueue_function[_invalid_kernel[5]](
            _device_span[mut=False](tree.bounds),
            _device_span[mut=False](tree.nodes),
            _device_span[mut=True](output),
            grid_dim=1,
            block_dim=32,
        )
        ctx.synchronize()
        with output.map_to_host() as data:
            for i in range(5):
                assert_equal(data[i * 2], UInt32(0))
                assert_equal(data[i * 2 + 1], UInt32(99))
        boxes[0] = AABB[.WORLD].invalid()
        var rejected = False
        try:
            tree.refit(ctx, boxes)
        except:
            rejected = True
        assert_true(rejected)
        with tree.bounds.map_to_host() as data:
            assert_equal(data[0], Float32(0.0))
            assert_equal(data[3], Float32(1.0))
        var valid_output = ctx.enqueue_create_buffer[.uint32](4)
        ctx.enqueue_function[_query_kernel[4]](
            _device_span[mut=False](tree.bounds),
            _device_span[mut=False](tree.nodes),
            _device_span[mut=True](valid_output),
            UInt32(0),
            Int32(1),
            grid_dim=1,
            block_dim=32,
        )
        ctx.synchronize()
        with valid_output.map_to_host() as data:
            assert_equal(data[0], UInt32(1))
            assert_equal(data[1], UInt32(1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
