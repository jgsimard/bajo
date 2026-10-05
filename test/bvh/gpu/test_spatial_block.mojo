from std.testing import TestSuite, assert_equal, assert_true
from max.gpu import thread_idx, block_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from bajo.core import AABB, Point3f32, Vec3f32, Rayf32
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.spatial import SpatialBoundsBvh, BoundsQueryView
from bajo.bvh.spatial.block_query import BlockBoundsQuery
from bajo.bvh.spatial.gpu import GpuSpatialBoundsBvh
from bajo.bvh.gpu.utils import _device_span


def _block_kernel[
    block_size: Int, kind: Int
](
    bounds: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[UInt32, MutAnyOrigin],
    stats: MutSpan[UInt32, MutAnyOrigin],
    group_root: UInt32,
    item_count: Int32,
):
    var lane = Int(thread_idx.x)
    var block = Int(block_idx.x)
    var root = UInt32(0)
    if block == 1:
        root = group_root
    elif block == 2:
        root = EMPTY_LANE
    var view = BoundsQueryView[ImmutAnyOrigin, .WORLD](bounds, nodes)
    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](0.5, 0.5, -1.0), Vec3f32[.WORLD](0.0, 0.0, 1.0)
    )
    var query = view.query_aabb(
        AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0)), root
    )
    comptime if kind == 1:
        query = view.query_ray(ray, root)
    elif kind == 2:
        query = view.query_sphere(Point3f32[.WORLD](1.0), 0.4, root)
    elif kind == 3:
        query = view.query_capsule(
            Point3f32[.WORLD](0.5, 0.5, -1.0),
            Vec3f32[.WORLD](0.0, 0.0, 4.0),
            0.25,
            root,
        )
    if block == 3:
        query = view.query_sphere(Point3f32[.WORLD](0.0), -1.0)
    var batch = BlockBoundsQuery[ImmutAnyOrigin, .WORLD, block_size](query^)
    for i in range(lane, Int(item_count), block_size):
        output[block * Int(item_count) + i] = 0
    stats[block * block_size + lane] = 0
    barrier()
    while True:
        var item = UInt32(999)
        var valid = batch.next(item)
        if not batch.batch_valid:
            if item != 999 or valid:
                stats[block * block_size + lane] |= 1
            break
        if valid:
            output[block * Int(item_count) + Int(item)] += 1
        elif item != 999:
            stats[block * block_size + lane] |= 2
        # Partial batches must retain uniform batch_valid even in empty lanes.
        if not valid:
            stats[block * block_size + lane] |= 4
    var item = UInt32(777)
    var valid = batch.next(item)
    if valid or batch.batch_valid or item != 777:
        stats[block * block_size + lane] |= 8


def _verify[
    block_size: Int, kind: Int
](
    ctx: DeviceContext,
    cpu: SpatialBoundsBvh[.WORLD],
    gpu: GpuSpatialBoundsBvh[.WORLD],
) raises:
    var count = len(cpu.item_bounds)
    var output = ctx.enqueue_create_buffer[.uint32](count * 4)
    var stats = ctx.enqueue_create_buffer[.uint32](block_size * 4)
    ctx.enqueue_function[_block_kernel[block_size, kind]](
        _device_span[mut=False](gpu.bounds),
        _device_span[mut=False](gpu.nodes),
        _device_span[mut=True](output),
        _device_span[mut=True](stats),
        cpu.group_root(1),
        Int32(count),
        grid_dim=4,
        block_dim=block_size,
    )
    ctx.synchronize()
    with output.map_to_host() as actual:
        for block in range(4):
            var root = UInt32(0)
            if block == 1:
                root = cpu.group_root(1)
            elif block == 2:
                root = EMPTY_LANE
            var view = cpu.view()
            var query = view.query_aabb(
                AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0)), root
            )
            comptime if kind == 1:
                query = view.query_ray(
                    Rayf32[.WORLD](
                        Point3f32[.WORLD](0.5, 0.5, -1.0),
                        Vec3f32[.WORLD](0.0, 0.0, 1.0),
                    ),
                    root,
                )
            elif kind == 2:
                query = view.query_sphere(Point3f32[.WORLD](1.0), 0.4, root)
            elif kind == 3:
                query = view.query_capsule(
                    Point3f32[.WORLD](0.5, 0.5, -1.0),
                    Vec3f32[.WORLD](0.0, 0.0, 4.0),
                    0.25,
                    root,
                )
            if block == 3:
                query = view.query_sphere(Point3f32[.WORLD](0.0), -1.0)
            var expected = List[UInt32](length=count, fill=0)
            var item = UInt32(0)
            while query.next(item):
                expected[Int(item)] += 1
            for i in range(count):
                assert_equal(actual[block * count + i], expected[i])
    with stats.map_to_host() as actual:
        for i in range(len(actual)):
            assert_equal(actual[i] & UInt32(11), UInt32(0))


def test_block_queries_multiple_blocks_groups_partial_batches_and_updates() raises:
    var boxes = List[AABB[.WORLD]]()
    var groups = List[UInt32]()
    for i in range(79):
        var p = Point3f32[.WORLD](
            Float32(i % 7) * 0.15, Float32(i % 5) * 0.2, Float32(i % 3) * 0.4
        )
        boxes.append(AABB(p, p + Vec3f32[.WORLD](0.2)))
        groups.append(UInt32(i % 2))
    var cpu = SpatialBoundsBvh(boxes, groups)
    with DeviceContext() as ctx:
        var gpu = GpuSpatialBoundsBvh(ctx, cpu.copy())
        for phase in range(3):
            if phase > 0:
                boxes[0] = AABB(
                    Point3f32[.WORLD](Float32(phase)),
                    Point3f32[.WORLD](Float32(phase) + 0.25),
                )
                if phase == 1:
                    cpu.refit(boxes)
                    gpu.refit(ctx, boxes)
                else:
                    cpu.rebuild(boxes)
                    gpu.rebuild(ctx, boxes)
            _verify[32, 0](ctx, cpu, gpu)
            _verify[32, 1](ctx, cpu, gpu)
            _verify[32, 2](ctx, cpu, gpu)
            _verify[32, 3](ctx, cpu, gpu)
            _verify[64, 0](ctx, cpu, gpu)


def test_block_queries_empty_tree() raises:
    var cpu = SpatialBoundsBvh[.WORLD](List[AABB[.WORLD]]())
    with DeviceContext() as ctx:
        var gpu = GpuSpatialBoundsBvh(ctx, cpu.copy())
        _verify[32, 0](ctx, cpu, gpu)


def _cutoff_kernel[
    block_size: Int
](
    bounds: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[UInt32, MutAnyOrigin],
):
    var lane = Int(thread_idx.x)
    var view = BoundsQueryView[ImmutAnyOrigin, .WORLD](bounds, nodes)
    var query = view.query_ray(
        Rayf32[.WORLD](
            Point3f32[.WORLD](0.5, 0.5, 0.0), Vec3f32[.WORLD](0.0, 0.0, 1.0)
        )
    )
    var batch = BlockBoundsQuery[ImmutAnyOrigin, .WORLD, block_size](query^)
    output[lane] = 0
    var item = UInt32(999)
    var valid = batch.next(item, 2.0)
    if not batch.batch_valid:
        output[lane] |= 1
    if valid and item != 0:
        output[lane] |= 2
    if not valid and item != 999:
        output[lane] |= 4
    item = 777
    valid = batch.next(item, 10.0)  # Increase must retain the earlier cutoff.
    if valid or batch.batch_valid or item != 777:
        output[lane] |= 8


def test_block_ray_cutoffs_remain_monotonic() raises:
    var boxes = List[AABB[.WORLD]]()
    for i in range(59):
        var z = Float32(1.0 if i == 0 else 5.0 + Float32(i) * 0.1)
        boxes.append(
            AABB(
                Point3f32[.WORLD](0.0, 0.0, z),
                Point3f32[.WORLD](1.0, 1.0, z + 0.25),
            )
        )
    with DeviceContext() as ctx:
        var gpu = GpuSpatialBoundsBvh(ctx, SpatialBoundsBvh(boxes))
        var output = ctx.enqueue_create_buffer[.uint32](32)
        ctx.enqueue_function[_cutoff_kernel[32]](
            _device_span[mut=False](gpu.bounds),
            _device_span[mut=False](gpu.nodes),
            _device_span[mut=True](output),
            grid_dim=1,
            block_dim=32,
        )
        ctx.synchronize()
        with output.map_to_host() as actual:
            for lane in range(32):
                assert_equal(actual[lane], UInt32(0))


def _wrong_shape_kernel[
    block_size: Int
](
    bounds: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[UInt32, MutAnyOrigin],
):
    var view = BoundsQueryView[ImmutAnyOrigin, .WORLD](bounds, nodes)
    var query = view.query_aabb(
        AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0))
    )
    var batch = BlockBoundsQuery[ImmutAnyOrigin, .WORLD, block_size](query^)
    var index = UInt32(99)
    var valid = batch.next(index)
    output[thread_idx.x] = UInt32(valid or batch.batch_valid or index != 99)


def test_block_wrong_launch_shape_returns_empty_without_scratch_overflow() raises:
    var boxes: List[AABB[.WORLD]] = [
        AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0))
    ]
    with DeviceContext() as ctx:
        var gpu = GpuSpatialBoundsBvh(ctx, SpatialBoundsBvh(boxes))
        var output = ctx.enqueue_create_buffer[.uint32](64)
        ctx.enqueue_function[_wrong_shape_kernel[32]](
            _device_span[mut=False](gpu.bounds),
            _device_span[mut=False](gpu.nodes),
            _device_span[mut=True](output),
            grid_dim=1,
            block_dim=64,
        )
        ctx.synchronize()
        with output.map_to_host() as actual:
            for lane in range(64):
                assert_equal(actual[lane], UInt32(0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
