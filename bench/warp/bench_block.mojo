"""Matched CPU batch fallback and GPU cooperative bounds queries."""
from std.sys import argv
from std.time import perf_counter_ns
from std.math import ceildiv
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.host import DeviceContext
from bajo.core import AABB, Point3f32, Vec3f32
from bajo.bvh.spatial import SpatialBoundsBvh, BoundsQueryView, BoundsQueryBatch
from bajo.bvh.spatial.block_query import BlockBoundsQuery
from bajo.bvh.spatial.gpu import GpuSpatialBoundsBvh
from bajo.bvh.gpu.utils import _device_span, upload_list
from bench.warp.bench_spatial import ITEMS, QUERIES, REPEATS, query_for
from bench.warp.bench_bvh import write_floats


def zero_output(output: MutSpan[Float32, MutAnyOrigin]):
    var i = Int(global_idx.x)
    if i < QUERIES * ITEMS:
        output[i] = 0.0


def block_query[
    kind: Int, width: Int
](
    bounds: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    params: ImmSpan[Float32, ImmutAnyOrigin],
    output: MutSpan[Float32, MutAnyOrigin],
):
    var q = Int(block_idx.x)
    var view = BoundsQueryView[ImmutAnyOrigin, .WORLD](bounds, nodes)
    var batch = BlockBoundsQuery[ImmutAnyOrigin, .WORLD, width](
        query_for[kind](view, params, q, UInt32(0))
    )
    while True:
        var item = UInt32(0)
        var valid = batch.next(item)
        if not batch.batch_valid:
            break
        if valid:
            output[q * ITEMS + Int(item)] += 1.0


def cpu_case[
    kind: Int, width: Int
](
    tree: SpatialBoundsBvh[.WORLD], params: ImmSpan[Float32, _], prefix: String
) raises:
    var output = List[Float32](length=QUERIES * ITEMS, fill=0.0)
    var buffer = List[UInt32](length=width, fill=0)
    var phase = String(t"query-{kind}-{width}")
    for repeat in range(REPEATS + 2):
        var start = perf_counter_ns()
        for i in range(len(output)):
            output[i] = 0.0
        for q in range(QUERIES):
            var batch = BoundsQueryBatch(
                query_for[kind](tree.view(), params, q, UInt32(0))
            )
            while True:
                var count = batch.next(buffer)
                if count == 0:
                    break
                for i in range(count):
                    output[q * ITEMS + Int(buffer[i])] += 1.0
        var elapsed = Int(perf_counter_ns() - start)
        if repeat >= 2:
            print(t"SAMPLE\tcpu\t{phase}\t{elapsed}")
    write_floats(prefix + "-bajo-cpu-" + phase + ".sets", output)


def gpu_case[
    kind: Int, width: Int
](
    mut ctx: DeviceContext,
    tree: GpuSpatialBoundsBvh[.WORLD],
    params: ImmSpan[Float32, _],
    prefix: String,
) raises:
    var dp = upload_list(ctx, List(params))
    var output = ctx.enqueue_create_buffer[.float32](QUERIES * ITEMS)
    var phase = String(t"query-{kind}-{width}")
    ctx.synchronize()
    for repeat in range(REPEATS + 2):
        var start = perf_counter_ns()
        ctx.enqueue_function[zero_output](
            _device_span[mut=True](output),
            grid_dim=ceildiv(QUERIES * ITEMS, 128),
            block_dim=128,
        )
        ctx.enqueue_function[block_query[kind, width]](
            _device_span[mut=False](tree.bounds),
            _device_span[mut=False](tree.nodes),
            _device_span[mut=False](dp),
            _device_span[mut=True](output),
            grid_dim=QUERIES,
            block_dim=width,
        )
        ctx.synchronize()
        var elapsed = Int(perf_counter_ns() - start)
        if repeat >= 2:
            print(t"SAMPLE\tcuda\t{phase}\t{elapsed}")
    var host = List[Float32](length=QUERIES * ITEMS, fill=0.0)
    with output.map_to_host() as data:
        for i in range(len(host)):
            host[i] = data[i]
    write_floats(prefix + "-bajo-cuda-" + phase + ".sets", host)


def main() raises:
    var args = argv()
    var mode = args[1]
    var prefix = args[2]
    var boxes = List[AABB[.WORLD]]()
    var flat = List[Float32]()
    for i in range(ITEMS):
        var p = Point3f32[.WORLD](
            Float32(i % 8) * 0.21113,
            Float32((i // 8) % 8) * 0.21113,
            Float32(i // 64) * 0.33117,
        )
        var box = AABB(p, p + Vec3f32[.WORLD](0.14637))
        boxes.append(box)
        flat.extend(
            [
                box._min.x,
                box._min.y,
                box._min.z,
                box._max.x,
                box._max.y,
                box._max.z,
            ]
        )
    var params = List[Float32]()
    for q in range(QUERIES):
        params.extend(
            [
                Float32(q % 13) * 0.17347 - 0.23117,
                Float32(q % 11) * 0.19731 - 0.17137,
                Float32(q % 17) * 0.10113 - 0.29117,
                Float32(0.31113),
                Float32(0.53117),
                Float32(1.37113),
                Float32(0.11731) + Float32(q % 3) * 0.08793,
            ]
        )
    if mode == "fixtures":
        write_floats(prefix + ".boxes", flat)
        write_floats(prefix + ".queries", params)
        return
    var tree = SpatialBoundsBvh(boxes)
    if mode == "cpu":
        cpu_case[0, 32](tree, params, prefix)
        cpu_case[1, 32](tree, params, prefix)
        cpu_case[0, 64](tree, params, prefix)
        cpu_case[1, 64](tree, params, prefix)
    else:
        with DeviceContext() as ctx:
            var gpu = GpuSpatialBoundsBvh(ctx, tree^)
            gpu_case[0, 32](ctx, gpu, params, prefix)
            gpu_case[1, 32](ctx, gpu, params, prefix)
            gpu_case[0, 64](ctx, gpu, params, prefix)
            gpu_case[1, 64](ctx, gpu, params, prefix)
