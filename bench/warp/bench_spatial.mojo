"""Matched general bounds queries, construction, refit and rebuild baseline."""
from std.sys import argv
from std.time import perf_counter_ns
from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import AABB, Point3f32, Vec3f32, Rayf32
from bajo.bvh.spatial import SpatialBoundsBvh, BoundsQueryView, BoundsQuery
from bajo.bvh.spatial.gpu import GpuSpatialBoundsBvh
from bajo.bvh.gpu.utils import _device_span, upload_list
from bench.warp.bench_bvh import write_floats

comptime ITEMS = 256
comptime QUERIES = 512
comptime REPEATS = 7


def query_for[
    kind: Int, origin: ImmOrigin
](
    view: BoundsQueryView[origin, .WORLD],
    params: ImmSpan[Float32, _],
    q: Int,
    root: UInt32,
) -> BoundsQuery[origin, .WORLD]:
    var p = Point3f32[.WORLD](
        params[q * 7], params[q * 7 + 1], params[q * 7 + 2]
    )
    var d = Vec3f32[.WORLD](
        params[q * 7 + 3], params[q * 7 + 4], params[q * 7 + 5]
    )
    var radius = params[q * 7 + 6]
    comptime if kind == 0:
        return view.query_aabb(
            AABB(p - Vec3f32[.WORLD](radius), p + Vec3f32[.WORLD](radius)), root
        )
    elif kind == 1:
        return view.query_ray(Rayf32[.WORLD](p, d, t_max=5.0), root)
    elif kind == 2:
        return view.query_sphere(p, radius, root)
    else:
        return view.query_capsule(p, d, radius, root)


def query_one[
    kind: Int, grouped: Bool
](
    view: BoundsQueryView[_, .WORLD],
    params: ImmSpan[Float32, _],
    roots: ImmSpan[UInt32, _],
    output: MutSpan[Float32, _],
    q: Int,
):
    var root = UInt32(0)
    comptime if grouped:
        root = roots[q % 4]
    for item in range(ITEMS):
        output[q * ITEMS + item] = 0.0
    var query = query_for[kind](view, params, q, root)
    var item = UInt32(0)
    while query.next(item, Float32(5.0 if kind == 1 else 1.0)):
        output[q * ITEMS + Int(item)] += 1.0


def gpu_query[
    kind: Int, grouped: Bool
](
    bounds: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    params: ImmSpan[Float32, ImmutAnyOrigin],
    roots: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[Float32, MutAnyOrigin],
):
    var q = Int(global_idx.x)
    if q < QUERIES:
        query_one[kind, grouped](
            BoundsQueryView[ImmutAnyOrigin, .WORLD](bounds, nodes),
            params,
            roots,
            output,
            q,
        )


def cpu_query_case[
    kind: Int, grouped: Bool
](
    tree: SpatialBoundsBvh[.WORLD], params: ImmSpan[Float32, _], prefix: String
) raises:
    var output = List[Float32](length=QUERIES * ITEMS, fill=0.0)
    var roots: List[UInt32] = [
        tree.group_root(0),
        tree.group_root(1),
        tree.group_root(2),
        tree.group_root(3),
    ]
    var phase = String(t"query-{kind}-{Int(grouped)}")
    for repeat in range(REPEATS + 2):
        var start = perf_counter_ns()
        for q in range(QUERIES):
            query_one[kind, grouped](tree.view(), params, roots, output, q)
        var elapsed = Int(perf_counter_ns() - start)
        if repeat >= 2:
            print(t"SAMPLE\tcpu\t{phase}\t{elapsed}")
    write_floats(prefix + "-bajo-cpu-" + phase + ".sets", output)


def gpu_query_case[
    kind: Int, grouped: Bool
](
    mut ctx: DeviceContext,
    tree: GpuSpatialBoundsBvh[.WORLD],
    params: ImmSpan[Float32, _],
    prefix: String,
) raises:
    var dp = upload_list(ctx, List(params))
    var roots: List[UInt32] = [
        tree.group_root(0),
        tree.group_root(1),
        tree.group_root(2),
        tree.group_root(3),
    ]
    var dr = upload_list(ctx, roots)
    var output = ctx.enqueue_create_buffer[.float32](QUERIES * ITEMS)
    var phase = String(t"query-{kind}-{Int(grouped)}")
    ctx.synchronize()
    for repeat in range(REPEATS + 2):
        var start = perf_counter_ns()
        ctx.enqueue_function[gpu_query[kind, grouped]](
            _device_span[mut=False](tree.bounds),
            _device_span[mut=False](tree.nodes),
            _device_span[mut=False](dp),
            _device_span[mut=False](dr),
            _device_span[mut=True](output),
            grid_dim=ceildiv(QUERIES, 128),
            block_dim=128,
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
    var groups = List[UInt32]()
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
        groups.append(UInt32(i % 4))
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
    var tree = SpatialBoundsBvh(boxes, groups)
    if mode == "cpu":
        for repeat in range(REPEATS + 2):
            var start = perf_counter_ns()
            var fresh = SpatialBoundsBvh(boxes, groups)
            var elapsed = Int(perf_counter_ns() - start)
            if repeat >= 2:
                print(t"SAMPLE\tcpu\tbuild\t{elapsed}")
            tree = fresh^
        for phase in range(2):
            for repeat in range(REPEATS + 2):
                var start = perf_counter_ns()
                if phase == 0:
                    tree.refit(boxes)
                else:
                    tree.rebuild(boxes)
                var elapsed = Int(perf_counter_ns() - start)
                if repeat >= 2:
                    print(t"SAMPLE\tcpu\tupdate-{phase}\t{elapsed}")
        cpu_query_case[0, False](tree, params, prefix)
        cpu_query_case[1, False](tree, params, prefix)
        cpu_query_case[2, False](tree, params, prefix)
        cpu_query_case[3, False](tree, params, prefix)
        cpu_query_case[0, True](tree, params, prefix)
        cpu_query_case[1, True](tree, params, prefix)
        cpu_query_case[2, True](tree, params, prefix)
        cpu_query_case[3, True](tree, params, prefix)
    elif mode == "cuda":
        with DeviceContext() as ctx:
            for repeat in range(REPEATS + 2):
                var start = perf_counter_ns()
                var fresh = GpuSpatialBoundsBvh(
                    ctx, SpatialBoundsBvh(boxes, groups)
                )
                var elapsed = Int(perf_counter_ns() - start)
                if repeat >= 2:
                    print(t"SAMPLE\tcuda\tbuild\t{elapsed}")
            var gpu = GpuSpatialBoundsBvh(ctx, tree^)
            var input = upload_list(ctx, flat)
            ctx.synchronize()
            for phase in range(2):
                for repeat in range(REPEATS + 2):
                    var start = perf_counter_ns()
                    if phase == 0:
                        gpu.enqueue_update[False](ctx, input)
                    else:
                        gpu.enqueue_update[True](ctx, input)
                    ctx.synchronize()
                    var elapsed = Int(perf_counter_ns() - start)
                    if repeat >= 2:
                        print(t"SAMPLE\tcuda\tupdate-{phase}\t{elapsed}")
                gpu.check_update(ctx)
            gpu_query_case[0, False](ctx, gpu, params, prefix)
            gpu_query_case[1, False](ctx, gpu, params, prefix)
            gpu_query_case[2, False](ctx, gpu, params, prefix)
            gpu_query_case[3, False](ctx, gpu, params, prefix)
            gpu_query_case[0, True](ctx, gpu, params, prefix)
            gpu_query_case[1, True](ctx, gpu, params, prefix)
            gpu_query_case[2, True](ctx, gpu, params, prefix)
            gpu_query_case[3, True](ctx, gpu, params, prefix)
