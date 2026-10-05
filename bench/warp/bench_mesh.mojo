"""Matched forward indexed mesh queries; no differentiation."""
from std.sys import argv
from std.time import perf_counter_ns
from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import AABB, Point3f32, Vec3f32, Rayf32
from bajo.bvh.spatial import (
    SpatialMesh,
    MeshQueryView,
    BoundsQueryView,
    MeshPointResult,
)
from bajo.bvh.spatial.gpu_mesh import GpuSpatialMesh
from bajo.bvh.gpu.utils import _device_span, upload_list
from bench.warp.bench_bvh import write_floats

comptime ITEMS = 256
comptime QUERIES = 256
comptime REPEATS = 7


def query_one[
    kind: Int, grouped: Bool
](
    view: MeshQueryView[_, .WORLD],
    params: ImmSpan[Float32, _],
    roots: ImmSpan[UInt32, _],
    output: MutSpan[Float32, _],
    q: Int,
):
    comptime WIDTH = ITEMS if kind >= 8 else 10
    var row = output[q * WIDTH : (q + 1) * WIDTH]
    for i in range(WIDTH):
        row[i] = 0.0
    var root = UInt32(0)
    comptime if grouped:
        root = roots[q % 4]
    var p = Point3f32[.WORLD](
        params[q * 7], params[q * 7 + 1], params[q * 7 + 2]
    )
    var d = Vec3f32[.WORLD](
        params[q * 7 + 3], params[q * 7 + 4], params[q * 7 + 5]
    )
    comptime if kind < 5:
        var hit: MeshPointResult[.WORLD]
        comptime if kind == 0:
            hit = view.query_point_no_sign(p, 1000.0)
        elif kind == 1:
            hit = view.query_furthest_point_no_sign(p, 0.0)
        elif kind == 2:
            hit = view.query_point_sign_normal(p, 1000.0)
        elif kind == 3:
            hit = view.query_point_sign_parity(p, 1000.0)
        else:
            hit = view.query_point_sign_winding_number(p, 1000.0)
        row[0] = Float32(1 if hit.result else 0)
        if hit.result:
            row[1] = hit.point.x
            row[2] = hit.point.y
            row[3] = hit.point.z
            row[4] = Float32(hit.distance)
            var velocity = view.eval_velocity(hit.face, hit.u, hit.v)
            row[5] = velocity.x
            row[6] = velocity.y
            row[7] = velocity.z
            row[8] = hit.sign
            row[9] = Float32(hit.face)
    elif kind == 5:
        var hit = view.query_ray(Rayf32[.WORLD](p, d, t_max=1000.0), root)
        row[0] = Float32(1 if hit.result else 0)
        if hit.result:
            row[1] = Float32(hit.t)
            row[2] = hit.normal.x
            row[3] = hit.normal.y
            row[4] = hit.normal.z
            var position = view.eval_position(hit.face, hit.u, hit.v)
            row[5] = position.x
            row[6] = position.y
            row[7] = position.z
            row[8] = Float32(1 if hit.front_face else 0)
            row[9] = Float32(hit.face)
    elif kind == 6:
        row[0] = Float32(
            view.query_ray_count_intersections(Rayf32[.WORLD](p, d), root)
        )
    elif kind == 7:
        row[0] = Float32(
            1 if view.query_ray_anyhit(
                Rayf32[.WORLD](p, d, t_max=1000.0), root
            ) else 0
        )
    elif kind == 8:
        var query = view.query_sphere(p, params[q * 7 + 6], root)
        var face = UInt32(0)
        while query.next(face):
            row[Int(face)] += 1.0
    else:
        var r = Vec3f32[.WORLD](params[q * 7 + 6])
        var query = view.query_aabb(AABB(p - r, p + r), root)
        var face = UInt32(0)
        while query.next(face):
            row[Int(face)] += 1.0


def gpu_query[
    kind: Int, grouped: Bool
](
    points: ImmSpan[Float32, ImmutAnyOrigin],
    indices: ImmSpan[UInt32, ImmutAnyOrigin],
    velocities: ImmSpan[Float32, ImmutAnyOrigin],
    bounds: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    params: ImmSpan[Float32, ImmutAnyOrigin],
    roots: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[Float32, MutAnyOrigin],
):
    var q = Int(global_idx.x)
    if q < QUERIES:
        query_one[kind, grouped](
            MeshQueryView[ImmutAnyOrigin, .WORLD](
                points,
                indices,
                velocities,
                BoundsQueryView[ImmutAnyOrigin, .WORLD](bounds, nodes),
            ),
            params,
            roots,
            output,
            q,
        )


def cpu_case[
    kind: Int, grouped: Bool
](
    tree: SpatialMesh[.WORLD], params: ImmSpan[Float32, _], prefix: String
) raises:
    comptime WIDTH = ITEMS if kind >= 8 else 10
    var output = List[Float32](length=QUERIES * WIDTH, fill=0.0)
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
    write_floats(prefix + "-bajo-cpu-" + phase + ".results", output)


def gpu_case[
    kind: Int, grouped: Bool
](
    mut ctx: DeviceContext,
    tree: GpuSpatialMesh[.WORLD],
    params: ImmSpan[Float32, _],
    prefix: String,
) raises:
    comptime WIDTH = ITEMS if kind >= 8 else 10
    var dp = upload_list(ctx, List(params))
    var roots: List[UInt32] = [
        tree.group_root(0),
        tree.group_root(1),
        tree.group_root(2),
        tree.group_root(3),
    ]
    var dr = upload_list(ctx, roots)
    var output = ctx.enqueue_create_buffer[.float32](QUERIES * WIDTH)
    var phase = String(t"query-{kind}-{Int(grouped)}")
    ctx.synchronize()
    for repeat in range(REPEATS + 2):
        var start = perf_counter_ns()
        ctx.enqueue_function[gpu_query[kind, grouped]](
            _device_span[mut=False](tree.points),
            _device_span[mut=False](tree.indices),
            _device_span[mut=False](tree.velocities),
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
    var host = List[Float32](length=QUERIES * WIDTH, fill=0.0)
    with output.map_to_host() as data:
        for i in range(len(host)):
            host[i] = data[i]
    write_floats(prefix + "-bajo-cuda-" + phase + ".results", host)


def main() raises:
    var args = argv()
    var mode = args[1]
    var prefix = args[2]
    var points = List[Point3f32[.WORLD]]()
    var indices = List[UInt32]()
    var velocities = List[Vec3f32[.WORLD]]()
    var groups = List[UInt32]()
    var flat = List[Float32]()
    var vi = List[Float32]()
    var ids = List[Float32]()
    var origins = List[Point3f32[.WORLD]]()
    var corners: List[Vec3f32[.WORLD]] = [
        Vec3f32[.WORLD](0.0),
        Vec3f32[.WORLD](0.14637, 0.0, 0.0),
        Vec3f32[.WORLD](0.0, 0.14637, 0.0),
        Vec3f32[.WORLD](0.0, 0.0, 0.14637),
    ]
    var order: List[UInt32] = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
    for i in range(64):
        var p = Point3f32[.WORLD](
            Float32(i % 4) * 0.43113,
            Float32((i // 4) % 4) * 0.43113,
            Float32(i // 16) * 0.43113,
        )
        origins.append(p)
        for c in corners:
            var vertex = p + c
            points.append(vertex)
            flat.extend([vertex.x, vertex.y, vertex.z])
            var velocity = Vec3f32[.WORLD](
                vertex.x * 2.0 + 1.0, vertex.y * 3.0 - 1.0, vertex.z * 4.0 + 0.5
            )
            velocities.append(velocity)
            vi.extend([velocity.x, velocity.y, velocity.z])
        for index in order:
            indices.append(UInt32(i * 4) + index)
            ids.append(Float32(UInt32(i * 4) + index))
        for _ in range(4):
            groups.append(UInt32(i % 4))
    var offsets: List[Vec3f32[.WORLD]] = [
        Vec3f32[.WORLD](0.01913, 0.02317, 0.03111),
        Vec3f32[.WORLD](-0.02713, 0.03717, 0.04113),
        Vec3f32[.WORLD](0.12713, 0.07917, 0.03113),
        Vec3f32[.WORLD](0.07113, 0.01717, -0.12313),
    ]
    var params = List[Float32]()
    for q in range(QUERIES):
        var p = origins[q % 64] + offsets[q // 64]
        params.extend(
            [
                p.x,
                p.y,
                p.z,
                Float32(0.13113),
                Float32(0.27117),
                Float32(1.01113),
                Float32(0.04913 + Float32(q % 3) * 0.03717),
            ]
        )
    if mode == "fixtures":
        write_floats(prefix + ".points", flat)
        write_floats(prefix + ".indices", ids)
        write_floats(prefix + ".velocities", vi)
        write_floats(prefix + ".queries", params)
        return
    var tree = SpatialMesh(points, indices, groups, velocities)
    if mode == "cpu":
        for repeat in range(REPEATS + 2):
            var start = perf_counter_ns()
            tree.refit(points)
            var elapsed = Int(perf_counter_ns() - start)
            if repeat >= 2:
                print(t"SAMPLE\tcpu\trefit\t{elapsed}")
        comptime for kind in range(10):
            comptime if kind != 4:
                cpu_case[kind, False](tree, params, prefix)
        var winding_tree = SpatialMesh(
            points, indices, List[UInt32](), velocities
        )
        cpu_case[4, False](winding_tree, params, prefix)
        cpu_case[5, True](tree, params, prefix)
        cpu_case[6, True](tree, params, prefix)
        cpu_case[7, True](tree, params, prefix)
        cpu_case[9, True](tree, params, prefix)
    else:
        with DeviceContext() as ctx:
            var gpu = GpuSpatialMesh(ctx, tree^)
            var input = upload_list(ctx, flat)
            ctx.synchronize()
            for repeat in range(REPEATS + 2):
                var start = perf_counter_ns()
                gpu.enqueue_update[False](ctx, input)
                ctx.synchronize()
                var elapsed = Int(perf_counter_ns() - start)
                if repeat >= 2:
                    print(t"SAMPLE\tcuda\trefit\t{elapsed}")
            gpu.check_update(ctx)
            comptime for kind in range(10):
                comptime if kind != 4:
                    gpu_case[kind, False](ctx, gpu, params, prefix)
            var winding_gpu = GpuSpatialMesh(
                ctx, SpatialMesh(points, indices, List[UInt32](), velocities)
            )
            gpu_case[4, False](ctx, winding_gpu, params, prefix)
            gpu_case[5, True](ctx, gpu, params, prefix)
            gpu_case[6, True](ctx, gpu, params, prefix)
            gpu_case[7, True](ctx, gpu, params, prefix)
            gpu_case[9, True](ctx, gpu, params, prefix)
