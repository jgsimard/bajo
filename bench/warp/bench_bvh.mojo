"""Matched Warp fixtures, fresh builds, and precomputed closest-hit queries."""

from std.io.file_descriptor import FileDescriptor
from std.math import ceildiv, max
from std.memory import bitcast
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import Point3f32, Vec3f32, Rayf32
from bajo.parser.obj.pack import pack_obj_triangles
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.cpu import (
    build_cpu_triangle_blas_set,
    trace_blas_set,
    CpuBvhBuildMethod,
)
from bajo.bvh.constants import f32_max, TraceMode
from bajo.bvh.gpu.utils import upload_vertices, upload_list
from bajo.bvh.gpu.triangle_trace import trace_cwbvh8_triangles
from bajo.benchmark.bvh_fixtures import make_camera_rays_and_params
from bajo.benchmark.gpu_bvh_fixtures import create_cwbvh8_bench_arena

comptime REPEATS = 11


def write_floats(path: String, data: ImmSpan[Float32, _]) raises:
    var bytes = List[UInt8](length=len(data) * 4, fill=0)
    for i in range(len(data)):
        var word = bitcast[.uint32](data[i])
        for j in range(4):
            bytes[i * 4 + j] = UInt8((word >> UInt32(j * 8)) & UInt32(255))
    with open(path, "w") as file:
        var fd = FileDescriptor(file)
        fd.write_bytes(bytes)


def cpu_case[
    method: CpuBvhBuildMethod
](
    vertices: ImmSpan[Point3f32[.WORLD], _],
    rays: ImmSpan[Rayf32[.WORLD], _],
    prefix: String,
) raises:
    var label = String(t"cpu-{method.name()}-n16-l16")
    var bvh = build_cpu_triangle_blas_set[16, 16, method, .WORLD](vertices)
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        var fresh = build_cpu_triangle_blas_set[16, 16, method, .WORLD](
            vertices
        )
        print(t"SAMPLE\t{label}\tbuild\t{Int(perf_counter_ns() - start)}")
        bvh = fresh^
    var hits = List[Float32](length=len(rays), fill=f32_max)
    for repeat in range(REPEATS + 2):
        var start = perf_counter_ns()
        for i in range(len(rays)):
            hits[i] = trace_blas_set[16, 16, .CLOSEST_HIT, .WORLD](
                bvh, UInt32(0), rays[i]
            ).t
        var elapsed = Int(perf_counter_ns() - start)
        if repeat >= 2:
            print(t"SAMPLE\t{label}\ttrace\t{elapsed}")
    write_floats(prefix + "-" + label + ".hits", hits)
    for repeat in range(REPEATS + 2):
        var start = perf_counter_ns()
        for i in range(len(rays)):
            hits[i] = Float32(
                trace_blas_set[16, 16, .ANY_HIT, .WORLD](
                    bvh, UInt32(0), rays[i]
                ).is_occluded()[0]
            )
        var elapsed = Int(perf_counter_ns() - start)
        if repeat >= 2:
            print(t"SAMPLE\t{label}\tany\t{elapsed}")
    write_floats(prefix + "-" + label + ".any", hits)


def trace_kernel[
    max_leaf_size: Int, mode: TraceMode = .CLOSEST_HIT
](
    nodes: ImmPointer[Float32, ImmutAnyOrigin],
    triangles: ImmPointer[Float32, ImmutAnyOrigin],
    rays: ImmPointer[Float32, ImmutAnyOrigin],
    hits: MutPointer[Float32, MutAnyOrigin],
    count: Int32,
):
    var i = global_idx.x
    if i >= Int(count):
        return
    var base = i * 6
    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](
            rays[unsafe_offset=base],
            rays[unsafe_offset=base + 1],
            rays[unsafe_offset=base + 2],
        ),
        Vec3f32[.WORLD](
            rays[unsafe_offset=base + 3],
            rays[unsafe_offset=base + 4],
            rays[unsafe_offset=base + 5],
        ),
        0.0,
        f32_max,
    )
    var hit = trace_cwbvh8_triangles[.WORLD, mode, True, max_leaf_size](
        nodes, triangles, UInt32(0), ray
    )
    comptime if mode == .ANY_HIT:
        hits[unsafe_offset=i] = Float32(hit.is_occluded()[0])
    else:
        hits[unsafe_offset=i] = hit.t


def gpu_case[
    max_leaf_size: Int
](
    vertices: List[Point3f32[.WORLD]],
    packed_rays: List[Float32],
    prefix: String,
) raises:
    var label = String(t"cuda-hploc-cwbvh8-m{max_leaf_size}")
    with DeviceContext() as ctx:
        var dv = upload_vertices(ctx, vertices)
        var dr = upload_list(ctx, packed_rays)
        var count = len(packed_rays) // 6
        var dh = ctx.enqueue_create_buffer[.float32](count)
        ctx.synchronize()
        var arena = create_cwbvh8_bench_arena[max_leaf_size](ctx, dv)
        ctx.synchronize()
        arena.finish_synchronized()
        for _ in range(REPEATS):
            var start = perf_counter_ns()
            var fresh = create_cwbvh8_bench_arena[max_leaf_size](ctx, dv)
            ctx.synchronize()
            var elapsed = Int(perf_counter_ns() - start)
            fresh.finish_synchronized()
            print(t"SAMPLE\t{label}\tbuild\t{elapsed}")
        for repeat in range(REPEATS + 2):
            var start = perf_counter_ns()
            ctx.enqueue_function[trace_kernel[max_leaf_size]](
                arena.nodes,
                arena.triangles,
                dr,
                dh,
                Int32(count),
                grid_dim=ceildiv(count, 128),
                block_dim=128,
            )
            ctx.synchronize()
            var elapsed = Int(perf_counter_ns() - start)
            if repeat >= 2:
                print(t"SAMPLE\t{label}\ttrace\t{elapsed}")
        var hits = List[Float32](capacity=count)
        with dh.map_to_host() as host:
            for i in range(count):
                hits.append(host[i])
        write_floats(prefix + "-" + label + ".hits", hits)
        for repeat in range(REPEATS + 2):
            var start = perf_counter_ns()
            ctx.enqueue_function[trace_kernel[max_leaf_size, .ANY_HIT]](
                arena.nodes,
                arena.triangles,
                dr,
                dh,
                Int32(count),
                grid_dim=ceildiv(count, 128),
                block_dim=128,
            )
            ctx.synchronize()
            var elapsed = Int(perf_counter_ns() - start)
            if repeat >= 2:
                print(t"SAMPLE\t{label}\tany\t{elapsed}")
        hits.clear()
        with dh.map_to_host() as host:
            for i in range(count):
                hits.append(host[i])
        write_floats(prefix + "-" + label + ".any", hits)


def main() raises:
    var args = argv()
    if len(args) != 4:
        raise Error("usage: bench_bvh cpu|cuda|fixtures mesh.obj output-prefix")
    var vertices = pack_obj_triangles[.WORLD](args[2])
    # Fit both meshes to the same scale so tiny source units do not turn the
    # camera workload into almost entirely root-AABB misses.
    var bounds = compute_bounds(vertices)
    var extent = bounds.extent()
    var center = bounds.centroid()
    var scale = Float32(2.0) / max(max(extent.x, extent.y), extent.z)
    for i in range(len(vertices)):
        vertices[i] = Point3f32[.WORLD](
            (vertices[i].x - center.x) * scale,
            (vertices[i].y - center.y) * scale,
            (vertices[i].z - center.z) * scale,
        )
    var camera = make_camera_rays_and_params(
        compute_bounds(vertices), 512, 288, 1, 0.2
    )
    var packed_vertices = List[Float32](capacity=len(vertices) * 3)
    for v in vertices:
        packed_vertices.extend([v.x, v.y, v.z])
    var packed_rays = List[Float32](capacity=len(camera[0]) * 6)
    for ray in camera[0]:
        packed_rays.extend(
            [ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z]
        )
    if args[1] == "fixtures":
        write_floats(args[3] + ".vertices", packed_vertices)
        write_floats(args[3] + ".rays", packed_rays)
    elif args[1] == "cpu":
        cpu_case[.SAH](vertices, camera[0], args[3])
        cpu_case[.MEDIAN](vertices, camera[0], args[3])
        cpu_case[.LBVH](vertices, camera[0], args[3])
        cpu_case[.HPLOC](vertices, camera[0], args[3])
    elif args[1] == "cuda":
        gpu_case[1](vertices, packed_rays, args[3])
        gpu_case[3](vertices, packed_rays, args[3])
    else:
        raise Error("unknown backend")
