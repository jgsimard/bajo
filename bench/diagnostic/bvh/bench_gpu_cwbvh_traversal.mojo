"""Batched CWBVH traversal with coherent, shuffled, and surface-bounce rays.

Build: mojo build -I . --target-accelerator sm_120 <this file> -o <binary>
Run: <binary> [OBJ path]. Geometry building and uploads are outside timing.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv, max
from std.memory import bitcast
from std.sys import argv
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns

from bajo.benchmark.bvh_fixtures import make_camera_rays_and_params
from bajo.benchmark.gpu_bvh_fixtures import create_cwbvh8_bench_arena
from bajo.bvh.constants import TraceMode
from bajo.bvh.gpu.ray_launch import _load_packed_ray, _store_packed_hit
from bajo.bvh.gpu.triangle_trace import trace_cwbvh8_triangles
from bajo.bvh.gpu.utils import upload_rays, upload_vertices
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Hit
from bajo.core import Rayf32, GeoKind, dot
from bajo.core.random import Rng, random_on_hemisphere
from bajo.parser.obj.pack import pack_obj_triangles

comptime WIDTH = 1024
comptime HEIGHT = 512
comptime BATCH = 8
comptime REPEATS = 21
comptime BLOCK_SIZE = get_defined_int["BLOCK_SIZE", 128]()


def _trace_kernel[
    max_leaf_size: Int, mode: TraceMode
](
    nodes: ImmPointer[Float32, ImmutAnyOrigin],
    triangles: ImmPointer[Float32, ImmutAnyOrigin],
    rays: ImmPointer[Float32, ImmutAnyOrigin],
    hits: MutPointer[Float32, MutAnyOrigin],
    count: Int32,
):
    var index = global_idx.x
    if index >= Int(count):
        return
    var ray = _load_packed_ray[.WORLD](rays, Int(count), index)
    var hit = trace_cwbvh8_triangles[.WORLD, mode, True, max_leaf_size](
        nodes, triangles, UInt32(0), ray
    )
    _store_packed_hit(hit, hits, Int(count), index)


def _measure[
    max_leaf_size: Int, mode: TraceMode
](
    mut ctx: DeviceContext,
    nodes: DeviceBuffer[.float32],
    triangles: DeviceBuffer[.float32],
    rays: DeviceBuffer[.float32],
    label: String,
) raises:
    var count = len(rays) / Rayf32.STRIDE
    var hits = ctx.enqueue_create_buffer[.float32](count * Hit.STRIDE)
    var times = List[Int]()
    for iteration in range(REPEATS + 5):
        var start = perf_counter_ns()
        for _ in range(BATCH):
            ctx.enqueue_function[_trace_kernel[max_leaf_size, mode]](
                nodes,
                triangles,
                rays,
                hits,
                Int32(count),
                grid_dim=ceildiv(count, BLOCK_SIZE),
                block_dim=BLOCK_SIZE,
            )
        ctx.synchronize()
        var elapsed = Int(perf_counter_ns() - start)
        if iteration >= 5:
            times.append(elapsed)
    sort(times)
    var hash = UInt64(0)
    with hits.map_to_host() as values:
        for index in range(len(values)):
            hash = (hash ^ UInt64(bitcast[.uint32](values[index]))) * UInt64(
                0x100000001B3
            )
    print(
        label,
        max_leaf_size,
        "closest" if comptime (mode == .CLOSEST_HIT) else "any",
        Float64(times[len(times) // 2]) / Float64(BATCH) / 1.0e6,
        Float64(times[0]) / Float64(BATCH) / 1.0e6,
        Float64(times[len(times) - 1]) / Float64(BATCH) / 1.0e6,
        hash,
    )


def _secondary_rays[
    max_leaf_size: Int
](
    mut ctx: DeviceContext,
    nodes: DeviceBuffer[.float32],
    triangles: DeviceBuffer[.float32],
    camera: List[Rayf32[.WORLD]],
    d_camera: DeviceBuffer[.float32],
    scene_scale: Float32,
) raises -> DeviceBuffer[.float32]:
    """Generate fixed-seed hemisphere rays from visible surface points."""
    var hits = ctx.enqueue_create_buffer[.float32](len(camera) * Hit.STRIDE)
    ctx.enqueue_function[_trace_kernel[max_leaf_size, TraceMode.CLOSEST_HIT]](
        nodes,
        triangles,
        d_camera,
        hits,
        Int32(len(camera)),
        grid_dim=ceildiv(len(camera), BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    ctx.synchronize()
    var rays = List[Rayf32[.WORLD]](capacity=len(camera))
    var rng = Rng(UInt64(42), UInt64(0))
    var epsilon = scene_scale * 1.0e-5
    with hits.map_to_host() as values:
        var span = Span(unsafe_ptr=values.unsafe_ptr(), length=len(values))
        var active = List[Int]()
        for index in range(len(camera)):
            if Hit[.WORLD].load(span, index).is_hit():
                active.append(index)
        if len(active) == 0:
            raise "surface-bounce benchmark requires visible geometry"
        for index in range(len(camera)):
            var source = active[(index * 1664525 + 1013904223) % len(active)]
            var hit = Hit[.WORLD].load(span, source)
            var normal = hit.normal.unsafe_convert[new_kind=GeoKind.VECTOR]()
            if dot(normal, camera[source].d) > 0.0:
                normal = -normal
            var origin = camera[source].at(hit.t) + epsilon * normal
            rays.append(
                Rayf32[.WORLD](
                    origin,
                    random_on_hemisphere[.WORLD](rng, normal),
                    epsilon,
                    scene_scale * 2.0,
                )
            )
    return upload_rays(ctx, rays)


def main() raises:
    var args = argv()
    var path = args[1] if len(args) > 1 else "assets/dragon/dragon.obj"
    var vertices = pack_obj_triangles[.WORLD](path)
    var bounds = compute_bounds(vertices)
    var extent = bounds.extent()
    var scene_scale = max(max(extent.x, extent.y), extent.z)
    var camera = make_camera_rays_and_params(
        bounds, WIDTH, HEIGHT, 1, 0.2, True
    )
    var shuffled = List[Rayf32[.WORLD]](capacity=WIDTH * HEIGHT)
    for index in range(WIDTH * HEIGHT):
        shuffled.append(
            camera[0][(index * 1664525 + 1013904223) & (WIDTH * HEIGHT - 1)]
        )
    print(path, len(vertices) // 3, "triangles", WIDTH * HEIGHT, "rays")
    print("ordering max_leaf mode median_ms min_ms max_ms full_hit_hash")
    with DeviceContext() as ctx:
        var d_vertices = upload_vertices(ctx, vertices)
        var d_coherent = upload_rays(ctx, camera[0])
        var d_shuffled = upload_rays(ctx, shuffled)
        comptime for max_leaf in [1, 3]:
            var arena = create_cwbvh8_bench_arena[max_leaf](ctx, d_vertices)
            ctx.synchronize()
            arena.finish_synchronized()
            var d_secondary = _secondary_rays[max_leaf](
                ctx,
                arena.nodes,
                arena.triangles,
                camera[0],
                d_coherent,
                scene_scale,
            )
            comptime for any_hit in [False, True]:
                comptime mode = TraceMode.ANY_HIT if any_hit else TraceMode.CLOSEST_HIT
                _measure[max_leaf, mode](
                    ctx, arena.nodes, arena.triangles, d_coherent, "coherent"
                )
                _measure[max_leaf, mode](
                    ctx, arena.nodes, arena.triangles, d_shuffled, "shuffled"
                )
                _measure[max_leaf, mode](
                    ctx, arena.nodes, arena.triangles, d_secondary, "secondary"
                )
