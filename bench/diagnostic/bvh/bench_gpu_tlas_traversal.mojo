"""Batched TLAS2 traversal through 108 rotated, scaled triangle instances.

Build: mojo build -I . --target-accelerator sm_120 <this file> -o <binary>
Default BLAS: CWBVH8/leaf1; -D COMPRESSED=0 selects ordinary-wide BVH4/leaf2.
Run: <binary> [OBJ path] [ordered: 0|1] [width] [height]. Defaults: 1024x512.
Width * height must be a positive power of two for the shuffle permutation.
Ordered timings include keys, OneSweep, indexed ray reads, traversal and
writing hits at original indices. Separate ray/hit buffers avoid a gather
pass. Builds, workspace allocation, uploads and readback are outside timing.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import max
from std.memory import bitcast
from std.sys import argv
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns

from bajo.benchmark.bvh_fixtures import make_camera_rays_and_params
from bajo.bvh.gpu.triangle_bvh import build_gpu_triangle_blas_set
from bajo.bvh.gpu import GpuBlasSet, GpuBvhLayout, GpuRayOrderWorkspace
from bajo.bvh.gpu.tlas import GpuTlas, build_gpu_tlas
from bajo.bvh.constants import TraceMode
from bajo.bvh.gpu.utils import upload_rays
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Hit, Instance
from bajo.core import Rayf32, GeoKind, dot, Affine3f32, Quat, Vec3f32, AABB
from bajo.core.random import Rng, random_on_hemisphere
from bajo.parser.obj.pack import pack_obj_triangles

comptime WIDTH = 1024
comptime HEIGHT = 512
comptime BATCH = 8
comptime REPEATS = 21
comptime COMPRESSED = get_defined_int["COMPRESSED", 1]() != 0
comptime LAYOUT = GpuBvhLayout.CWBVH8 if COMPRESSED else GpuBvhLayout.WIDE
comptime NODE_WIDTH = 8 if COMPRESSED else 4
comptime LEAF_WIDTH = 4 if COMPRESSED else 2
comptime Tlas = GpuTlas[.TRIANGLE, 2, NODE_WIDTH, 1, LEAF_WIDTH, LAYOUT]
comptime Blases = GpuBlasSet[.TRIANGLE, LAYOUT, NODE_WIDTH, LEAF_WIDTH]


def _make_instances(bounds: AABB[.LOCAL]) -> List[Instance]:
    var extent = bounds.extent()
    var spacing = max(max(extent.x, extent.y), extent.z) * 2.0
    var out = List[Instance]()
    var rng = Rng(123, 123)
    for y in range(9):
        for x in range(12):
            var idx = y * 12 + x
            var rotation = Quat.from_axis_angle(
                Vec3f32[.LOCAL](0, 1, 0), rng.f32(-1, 1)
            )
            var scale = Vec3f32[.LOCAL](1.0, 0.95, 1.05) * (
                1.0 + Float32(idx % 3) * 0.025
            )
            var translation = Vec3f32[.WORLD](
                (Float32(x) - 5.5) * spacing,
                (Float32(y) - 4.0) * spacing,
                Float32(idx % 4) * spacing * 0.05,
            )
            out.append(
                Instance(
                    Affine3f32[.LOCAL, .WORLD].from_rotation_scale_translation(
                        rotation, scale, translation
                    ),
                    UInt32(0),
                    bounds,
                    .TRIANGLE,
                )
            )
    return out^


def _measure[
    mode: TraceMode
](
    mut ctx: DeviceContext,
    tlas: Tlas,
    blases: Blases,
    rays: DeviceBuffer[.float32],
    label: String,
    bounds: AABB[.WORLD],
    ordered: Bool,
) raises:
    var count = len(rays) / Rayf32.STRIDE
    var hits = ctx.enqueue_create_buffer[.float32](count * Hit.STRIDE)
    var ordering = GpuRayOrderWorkspace(ctx, count)
    var times = List[Int]()
    ctx.synchronize()
    for iteration in range(REPEATS + 5):
        var start = perf_counter_ns()
        for _ in range(BATCH):
            if ordered:
                tlas.launch_ordered_rays[mode](
                    ctx, blases, ordering, rays, hits, count, bounds
                )
            else:
                tlas.launch_rays[mode](ctx, blases, rays, hits, count)
        ctx.synchronize()
        var elapsed = Int(perf_counter_ns() - start)
        if iteration >= 5:
            times.append(elapsed)
    sort(times)
    var hash = UInt64(0)
    var visibility_hash = UInt64(0)
    with hits.map_to_host() as values:
        for index in range(len(values)):
            hash = (hash ^ UInt64(bitcast[.uint32](values[index]))) * UInt64(
                0x100000001B3
            )
        var span = Span(unsafe_ptr=values.unsafe_ptr(), length=len(values))
        for index in range(count):
            var hit = Hit[.WORLD].load(span, index)
            var visible = hit.is_occluded() if comptime (
                mode == .ANY_HIT
            ) else hit.is_hit()
            visibility_hash = (visibility_hash ^ UInt64(visible)) * UInt64(
                0x100000001B3
            )
    print(
        label,
        NODE_WIDTH,
        "closest" if comptime (mode == .CLOSEST_HIT) else "any",
        Float64(times[len(times) // 2]) / Float64(BATCH) / 1.0e6,
        Float64(times[0]) / Float64(BATCH) / 1.0e6,
        Float64(times[len(times) - 1]) / Float64(BATCH) / 1.0e6,
        hash,
        visibility_hash,
    )


def _secondary_rays(
    mut ctx: DeviceContext,
    tlas: Tlas,
    blases: Blases,
    camera: List[Rayf32[.WORLD]],
    d_camera: DeviceBuffer[.float32],
    scene_scale: Float32,
) raises -> DeviceBuffer[.float32]:
    """Generate fixed-seed hemisphere rays from visible surface points."""
    var hits = ctx.enqueue_create_buffer[.float32](len(camera) * Hit.STRIDE)
    tlas.launch_rays[TraceMode.CLOSEST_HIT](
        ctx, blases, d_camera, hits, len(camera)
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
    var ordered = Int(args[2]) != 0 if len(args) > 2 else False
    var path = args[1] if len(args) > 1 else "assets/dragon/dragon.obj"
    var width = Int(args[3]) if len(args) > 3 else WIDTH
    var height = Int(args[4]) if len(args) > 4 else HEIGHT
    var ray_count = width * height
    if (
        width <= 0
        or height <= 0
        or ray_count > Int(0x7FFFFFFF)
        or (ray_count & (ray_count - 1)) != 0
    ):
        raise "resolution must produce a positive power-of-two ray count fitting Int32"
    var vertices = pack_obj_triangles[.LOCAL](path)
    var local_bounds = compute_bounds(vertices)
    var instances = _make_instances(local_bounds)
    var bounds = AABB[.WORLD].invalid()
    for instance in instances:
        bounds.grow(instance.bounds)
    var extent = bounds.extent()
    var scene_scale = max(max(extent.x, extent.y), extent.z)
    var camera = make_camera_rays_and_params(
        bounds, width, height, 1, 0.2, True
    )
    var shuffled = List[Rayf32[.WORLD]](capacity=ray_count)
    for index in range(ray_count):
        shuffled.append(
            camera[0][(index * 1664525 + 1013904223) & (ray_count - 1)]
        )
    print(
        path,
        len(vertices) // 3,
        "triangles",
        len(instances),
        "instances",
        ray_count,
        "rays",
        "ordered" if ordered else "direct",
    )
    print(
        "pattern node_width mode median_ms min_ms max_ms full_hit_hash"
        " visibility_hash"
    )
    with DeviceContext() as ctx:
        var d_coherent = upload_rays(ctx, camera[0])
        var d_shuffled = upload_rays(ctx, shuffled)
        var vertex_sets = [vertices.copy()]
        var blases = build_gpu_triangle_blas_set[
            NODE_WIDTH, LEAF_WIDTH, .HPLOC, LAYOUT, max_leaf_size=1
        ](ctx, vertex_sets)
        var tlas = build_gpu_tlas[
            .TRIANGLE, 2, NODE_WIDTH, 1, LEAF_WIDTH, .LBVH, LAYOUT
        ](ctx, instances)
        ctx.synchronize()
        var d_secondary = _secondary_rays(
            ctx, tlas, blases, camera[0], d_coherent, scene_scale
        )
        comptime for any_hit in [False, True]:
            comptime mode = TraceMode.ANY_HIT if any_hit else TraceMode.CLOSEST_HIT
            _measure[mode](
                ctx, tlas, blases, d_coherent, "coherent", bounds, ordered
            )
            _measure[mode](
                ctx, tlas, blases, d_shuffled, "shuffled", bounds, ordered
            )
            _measure[mode](
                ctx, tlas, blases, d_secondary, "secondary", bounds, ordered
            )
