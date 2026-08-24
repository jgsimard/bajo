"""Diagnostic GPU triangle traversal counters."""

from std.math import round
from std.sys import has_accelerator
from max.gpu.host import DeviceContext, DeviceBuffer

from bajo.bvh.gpu.trace import GpuTraversalStats
from bajo.bvh.gpu.triangle_bvh import build_gpu_triangle_bvh
from bajo.bvh.gpu.utils import upload_list, upload_vertices
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Hit
from bajo.parser.obj.pack import pack_obj_triangles
from bajo.benchmark.bvh_fixtures import make_camera_rays_and_params


comptime DEFAULT_OBJ_PATH = "./assets/dragon/dragon.obj"
comptime PRIMARY_WIDTH = 1280
comptime PRIMARY_HEIGHT = 640
comptime PRIMARY_VIEWS = 3
comptime FOV_SCALE = 0.2


def _collect_and_print_triangle_stats[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
](
    mut ctx: DeviceContext,
    d_vertices: DeviceBuffer[.float32],
    d_camera_params: DeviceBuffer[.float32],
    ray_count: Int,
) raises:
    var bvh = build_gpu_triangle_bvh[.WORLD, node_width, leaf_width](
        ctx, d_vertices
    )
    var d_hits = ctx.enqueue_create_buffer[.float32](ray_count * Hit.STRIDE)
    var d_stats = ctx.enqueue_create_buffer[.uint32](
        ray_count * GpuTraversalStats.STRIDE
    )

    bvh.launch_camera_instrumented(
        ctx,
        d_camera_params,
        d_hits,
        d_stats,
        ray_count,
        PRIMARY_WIDTH,
        PRIMARY_HEIGHT,
    )
    ctx.synchronize()

    var node_visits = UInt64(0)
    var internal_child_hits = UInt64(0)
    var leaf_blocks = UInt64(0)
    var primitive_tests = UInt64(0)
    var leaf_active_rays = UInt64(0)
    var max_node_visits = UInt32(0)
    var max_leaf_blocks = UInt32(0)
    var max_primitive_tests = UInt32(0)
    var max_stack_depth = UInt32(0)

    with d_stats.map_to_host() as stats:
        for ray_idx in range(ray_count):
            var base = ray_idx * GpuTraversalStats.STRIDE
            var ray_nodes = stats[base + GpuTraversalStats.NODE_VISITS]
            var ray_internal = stats[
                base + GpuTraversalStats.INTERNAL_CHILD_HITS
            ]
            var ray_leaves = stats[base + GpuTraversalStats.LEAF_BLOCKS]
            var ray_primitives = stats[base + GpuTraversalStats.PRIMITIVE_TESTS]
            var ray_stack = stats[base + GpuTraversalStats.MAX_STACK_DEPTH]

            node_visits += UInt64(ray_nodes)
            internal_child_hits += UInt64(ray_internal)
            leaf_blocks += UInt64(ray_leaves)
            primitive_tests += UInt64(ray_primitives)
            if ray_leaves > 0:
                leaf_active_rays += 1
            if ray_nodes > max_node_visits:
                max_node_visits = ray_nodes
            if ray_leaves > max_leaf_blocks:
                max_leaf_blocks = ray_leaves
            if ray_primitives > max_primitive_tests:
                max_primitive_tests = ray_primitives
            if ray_stack > max_stack_depth:
                max_stack_depth = ray_stack

    var rays = UInt64(ray_count)
    var nodes = round(Float64(node_visits) / Float64(rays), 3)
    var lanes = round(nodes * Float64(Int(node_width)), 3)
    var children = round(Float64(internal_child_hits) / Float64(rays), 3)
    var leaves = round(Float64(leaf_blocks) / Float64(rays), 3)
    var primitives = round(Float64(primitive_tests) / Float64(rays), 3)
    var leaf_active = round(
        100.0 * Float64(leaf_active_rays) / Float64(rays),
        2,
    )

    print(
        String(t"n{Int(node_width)}/l{Int(leaf_width)}").ascii_ljust(10),
        String(nodes).ascii_rjust(10),
        String(lanes).ascii_rjust(15),
        String(children).ascii_rjust(15),
        String(leaves).ascii_rjust(11),
        String(primitives).ascii_rjust(14),
        String(t"{leaf_active}%").ascii_rjust(11),
        String(max_node_visits).ascii_rjust(10),
        String(max_leaf_blocks).ascii_rjust(11),
        String(max_primitive_tests).ascii_rjust(14),
        String(max_stack_depth).ascii_rjust(10),
    )


def main() raises:
    comptime if not has_accelerator():
        raise "No compatible GPU found; skipped triangle traversal stats."

    print("GPU triangle traversal instrumentation")
    print(t"OBJ path: {DEFAULT_OBJ_PATH}")

    var vertices = pack_obj_triangles[.WORLD](DEFAULT_OBJ_PATH)
    var bounds = compute_bounds(vertices)
    var camera = make_camera_rays_and_params(
        bounds.unsafe_convert_frame[.WORLD](),
        PRIMARY_WIDTH,
        PRIMARY_HEIGHT,
        PRIMARY_VIEWS,
        FOV_SCALE,
    )
    var ray_count = len(camera[0])
    print(t"triangles: {len(vertices) / 3}")
    print(t"rays: {ray_count}\n")

    with DeviceContext() as ctx:
        var d_vertices = upload_vertices(ctx, vertices)
        var d_camera_params = upload_list(ctx, camera[1])
        ctx.synchronize()

        print(
            "config      nodes/ray  AABB lanes/ray  child hits/ray  "
            "leaves/ray  triangles/ray  leaf-active  max nodes  "
            "max leaves  max triangles  max stack"
        )
        print(
            "---------- ---------- --------------- --------------- "
            "----------- -------------- ----------- ---------- "
            "----------- -------------- ----------"
        )
        _collect_and_print_triangle_stats[2, 2](
            ctx, d_vertices, d_camera_params, ray_count
        )
        _collect_and_print_triangle_stats[2, 4](
            ctx, d_vertices, d_camera_params, ray_count
        )
        _collect_and_print_triangle_stats[4, 4](
            ctx, d_vertices, d_camera_params, ray_count
        )
        _collect_and_print_triangle_stats[8, 8](
            ctx, d_vertices, d_camera_params, ray_count
        )
