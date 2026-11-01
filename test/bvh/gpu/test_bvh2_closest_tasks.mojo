"""Closest- and any-hit traverse deep pending-leaf stacks against a triangle oracle."""
from std.testing import TestSuite, assert_equal, assert_almost_equal
from std.memory import bitcast
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.bvh.constants import EMPTY_LANE, TRI_LEAF_PACKED_STRIDE
from bajo.bvh.wide_meta import _pack_wide_meta
from bajo.bvh.gpu.trace import (
    trace_bounds_bvh,
    trace_ordered_bounds_bvh_state,
    GpuPrimitiveLeafState,
    _dispatch_primitive_leaf,
)
from bajo.bvh.gpu.triangle_trace import _intersect_triangle_leaf
from bajo.bvh.gpu.ray_launch import _load_packed_ray
from bajo.bvh.gpu.utils import upload_list, upload_rays
from bajo.core import Point3f32, Vec3f32, Rayf32
from bajo.core.intersect import intersect_ray_tri
from bajo.bvh.types import Hit


def _record_leaf_order(
    visits: Pointer[UInt32, MutUntrackedOrigin],
    data: UInt32,
    count: UInt32,
    ray: Rayf32[.WORLD],
    mut hit: Hit[.WORLD],
) -> Bool:
    var next = Int(visits[unsafe_offset=0])
    visits[unsafe_offset=next + 1] = data
    visits[unsafe_offset=0] += 1
    return False


def _record_order_kernel[
    width: SIMDLength
](
    nodes: Pointer[Float32, ImmutAnyOrigin],
    visits: Pointer[UInt32, MutAnyOrigin],
):
    if global_idx.x != 0:
        return
    visits[unsafe_offset=0] = 0
    _ = trace_ordered_bounds_bvh_state[
        .WORLD, width, Pointer[UInt32, MutUntrackedOrigin], _record_leaf_order
    ](
        nodes,
        visits.unsafe_origin_cast[MutUntrackedOrigin](),
        UInt32(0),
        Rayf32[.WORLD](Point3f32[.WORLD](0), Vec3f32[.WORLD](0, 0, 1)),
    )


def _assert_leaf_order[width: SIMDLength](expected: List[UInt32]) raises:
    var nodes = List[Float32](length=7 * width, fill=0.0)
    for lane in range(width):
        var distance = Float32((width - lane) // 2 + 1)
        nodes[lane] = -1.0
        nodes[width + lane] = -1.0
        nodes[2 * width + lane] = distance
        nodes[3 * width + lane] = 1.0
        nodes[4 * width + lane] = 1.0
        nodes[5 * width + lane] = distance + 0.25
        nodes[6 * width + lane] = bitcast[.float32](
            _pack_wide_meta(UInt32(lane), 1)
        )
    with DeviceContext() as ctx:
        var device_nodes = upload_list(ctx, nodes)
        var visits = ctx.enqueue_create_buffer[.uint32](width + 1)
        ctx.enqueue_function[_record_order_kernel[width]](
            device_nodes,
            visits,
            grid_dim=1,
            block_dim=32,
        )
        with visits.map_to_host() as output:
            assert_equal(output[0], UInt32(width))
            for i, value in enumerate(expected):
                assert_equal(output[i + 1], value)


def test_ordered_binary_and_wide_leaf_visits_preserve_equal_distance_order() raises:
    _assert_leaf_order[2]([1, 0])
    _assert_leaf_order[4]([3, 1, 2, 0])
    _assert_leaf_order[8]([7, 5, 6, 3, 4, 1, 2, 0])


def _trace_kernel[
    any_hit: Bool
](
    nodes: Pointer[Float32, ImmutAnyOrigin],
    leaves: Pointer[Float32, ImmutAnyOrigin],
    rays: Pointer[Float32, ImmutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    count: Int32,
):
    var idx = global_idx.x
    if idx >= Int(count):
        return
    comptime if any_hit:
        var hit = trace_bounds_bvh[
            .WORLD,
            2,
            .ANY_HIT,
            _intersect_triangle_leaf[.WORLD, 1, .ANY_HIT],
            compact_bvh2=True,
        ](
            nodes,
            leaves,
            UInt32(0),
            _load_packed_ray[.WORLD](rays, Int(count), idx),
        )
        output[unsafe_offset=idx * 4] = bitcast[.float32](
            UInt32(hit.is_occluded())
        )
    else:
        var hit = trace_ordered_bounds_bvh_state[
            .WORLD,
            2,
            GpuPrimitiveLeafState,
            _dispatch_primitive_leaf[
                .WORLD,
                .CLOSEST_HIT,
                _intersect_triangle_leaf[.WORLD, 1, .CLOSEST_HIT],
            ],
        ](
            nodes,
            GpuPrimitiveLeafState(
                leaves.unsafe_origin_cast[ImmUntrackedOrigin]()
            ),
            UInt32(0),
            _load_packed_ray[.WORLD](rays, Int(count), idx),
        )
        output[unsafe_offset=idx * 4] = bitcast[.float32](hit.prim)
        output[unsafe_offset=idx * 4 + 1] = hit.t
        output[unsafe_offset=idx * 4 + 2] = hit.u
        output[unsafe_offset=idx * 4 + 3] = hit.v


def _assert_deep_tree[near_hit: Bool, far_hit: Bool = False]() raises:
    comptime depth = 32
    var nodes = List[Float32](length=(depth + 1) * 14, fill=0.0)
    var vertices = List[Point3f32[.WORLD]]()
    var leaves = List[Float32](
        length=(depth + 1) * TRI_LEAF_PACKED_STRIDE, fill=0.0
    )
    for node in range(depth + 1):
        # The near internal chain overlaps all conservative far-leaf bounds.
        for lane in range(2):
            var base = node * 14 + lane
            nodes[base + 0] = -4.0
            nodes[base + 2] = -1.0
            nodes[base + 4] = Float32(1.0) if lane == 0 else Float32(40.0)
            nodes[base + 6] = 4.0
            nodes[base + 8] = 1.0
            nodes[base + 10] = 42.0
        var meta0 = _pack_wide_meta(
            UInt32(node + 1), 0
        ) if node < depth else _pack_wide_meta(UInt32(depth), 1)
        var meta1 = (
            _pack_wide_meta(UInt32(node), 1) if node < depth else EMPTY_LANE
        )
        nodes[node * 14 + 12] = bitcast[.float32](meta0)
        nodes[node * 14 + 13] = bitcast[.float32](meta1)
        var x = Float32(-0.5) if (near_hit and node == depth) or (
            far_hit and node == depth // 2
        ) else Float32(2.0)
        var z = Float32(2.0) if node == depth else Float32(40.0)
        vertices.append(Point3f32[.WORLD](x, -0.5, z))
        vertices.append(Point3f32[.WORLD](x + 1.0, -0.5, z))
        vertices.append(Point3f32[.WORLD](x, 0.5, z))
        var base = node * TRI_LEAF_PACKED_STRIDE
        leaves[base + 0] = x
        leaves[base + 1] = -0.5
        leaves[base + 2] = z
        leaves[base + 3] = bitcast[.float32](UInt32(node))
        leaves[base + 4] = 1.0
        leaves[base + 9] = 1.0
    var rays = List[Rayf32[.WORLD]]()
    var expected_prims = List[UInt32]()
    var expected_t = List[Float32]()
    var expected_u = List[Float32]()
    var expected_v = List[Float32]()
    for sign in [Float32(0.0), Float32(-0.0)]:
        for t_max in [Float32(1.9), Float32(5.0), Float32(100.0)]:
            var ray = Rayf32[.WORLD](
                Point3f32[.WORLD](0.0),
                Vec3f32[.WORLD](sign, 0.0, 1.0),
                0.001,
                t_max,
            )
            rays.append(ray)
            var prim = EMPTY_LANE
            var closest_t = ray.t_max
            var u = Float32(0.0)
            var v = Float32(0.0)
            for tri in range(depth + 1):
                var candidate = intersect_ray_tri(
                    ray.o,
                    ray.d,
                    vertices[3 * tri],
                    vertices[3 * tri + 1],
                    vertices[3 * tri + 2],
                    closest_t,
                    ray.t_min,
                )
                if candidate.mask[0]:
                    prim = UInt32(tri)
                    closest_t = candidate.t[0]
                    u = candidate.u[0]
                    v = candidate.v[0]
            expected_prims.append(prim)
            expected_t.append(closest_t)
            expected_u.append(u)
            expected_v.append(v)
    with DeviceContext() as ctx:
        var d_nodes = upload_list(ctx, nodes)
        var d_leaves = upload_list(ctx, leaves)
        var d_rays = upload_rays(ctx, rays)
        var output = ctx.enqueue_create_buffer[.float32](len(rays) * 4)
        # Ordered closest-hit and the unchanged shadow traversal share input.
        comptime for variant in range(2):
            comptime any_hit = variant == 1
            ctx.enqueue_function[_trace_kernel[any_hit]](
                d_nodes,
                d_leaves,
                d_rays,
                output,
                Int32(len(rays)),
                grid_dim=1,
                block_dim=64,
            )
            ctx.synchronize()
            with output.map_to_host() as host:
                for i, value in enumerate(expected_prims):
                    comptime if any_hit:
                        assert_equal(
                            bitcast[.uint32](host[i * 4]),
                            UInt32(value != EMPTY_LANE),
                        )
                    else:
                        assert_equal(bitcast[.uint32](host[i * 4]), value)
                        assert_almost_equal(host[i * 4 + 1], expected_t[i])
                        assert_almost_equal(host[i * 4 + 2], expected_u[i])
                        assert_almost_equal(host[i * 4 + 3], expected_v[i])


def test_gpu_bvh2_closest_deep_chain_finds_near_triangle() raises:
    _assert_deep_tree[True]()


def test_gpu_bvh2_closest_deep_chain_unwinds_deferred_misses() raises:
    _assert_deep_tree[False]()


def test_gpu_bvh2_closest_deep_chain_finds_deferred_triangle() raises:
    _assert_deep_tree[False, True]()


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
