"""Closest-hit matches triangle reference through deep pending-leaf stacks."""
from std.testing import TestSuite, assert_equal, assert_almost_equal
from std.memory import bitcast
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.bvh.constants import EMPTY_LANE, TRI_LEAF_PACKED_STRIDE
from bajo.bvh.wide_meta import _pack_wide_meta
from bajo.bvh.gpu.trace import (
    trace_bounds_bvh_state,
    GpuPrimitiveLeafState,
    _dispatch_primitive_leaf,
)
from bajo.bvh.gpu.triangle_trace import _intersect_triangle_leaf
from bajo.bvh.gpu.ray_launch import _load_packed_ray
from bajo.bvh.gpu.utils import upload_list, upload_rays
from bajo.core import Point3f32, Vec3f32, Rayf32
from bajo.core.intersect import intersect_ray_tri


def _closest_kernel[
    policy: Int
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
    var hit = trace_bounds_bvh_state[
        .WORLD,
        2,
        .CLOSEST_HIT,
        GpuPrimitiveLeafState,
        _dispatch_primitive_leaf[
            .WORLD,
            .CLOSEST_HIT,
            _intersect_triangle_leaf[.WORLD, 1, .CLOSEST_HIT],
        ],
        True,
        True,
        policy,
    ](
        nodes,
        GpuPrimitiveLeafState(leaves.unsafe_origin_cast[ImmUntrackedOrigin]()),
        UInt32(0),
        _load_packed_ray[.WORLD](rays, Int(count), idx),
    )
    output[unsafe_offset=idx * 4] = bitcast[.float32](hit.prim)
    output[unsafe_offset=idx * 4 + 1] = hit.t
    output[unsafe_offset=idx * 4 + 2] = hit.u
    output[unsafe_offset=idx * 4 + 3] = hit.v


def _assert_deep_tree[
    policy: Int, near_hit: Bool, far_hit: Bool = False
]() raises:
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
        ctx.enqueue_function[_closest_kernel[policy]](
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
                assert_equal(bitcast[.uint32](host[i * 4]), value)
                assert_almost_equal(host[i * 4 + 1], expected_t[i])
                assert_almost_equal(host[i * 4 + 2], expected_u[i])
                assert_almost_equal(host[i * 4 + 3], expected_v[i])


def test_gpu_bvh2_closest_deep_chain_finds_near_triangle() raises:
    _assert_deep_tree[1, True]()


def test_gpu_bvh2_closest_deep_chain_unwinds_deferred_misses() raises:
    _assert_deep_tree[1, False]()


def test_gpu_bvh2_closest_deep_chain_finds_deferred_triangle() raises:
    _assert_deep_tree[1, False, True]()


def test_gpu_bvh2_closest_separate_deep_chain_finds_near_triangle() raises:
    _assert_deep_tree[2, True]()


def test_gpu_bvh2_closest_separate_deep_chain_unwinds_deferred_misses() raises:
    _assert_deep_tree[2, False]()


def test_gpu_bvh2_closest_separate_deep_chain_finds_deferred_triangle() raises:
    _assert_deep_tree[2, False, True]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
