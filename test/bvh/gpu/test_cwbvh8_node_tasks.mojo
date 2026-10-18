"""Packed slab decoding against the retained decoder at ray boundaries."""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.memory import bitcast
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal

from bajo.bvh.constants import f32_max
from bajo.bvh.gpu.compressed_bounds_bvh import (
    CWBVH_NODE_WORDS,
    CWBVH_META_BASE,
    CWBVH_QUANTIZED_BASE,
    _intersect_cwbvh8_node_tasks,
    _intersect_cwbvh8_node_tasks_legacy,
)
from bajo.bvh.gpu.ray_launch import _load_packed_ray
from bajo.bvh.gpu.utils import upload_list, upload_rays
from bajo.core import Point3f32, Rayf32, Vec3f32


def _compare_tasks[
    max_leaf_size: Int
](
    nodes: ImmPointer[Float32, ImmutAnyOrigin],
    rays: ImmPointer[Float32, ImmutAnyOrigin],
    results: MutPointer[UInt32, MutAnyOrigin],
    ray_count: Int32,
    node_count: Int32,
):
    var index = global_idx.x
    if index >= Int(ray_count) * Int(node_count):
        return
    var node = UInt32(index % Int(node_count))
    var ray = _load_packed_ray[.WORLD](
        rays, Int(ray_count), index // Int(node_count)
    )
    var rcp = ray.reciprocal_direction()
    var octant = (
        (UInt32(ray.d.x < 0) << 2)
        | (UInt32(ray.d.y < 0) << 1)
        | UInt32(ray.d.z < 0)
    )
    var packed = _intersect_cwbvh8_node_tasks[.WORLD, max_leaf_size](
        nodes, node, ray, rcp.x, rcp.y, rcp.z, ray.t_max, UInt32(7) - octant
    )
    var legacy = _intersect_cwbvh8_node_tasks_legacy[.WORLD](
        nodes, node, ray, rcp.x, rcp.y, rcp.z, ray.t_max, UInt32(7) - octant
    )
    results[unsafe_offset=index * 8] = packed.child_base
    results[unsafe_offset=index * 8 + 1] = packed.triangle_base
    results[unsafe_offset=index * 8 + 2] = packed.node_group_mask
    results[unsafe_offset=index * 8 + 3] = packed.triangle_group_mask
    results[unsafe_offset=index * 8 + 4] = legacy.child_base
    results[unsafe_offset=index * 8 + 5] = legacy.triangle_base
    results[unsafe_offset=index * 8 + 6] = legacy.node_group_mask
    results[unsafe_offset=index * 8 + 7] = legacy.triangle_group_mask


def _assert_decoders_match[max_leaf_size: Int]() raises:
    # Zero through eight active slots, mixing internal and triangle tasks.
    var nodes = List[Float32](length=9 * CWBVH_NODE_WORDS, fill=0.0)
    for active in range(9):
        var base = active * CWBVH_NODE_WORDS
        nodes[base] = -1.25
        nodes[base + 1] = -0.125
        nodes[base + 2] = 0.5
        nodes[base + 4] = bitcast[.float32](UInt32(17))
        nodes[base + 5] = bitcast[.float32](UInt32(51))
        # Empty slots may contain arbitrary quantized endpoints. Their zero
        # metadata must suppress tasks even when those endpoints intersect.
        for word in range(CWBVH_QUANTIZED_BASE, CWBVH_NODE_WORDS):
            nodes[base + word] = bitcast[.float32](UInt32(0xD6C49327))
        var imask = UInt32(0)
        var triangle_offset = 0
        for lane in range(active):
            var meta: UInt32
            if lane % 3 == 0:
                imask |= UInt32(1) << UInt32(lane)
                meta = UInt32(32 + 24 + lane)
            else:
                var count = 1 + lane % max_leaf_size
                meta = UInt32(triangle_offset) | (
                    ((UInt32(1) << UInt32(count)) - 1) << 5
                )
                triangle_offset += count
            var slot = base + CWBVH_META_BASE + lane // 4
            nodes[slot] = bitcast[.float32](
                bitcast[.uint32](nodes[slot]) | (meta << UInt32((lane % 4) * 8))
            )
            for plane in range(6):
                var lo = lane * (1 + plane % 3)
                var quantized = lo if plane < 3 else lo + 7 + plane
                var qslot = base + CWBVH_QUANTIZED_BASE + plane * 2 + lane // 4
                nodes[qslot] = bitcast[.float32](
                    (
                        bitcast[.uint32](nodes[qslot])
                        & ~(UInt32(0xFF) << UInt32((lane % 4) * 8))
                    )
                    | (UInt32(quantized) << UInt32((lane % 4) * 8))
                )
        nodes[base + 3] = bitcast[.float32](
            UInt32(125)
            | (UInt32(120) << 8)
            | (UInt32(122) << 16)
            | (imask << 24)
        )
    var rays = List[Rayf32[.WORLD]]()
    for x in [-2.0, -1.25, 0.0, 2.0, 8.0]:
        for y in [-0.25, -0.125, 0.0, 0.25]:
            for z in [0.0, 0.5, 1.0, 4.0]:
                for dx in [-1.0, -1.0e-12, -0.0, 1.0e-12, 1.0]:
                    for dy in [-1.0, -1.0e-12, -0.0, 1.0e-12, 1.0]:
                        for dz in [-1.0, -1.0e-12, -0.0, 1.0e-12, 1.0]:
                            for limit in [
                                Float32(0),
                                1.0e-9,
                                0.25,
                                1.0,
                                4.0,
                                f32_max,
                            ]:
                                rays.append(
                                    Rayf32[.WORLD](
                                        Point3f32[.WORLD](
                                            Float32(x), Float32(y), Float32(z)
                                        ),
                                        Vec3f32[.WORLD](
                                            Float32(dx),
                                            Float32(dy),
                                            Float32(dz),
                                        ),
                                        0.0,
                                        limit,
                                    )
                                )
    with DeviceContext() as ctx:
        var results = ctx.enqueue_create_buffer[.uint32](len(rays) * 9 * 8)
        ctx.enqueue_function[_compare_tasks[max_leaf_size]](
            upload_list(ctx, nodes),
            upload_rays(ctx, rays),
            results,
            Int32(len(rays)),
            Int32(9),
            grid_dim=ceildiv(len(rays) * 9, 128),
            block_dim=128,
        )
        ctx.synchronize()
        with results.map_to_host() as values:
            for index in range(len(rays) * 9):
                for field in range(4):
                    assert_equal(
                        values[index * 8 + field], values[index * 8 + field + 4]
                    )


def test_packed_slab_boundaries_match_legacy_decoder() raises:
    _assert_decoders_match[1]()
    _assert_decoders_match[2]()
    _assert_decoders_match[3]()


def run_tests[_discovery: Bool = True]() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
