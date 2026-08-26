"""Streaming Float32 BVH8 versus conservatively quantized 80-byte node test."""

from std.benchmark import keep
from std.math import round
from std.memory import bitcast, pack_bits
from std.time import perf_counter_ns

from bajo.bvh.constants import WideNode, f32_max
from bajo.bvh.gpu.compressed_bounds_bvh import (
    CWBVH_META_BASE,
    CWBVH_NODE_WORDS,
    CWBVH_QUANTIZED_BASE,
    _intersect_cwbvh8_node_tasks,
    _power_of_two_scale_exponent,
    _quantize_lower,
    _quantize_upper,
    _scale_from_exponent,
)
from bajo.core import (
    AxisAlignedBoundingBox,
    Point3,
    Point3f32,
    Rayf32,
    Vec3,
    Vec3f32,
)
from bajo.core.intersect import intersect_ray_aabb_octant_fma
from bajo.core.utils import ns_to_ms


comptime WIDTH = 8
comptime NODE_COUNT = 65536
comptime STREAM_PASSES = 32
comptime REPEATS = 7


def _median(values: List[Int]) -> Int:
    var ordered = values.copy()
    sort(ordered)
    return ordered[(len(ordered) - 1) >> 1]


def _pack_bytes(values: SIMD[.uint8, WIDTH]) -> SIMD[.uint32, 2]:
    return bitcast[.uint32, 2](values)


def _make_nodes() -> Tuple[List[Float32], List[Float32]]:
    var float_nodes = List[Float32](
        length=NODE_COUNT * WIDTH * WideNode.CHILD_STRIDE, fill=0.0
    )
    var compressed = List[Float32](
        length=NODE_COUNT * CWBVH_NODE_WORDS, fill=0.0
    )
    var compressed_u32 = compressed.unsafe_ptr().unsafe_bitcast[UInt32]()

    for node_idx in range(NODE_COUNT):
        var node_base = node_idx * WIDTH * WideNode.CHILD_STRIDE
        var compressed_base = node_idx * CWBVH_NODE_WORDS
        var px = Float32((node_idx % 257) - 128) * 0.25
        var py = Float32(((node_idx / 257) % 257) - 128) * 0.25
        var pz = Float32(node_idx % 31) * 0.1 + 1.0
        compressed[compressed_base + 0] = px
        compressed[compressed_base + 1] = py
        compressed[compressed_base + 2] = pz

        var extent_x = Float32(8) * 0.3 + 0.2
        var extent_y = Float32(8) * 0.2 + 0.2
        var extent_z = Float32(8) * 0.1 + 0.2
        var exp_x = _power_of_two_scale_exponent(extent_x)
        var exp_y = _power_of_two_scale_exponent(extent_y)
        var exp_z = _power_of_two_scale_exponent(extent_z)
        var scale_x = _scale_from_exponent(exp_x)
        var scale_y = _scale_from_exponent(exp_y)
        var scale_z = _scale_from_exponent(exp_z)
        compressed_u32[unsafe_offset=compressed_base + 3] = (
            exp_x | (exp_y << 8) | (exp_z << 16)
        )
        compressed_u32[unsafe_offset=compressed_base + 4] = UInt32(0)
        compressed_u32[unsafe_offset=compressed_base + 5] = UInt32(0)

        var meta = SIMD[.uint8, WIDTH](0)
        var qmin_x = SIMD[.uint8, WIDTH](0)
        var qmin_y = SIMD[.uint8, WIDTH](0)
        var qmin_z = SIMD[.uint8, WIDTH](0)
        var qmax_x = SIMD[.uint8, WIDTH](0)
        var qmax_y = SIMD[.uint8, WIDTH](0)
        var qmax_z = SIMD[.uint8, WIDTH](0)
        comptime for lane in range(WIDTH):
            var min_x = px + Float32(lane) * 0.3
            var min_y = py + Float32(lane) * 0.2
            var min_z = pz + Float32(lane) * 0.1
            var max_x = min_x + 0.2
            var max_y = min_y + 0.2
            var max_z = min_z + 0.2
            float_nodes[node_base + WideNode.MIN_X * WIDTH + lane] = min_x
            float_nodes[node_base + WideNode.MIN_Y * WIDTH + lane] = min_y
            float_nodes[node_base + WideNode.MIN_Z * WIDTH + lane] = min_z
            float_nodes[node_base + WideNode.MAX_X * WIDTH + lane] = max_x
            float_nodes[node_base + WideNode.MAX_Y * WIDTH + lane] = max_y
            float_nodes[node_base + WideNode.MAX_Z * WIDTH + lane] = max_z
            meta[lane] = UInt8(UInt32(0x20) | UInt32(lane))
            qmin_x[lane] = UInt8(_quantize_lower(min_x, px, scale_x))
            qmin_y[lane] = UInt8(_quantize_lower(min_y, py, scale_y))
            qmin_z[lane] = UInt8(_quantize_lower(min_z, pz, scale_z))
            qmax_x[lane] = UInt8(_quantize_upper(max_x, px, scale_x))
            qmax_y[lane] = UInt8(_quantize_upper(max_y, py, scale_y))
            qmax_z[lane] = UInt8(_quantize_upper(max_z, pz, scale_z))

        var packed_meta = _pack_bytes(meta)
        compressed_u32[
            unsafe_offset=compressed_base + CWBVH_META_BASE
        ] = packed_meta[0]
        compressed_u32[
            unsafe_offset=compressed_base + CWBVH_META_BASE + 1
        ] = packed_meta[1]
        var planes = [qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z]
        for plane in range(6):
            var packed = _pack_bytes(planes[plane])
            compressed_u32[
                unsafe_offset=compressed_base + CWBVH_QUANTIZED_BASE + plane * 2
            ] = packed[0]
            compressed_u32[
                unsafe_offset=(
                    compressed_base + CWBVH_QUANTIZED_BASE + plane * 2 + 1
                )
            ] = packed[1]

    return (float_nodes^, compressed^)


def _trace_float(nodes: List[Float32], ray: Rayf32[.WORLD]) -> UInt64:
    var rcp_d = ray.reciprocal_direction[WIDTH]()
    var origin = ray.origin[WIDTH]()
    var origin_rcp_d = Vec3[.float32, .WORLD, WIDTH](
        origin.x * rcp_d.x, origin.y * rcp_d.y, origin.z * rcp_d.z
    )
    var checksum = UInt64(0)
    for _ in range(STREAM_PASSES):
        for node_idx in range(NODE_COUNT):
            var base = node_idx * WIDTH * WideNode.CHILD_STRIDE
            var aabb = AxisAlignedBoundingBox[.float32, .WORLD, WIDTH](
                Point3[.float32, .WORLD, WIDTH](
                    nodes.unsafe_ptr().unsafe_load[width=WIDTH](
                        base + WideNode.MIN_X * WIDTH
                    ),
                    nodes.unsafe_ptr().unsafe_load[width=WIDTH](
                        base + WideNode.MIN_Y * WIDTH
                    ),
                    nodes.unsafe_ptr().unsafe_load[width=WIDTH](
                        base + WideNode.MIN_Z * WIDTH
                    ),
                ),
                Point3[.float32, .WORLD, WIDTH](
                    nodes.unsafe_ptr().unsafe_load[width=WIDTH](
                        base + WideNode.MAX_X * WIDTH
                    ),
                    nodes.unsafe_ptr().unsafe_load[width=WIDTH](
                        base + WideNode.MAX_Y * WIDTH
                    ),
                    nodes.unsafe_ptr().unsafe_load[width=WIDTH](
                        base + WideNode.MAX_Z * WIDTH
                    ),
                ),
            )
            var hit = intersect_ray_aabb_octant_fma[
                positive_x=True, positive_y=True, positive_z=True
            ](origin_rcp_d, rcp_d, aabb, SIMD[.float32, WIDTH](f32_max))
            checksum += UInt64(pack_bits(hit.mask))
    return checksum


def _trace_quantized(nodes: List[Float32], ray: Rayf32[.WORLD]) -> UInt64:
    var rcp = ray.reciprocal_direction[1]()
    var checksum = UInt64(0)
    for _ in range(STREAM_PASSES):
        for node_idx in range(NODE_COUNT):
            var tasks = _intersect_cwbvh8_node_tasks(
                nodes.unsafe_ptr(),
                UInt32(node_idx),
                ray,
                rcp.x[0],
                rcp.y[0],
                rcp.z[0],
                f32_max,
                UInt32(0),
            )
            checksum += UInt64(tasks.triangle_group_mask)
    return checksum


def main():
    var nodes = _make_nodes()
    var float_nodes = nodes[0].copy()
    var compressed_nodes = nodes[1].copy()
    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Vec3f32[.WORLD](0.01, 0.02, 1.0),
    )
    var float_times = List[Int](capacity=REPEATS)
    var quantized_times = List[Int](capacity=REPEATS)
    var float_checksum = UInt64(0)
    var quantized_checksum = UInt64(0)
    for _ in range(REPEATS):
        var start = perf_counter_ns()
        float_checksum = _trace_float(float_nodes, ray)
        float_times.append(Int(perf_counter_ns() - start))
        start = perf_counter_ns()
        quantized_checksum = _trace_quantized(compressed_nodes, ray)
        quantized_times.append(Int(perf_counter_ns() - start))
    keep(float_checksum)
    keep(quantized_checksum)
    print("CPU BVH8 streaming node microbenchmark")
    print(t"Nodes: {NODE_COUNT}; passes: {STREAM_PASSES}")
    print(
        t"Float32: {round(ns_to_ms(_median(float_times)), 3)} ms; "
        t"{len(float_nodes) * 4} bytes; checksum={float_checksum}"
    )
    print(
        t"Quantized: {round(ns_to_ms(_median(quantized_times)), 3)} ms; "
        t"{len(compressed_nodes) * 4} bytes; checksum={quantized_checksum}"
    )
