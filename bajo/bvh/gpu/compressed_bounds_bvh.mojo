from std.atomic import Atomic, Ordering
from std.gpu import global_idx
from std.math import ceil, ceildiv, floor, fma, max, min, clamp
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import (
    EMPTY_LANE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    WideNode,
    f32_max,
)
from bajo.bvh.gpu.builder.binary_layout import _segment_for_item
from bajo.bvh.gpu.utils import _device_span
from bajo.bvh.wide_meta import (
    _wide_meta_count,
    _wide_meta_data,
    _wide_node_index,
)
from bajo.core import Frame, Rayf32
from bajo.core.utils import fmax, fmin


# Ylitie, Karras, Laine, HPG 2017, Figure 4.
comptime CWBVH_WIDTH = 8
comptime CWBVH_MAX_LEAF_SIZE = 3
comptime CWBVH_LEAF_STORAGE_WIDTH = 4
comptime CWBVH_NODE_WORDS = 20
comptime CWBVH_NODE_BYTES = 80
comptime CWBVH_TRIANGLE_WORDS = 12

comptime CWBVH_PX = 0
comptime CWBVH_PY = 1
comptime CWBVH_PZ = 2
comptime CWBVH_EXP_IMASK = 3
comptime CWBVH_CHILD_BASE = 4
comptime CWBVH_TRIANGLE_BASE = 5
comptime CWBVH_META_BASE = 6
comptime CWBVH_QUANTIZED_BASE = 8


@always_inline
def _power_of_two_scale_exponent(extent: Float32) -> UInt32:
    if extent <= 0.0:
        return 127  # 2^0

    var required = extent / 255.0
    var bits = bitcast[DType.uint32](required)
    var exponent = (bits >> 23) & 0xFF
    if (bits & 0x007FFFFF) != 0:
        exponent += 1
    return clamp(exponent, 1, 254)


@always_inline
def _scale_from_exponent(exponent: UInt32) -> Float32:
    return bitcast[DType.float32](exponent << 23)


@always_inline
def _quantize_lower(value: Float32, start: Float32, scale: Float32) -> UInt32:
    var q = Int(floor((value - start) / scale))
    q = clamp(q, 0, 255)
    if fma(Float32(q), scale, start) > value and q > 0:
        q -= 1
    return UInt32(q)


@always_inline
def _quantize_upper(value: Float32, start: Float32, scale: Float32) -> UInt32:
    var q = Int(ceil((value - start) / scale))
    q = clamp(q, 0, 255)
    if fma(Float32(q), scale, start) < value and q < 255:
        q += 1
    return UInt32(q)


@always_inline
def _unary_triangle_count(count: UInt32) -> UInt32:
    # Paper encoding: 1 -> 001, 2 -> 011, 3 -> 111 in the high bits.
    return ((1 << count) - 1) << 5


def _encode_cwbvh8_node[
    leaf_width: SIMDLength,
](
    wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    leaf_block_indices: Pointer[UInt32, ImmutAnyOrigin],
    cwbvh_nodes: Pointer[Float32, MutAnyOrigin],
    compact_primitive_ids: Pointer[UInt32, MutAnyOrigin],
    triangle_counter: Pointer[UInt32, MutAnyOrigin],
    node_idx_i: Int,
):
    comptime assert leaf_width == CWBVH_LEAF_STORAGE_WIDTH
    var node_idx = UInt32(node_idx_i)
    var lo_x = f32_max
    var lo_y = f32_max
    var lo_z = f32_max
    var hi_x = -f32_max
    var hi_y = -f32_max
    var hi_z = -f32_max
    var metadata = SIMD[DType.uint32, CWBVH_WIDTH](0)
    var leaf_triangle_count = UInt32(0)
    var child_base = UInt32(0)
    var internal_count = UInt32(0)
    var imask = UInt32(0)
    var valid_count = 0

    comptime for lane in range(CWBVH_WIDTH):
        var meta = wide_nodes.unsafe_bitcast[UInt32]()[
            unsafe_offset=_wide_node_index[CWBVH_WIDTH](
                node_idx, WideNode.META, lane
            )
        ]
        metadata[lane] = meta
        var count = _wide_meta_count(meta)
        if count == EMPTY_LANE:
            continue

        valid_count += 1
        lo_x = min(
            lo_x,
            wide_nodes[
                unsafe_offset=_wide_node_index[CWBVH_WIDTH](
                    node_idx, WideNode.MIN_X, lane
                )
            ],
        )
        lo_y = min(
            lo_y,
            wide_nodes[
                unsafe_offset=_wide_node_index[CWBVH_WIDTH](
                    node_idx, WideNode.MIN_Y, lane
                )
            ],
        )
        lo_z = min(
            lo_z,
            wide_nodes[
                unsafe_offset=_wide_node_index[CWBVH_WIDTH](
                    node_idx, WideNode.MIN_Z, lane
                )
            ],
        )
        hi_x = max(
            hi_x,
            wide_nodes[
                unsafe_offset=_wide_node_index[CWBVH_WIDTH](
                    node_idx, WideNode.MAX_X, lane
                )
            ],
        )
        hi_y = max(
            hi_y,
            wide_nodes[
                unsafe_offset=_wide_node_index[CWBVH_WIDTH](
                    node_idx, WideNode.MAX_Y, lane
                )
            ],
        )
        hi_z = max(
            hi_z,
            wide_nodes[
                unsafe_offset=_wide_node_index[CWBVH_WIDTH](
                    node_idx, WideNode.MAX_Z, lane
                )
            ],
        )

        if count == 0:
            var child = _wide_meta_data(meta)
            if internal_count == 0:
                child_base = child
            debug_assert["safe", _use_compiler_assume=True](
                child == child_base + internal_count,
                "CWBVH8 requires contiguous internal children",
            )
            internal_count += 1
            imask |= UInt32(1) << UInt32(lane)
        else:
            debug_assert["safe", _use_compiler_assume=True](
                count <= UInt32(CWBVH_MAX_LEAF_SIZE)
            )
            leaf_triangle_count += count

    debug_assert["safe", _use_compiler_assume=True](valid_count > 0)
    debug_assert["safe", _use_compiler_assume=True](
        leaf_triangle_count <= UInt32(24),
        "CWBVH8 supports at most 24 triangles referenced by one node",
    )

    var triangle_base = Atomic.fetch_add[ordering=Ordering.RELAXED](
        triangle_counter, leaf_triangle_count
    )
    var exponent_x = _power_of_two_scale_exponent(hi_x - lo_x)
    var exponent_y = _power_of_two_scale_exponent(hi_y - lo_y)
    var exponent_z = _power_of_two_scale_exponent(hi_z - lo_z)
    var scale_x = _scale_from_exponent(exponent_x)
    var scale_y = _scale_from_exponent(exponent_y)
    var scale_z = _scale_from_exponent(exponent_z)
    if fma(Float32(255), scale_x, lo_x) < hi_x and exponent_x < UInt32(254):
        exponent_x += 1
        scale_x = _scale_from_exponent(exponent_x)
    if fma(Float32(255), scale_y, lo_y) < hi_y and exponent_y < UInt32(254):
        exponent_y += 1
        scale_y = _scale_from_exponent(exponent_y)
    if fma(Float32(255), scale_z, lo_z) < hi_z and exponent_z < UInt32(254):
        exponent_z += 1
        scale_z = _scale_from_exponent(exponent_z)

    var base = node_idx_i * CWBVH_NODE_WORDS
    cwbvh_nodes[unsafe_offset=base + CWBVH_PX] = lo_x
    cwbvh_nodes[unsafe_offset=base + CWBVH_PY] = lo_y
    cwbvh_nodes[unsafe_offset=base + CWBVH_PZ] = lo_z
    var cwbvh_u32 = cwbvh_nodes.unsafe_bitcast[UInt32]()
    cwbvh_u32[unsafe_offset=base + CWBVH_EXP_IMASK] = (
        exponent_x
        | (exponent_y << UInt32(8))
        | (exponent_z << UInt32(16))
        | (imask << UInt32(24))
    )
    cwbvh_u32[unsafe_offset=base + CWBVH_CHILD_BASE] = child_base
    cwbvh_u32[unsafe_offset=base + CWBVH_TRIANGLE_BASE] = triangle_base

    var leaf_offset = UInt32(0)
    var packed_meta0 = UInt32(0)
    var packed_meta1 = UInt32(0)
    comptime for lane in range(CWBVH_WIDTH):
        var source_meta = metadata[lane]
        var count = _wide_meta_count(source_meta)
        var byte = UInt32(0)
        if count == 0:
            # Published internal encoding: 001 in the high bits and slot+24
            # in the low five bits. imask distinguishes it from a leaf.
            byte = UInt32(0x20) | (UInt32(24) + UInt32(lane))
        elif count != EMPTY_LANE:
            byte = _unary_triangle_count(count) | leaf_offset
            var leaf_block = _wide_meta_data(source_meta)
            for item in range(Int(count)):
                compact_primitive_ids[
                    unsafe_offset=Int(triangle_base + leaf_offset) + item
                ] = leaf_block_indices[
                    unsafe_offset=Int(leaf_block) * leaf_width + item
                ]
            leaf_offset += count

        comptime if lane < 4:
            packed_meta0 |= byte << UInt32(lane * 8)
        else:
            packed_meta1 |= byte << UInt32((lane - 4) * 8)

    cwbvh_u32[unsafe_offset=base + CWBVH_META_BASE + 0] = packed_meta0
    cwbvh_u32[unsafe_offset=base + CWBVH_META_BASE + 1] = packed_meta1

    comptime for plane in range(6):
        comptime for group in range(2):
            var packed = UInt32(0)
            comptime for byte_i in range(4):
                comptime lane = group * 4 + byte_i
                var q = UInt32(0)
                if _wide_meta_count(metadata[lane]) != EMPTY_LANE:
                    comptime if plane == 0:
                        q = _quantize_lower(
                            wide_nodes[
                                unsafe_offset=_wide_node_index[CWBVH_WIDTH](
                                    node_idx, WideNode.MIN_X, lane
                                )
                            ],
                            lo_x,
                            scale_x,
                        )
                    elif plane == 1:
                        q = _quantize_lower(
                            wide_nodes[
                                unsafe_offset=_wide_node_index[CWBVH_WIDTH](
                                    node_idx, WideNode.MIN_Y, lane
                                )
                            ],
                            lo_y,
                            scale_y,
                        )
                    elif plane == 2:
                        q = _quantize_lower(
                            wide_nodes[
                                unsafe_offset=_wide_node_index[CWBVH_WIDTH](
                                    node_idx, WideNode.MIN_Z, lane
                                )
                            ],
                            lo_z,
                            scale_z,
                        )
                    elif plane == 3:
                        q = _quantize_upper(
                            wide_nodes[
                                unsafe_offset=_wide_node_index[CWBVH_WIDTH](
                                    node_idx, WideNode.MAX_X, lane
                                )
                            ],
                            lo_x,
                            scale_x,
                        )
                    elif plane == 4:
                        q = _quantize_upper(
                            wide_nodes[
                                unsafe_offset=_wide_node_index[CWBVH_WIDTH](
                                    node_idx, WideNode.MAX_Y, lane
                                )
                            ],
                            lo_y,
                            scale_y,
                        )
                    else:
                        q = _quantize_upper(
                            wide_nodes[
                                unsafe_offset=_wide_node_index[CWBVH_WIDTH](
                                    node_idx, WideNode.MAX_Z, lane
                                )
                            ],
                            lo_z,
                            scale_z,
                        )
                packed |= q << UInt32(byte_i * 8)
            cwbvh_u32[
                unsafe_offset=base + CWBVH_QUANTIZED_BASE + plane * 2 + group
            ] = packed


def encode_segmented_cwbvh8_nodes_kernel[
    leaf_width: SIMDLength,
](
    wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    leaf_block_indices: Pointer[UInt32, ImmutAnyOrigin],
    node_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    leaf_block_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    primitive_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    node_counts: Pointer[UInt32, ImmutAnyOrigin],
    cwbvh_nodes: Pointer[Float32, MutAnyOrigin],
    compact_primitive_ids: Pointer[UInt32, MutAnyOrigin],
    triangle_counters: Pointer[UInt32, MutAnyOrigin],
):
    """Encode packed wide segments while retaining local CWBVH indices."""
    comptime assert leaf_width == CWBVH_LEAF_STORAGE_WIDTH
    var physical_node = global_idx.x
    var node_capacity = Int(
        node_segment_offsets.unsafe_get(len(node_segment_offsets) - 1)
    )
    if physical_node >= node_capacity:
        return

    var segment_idx = _segment_for_item(node_segment_offsets, physical_node)
    var node_begin = Int(node_segment_offsets.unsafe_get(segment_idx))
    var local_node = physical_node - node_begin
    if local_node >= Int(node_counts[unsafe_offset=segment_idx]):
        return

    var leaf_block_begin = Int(
        leaf_block_segment_offsets.unsafe_get(segment_idx)
    )
    var primitive_begin = Int(primitive_segment_offsets.unsafe_get(segment_idx))
    _encode_cwbvh8_node[leaf_width](
        wide_nodes.unsafe_offset(
            node_begin * CWBVH_WIDTH * WideNode.CHILD_STRIDE
        ),
        leaf_block_indices.unsafe_offset(leaf_block_begin * leaf_width),
        cwbvh_nodes.unsafe_offset(node_begin * CWBVH_NODE_WORDS),
        compact_primitive_ids.unsafe_offset(primitive_begin),
        triangle_counters.unsafe_offset(segment_idx),
        local_node,
    )


def pack_segmented_cwbvh_triangles_kernel(
    vertices: Pointer[Float32, ImmutAnyOrigin],
    primitive_ids: Pointer[UInt32, ImmutAnyOrigin],
    primitive_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    triangles: Pointer[Float32, MutAnyOrigin],
    triangle_count: Int32,
):
    var triangle_idx = global_idx.x
    if triangle_idx >= Int(triangle_count):
        return

    var segment_idx = _segment_for_item(primitive_segment_offsets, triangle_idx)
    var primitive_begin = Int(primitive_segment_offsets.unsafe_get(segment_idx))
    var prim = primitive_ids[unsafe_offset=triangle_idx]
    var src = Int(prim) * 9
    var dst = triangle_idx * CWBVH_TRIANGLE_WORDS
    var v0x = vertices[unsafe_offset=src + 0]
    var v0y = vertices[unsafe_offset=src + 1]
    var v0z = vertices[unsafe_offset=src + 2]
    triangles[unsafe_offset=dst + 0] = vertices[unsafe_offset=src + 3] - v0x
    triangles[unsafe_offset=dst + 1] = vertices[unsafe_offset=src + 4] - v0y
    triangles[unsafe_offset=dst + 2] = vertices[unsafe_offset=src + 5] - v0z
    triangles[unsafe_offset=dst + 3] = 0.0
    triangles[unsafe_offset=dst + 4] = vertices[unsafe_offset=src + 6] - v0x
    triangles[unsafe_offset=dst + 5] = vertices[unsafe_offset=src + 7] - v0y
    triangles[unsafe_offset=dst + 6] = vertices[unsafe_offset=src + 8] - v0z
    triangles[unsafe_offset=dst + 7] = 0.0
    triangles[unsafe_offset=dst + 8] = v0x
    triangles[unsafe_offset=dst + 9] = v0y
    triangles[unsafe_offset=dst + 10] = v0z
    triangles.unsafe_bitcast[UInt32]()[unsafe_offset=dst + 11] = prim - UInt32(
        primitive_begin
    )


def enqueue_segmented_cwbvh8_representation[
    leaf_width: SIMDLength,
](
    mut ctx: DeviceContext,
    wide_nodes: DeviceBuffer[.float32],
    leaf_block_indices: DeviceBuffer[.uint32],
    node_segment_offsets: DeviceBuffer[.uint32],
    leaf_block_segment_offsets: DeviceBuffer[.uint32],
    primitive_segment_offsets: DeviceBuffer[.uint32],
    node_counts: DeviceBuffer[.uint32],
    vertices: DeviceBuffer[.float32],
    cwbvh_nodes: DeviceBuffer[.float32],
    triangles: DeviceBuffer[.float32],
) raises -> DeviceBuffer[.uint32]:
    """Queue one CWBVH8 encoding and triangle pack for all segments."""
    comptime assert leaf_width == CWBVH_LEAF_STORAGE_WIDTH
    var segment_count = len(primitive_segment_offsets) - 1
    var node_capacity = len(cwbvh_nodes) / CWBVH_NODE_WORDS
    var triangle_count = len(triangles) / CWBVH_TRIANGLE_WORDS
    var compact_primitive_ids = ctx.enqueue_create_buffer[.uint32](
        triangle_count
    )
    var triangle_counters = ctx.enqueue_create_buffer[.uint32](
        segment_count
    )
    ctx.enqueue_memset(triangle_counters, 0)
    ctx.enqueue_function[encode_segmented_cwbvh8_nodes_kernel[leaf_width]](
        wide_nodes,
        leaf_block_indices,
        _device_span[mut=False](node_segment_offsets),
        _device_span[mut=False](leaf_block_segment_offsets),
        _device_span[mut=False](primitive_segment_offsets),
        node_counts,
        cwbvh_nodes,
        compact_primitive_ids,
        triangle_counters,
        grid_dim=ceildiv(node_capacity, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    ctx.enqueue_function[pack_segmented_cwbvh_triangles_kernel](
        vertices,
        compact_primitive_ids,
        _device_span[mut=False](primitive_segment_offsets),
        triangles,
        Int32(triangle_count),
        grid_dim=ceildiv(triangle_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    return triangle_counters^


@fieldwise_init
struct Cwbvh8NodeTasks(TrivialRegisterPassable):
    """Native CWBVH8 traversal tasks emitted by one compressed node."""

    var child_base: UInt32
    var triangle_base: UInt32
    var node_group_mask: UInt32
    var triangle_group_mask: UInt32


@always_inline
def _intersect_cwbvh8_node_tasks[
    frame: Frame,
](
    cwbvh_nodes: ImmPointer[Float32, _],
    node_idx: UInt32,
    ray: Rayf32[frame],
    rcp_x: Float32,
    rcp_y: Float32,
    rcp_z: Float32,
    t_max: Float32,
    octant_inverse: UInt32,
) -> Cwbvh8NodeTasks:
    """Decode one 80-byte node directly into the paper's task masks."""
    var base = Int(node_idx) * CWBVH_NODE_WORDS
    var px = cwbvh_nodes[unsafe_offset=base + CWBVH_PX]
    var py = cwbvh_nodes[unsafe_offset=base + CWBVH_PY]
    var pz = cwbvh_nodes[unsafe_offset=base + CWBVH_PZ]
    var cwbvh_u32 = cwbvh_nodes.unsafe_bitcast[UInt32]()
    var exp_imask = cwbvh_u32[unsafe_offset=base + CWBVH_EXP_IMASK]
    var scale_x = _scale_from_exponent(exp_imask & UInt32(0xFF))
    var scale_y = _scale_from_exponent((exp_imask >> UInt32(8)) & UInt32(0xFF))
    var scale_z = _scale_from_exponent((exp_imask >> UInt32(16)) & UInt32(0xFF))
    var imask = exp_imask >> UInt32(24)
    var child_base = cwbvh_u32[unsafe_offset=base + CWBVH_CHILD_BASE]
    var triangle_base = cwbvh_u32[unsafe_offset=base + CWBVH_TRIANGLE_BASE]
    var meta_words = SIMD[DType.uint32, 2](
        cwbvh_u32[unsafe_offset=base + CWBVH_META_BASE + 0],
        cwbvh_u32[unsafe_offset=base + CWBVH_META_BASE + 1],
    )

    var qlo_x = SIMD[DType.float32, CWBVH_WIDTH](0.0)
    var qlo_y = SIMD[DType.float32, CWBVH_WIDTH](0.0)
    var qlo_z = SIMD[DType.float32, CWBVH_WIDTH](0.0)
    var qhi_x = SIMD[DType.float32, CWBVH_WIDTH](0.0)
    var qhi_y = SIMD[DType.float32, CWBVH_WIDTH](0.0)
    var qhi_z = SIMD[DType.float32, CWBVH_WIDTH](0.0)
    comptime for lane in range(CWBVH_WIDTH):
        comptime qgroup = lane / 4
        comptime qshift = UInt32((lane % 4) * 8)
        comptime for plane in range(6):
            var packed = cwbvh_u32[
                unsafe_offset=base + CWBVH_QUANTIZED_BASE + plane * 2 + qgroup
            ]
            var q = Float32((packed >> qshift) & UInt32(0xFF))
            comptime if plane == 0:
                qlo_x[lane] = q
            elif plane == 1:
                qlo_y[lane] = q
            elif plane == 2:
                qlo_z[lane] = q
            elif plane == 3:
                qhi_x[lane] = q
            elif plane == 4:
                qhi_y[lane] = q
            else:
                qhi_z[lane] = q

    var near_x = qlo_x
    var near_y = qlo_y
    var near_z = qlo_z
    var far_x = qhi_x
    var far_y = qhi_y
    var far_z = qhi_z
    if ray.d.x < 0.0:
        near_x = qhi_x
        far_x = qlo_x
    if ray.d.y < 0.0:
        near_y = qhi_y
        far_y = qlo_y
    if ray.d.z < 0.0:
        near_z = qhi_z
        far_z = qlo_z

    var idir_x = scale_x * rcp_x
    var idir_y = scale_y * rcp_y
    var idir_z = scale_z * rcp_z
    var origin_x = (px - ray.o.x) * rcp_x
    var origin_y = (py - ray.o.y) * rcp_y
    var origin_z = (pz - ray.o.z) * rcp_z
    var tnear_x = fma(near_x, idir_x, origin_x)
    var tnear_y = fma(near_y, idir_y, origin_y)
    var tnear_z = fma(near_z, idir_z, origin_z)
    var tfar_x = fma(far_x, idir_x, origin_x)
    var tfar_y = fma(far_y, idir_y, origin_y)
    var tfar_z = fma(far_z, idir_z, origin_z)
    var tnear = fmax(fmax(tnear_x, tnear_y), fmax(tnear_z, 0.0))
    var tfar = fmin(
        fmin(tfar_x, tfar_y),
        fmin(tfar_z, SIMD[DType.float32, CWBVH_WIDTH](t_max)),
    )
    var bounds_mask = tnear.le(tfar)
    var hitmask = UInt32(0)
    comptime for lane in range(CWBVH_WIDTH):
        comptime group = lane / 4
        comptime shift = UInt32((lane % 4) * 8)
        var byte = (meta_words[group] >> shift) & UInt32(0xFF)
        if byte != 0 and bounds_mask[lane]:
            var bit_index = byte & UInt32(0x1F)
            if (imask & (UInt32(1) << UInt32(lane))) != 0:
                bit_index ^= octant_inverse
            hitmask |= (byte >> UInt32(5)) << bit_index

    return Cwbvh8NodeTasks(
        child_base,
        triangle_base,
        (hitmask & UInt32(0xFF000000)) | imask,
        hitmask & UInt32(0x00FFFFFF),
    )
