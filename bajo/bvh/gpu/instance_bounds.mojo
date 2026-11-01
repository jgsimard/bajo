"""Refine world instance bounds from the built BLAS root frontier."""

from std.math import abs, ceildiv, fma, min, max
from std.memory import bitcast
from max.gpu import global_idx
from max.gpu.host import DeviceContext

from bajo.core import AABB, Affine3f32, Point3f32
from bajo.bvh.constants import EMPTY_LANE, WideNode, PrimitiveKind
from bajo.bvh.types import BlasDesc, Instance
from bajo.bvh.gpu.blas_storage import GpuBlasSet, GpuBvhLayout
from bajo.bvh.gpu.compressed_bounds_bvh import (
    CWBVH_NODE_WORDS,
    CWBVH_EXP_IMASK,
    CWBVH_META_BASE,
    CWBVH_QUANTIZED_BASE,
    _scale_from_exponent,
)


def _instance_transform_mixes_axes(
    transform: Affine3f32[.LOCAL, .WORLD],
) -> Bool:
    """Only rows mixing local axes can tighten a transformed root union."""
    return (
        (transform.m00 != 0.0 and transform.m01 != 0.0)
        or (transform.m00 != 0.0 and transform.m02 != 0.0)
        or (transform.m01 != 0.0 and transform.m02 != 0.0)
        or (transform.m10 != 0.0 and transform.m11 != 0.0)
        or (transform.m10 != 0.0 and transform.m12 != 0.0)
        or (transform.m11 != 0.0 and transform.m12 != 0.0)
        or (transform.m20 != 0.0 and transform.m21 != 0.0)
        or (transform.m20 != 0.0 and transform.m22 != 0.0)
        or (transform.m21 != 0.0 and transform.m22 != 0.0)
    )


def _extract_blas_root_children[
    width: SIMDLength,
    layout: GpuBvhLayout,
](
    descriptors: Pointer[UInt32, ImmutAnyOrigin],
    nodes: Pointer[Float32, ImmutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    count: Int32,
):
    var index = global_idx.x
    if index >= Int(count) * width:
        return
    var desc = BlasDesc.load(descriptors, UInt32(index // width))
    var lane = index % width
    var bounds = AABB[.LOCAL].invalid()
    var active = False
    if desc.prim_count != 0:
        comptime if layout.compressed:
            comptime assert width == 8
            var base = (
                Int(desc.node_f32_base) + Int(desc.root_idx) * CWBVH_NODE_WORDS
            )
            var words = nodes.unsafe_bitcast[UInt32]()
            var byte = (
                words[unsafe_offset=base + CWBVH_META_BASE + lane // 4]
                >> UInt32((lane % 4) * 8)
            ) & UInt32(255)
            if byte != 0:
                active = True
                var exponents = words[unsafe_offset=base + CWBVH_EXP_IMASK]
                for axis in range(3):
                    var scale = _scale_from_exponent(
                        (exponents >> UInt32(axis * 8)) & UInt32(255)
                    )
                    var origin = nodes[unsafe_offset=base + axis]
                    var lower = words[
                        unsafe_offset=base
                        + CWBVH_QUANTIZED_BASE
                        + axis * 2
                        + lane // 4
                    ]
                    var upper = words[
                        unsafe_offset=base
                        + CWBVH_QUANTIZED_BASE
                        + (axis + 3) * 2
                        + lane // 4
                    ]
                    var shift = UInt32((lane % 4) * 8)
                    bounds._min[axis] = fma(
                        Float32((lower >> shift) & UInt32(255)), scale, origin
                    )
                    bounds._max[axis] = fma(
                        Float32((upper >> shift) & UInt32(255)), scale, origin
                    )
        else:
            var base = (
                Int(desc.node_f32_base)
                + Int(desc.root_idx) * width * WideNode.CHILD_STRIDE
            )
            var meta = nodes.unsafe_bitcast[UInt32]()[
                unsafe_offset=base + WideNode.META * width + lane
            ]
            if meta != EMPTY_LANE:
                active = True
                for axis in range(3):
                    bounds._min[axis] = nodes[
                        unsafe_offset=base + axis * width + lane
                    ]
                    bounds._max[axis] = nodes[
                        unsafe_offset=base + (axis + 3) * width + lane
                    ]
    for axis in range(3):
        output[unsafe_offset=index * 7 + axis] = bounds._min[axis]
        output[unsafe_offset=index * 7 + axis + 3] = bounds._max[axis]
    output[unsafe_offset=index * 7 + 6] = Float32(1 if active else 0)


def _outward_float32(value: Float64, upper: Bool) -> Float32:
    var rounded = Float32(value)
    if (upper and Float64(rounded) < value) or (
        not upper and Float64(rounded) > value
    ):
        if rounded == 0.0:
            return bitcast[.float32](UInt32(1) if upper else UInt32(0x80000001))
        var bits = bitcast[.uint32](rounded)
        if (rounded > 0.0) == upper:
            bits += 1
        else:
            bits -= 1
        rounded = bitcast[.float32](bits)
    return rounded


def _transform_instance_child_bounds(
    bounds: AABB[.LOCAL], transform: Affine3f32[.LOCAL, .WORLD]
) -> AABB[.WORLD]:
    """Accumulate intervals in Float64 and cover Float32 transform rounding.

    gamma(8) bounds the small affine evaluation's Float32 rounding, including
    cancellation through a sum-of-magnitudes bound. Subnormal absolute slack
    and outward conversion cover endpoint conversion/underflow as well.
    """
    var matrix = transform.flatten()
    var result = AABB[.WORLD].invalid()
    comptime u = Float64(5.9604644775390625e-8)
    comptime gamma = (8 * u) / (1 - 8 * u)
    for row in range(3):
        var lo = Float64(matrix[row * 4 + 3])
        var hi = lo
        var magnitude = abs(lo)
        for axis in range(3):
            var coefficient = Float64(matrix[row * 4 + axis])
            var a = coefficient * Float64(bounds._min[axis])
            var b = coefficient * Float64(bounds._max[axis])
            lo += min(a, b)
            hi += max(a, b)
            magnitude += max(abs(a), abs(b))
        var error = gamma * magnitude + Float64(8) * Float64(
            1.401298464324817e-45
        )
        result._min[row] = _outward_float32(lo - error, False)
        result._max[row] = _outward_float32(hi + error, True)
    return result


def refine_gpu_instance_bounds[
    kind: PrimitiveKind,
    layout: GpuBvhLayout,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    mut ctx: DeviceContext,
    blases: GpuBlasSet[kind, layout, node_width, leaf_width],
    instances: ImmSpan[Instance, _],
) raises -> List[Instance]:
    """Return instance copies with tighter conservative root-frontier bounds.

    Read back only root children, once per BLAS, rather than the full trees.
    Invalid/empty frontiers and non-improving candidates retain caller bounds.
    This adds one preparation synchronization, never per-ray work.
    """
    var result = List[Instance](capacity=len(instances))
    if len(instances) == 0 or blases.blas_count == 0:
        result.extend(instances)
        return result^
    var needs_refinement = False
    for instance in instances:
        if Int(instance.blas_idx) >= blases.blas_count:
            raise Error(
                "instance BLAS index out of range during bounds refinement"
            )
        needs_refinement |= _instance_transform_mixes_axes(instance.transform)
    if not needs_refinement:
        result.extend(instances)
        return result^
    var children = ctx.enqueue_create_buffer[.float32](
        blases.blas_count * node_width * 7
    )
    ctx.enqueue_function[_extract_blas_root_children[node_width, layout]](
        blases.descs,
        blases.nodes,
        children,
        Int32(blases.blas_count),
        grid_dim=ceildiv(blases.blas_count * node_width, 128),
        block_dim=128,
    )
    with children.map_to_host() as mapped:
        for instance in instances:
            if not _instance_transform_mixes_axes(instance.transform):
                result.append(instance.copy())
                continue
            var candidate = AABB[.WORLD].invalid()
            var complete = True
            for lane in range(node_width):
                var base = (Int(instance.blas_idx) * node_width + lane) * 7
                if mapped[base + 6] == 0.0:
                    continue
                var bounds = AABB[.LOCAL].load6(mapped.unsafe_ptr(), base)
                if not bounds.is_valid()[0]:
                    complete = False
                    break
                var transformed = _transform_instance_child_bounds(
                    bounds, instance.transform
                )
                if not transformed.is_valid()[0]:
                    complete = False
                    break
                candidate.grow(transformed)
            var refined = instance.copy()
            if (
                complete
                and candidate.is_valid()[0]
                and candidate.surface_area() < instance.bounds.surface_area()
            ):
                refined.bounds = candidate
            result.append(refined^)
    return result^
