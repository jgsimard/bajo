"""Refine world instance bounds from the built BLAS root frontier."""

from std.math import ceildiv, fma
from max.gpu import global_idx
from max.gpu.host import DeviceContext

from bajo.core import AABB, Point3f32
from bajo.core.aabb import transform_bounds_conservative
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
        comptime if layout == .CWBVH8:
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
        needs_refinement |= instance.transform.mixes_axes()
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
            if not instance.transform.mixes_axes():
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
                var transformed = transform_bounds_conservative(
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
