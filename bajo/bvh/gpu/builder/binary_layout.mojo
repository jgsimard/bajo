from std.math import max, ceildiv
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.core import AABB, Frame
from bajo.bvh.constants import (
    LBVH_LEAF_FLAG,
    LBVH_INDEX_MASK,
    BinaryBvhNode,
    REDUCED_BOUNDS_STRIDE,
    BOUNDS_REDUCE_CHUNK,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)
from bajo.bvh.gpu.utils import (
    GpuBuildTimings,
    GpuBVHValidation,
    _device_span,
    upload_list,
)
from bajo.bvh.gpu.validate import (
    validate_sorted_keys,
    validate_topology,
    validate_refit_bounds,
)


def _node_meta_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * BinaryBvhNode.META_STRIDE


def _node_bounds_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * BinaryBvhNode.BOUNDS_STRIDE


def _node_parent_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + BinaryBvhNode.PARENT


def _node_left_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + BinaryBvhNode.LEFT


def _node_right_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + BinaryBvhNode.RIGHT


def _node_left(
    node_meta: Span[mut=False, UInt32, _], node_idx: UInt32
) -> UInt32:
    var base = _node_meta_base(node_idx)
    debug_assert["safe", _use_compiler_assume=True](
        base >= 0 and base <= len(node_meta) - BinaryBvhNode.META_STRIDE,
        "binary node metadata is outside the input span",
    )
    return node_meta.unsafe_get(base + BinaryBvhNode.LEFT)


def _node_right(
    node_meta: Span[mut=False, UInt32, _], node_idx: UInt32
) -> UInt32:
    var base = _node_meta_base(node_idx)
    debug_assert["safe", _use_compiler_assume=True](
        base >= 0 and base <= len(node_meta) - BinaryBvhNode.META_STRIDE,
        "binary node metadata is outside the input span",
    )
    return node_meta.unsafe_get(base + BinaryBvhNode.RIGHT)


# Raw-pointer overloads remain for GPU kernels that have not yet crossed a
# length-carrying ABI boundary. They are removed as those kernels migrate.
def _node_left(
    node_meta: UnsafePointer[mut=False, UInt32, _], node_idx: UInt32
) -> UInt32:
    return node_meta[unsafe_offset=_node_left_index(node_idx)]


def _node_right(
    node_meta: UnsafePointer[mut=False, UInt32, _], node_idx: UInt32
) -> UInt32:
    return node_meta[unsafe_offset=_node_right_index(node_idx)]


def _is_encoded_leaf(encoded: UInt32) -> Bool:
    return (encoded & LBVH_LEAF_FLAG) != 0


def _encoded_index(encoded: UInt32) -> UInt32:
    return encoded & LBVH_INDEX_MASK


def _write_child_bounds(
    node_bounds: Span[mut=True, Float32, _],
    parent: UInt32,
    write_left: Bool,
    bounds: AABB,
):
    var b = _node_bounds_base(parent)
    if not write_left:
        b += 6
    bounds.store6(node_bounds, b)


def _load_and_union_node_bounds(
    node_bounds: Span[mut=False, Float32, _], parent: UInt32
) -> AABB[Frame.WORLD]:
    var b = _node_bounds_base(parent)
    b1 = AABB[Frame.WORLD].load6(node_bounds, b)
    b2 = AABB[Frame.WORLD].load6(node_bounds, b + 6)
    return AABB.merge(b1, b2)


def init_empty_bounds_kernel(bounds: Span[mut=True, Float32, MutAnyOrigin]):
    var node_count = len(bounds) / BinaryBvhNode.BOUNDS_STRIDE
    var i = global_idx.x
    if i >= node_count:
        return

    var b = i * BinaryBvhNode.BOUNDS_STRIDE
    var invalid = AABB[Frame.WORLD].invalid()
    invalid.store6(bounds, b)
    invalid.store6(bounds, b + AABB.STRIDE)


def compute_bounds_partials_kernel(
    leaf_bounds: Span[mut=False, Float32, ImmutAnyOrigin],
    out_partials: Span[mut=True, Float32, MutAnyOrigin],
):
    var leaf_count = len(leaf_bounds) / AABB.STRIDE
    var chunk = global_idx.x
    var first = chunk * BOUNDS_REDUCE_CHUNK
    if first >= leaf_count:
        return

    var last = min(first + BOUNDS_REDUCE_CHUNK, leaf_count)
    var bounds = AABB[Frame.WORLD].invalid()
    var centroid_bounds = AABB[Frame.WORLD].invalid()

    for leaf_idx in range(first, last):
        var b = leaf_idx * AABB.STRIDE
        var aabb = AABB[Frame.WORLD].load6(leaf_bounds, b)

        bounds.grow(aabb)
        centroid_bounds.grow(aabb.centroid())

    var out = chunk * REDUCED_BOUNDS_STRIDE
    bounds.store6(out_partials, out)
    centroid_bounds.store6(out_partials, out + AABB.STRIDE)


def reduce_bounds_partials_kernel(
    in_partials: Span[mut=False, Float32, ImmutAnyOrigin],
    out_partials: Span[mut=True, Float32, MutAnyOrigin],
    partial_count: Int32,
):
    var partial_count_int = Int(partial_count)
    var chunk = global_idx.x
    var first = chunk * BOUNDS_REDUCE_CHUNK
    if first >= partial_count_int:
        return

    var last = min(first + BOUNDS_REDUCE_CHUNK, partial_count_int)
    var bounds = AABB[Frame.WORLD].invalid()
    var centroid_bounds = AABB[Frame.WORLD].invalid()

    for i in range(first, last):
        var b = i * REDUCED_BOUNDS_STRIDE

        var partial_bounds = AABB[Frame.WORLD].load6(in_partials, b)
        var partial_centroid_bounds = AABB[Frame.WORLD].load6(
            in_partials, b + AABB.STRIDE
        )

        bounds.grow(partial_bounds)
        centroid_bounds.grow(partial_centroid_bounds)

    var out = chunk * REDUCED_BOUNDS_STRIDE
    bounds.store6(out_partials, out)
    centroid_bounds.store6(out_partials, out + AABB.STRIDE)


struct GpuBinaryBoundsBvh:
    var leaf_count: Int
    var internal_count: Int

    var blocks_leaves: Int
    var blocks_internal: Int
    var blocks_init: Int

    var bounds_device: DeviceBuffer[DType.float32]
    """[0..5]  = root bounds, [6..11] = centroid bounds."""
    var bounds_scratch_a: DeviceBuffer[DType.float32]
    var bounds_scratch_b: DeviceBuffer[DType.float32]

    var leaf_bounds: DeviceBuffer[DType.float32]
    var leaf_payloads: DeviceBuffer[DType.uint32]

    var keys: DeviceBuffer[DType.uint32]
    """Morton keys."""
    var leaf_ids: DeviceBuffer[DType.uint32]

    # Binary node layout:
    #   node_meta   : parent, left encoded child, right encoded child, fence/range end
    #   leaf_parent : parent internal node for each sorted leaf
    #   node_bounds : two child AABBs per internal node, 12 floats total
    #   node_flags  : refit synchronization flags
    var node_meta: DeviceBuffer[DType.uint32]
    var leaf_parent: DeviceBuffer[DType.uint32]
    var node_bounds: DeviceBuffer[DType.float32]
    var node_flags: DeviceBuffer[DType.uint32]
    var node_leaf_counts: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        leaf_bounds: DeviceBuffer[DType.float32],
        leaf_payloads: DeviceBuffer[DType.uint32],
    ) raises:
        self.leaf_count = len(leaf_payloads)
        debug_assert["safe", _use_compiler_assume=True](
            self.leaf_count > 0, "passed empty input."
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(leaf_bounds) == self.leaf_count * AABB.STRIDE,
            "leaf bounds buffer has the wrong length",
        )
        self.internal_count = self.leaf_count - 1

        var n_leaf = self.leaf_count
        var n_internal = max(self.internal_count, 1)

        self.blocks_leaves = ceildiv(n_leaf, GPU_BOUNDS_BVH_BLOCK_SIZE)
        self.blocks_internal = ceildiv(
            n_internal,
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        self.blocks_init = ceildiv(
            max(n_leaf, n_internal),
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        self.leaf_bounds = leaf_bounds
        self.leaf_payloads = leaf_payloads

        self.bounds_device = ctx.enqueue_create_buffer[DType.float32](
            REDUCED_BOUNDS_STRIDE
        )

        var partial_count = ceildiv(
            self.leaf_count,
            BOUNDS_REDUCE_CHUNK,
        )

        self.bounds_scratch_a = ctx.enqueue_create_buffer[DType.float32](
            max(partial_count, 1) * REDUCED_BOUNDS_STRIDE
        )
        self.bounds_scratch_b = ctx.enqueue_create_buffer[DType.float32](
            max(
                ceildiv(partial_count, BOUNDS_REDUCE_CHUNK),
                1,
            )
            * REDUCED_BOUNDS_STRIDE
        )

        var reduce_grid = ceildiv(
            partial_count,
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        ctx.enqueue_function[compute_bounds_partials_kernel](
            _device_span[mut=False](self.leaf_bounds),
            _device_span[mut=True](self.bounds_scratch_a),
            grid_dim=reduce_grid,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        var in_buf = self.bounds_scratch_a.copy()
        var out_buf = self.bounds_scratch_b.copy()
        var count = partial_count

        while count > 1:
            var next_count = ceildiv(
                count,
                BOUNDS_REDUCE_CHUNK,
            )
            var grid = ceildiv(
                next_count,
                GPU_BOUNDS_BVH_BLOCK_SIZE,
            )

            ctx.enqueue_function[reduce_bounds_partials_kernel](
                _device_span[mut=False](in_buf),
                _device_span[mut=True](out_buf),
                Int32(count),
                grid_dim=grid,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )

            swap(in_buf, out_buf)
            count = next_count
        in_buf.enqueue_copy_to(self.bounds_device)

        self.keys = ctx.enqueue_create_buffer[DType.uint32](n_leaf)
        self.leaf_ids = ctx.enqueue_create_buffer[DType.uint32](n_leaf)

        self.node_meta = ctx.enqueue_create_buffer[DType.uint32](
            n_internal * BinaryBvhNode.META_STRIDE
        )
        self.leaf_parent = ctx.enqueue_create_buffer[DType.uint32](n_leaf)
        self.node_leaf_counts = ctx.enqueue_create_buffer[DType.uint32](
            n_internal
        )
        self.node_bounds = ctx.enqueue_create_buffer[DType.float32](
            n_internal * BinaryBvhNode.BOUNDS_STRIDE
        )
        self.node_flags = ctx.enqueue_create_buffer[DType.uint32](n_internal)

    def root_bounds(self) raises -> AABB[Frame.WORLD]:
        with self.bounds_device.map_to_host() as h:
            return AABB[Frame.WORLD].load6(
                Span(unsafe_ptr=h.unsafe_ptr(), length=len(h)), 0
            )

    def centroid_bounds(self) raises -> AABB[Frame.WORLD]:
        with self.bounds_device.map_to_host() as h:
            return AABB[Frame.WORLD].load6(
                Span(unsafe_ptr=h.unsafe_ptr(), length=len(h)), AABB.STRIDE
            )

    def validate(self, bounds: AABB) raises -> GpuBVHValidation:
        var sorted_validation = validate_sorted_keys(
            self.keys,
            self.leaf_ids,
            self.leaf_count,
        )

        if self.leaf_count <= 1:
            return GpuBVHValidation(
                sorted_validation.sorted_ok,
                sorted_validation.values_ok,
                True,
                UInt32(1),
                UInt32(0),
                True,
                0.0,
                UInt32(0),
                sorted_validation.guard,
            )

        var topo_validation = validate_topology(
            self.node_meta,
            self.leaf_parent,
            self.leaf_count,
        )
        var refit_validation = validate_refit_bounds(
            self.node_bounds,
            self.node_flags,
            self.node_meta,
            self.leaf_count,
            bounds,
        )
        var guard = (
            sorted_validation.guard
            + topo_validation.guard
            + refit_validation.guard
        )

        return GpuBVHValidation(
            sorted_validation.sorted_ok,
            sorted_validation.values_ok,
            topo_validation.ok,
            topo_validation.root_count,
            topo_validation.root_idx,
            refit_validation.ok,
            refit_validation.diff,
            refit_validation.root_idx,
            guard,
        )
