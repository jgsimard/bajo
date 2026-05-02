from std.gpu import DeviceBuffer
from std.gpu.host import DeviceContext
from std.math import abs
from std.utils.numerics import max_finite

from bajo.core.bvh.cpu.tlas import Tlas


comptime f32_max = max_finite[DType.float32]()

comptime GPU_TLAS_NODE_META_STRIDE = 4
comptime GPU_TLAS_NODE_BOUNDS_STRIDE = 6
comptime GPU_TLAS_INSTANCE_META_STRIDE = 4
comptime GPU_TLAS_TRANSFORM_STRIDE = 16
comptime GPU_TLAS_INSTANCE_BOUNDS_STRIDE = 6

comptime GPU_TLAS_INTERNAL_FLAG = UInt32(0)
comptime GPU_TLAS_LEAF_FLAG = UInt32(1)
comptime GPU_TLAS_INVALID = UInt32(0xFFFFFFFF)


@always_inline
def _nonzero_count(count: Int) -> Int:
    if count <= 0:
        return 1
    return count


@always_inline
def _matrix_elem(tlas: Tlas, inst_idx: Int, flat_idx: Int) -> Float32:
    var row = flat_idx // 4
    var col = flat_idx - row * 4
    return tlas.instances[inst_idx].transform[row][col]


@always_inline
def _inv_matrix_elem(tlas: Tlas, inst_idx: Int, flat_idx: Int) -> Float32:
    var row = flat_idx // 4
    var col = flat_idx - row * 4
    return tlas.instances[inst_idx].inv_transform[row][col]


@always_inline
def _almost_equal(a: Float32, b: Float32, eps: Float64 = 1.0e-6) -> Bool:
    return abs(Float64(a - b)) <= eps


@always_inline
def _first_bad(first_bad: Int, idx: Int) -> Int:
    if first_bad < 0:
        return idx
    return first_bad


@fieldwise_init
struct GpuTlasValidation(TrivialRegisterPassable):
    var ok: Bool
    var node_count: UInt32
    var inst_count: UInt32
    var leaf_count: UInt32
    var internal_count: UInt32
    var leaf_instance_sum: UInt32
    var first_bad: Int
    var checksum: UInt64

    @always_inline
    def __init__(out self):
        self.ok = False
        self.node_count = 0
        self.inst_count = 0
        self.leaf_count = 0
        self.internal_count = 0
        self.leaf_instance_sum = 0
        self.first_bad = -1
        self.checksum = 0


struct GpuTlasLayout:
    """Uploaded GPU TLAS buffers.

    Phase B deliberately uploads only data. It does not add traversal kernels.

    Node layout:
    - `node_meta[node * 4 + 0]`: `left_first`
        - internal: left child node index
        - leaf: first index in `inst_indices`
    - `node_meta[node * 4 + 1]`: `tri_count`
        - internal: 0
        - leaf: instance count
    - `node_meta[node * 4 + 2]`: parent/reserved, currently `GPU_TLAS_INVALID`
    - `node_meta[node * 4 + 3]`: 0 internal, 1 leaf

    Node bounds layout:
    - `node_bounds[node * 6 + 0:3]`: min xyz
    - `node_bounds[node * 6 + 3:6]`: max xyz

    Instance layout:
    - `inst_indices` preserves the CPU TLAS leaf ranges.
    - `inst_meta[inst * 4 + 0]`: BLAS index.
    - `inst_meta[inst * 4 + 1]`: original instance id.
    - `inst_meta[inst * 4 + 2:4]`: reserved.
    - transforms are row-major Mat44f32 flattened to 16 floats.
    - instance bounds are min xyz then max xyz.
    """

    var node_count: Int
    var inst_count: Int

    var node_meta: DeviceBuffer[DType.uint32]
    var node_bounds: DeviceBuffer[DType.float32]
    var inst_indices: DeviceBuffer[DType.uint32]
    var inst_meta: DeviceBuffer[DType.uint32]
    var inst_transform: DeviceBuffer[DType.float32]
    var inst_inv_transform: DeviceBuffer[DType.float32]
    var inst_bounds: DeviceBuffer[DType.float32]

    def __init__(out self, mut ctx: DeviceContext, tlas: Tlas) raises:
        self.node_count = Int(tlas.nodes_used)
        self.inst_count = Int(tlas.inst_count)

        self.node_meta = ctx.enqueue_create_buffer[DType.uint32](
            _nonzero_count(self.node_count * GPU_TLAS_NODE_META_STRIDE)
        )
        self.node_bounds = ctx.enqueue_create_buffer[DType.float32](
            _nonzero_count(self.node_count * GPU_TLAS_NODE_BOUNDS_STRIDE)
        )
        self.inst_indices = ctx.enqueue_create_buffer[DType.uint32](
            _nonzero_count(self.inst_count)
        )
        self.inst_meta = ctx.enqueue_create_buffer[DType.uint32](
            _nonzero_count(self.inst_count * GPU_TLAS_INSTANCE_META_STRIDE)
        )
        self.inst_transform = ctx.enqueue_create_buffer[DType.float32](
            _nonzero_count(self.inst_count * GPU_TLAS_TRANSFORM_STRIDE)
        )
        self.inst_inv_transform = ctx.enqueue_create_buffer[DType.float32](
            _nonzero_count(self.inst_count * GPU_TLAS_TRANSFORM_STRIDE)
        )
        self.inst_bounds = ctx.enqueue_create_buffer[DType.float32](
            _nonzero_count(self.inst_count * GPU_TLAS_INSTANCE_BOUNDS_STRIDE)
        )

        self.upload(ctx, tlas)

    def upload(mut self, ctx: DeviceContext, tlas: Tlas) raises:
        """Upload a CPU `Tlas` into the Phase B flat GPU layout."""
        self.node_count = Int(tlas.nodes_used)
        self.inst_count = Int(tlas.inst_count)

        with self.node_meta.map_to_host() as h:
            for i in range(self.node_count):
                ref node = tlas.tlas_nodes[i]
                var base = i * GPU_TLAS_NODE_META_STRIDE
                h[base + 0] = node.left_first
                h[base + 1] = node.tri_count
                h[base + 2] = GPU_TLAS_INVALID
                if node.is_leaf():
                    h[base + 3] = GPU_TLAS_LEAF_FLAG
                else:
                    h[base + 3] = GPU_TLAS_INTERNAL_FLAG

        with self.node_bounds.map_to_host() as h:
            for i in range(self.node_count):
                ref node = tlas.tlas_nodes[i]
                var base = i * GPU_TLAS_NODE_BOUNDS_STRIDE
                h[base + 0] = node.aabb._min.x()
                h[base + 1] = node.aabb._min.y()
                h[base + 2] = node.aabb._min.z()
                h[base + 3] = node.aabb._max.x()
                h[base + 4] = node.aabb._max.y()
                h[base + 5] = node.aabb._max.z()

        with self.inst_indices.map_to_host() as h:
            for i in range(self.inst_count):
                h[i] = tlas.inst_indices[i]

        with self.inst_meta.map_to_host() as h:
            for i in range(self.inst_count):
                ref inst = tlas.instances[i]
                var base = i * GPU_TLAS_INSTANCE_META_STRIDE
                h[base + 0] = inst.blas_idx
                h[base + 1] = UInt32(i)
                h[base + 2] = 0
                h[base + 3] = 0

        with self.inst_transform.map_to_host() as h:
            for i in range(self.inst_count):
                var base = i * GPU_TLAS_TRANSFORM_STRIDE
                comptime for j in range(GPU_TLAS_TRANSFORM_STRIDE):
                    h[base + j] = _matrix_elem(tlas, i, j)

        with self.inst_inv_transform.map_to_host() as h:
            for i in range(self.inst_count):
                var base = i * GPU_TLAS_TRANSFORM_STRIDE
                comptime for j in range(GPU_TLAS_TRANSFORM_STRIDE):
                    h[base + j] = _inv_matrix_elem(tlas, i, j)

        with self.inst_bounds.map_to_host() as h:
            for i in range(self.inst_count):
                ref inst = tlas.instances[i]
                var base = i * GPU_TLAS_INSTANCE_BOUNDS_STRIDE
                h[base + 0] = inst.bounds_min.x()
                h[base + 1] = inst.bounds_min.y()
                h[base + 2] = inst.bounds_min.z()
                h[base + 3] = inst.bounds_max.x()
                h[base + 4] = inst.bounds_max.y()
                h[base + 5] = inst.bounds_max.z()

        ctx.synchronize()

    def validate(self, tlas: Tlas) raises -> GpuTlasValidation:
        return validate_gpu_tlas_layout(self, tlas)


@always_inline
def _expected_flag(tlas: Tlas, node_idx: Int) -> UInt32:
    if tlas.tlas_nodes[node_idx].is_leaf():
        return GPU_TLAS_LEAF_FLAG
    return GPU_TLAS_INTERNAL_FLAG


def validate_gpu_tlas_layout(
    layout: GpuTlasLayout,
    tlas: Tlas,
) raises -> GpuTlasValidation:
    """Host-side validation of the uploaded TLAS buffers.

    This intentionally validates data movement only. Traversal correctness stays
    in CPU Phase A and GPU traversal gets tested in Phase C.
    """
    var out = GpuTlasValidation()
    out.ok = True
    out.node_count = UInt32(layout.node_count)
    out.inst_count = UInt32(layout.inst_count)

    if layout.node_count != Int(tlas.nodes_used):
        out.ok = False
        out.first_bad = _first_bad(out.first_bad, -100)
    if layout.inst_count != Int(tlas.inst_count):
        out.ok = False
        out.first_bad = _first_bad(out.first_bad, -101)

    with layout.node_meta.map_to_host() as h:
        for i in range(layout.node_count):
            ref node = tlas.tlas_nodes[i]
            var base = i * GPU_TLAS_NODE_META_STRIDE

            if h[base + 0] != node.left_first:
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 0)
            if h[base + 1] != node.tri_count:
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 1)
            if h[base + 2] != GPU_TLAS_INVALID:
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 2)
            if h[base + 3] != _expected_flag(tlas, i):
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 3)

            out.checksum += UInt64(h[base + 0])
            out.checksum += UInt64(h[base + 1])
            out.checksum += UInt64(h[base + 3])

            if node.is_leaf():
                out.leaf_count += 1
                out.leaf_instance_sum += node.tri_count
            else:
                out.internal_count += 1

    if out.leaf_instance_sum != UInt32(layout.inst_count):
        out.ok = False
        out.first_bad = _first_bad(out.first_bad, -102)

    with layout.node_bounds.map_to_host() as h:
        for i in range(layout.node_count):
            ref node = tlas.tlas_nodes[i]
            var base = i * GPU_TLAS_NODE_BOUNDS_STRIDE

            if not _almost_equal(h[base + 0], node.aabb._min.x()):
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 0)
            if not _almost_equal(h[base + 1], node.aabb._min.y()):
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 1)
            if not _almost_equal(h[base + 2], node.aabb._min.z()):
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 2)
            if not _almost_equal(h[base + 3], node.aabb._max.x()):
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 3)
            if not _almost_equal(h[base + 4], node.aabb._max.y()):
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 4)
            if not _almost_equal(h[base + 5], node.aabb._max.z()):
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 5)

            out.checksum += UInt64(abs(Float64(h[base + 0])) * 1000.0)
            out.checksum += UInt64(abs(Float64(h[base + 1])) * 1000.0)
            out.checksum += UInt64(abs(Float64(h[base + 2])) * 1000.0)
            out.checksum += UInt64(abs(Float64(h[base + 3])) * 1000.0)
            out.checksum += UInt64(abs(Float64(h[base + 4])) * 1000.0)
            out.checksum += UInt64(abs(Float64(h[base + 5])) * 1000.0)

    with layout.inst_indices.map_to_host() as h:
        for i in range(layout.inst_count):
            if h[i] != tlas.inst_indices[i]:
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, i)
            out.checksum += UInt64(h[i])

    with layout.inst_meta.map_to_host() as h:
        for i in range(layout.inst_count):
            ref inst = tlas.instances[i]
            var base = i * GPU_TLAS_INSTANCE_META_STRIDE
            if h[base + 0] != inst.blas_idx:
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 0)
            if h[base + 1] != UInt32(i):
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 1)
            if h[base + 2] != 0 or h[base + 3] != 0:
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 2)

            out.checksum += UInt64(h[base + 0])
            out.checksum += UInt64(h[base + 1])

    with layout.inst_transform.map_to_host() as h:
        for i in range(layout.inst_count):
            var base = i * GPU_TLAS_TRANSFORM_STRIDE
            comptime for j in range(GPU_TLAS_TRANSFORM_STRIDE):
                var expected = _matrix_elem(tlas, i, j)
                if not _almost_equal(h[base + j], expected):
                    out.ok = False
                    out.first_bad = _first_bad(out.first_bad, base + j)
                out.checksum += UInt64(abs(Float64(h[base + j])) * 1000.0)

    with layout.inst_inv_transform.map_to_host() as h:
        for i in range(layout.inst_count):
            var base = i * GPU_TLAS_TRANSFORM_STRIDE
            comptime for j in range(GPU_TLAS_TRANSFORM_STRIDE):
                var expected = _inv_matrix_elem(tlas, i, j)
                if not _almost_equal(h[base + j], expected):
                    out.ok = False
                    out.first_bad = _first_bad(out.first_bad, base + j)
                out.checksum += UInt64(abs(Float64(h[base + j])) * 1000.0)

    with layout.inst_bounds.map_to_host() as h:
        for i in range(layout.inst_count):
            ref inst = tlas.instances[i]
            var base = i * GPU_TLAS_INSTANCE_BOUNDS_STRIDE

            if not _almost_equal(h[base + 0], inst.bounds_min.x()):
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 0)
            if not _almost_equal(h[base + 1], inst.bounds_min.y()):
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 1)
            if not _almost_equal(h[base + 2], inst.bounds_min.z()):
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 2)
            if not _almost_equal(h[base + 3], inst.bounds_max.x()):
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 3)
            if not _almost_equal(h[base + 4], inst.bounds_max.y()):
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 4)
            if not _almost_equal(h[base + 5], inst.bounds_max.z()):
                out.ok = False
                out.first_bad = _first_bad(out.first_bad, base + 5)

            out.checksum += UInt64(abs(Float64(h[base + 0])) * 1000.0)
            out.checksum += UInt64(abs(Float64(h[base + 1])) * 1000.0)
            out.checksum += UInt64(abs(Float64(h[base + 2])) * 1000.0)
            out.checksum += UInt64(abs(Float64(h[base + 3])) * 1000.0)
            out.checksum += UInt64(abs(Float64(h[base + 4])) * 1000.0)
            out.checksum += UInt64(abs(Float64(h[base + 5])) * 1000.0)

    return out^
