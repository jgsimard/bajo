"""Reusable GPU storage for grouping incoherent world-space ray batches."""

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv, clamp, max

from bajo.bvh.gpu.ray_launch import _load_packed_ray
from bajo.core import AABB, Rayf32
from bajo.core.morton import morton3
from bajo.sort.gpu.onesweep import OneSweepWorkspace, onesweep_radix_sort_pairs


def _ray_order_keys(
    rays: ImmPointer[Float32, ImmutAnyOrigin],
    keys: MutPointer[UInt32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    count: Int32,
    min_x: Float32,
    min_y: Float32,
    min_z: Float32,
    inv_x: Float32,
    inv_y: Float32,
    inv_z: Float32,
):
    var index = global_idx.x
    if index >= Int(count):
        return
    var ray = _load_packed_ray[.WORLD](rays, Int(count), index)
    # Clamp before converting to unsigned Morton coordinates. Origins outside
    # the scene bounds and degenerate axes still produce valid sorting keys.
    var x = clamp((ray.o.x - min_x) * inv_x, Float32(0), Float32(1))
    var y = clamp((ray.o.y - min_y) * inv_y, Float32(0), Float32(1))
    var z = clamp((ray.o.z - min_z) * inv_z, Float32(0), Float32(1))
    var origin_key = morton3[1, 256](x, y, z)
    var octant = (
        UInt32(ray.d.x < 0)
        | (UInt32(ray.d.y < 0) << 1)
        | (UInt32(ray.d.z < 0) << 2)
    )
    keys[unsafe_offset=index] = (octant << 24) | origin_key
    indices[unsafe_offset=index] = UInt32(index)


def _copy_ray_buffer(
    source: ImmPointer[Float32, ImmutAnyOrigin],
    destination: MutPointer[Float32, MutAnyOrigin],
    count: Int32,
):
    # The ray stride is eight words, so every copy has complete four-word
    # groups. Sources can be unaligned views; the cached destination is aligned.
    var index = global_idx.x * 4
    if index >= Int(count) * Rayf32.STRIDE:
        return
    var words = source.unsafe_bitcast[UInt32]().unsafe_load[
        width=4, alignment=4
    ](index)
    destination.unsafe_bitcast[UInt32]().unsafe_store[width=4, alignment=16](
        index, words
    )


struct GpuRayOrderWorkspace:
    """Preallocated OneSweep storage for opt-in incoherent TLAS traversal.

    Rays are grouped by direction octant, then origin Morton code.
    Traversal reads the original rays through the sorted indices and writes
    hits at their original indices. Reuse this workspace on the same GPU
    stream. Separate input/output buffers need no allocation or synchronization
    during enqueueing. Overlapping buffers allocate a cached input copy on
    first use and reuse it thereafter. Camera rays usually run faster without
    ordering.
    """

    var _capacity: Int
    var _keys: DeviceBuffer[.uint32]
    var _indices: DeviceBuffer[.uint32]
    var _alias_rays: Optional[DeviceBuffer[.float32]]
    var _sort: OneSweepWorkspace[
        .uint32, .uint32, BLOCK_SIZE=512, KEYS_PER_THREAD=15
    ]

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        debug_assert["safe", _use_compiler_assume=True](
            capacity > 0 and capacity <= Int(0x7FFFFFFF),
            "ray ordering capacity must fit a positive Int32",
        )
        self._capacity = capacity
        self._keys = ctx.enqueue_create_buffer[.uint32](capacity)
        self._indices = ctx.enqueue_create_buffer[.uint32](capacity)
        self._alias_rays = Optional[DeviceBuffer[.float32]]()
        self._sort = OneSweepWorkspace[
            .uint32, .uint32, BLOCK_SIZE=512, KEYS_PER_THREAD=15
        ](ctx, capacity)

    def _enqueue_order(
        mut self,
        ctx: DeviceContext,
        source: DeviceBuffer[.float32],
        count: Int,
        bounds: AABB[.WORLD],
    ) raises:
        debug_assert["safe", _use_compiler_assume=True](
            count <= self._capacity, "ray ordering workspace is too small"
        )
        debug_assert["safe", _use_compiler_assume=True](
            count > 0 and len(source) >= count * Rayf32.STRIDE,
            "packed ray input buffer is too short",
        )
        var extent = bounds.extent()
        debug_assert["safe", _use_compiler_assume=True](
            extent.x >= 0 and extent.y >= 0 and extent.z >= 0,
            "ray ordering requires valid scene bounds",
        )
        ctx.enqueue_function[_ray_order_keys](
            source,
            self._keys,
            self._indices,
            Int32(count),
            bounds._min.x,
            bounds._min.y,
            bounds._min.z,
            Float32(1) / max(extent.x, Float32(1.0e-10)),
            Float32(1) / max(extent.y, Float32(1.0e-10)),
            Float32(1) / max(extent.z, Float32(1.0e-10)),
            grid_dim=ceildiv(count, 256),
            block_dim=256,
        )
        onesweep_radix_sort_pairs[
            .uint32, .uint32, BINNING_TPB=512, KEYS_PER_THREAD=15
        ](ctx, self._sort, self._keys, self._indices, count)

    def _enqueue_alias_copy(
        mut self,
        ctx: DeviceContext,
        source: DeviceBuffer[.float32],
        count: Int,
    ) raises:
        if not self._alias_rays:
            self._alias_rays = ctx.enqueue_create_buffer[.float32](
                self._capacity * Rayf32.STRIDE
            )
        ctx.enqueue_function[_copy_ray_buffer](
            source,
            self._alias_rays.value(),
            Int32(count),
            grid_dim=ceildiv(count * Rayf32.STRIDE // 4, 256),
            block_dim=256,
        )
