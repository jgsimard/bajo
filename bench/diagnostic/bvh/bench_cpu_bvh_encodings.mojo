"""Diagnostic sweep of alternative CPU BVH encodings."""

from std.bit import count_trailing_zeros
from std.math import ceil, floor, fma, round
from std.memory import bitcast, pack_bits
from std.sys import size_of
from std.time import perf_counter_ns

from bajo.bvh.constants import CPU_STACK_SIZE, EMPTY_LANE, TRACE, f32_max
from bajo.bvh.cpu.bounds_bvh import (
    WideBvhNode,
)
from bajo.bvh.tagged_ref import (
    decode_ref_index,
    encode_leaf_ref,
    is_leaf_ref,
)
from bajo.bvh.cpu.trace import (
    CpuBvhTraversalStats,
    _count_true_lanes,
    _pack_pending_task,
    _pending_task_ref,
    _pending_task_t,
)
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Hit
from bajo.core import (
    AxisAlignedBoundingBox,
    Frame,
    Point3,
    Point3f32,
    Vec3,
    Vec3f32,
    Rayf32,
)
from bajo.core.intersect import (
    intersect_ray_aabb_octant_fma,
    intersect_ray_tri_edges,
    intersect_ray_tri_edges_scaled,
)
from bajo.core.utils import ns_to_mrays_per_s
from bajo.obj.pack import pack_obj_triangles
from bench.bvh.fixtures import (
    make_camera_rays_and_params,
    make_depth_overlap_rays,
    make_depth_overlap_triangles,
    make_grid_triangles,
    make_hit_and_miss_rays,
    permute_rays,
    select_and_repeat_hit_rays,
)


comptime OBJ_PATH = "./assets/dragon/dragon.obj"
comptime WIDTH = 16
comptime TIMING_REPEATS = 4
comptime COMPACT_LEAF_REF_BIT = UInt32(0x40000000)
comptime COMPACT_LEAF_REF_INDEX_MASK = UInt32(0x3FFFFFFF)


comptime BoundsF32 = AxisAlignedBoundingBox[DType.float32, Frame.WORLD, WIDTH]
comptime BoundsF16 = AxisAlignedBoundingBox[DType.float16, Frame.WORLD, WIDTH]
comptime BoundsU16 = AxisAlignedBoundingBox[DType.uint16, Frame.WORLD, WIDTH]
comptime BoundsU8 = AxisAlignedBoundingBox[DType.uint8, Frame.WORLD, WIDTH]


@fieldwise_init
struct Float16Node(Copyable):
    var bounds: BoundsF16
    var data: SIMD[DType.uint32, WIDTH]


@fieldwise_init
struct GlobalUInt16Node(Copyable):
    var bounds: BoundsU16
    var data: SIMD[DType.uint32, WIDTH]


@fieldwise_init
struct RelativeUInt16Node(Copyable):
    var data: SIMD[DType.uint32, WIDTH]
    var start: Point3f32[Frame.WORLD]
    var scale: Vec3f32[Frame.WORLD]
    var child_mask: UInt32
    var bounds: BoundsU16


@fieldwise_init
struct RelativeUInt8Node(Copyable):
    var data: SIMD[DType.uint32, WIDTH]
    var start: Point3f32[Frame.WORLD]
    var scale: Vec3f32[Frame.WORLD]
    var child_mask: UInt32
    var bounds: BoundsU8


@fieldwise_init
struct CompactLeafNode(Copyable):
    var start: Point3f32[Frame.WORLD]
    var scale: Vec3f32[Frame.WORLD]
    var first_leaf: UInt32
    var child_mask: UInt32
    var bounds: BoundsU8


struct EncodedTrees(Copyable):
    var float16_nodes: List[Float16Node]
    var global_u16_nodes: List[GlobalUInt16Node]
    var relative_u16_nodes: List[RelativeUInt16Node]
    var relative_u8_nodes: List[RelativeUInt8Node]
    var hybrid_exact_nodes: List[WideBvhNode[Frame.WORLD, WIDTH]]
    var hybrid_exact_masks: List[UInt32]
    var hybrid_leaf_nodes: List[CompactLeafNode]
    var hybrid_root_ref: UInt32
    var global_start: Point3f32[Frame.WORLD]
    var global_scale: Vec3f32[Frame.WORLD]

    def __init__(out self, bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH]):
        var node_count = len(bvh.tree.nodes)
        self.float16_nodes = List[Float16Node](capacity=node_count)
        self.global_u16_nodes = List[GlobalUInt16Node](capacity=node_count)
        self.relative_u16_nodes = List[RelativeUInt16Node](capacity=node_count)
        self.relative_u8_nodes = List[RelativeUInt8Node](capacity=node_count)
        self.hybrid_exact_nodes = []
        self.hybrid_exact_masks = []
        self.hybrid_leaf_nodes = []
        self.hybrid_root_ref = 0

        var root = bvh.tree.root_bounds()
        self.global_start = root._min
        self.global_scale = Vec3f32[Frame.WORLD](
            _quant_scale(root._min.x, root._max.x, 65535.0),
            _quant_scale(root._min.y, root._max.y, 65535.0),
            _quant_scale(root._min.z, root._max.z, 65535.0),
        )

        for node_idx in range(node_count):
            ref node = bvh.tree.nodes[node_idx]
            var child_mask = bvh.tree.child_masks[node_idx]
            var domain = _node_domain(node.aabb, child_mask)
            var start = domain[0]
            var extent = domain[1] - domain[0]
            var scale_u16 = Vec3f32[Frame.WORLD](
                _quant_scale(0.0, extent.x, 65535.0),
                _quant_scale(0.0, extent.y, 65535.0),
                _quant_scale(0.0, extent.z, 65535.0),
            )
            var scale_u8 = Vec3f32[Frame.WORLD](
                _quant_scale(0.0, extent.x, 255.0),
                _quant_scale(0.0, extent.y, 255.0),
                _quant_scale(0.0, extent.z, 255.0),
            )

            self.float16_nodes.append(
                Float16Node(_encode_float16(node.aabb), node.data)
            )
            self.global_u16_nodes.append(
                GlobalUInt16Node(
                    _encode_u16(
                        node.aabb,
                        child_mask,
                        self.global_start,
                        self.global_scale,
                    ),
                    node.data,
                )
            )
            self.relative_u16_nodes.append(
                RelativeUInt16Node(
                    node.data,
                    start,
                    scale_u16,
                    child_mask,
                    _encode_u16(node.aabb, child_mask, start, scale_u16),
                )
            )
            self.relative_u8_nodes.append(
                RelativeUInt8Node(
                    node.data,
                    start,
                    scale_u8,
                    child_mask,
                    _encode_u8(node.aabb, child_mask, start, scale_u8),
                )
            )

        self._build_hybrid(bvh)

    def _build_hybrid(mut self, bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH]):
        var node_count = len(bvh.tree.nodes)
        var remap = List[UInt32](length=node_count, fill=UInt32(0))
        var all_leaf = List[Bool](length=node_count, fill=False)
        var exact_count = UInt32(0)
        var compact_count = UInt32(0)

        for node_idx in range(node_count):
            ref node = bvh.tree.nodes[node_idx]
            var child_mask = bvh.tree.child_masks[node_idx]
            var only_leaves = child_mask != 0
            comptime for lane in range(WIDTH):
                if (child_mask & (UInt32(1) << UInt32(lane))) != 0:
                    only_leaves &= is_leaf_ref(node.data[lane])

            all_leaf[node_idx] = only_leaves
            if only_leaves:
                remap[node_idx] = COMPACT_LEAF_REF_BIT | compact_count
                compact_count += 1
            else:
                remap[node_idx] = exact_count
                exact_count += 1

        self.hybrid_exact_nodes = List[WideBvhNode[Frame.WORLD, WIDTH]](
            capacity=Int(exact_count)
        )
        self.hybrid_exact_masks = List[UInt32](capacity=Int(exact_count))
        self.hybrid_leaf_nodes = List[CompactLeafNode](
            capacity=Int(compact_count)
        )

        for node_idx in range(node_count):
            ref source = bvh.tree.nodes[node_idx]
            var child_mask = bvh.tree.child_masks[node_idx]
            if all_leaf[node_idx]:
                var domain = _node_domain(source.aabb, child_mask)
                var start = domain[0]
                var extent = domain[1] - domain[0]
                var scale = Vec3f32[Frame.WORLD](
                    _quant_scale(0.0, extent.x, 255.0),
                    _quant_scale(0.0, extent.y, 255.0),
                    _quant_scale(0.0, extent.z, 255.0),
                )
                var first_leaf = decode_ref_index(source.data[0])
                comptime for lane in range(WIDTH):
                    if (child_mask & (UInt32(1) << UInt32(lane))) != 0:
                        debug_assert["safe", _use_compiler_assume=True](
                            decode_ref_index(source.data[lane])
                            == first_leaf + UInt32(lane),
                            "typed all-leaf blocks are not consecutive",
                        )
                self.hybrid_leaf_nodes.append(
                    CompactLeafNode(
                        start,
                        scale,
                        first_leaf,
                        child_mask,
                        _encode_u8(source.aabb, child_mask, start, scale),
                    )
                )
            else:
                var exact = source.copy()
                comptime for lane in range(WIDTH):
                    var child_ref = exact.data[lane]
                    if child_ref != EMPTY_LANE and not is_leaf_ref(child_ref):
                        exact.data[lane] = remap[Int(child_ref)]
                self.hybrid_exact_nodes.append(exact^)
                self.hybrid_exact_masks.append(child_mask)

        self.hybrid_root_ref = remap[0]

    def bytes[encoding: String](self) -> Int:
        comptime if encoding == "f32":
            return 0
        elif encoding == "f16":
            return len(self.float16_nodes) * size_of[Float16Node]()
        elif encoding == "global_u16":
            return len(self.global_u16_nodes) * size_of[GlobalUInt16Node]()
        elif encoding == "relative_u16":
            return len(self.relative_u16_nodes) * size_of[RelativeUInt16Node]()
        elif encoding == "relative_u8":
            return len(self.relative_u8_nodes) * size_of[RelativeUInt8Node]()
        else:
            comptime assert encoding == "hybrid_u8"
            return (
                len(self.hybrid_exact_nodes)
                * size_of[WideBvhNode[Frame.WORLD, WIDTH]]()
                + len(self.hybrid_exact_masks) * 4
                + len(self.hybrid_leaf_nodes) * size_of[CompactLeafNode]()
            )


def _quant_scale(lo: Float32, hi: Float32, max_quant: Float32) -> Float32:
    var extent = hi - lo
    if extent <= 0.0:
        return Float32(2.350988701644575e-38)
    # Expand by two Float32 ulps, as in Embree's conservative quantizer.
    var scale = extent * Float32(1.000000238418579) / max_quant
    if fma(scale, max_quant, lo) < hi:
        scale *= Float32(1.0000001192092896)
    return scale


def _node_domain(
    bounds: BoundsF32, child_mask: UInt32
) -> Tuple[Point3f32[Frame.WORLD], Point3f32[Frame.WORLD]]:
    var lo = Point3f32[Frame.WORLD](f32_max)
    var hi = Point3f32[Frame.WORLD](-f32_max)
    comptime for lane in range(WIDTH):
        if (child_mask & (UInt32(1) << UInt32(lane))) != 0:
            if bounds._min.x[lane] < lo.x:
                lo.x = bounds._min.x[lane]
            if bounds._min.y[lane] < lo.y:
                lo.y = bounds._min.y[lane]
            if bounds._min.z[lane] < lo.z:
                lo.z = bounds._min.z[lane]
            if bounds._max.x[lane] > hi.x:
                hi.x = bounds._max.x[lane]
            if bounds._max.y[lane] > hi.y:
                hi.y = bounds._max.y[lane]
            if bounds._max.z[lane] > hi.z:
                hi.z = bounds._max.z[lane]
    return (lo, hi)


def _quantize_lower(
    value: Float32, start: Float32, scale: Float32, max_quant: Int
) -> UInt32:
    var q = Int(floor((value - start) / scale))
    if q < 0:
        q = 0
    if q > max_quant:
        q = max_quant
    if fma(Float32(q), scale, start) > value and q > 0:
        q -= 1
    return UInt32(q)


def _quantize_upper(
    value: Float32, start: Float32, scale: Float32, max_quant: Int
) -> UInt32:
    var q = Int(ceil((value - start) / scale))
    if q < 0:
        q = 0
    if q > max_quant:
        q = max_quant
    if fma(Float32(q), scale, start) < value and q < max_quant:
        q += 1
    return UInt32(q)


def _encode_u8(
    bounds: BoundsF32,
    child_mask: UInt32,
    start: Point3f32[Frame.WORLD],
    scale: Vec3f32[Frame.WORLD],
) -> BoundsU8:
    var out = BoundsU8(
        Point3[DType.uint8, Frame.WORLD, WIDTH](UInt8(255)),
        Point3[DType.uint8, Frame.WORLD, WIDTH](UInt8(0)),
    )
    comptime for lane in range(WIDTH):
        if (child_mask & (UInt32(1) << UInt32(lane))) != 0:
            out._min.x[lane] = UInt8(
                _quantize_lower(bounds._min.x[lane], start.x, scale.x, 255)
            )
            out._min.y[lane] = UInt8(
                _quantize_lower(bounds._min.y[lane], start.y, scale.y, 255)
            )
            out._min.z[lane] = UInt8(
                _quantize_lower(bounds._min.z[lane], start.z, scale.z, 255)
            )
            out._max.x[lane] = UInt8(
                _quantize_upper(bounds._max.x[lane], start.x, scale.x, 255)
            )
            out._max.y[lane] = UInt8(
                _quantize_upper(bounds._max.y[lane], start.y, scale.y, 255)
            )
            out._max.z[lane] = UInt8(
                _quantize_upper(bounds._max.z[lane], start.z, scale.z, 255)
            )
            debug_assert["safe", _use_compiler_assume=True](
                fma(Float32(out._min.x[lane]), scale.x, start.x)
                <= bounds._min.x[lane]
                and fma(Float32(out._min.y[lane]), scale.y, start.y)
                <= bounds._min.y[lane]
                and fma(Float32(out._min.z[lane]), scale.z, start.z)
                <= bounds._min.z[lane]
                and fma(Float32(out._max.x[lane]), scale.x, start.x)
                >= bounds._max.x[lane]
                and fma(Float32(out._max.y[lane]), scale.y, start.y)
                >= bounds._max.y[lane]
                and fma(Float32(out._max.z[lane]), scale.z, start.z)
                >= bounds._max.z[lane],
                "UInt8 decoded bounds are not conservative",
            )
    return out


def _encode_u16(
    bounds: BoundsF32,
    child_mask: UInt32,
    start: Point3f32[Frame.WORLD],
    scale: Vec3f32[Frame.WORLD],
) -> BoundsU16:
    var out = BoundsU16(
        Point3[DType.uint16, Frame.WORLD, WIDTH](UInt16(65535)),
        Point3[DType.uint16, Frame.WORLD, WIDTH](UInt16(0)),
    )
    comptime for lane in range(WIDTH):
        if (child_mask & (UInt32(1) << UInt32(lane))) != 0:
            out._min.x[lane] = UInt16(
                _quantize_lower(bounds._min.x[lane], start.x, scale.x, 65535)
            )
            out._min.y[lane] = UInt16(
                _quantize_lower(bounds._min.y[lane], start.y, scale.y, 65535)
            )
            out._min.z[lane] = UInt16(
                _quantize_lower(bounds._min.z[lane], start.z, scale.z, 65535)
            )
            out._max.x[lane] = UInt16(
                _quantize_upper(bounds._max.x[lane], start.x, scale.x, 65535)
            )
            out._max.y[lane] = UInt16(
                _quantize_upper(bounds._max.y[lane], start.y, scale.y, 65535)
            )
            out._max.z[lane] = UInt16(
                _quantize_upper(bounds._max.z[lane], start.z, scale.z, 65535)
            )
            debug_assert["safe", _use_compiler_assume=True](
                fma(Float32(out._min.x[lane]), scale.x, start.x)
                <= bounds._min.x[lane]
                and fma(Float32(out._min.y[lane]), scale.y, start.y)
                <= bounds._min.y[lane]
                and fma(Float32(out._min.z[lane]), scale.z, start.z)
                <= bounds._min.z[lane]
                and fma(Float32(out._max.x[lane]), scale.x, start.x)
                >= bounds._max.x[lane]
                and fma(Float32(out._max.y[lane]), scale.y, start.y)
                >= bounds._max.y[lane]
                and fma(Float32(out._max.z[lane]), scale.z, start.z)
                >= bounds._max.z[lane],
                "UInt16 decoded bounds are not conservative",
            )
    return out


def _half_from_bits(bits: UInt16) -> Float16:
    return bitcast[DType.float16, 1](SIMD[DType.uint16, 1](bits))[0]


def _half_bits(value: Float16) -> UInt16:
    return bitcast[DType.uint16, 1](SIMD[DType.float16, 1](value))[0]


def _half_next_down(value: Float16) -> Float16:
    var bits = _half_bits(value)
    if bits == UInt16(0):
        return _half_from_bits(UInt16(0x8001))
    if (bits & UInt16(0x8000)) != 0:
        return _half_from_bits(bits + 1)
    return _half_from_bits(bits - 1)


def _half_next_up(value: Float16) -> Float16:
    var bits = _half_bits(value)
    if bits == UInt16(0x8000):
        return _half_from_bits(UInt16(1))
    if (bits & UInt16(0x8000)) != 0:
        return _half_from_bits(bits - 1)
    return _half_from_bits(bits + 1)


def _half_lower(value: Float32) -> Float16:
    var out = Float16(value)
    if Float32(out) > value:
        out = _half_next_down(out)
    return out


def _half_upper(value: Float32) -> Float16:
    var out = Float16(value)
    if Float32(out) < value:
        out = _half_next_up(out)
    return out


def _encode_float16(bounds: BoundsF32) -> BoundsF16:
    var out = BoundsF16(
        Point3[DType.float16, Frame.WORLD, WIDTH](0.0),
        Point3[DType.float16, Frame.WORLD, WIDTH](0.0),
    )
    comptime for lane in range(WIDTH):
        out._min.x[lane] = _half_lower(bounds._min.x[lane])
        out._min.y[lane] = _half_lower(bounds._min.y[lane])
        out._min.z[lane] = _half_lower(bounds._min.z[lane])
        out._max.x[lane] = _half_upper(bounds._max.x[lane])
        out._max.y[lane] = _half_upper(bounds._max.y[lane])
        out._max.z[lane] = _half_upper(bounds._max.z[lane])
        debug_assert["safe", _use_compiler_assume=True](
            Float32(out._min.x[lane]) <= bounds._min.x[lane]
            and Float32(out._min.y[lane]) <= bounds._min.y[lane]
            and Float32(out._min.z[lane]) <= bounds._min.z[lane]
            and Float32(out._max.x[lane]) >= bounds._max.x[lane]
            and Float32(out._max.y[lane]) >= bounds._max.y[lane]
            and Float32(out._max.z[lane]) >= bounds._max.z[lane],
            "Float16 decoded bounds are not conservative",
        )
    return out


@always_inline
def _decode_f16(node: Float16Node) -> BoundsF32:
    return BoundsF32(
        Point3[DType.float32, Frame.WORLD, WIDTH](
            node.bounds._min.x.cast[DType.float32](),
            node.bounds._min.y.cast[DType.float32](),
            node.bounds._min.z.cast[DType.float32](),
        ),
        Point3[DType.float32, Frame.WORLD, WIDTH](
            node.bounds._max.x.cast[DType.float32](),
            node.bounds._max.y.cast[DType.float32](),
            node.bounds._max.z.cast[DType.float32](),
        ),
    )


@always_inline
def _decode_u16(
    bounds: BoundsU16,
    start: Point3f32[Frame.WORLD],
    scale: Vec3f32[Frame.WORLD],
) -> BoundsF32:
    return BoundsF32(
        Point3[DType.float32, Frame.WORLD, WIDTH](
            fma(bounds._min.x.cast[DType.float32](), scale.x, start.x),
            fma(bounds._min.y.cast[DType.float32](), scale.y, start.y),
            fma(bounds._min.z.cast[DType.float32](), scale.z, start.z),
        ),
        Point3[DType.float32, Frame.WORLD, WIDTH](
            fma(bounds._max.x.cast[DType.float32](), scale.x, start.x),
            fma(bounds._max.y.cast[DType.float32](), scale.y, start.y),
            fma(bounds._max.z.cast[DType.float32](), scale.z, start.z),
        ),
    )


@always_inline
def _decode_u8(
    bounds: BoundsU8,
    start: Point3f32[Frame.WORLD],
    scale: Vec3f32[Frame.WORLD],
) -> BoundsF32:
    return BoundsF32(
        Point3[DType.float32, Frame.WORLD, WIDTH](
            fma(bounds._min.x.cast[DType.float32](), scale.x, start.x),
            fma(bounds._min.y.cast[DType.float32](), scale.y, start.y),
            fma(bounds._min.z.cast[DType.float32](), scale.z, start.z),
        ),
        Point3[DType.float32, Frame.WORLD, WIDTH](
            fma(bounds._max.x.cast[DType.float32](), scale.x, start.x),
            fma(bounds._max.y.cast[DType.float32](), scale.y, start.y),
            fma(bounds._max.z.cast[DType.float32](), scale.z, start.z),
        ),
    )


@always_inline
def _load_encoded_node[
    encoding: String
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    current_ref: UInt32,
    mut bounds: BoundsF32,
    mut data: SIMD[DType.uint32, WIDTH],
    mut child_mask: UInt32,
    mut compact_first_leaf: UInt32,
    mut is_compact_leaf_node: Bool,
):
    data = SIMD[DType.uint32, WIDTH](EMPTY_LANE)
    compact_first_leaf = 0
    is_compact_leaf_node = False
    comptime if encoding == "f32":
        ref node = bvh.tree.nodes.unsafe_get(Int(current_ref))
        bounds = node.aabb
        data = node.data
        child_mask = bvh.tree.child_masks.unsafe_get(Int(current_ref))
    elif encoding == "f16":
        ref node = encoded.float16_nodes.unsafe_get(Int(current_ref))
        bounds = _decode_f16(node)
        data = node.data
        child_mask = bvh.tree.child_masks.unsafe_get(Int(current_ref))
    elif encoding == "global_u16":
        ref node = encoded.global_u16_nodes.unsafe_get(Int(current_ref))
        bounds = _decode_u16(
            node.bounds, encoded.global_start, encoded.global_scale
        )
        data = node.data
        child_mask = bvh.tree.child_masks.unsafe_get(Int(current_ref))
    elif encoding == "relative_u16":
        ref node = encoded.relative_u16_nodes.unsafe_get(Int(current_ref))
        bounds = _decode_u16(node.bounds, node.start, node.scale)
        data = node.data
        child_mask = node.child_mask
    elif encoding == "relative_u8":
        ref node = encoded.relative_u8_nodes.unsafe_get(Int(current_ref))
        bounds = _decode_u8(node.bounds, node.start, node.scale)
        data = node.data
        child_mask = node.child_mask
    else:
        comptime assert encoding == "hybrid_u8"
        if (current_ref & COMPACT_LEAF_REF_BIT) != 0:
            ref node = encoded.hybrid_leaf_nodes.unsafe_get(
                Int(current_ref & COMPACT_LEAF_REF_INDEX_MASK)
            )
            bounds = _decode_u8(node.bounds, node.start, node.scale)
            child_mask = node.child_mask
            compact_first_leaf = node.first_leaf
            is_compact_leaf_node = True
        else:
            ref node = encoded.hybrid_exact_nodes.unsafe_get(Int(current_ref))
            bounds = node.aabb
            data = node.data
            child_mask = encoded.hybrid_exact_masks.unsafe_get(Int(current_ref))


@always_inline
def _trace_encoded_octant[
    encoding: String,
    collect_stats: Bool,
    positive_x: Bool,
    positive_y: Bool,
    positive_z: Bool,
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    ray: Rayf32[Frame.WORLD],
    mut stats: CpuBvhTraversalStats,
) -> Hit[Frame.WORLD]:
    var hit = Hit[Frame.WORLD].miss(ray.t_max)
    var ordered_stack = Array[UInt64, CPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var current_ref = UInt32(0)
    comptime if encoding == "hybrid_u8":
        current_ref = encoded.hybrid_root_ref

    var O = ray.origin[WIDTH]()
    var D = ray.direction[WIDTH]()
    var rcp_d = ray.rcp_direction[WIDTH]()
    var origin_rcp_d = Vec3[DType.float32, Frame.WORLD, WIDTH](
        O.x * rcp_d.x, O.y * rcp_d.y, O.z * rcp_d.z
    )

    @always_inline
    def push_pending(
        child_ref: UInt32, child_t: Float32
    ) {imm, mut ordered_stack, mut stack_ptr, mut stats}:
        var task = _pack_pending_task(child_ref, child_t)
        var insert_idx = stack_ptr
        while insert_idx > 0:
            var previous_idx = insert_idx - 1
            var previous = ordered_stack.unsafe_get(previous_idx)
            if previous >= task:
                break
            ordered_stack.unsafe_get(insert_idx) = previous
            insert_idx = previous_idx
        ordered_stack.unsafe_get(insert_idx) = task
        stack_ptr += 1
        comptime if collect_stats:
            stats.stack_pushes += 1
            stats.stack_insertion_shifts += stack_ptr - insert_idx - 1
            if stack_ptr > stats.max_stack_depth:
                stats.max_stack_depth = stack_ptr

    @always_inline
    def intersect_leaf(leaf_idx: UInt32) {imm, mut stats, mut hit}:
        comptime if collect_stats:
            stats.leaf_blocks += 1
            stats.primitive_packet_lanes += WIDTH
        ref block = bvh.leaf_blocks.unsafe_get(Int(leaf_idx))
        var scaled = intersect_ray_tri_edges_scaled(
            O,
            D,
            block.v0,
            block.e1,
            block.e2,
            hit.t,
            ray.t_min,
        )
        var mask = scaled.mask & block.prim_indices.ne(EMPTY_LANE)
        comptime if collect_stats:
            stats.valid_primitives += _count_true_lanes(
                block.prim_indices.ne(EMPTY_LANE)
            )
            stats.primitive_hit_candidates += _count_true_lanes(mask)
        if not mask.reduce_or():
            return

        var bits = UInt32(pack_bits(mask))
        var lane = Int(count_trailing_zeros(bits))
        bits &= bits - 1
        while bits != 0:
            var candidate = Int(count_trailing_zeros(bits))
            bits &= bits - 1
            if (
                scaled.t_scaled[candidate] * scaled.abs_det[lane]
                < scaled.t_scaled[lane] * scaled.abs_det[candidate]
            ):
                lane = candidate
        hit.t = scaled.t_scaled[lane] / scaled.abs_det[lane]
        hit.prim = block.prim_indices[lane]
        comptime if collect_stats:
            stats.closer_hit_updates += 1

    while True:
        if is_leaf_ref(current_ref):
            intersect_leaf(decode_ref_index(current_ref))
        else:
            comptime if collect_stats:
                stats.internal_nodes += 1
                stats.aabb_packet_lanes += WIDTH

            var bounds: BoundsF32
            var data = SIMD[DType.uint32, WIDTH](EMPTY_LANE)
            var child_mask: UInt32
            var compact_first_leaf = UInt32(0)
            var is_compact_leaf_node = False

            comptime if encoding == "f32":
                ref node = bvh.tree.nodes.unsafe_get(Int(current_ref))
                bounds = node.aabb
                data = node.data
                child_mask = bvh.tree.child_masks.unsafe_get(Int(current_ref))
            elif encoding == "f16":
                ref node = encoded.float16_nodes.unsafe_get(Int(current_ref))
                bounds = _decode_f16(node)
                data = node.data
                child_mask = bvh.tree.child_masks.unsafe_get(Int(current_ref))
            elif encoding == "global_u16":
                ref node = encoded.global_u16_nodes.unsafe_get(Int(current_ref))
                bounds = _decode_u16(
                    node.bounds, encoded.global_start, encoded.global_scale
                )
                data = node.data
                child_mask = bvh.tree.child_masks.unsafe_get(Int(current_ref))
            elif encoding == "relative_u16":
                ref node = encoded.relative_u16_nodes.unsafe_get(
                    Int(current_ref)
                )
                bounds = _decode_u16(node.bounds, node.start, node.scale)
                data = node.data
                child_mask = node.child_mask
            elif encoding == "relative_u8":
                ref node = encoded.relative_u8_nodes.unsafe_get(
                    Int(current_ref)
                )
                bounds = _decode_u8(node.bounds, node.start, node.scale)
                data = node.data
                child_mask = node.child_mask
            else:
                comptime assert encoding == "hybrid_u8"
                if (current_ref & COMPACT_LEAF_REF_BIT) != 0:
                    ref node = encoded.hybrid_leaf_nodes.unsafe_get(
                        Int(current_ref & COMPACT_LEAF_REF_INDEX_MASK)
                    )
                    bounds = _decode_u8(node.bounds, node.start, node.scale)
                    child_mask = node.child_mask
                    compact_first_leaf = node.first_leaf
                    is_compact_leaf_node = True
                else:
                    ref node = encoded.hybrid_exact_nodes.unsafe_get(
                        Int(current_ref)
                    )
                    bounds = node.aabb
                    data = node.data
                    child_mask = encoded.hybrid_exact_masks.unsafe_get(
                        Int(current_ref)
                    )

            var aabb_hit = intersect_ray_aabb_octant_fma[
                positive_x=positive_x,
                positive_y=positive_y,
                positive_z=positive_z,
            ](origin_rcp_d, rcp_d, bounds, hit.t)
            var mask = aabb_hit.mask
            var bits = UInt32(pack_bits(mask)) & child_mask
            comptime if collect_stats:
                var active_lanes = 0
                var active_bits = child_mask
                while active_bits != 0:
                    active_bits &= active_bits - 1
                    active_lanes += 1
                stats.active_child_lanes += active_lanes
                var hit_lanes = 0
                var count_bits = bits
                while count_bits != 0:
                    count_bits &= count_bits - 1
                    hit_lanes += 1
                stats.aabb_hit_lanes += hit_lanes
                if hit_lanes > 0:
                    stats.nodes_with_hits += 1

            var has_nearest = False
            var nearest_ref = UInt32(0)
            var nearest_t = Float32(0.0)
            while bits != 0:
                var lane = Int(count_trailing_zeros(bits))
                bits &= bits - 1
                var child_ref = data[lane]
                comptime if encoding == "hybrid_u8":
                    if is_compact_leaf_node:
                        child_ref = encode_leaf_ref(
                            compact_first_leaf + UInt32(lane)
                        )
                var child_t = aabb_hit.t[lane]
                if child_t <= hit.t:
                    if not has_nearest:
                        nearest_ref = child_ref
                        nearest_t = child_t
                        has_nearest = True
                    elif child_t < nearest_t:
                        push_pending(nearest_ref, nearest_t)
                        nearest_ref = child_ref
                        nearest_t = child_t
                    else:
                        push_pending(child_ref, child_t)

            if has_nearest:
                if stack_ptr > 0:
                    var pending_idx = stack_ptr - 1
                    var pending = ordered_stack.unsafe_get(pending_idx)
                    if _pending_task_t(pending) < nearest_t:
                        stack_ptr = pending_idx
                        comptime if collect_stats:
                            stats.stack_pops += 1
                        push_pending(nearest_ref, nearest_t)
                        current_ref = _pending_task_ref(pending)
                        continue
                current_ref = nearest_ref
                continue

        var found_pending = False
        if stack_ptr > 0:
            var next_idx = stack_ptr - 1
            var task = ordered_stack.unsafe_get(next_idx)
            if _pending_task_t(task) <= hit.t:
                stack_ptr = next_idx
                comptime if collect_stats:
                    stats.stack_pops += 1
                current_ref = _pending_task_ref(task)
                found_pending = True
            else:
                comptime if collect_stats:
                    stats.stack_pruned_tasks += stack_ptr
                stack_ptr = 0
        if not found_pending:
            break
    return hit


@always_inline
def trace_encoded[
    encoding: String, collect_stats: Bool = False
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    ray: Rayf32[Frame.WORLD],
    mut stats: CpuBvhTraversalStats,
) -> Hit[Frame.WORLD]:
    comptime if collect_stats:
        stats.rays += 1

    @always_inline
    def octant[
        positive_x: Bool, positive_y: Bool, positive_z: Bool
    ]() {imm, mut stats} -> Hit[Frame.WORLD]:
        return _trace_encoded_octant[
            encoding,
            collect_stats,
            positive_x,
            positive_y,
            positive_z,
        ](bvh, encoded, ray, stats)

    var px = ray.d.x >= 0.0
    var py = ray.d.y >= 0.0
    var pz = ray.d.z >= 0.0
    if px:
        if py:
            if pz:
                return octant[True, True, True]()
            return octant[True, True, False]()
        if pz:
            return octant[True, False, True]()
        return octant[True, False, False]()
    if py:
        if pz:
            return octant[False, True, True]()
        return octant[False, True, False]()
    if pz:
        return octant[False, False, True]()
    return octant[False, False, False]()


@always_inline
def _trace_any_encoded_octant[
    encoding: String,
    collect_stats: Bool,
    positive_x: Bool,
    positive_y: Bool,
    positive_z: Bool,
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    ray: Rayf32[Frame.WORLD],
    mut stats: CpuBvhTraversalStats,
) -> Hit[Frame.WORLD]:
    var hit = Hit[Frame.WORLD].miss(ray.t_max)
    var stack = Array[UInt32, CPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0
    var current_ref = UInt32(0)
    comptime if encoding == "hybrid_u8":
        current_ref = encoded.hybrid_root_ref

    var O = ray.origin[WIDTH]()
    var D = ray.direction[WIDTH]()
    var rcp_d = ray.rcp_direction[WIDTH]()
    var origin_rcp_d = Vec3[DType.float32, Frame.WORLD, WIDTH](
        O.x * rcp_d.x, O.y * rcp_d.y, O.z * rcp_d.z
    )

    @always_inline
    def leaf_hit(leaf_idx: UInt32) {imm, mut stats} -> Bool:
        comptime if collect_stats:
            stats.leaf_blocks += 1
            stats.primitive_packet_lanes += WIDTH
        ref block = bvh.leaf_blocks.unsafe_get(Int(leaf_idx))
        var tri_hit = intersect_ray_tri_edges(
            O,
            D,
            block.v0,
            block.e1,
            block.e2,
            ray.t_max,
            ray.t_min,
        )
        var mask = tri_hit.mask & block.prim_indices.ne(EMPTY_LANE)
        comptime if collect_stats:
            stats.valid_primitives += _count_true_lanes(
                block.prim_indices.ne(EMPTY_LANE)
            )
            stats.primitive_hit_candidates += _count_true_lanes(mask)
        return mask.reduce_or()

    while True:
        comptime if collect_stats:
            stats.internal_nodes += 1
            stats.aabb_packet_lanes += WIDTH

        var bounds = BoundsF32.invalid()
        var data = SIMD[DType.uint32, WIDTH](EMPTY_LANE)
        var child_mask = UInt32(0)
        var compact_first_leaf = UInt32(0)
        var is_compact_leaf_node = False
        _load_encoded_node[encoding](
            bvh,
            encoded,
            current_ref,
            bounds,
            data,
            child_mask,
            compact_first_leaf,
            is_compact_leaf_node,
        )

        var aabb_hit = intersect_ray_aabb_octant_fma[
            positive_x=positive_x,
            positive_y=positive_y,
            positive_z=positive_z,
        ](origin_rcp_d, rcp_d, bounds, ray.t_max)
        var bits = UInt32(pack_bits(aabb_hit.mask)) & child_mask
        comptime if collect_stats:
            var hit_lanes = 0
            var count_bits = bits
            while count_bits != 0:
                count_bits &= count_bits - 1
                hit_lanes += 1
            stats.aabb_hit_lanes += hit_lanes
            if hit_lanes > 0:
                stats.nodes_with_hits += 1

        var has_next = False
        var next_ref = UInt32(0)
        while bits != 0:
            var lane = Int(count_trailing_zeros(bits))
            bits &= bits - 1
            var child_ref = data[lane]
            comptime if encoding == "hybrid_u8":
                if is_compact_leaf_node:
                    child_ref = encode_leaf_ref(
                        compact_first_leaf + UInt32(lane)
                    )

            if is_leaf_ref(child_ref):
                if leaf_hit(decode_ref_index(child_ref)):
                    comptime if collect_stats:
                        stats.any_hit_early_exits += 1
                    return Hit[Frame.WORLD].shadow_hit()
            else:
                if has_next:
                    stack.unsafe_get(stack_ptr) = next_ref
                    stack_ptr += 1
                    comptime if collect_stats:
                        stats.stack_pushes += 1
                        if stack_ptr > stats.max_stack_depth:
                            stats.max_stack_depth = stack_ptr
                next_ref = child_ref
                has_next = True

        if has_next:
            current_ref = next_ref
            continue
        if stack_ptr == 0:
            break
        stack_ptr -= 1
        comptime if collect_stats:
            stats.stack_pops += 1
        current_ref = stack.unsafe_get(stack_ptr)
    return hit


@always_inline
def trace_any_encoded[
    encoding: String, collect_stats: Bool = False
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    ray: Rayf32[Frame.WORLD],
    mut stats: CpuBvhTraversalStats,
) -> Hit[Frame.WORLD]:
    comptime if collect_stats:
        stats.rays += 1

    @always_inline
    def octant[
        positive_x: Bool, positive_y: Bool, positive_z: Bool
    ]() {imm, mut stats} -> Hit[Frame.WORLD]:
        return _trace_any_encoded_octant[
            encoding,
            collect_stats,
            positive_x,
            positive_y,
            positive_z,
        ](bvh, encoded, ray, stats)

    var px = ray.d.x >= 0.0
    var py = ray.d.y >= 0.0
    var pz = ray.d.z >= 0.0
    if px:
        if py:
            if pz:
                return octant[True, True, True]()
            return octant[True, True, False]()
        if pz:
            return octant[True, False, True]()
        return octant[True, False, False]()
    if py:
        if pz:
            return octant[False, True, True]()
        return octant[False, True, False]()
    if pz:
        return octant[False, False, True]()
    return octant[False, False, False]()


@fieldwise_init
struct BenchResult(Copyable):
    var ns: Int
    var checksum: Float64
    var hits: Int


def run_rays[
    encoding: String, collect_stats: Bool = False
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    rays: List[Rayf32[Frame.WORLD]],
    mut stats: CpuBvhTraversalStats,
) -> Tuple[Float64, Int]:
    var checksum = 0.0
    var hits = 0
    for ray in rays:
        var hit = trace_encoded[encoding, collect_stats](
            bvh, encoded, ray, stats
        )
        if hit.is_hit():
            checksum += Float64(hit.t) + Float64(hit.prim)
            hits += 1
    return (checksum, hits)


def benchmark[
    encoding: String
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    rays: List[Rayf32[Frame.WORLD]],
) -> BenchResult:
    var unused = CpuBvhTraversalStats()
    var summary = run_rays[encoding](bvh, encoded, rays, unused)
    var best_ns = Int.MAX
    for _ in range(TIMING_REPEATS):
        var t0 = perf_counter_ns()
        summary = run_rays[encoding](bvh, encoded, rays, unused)
        var elapsed = Int(perf_counter_ns() - t0)
        if elapsed < best_ns:
            best_ns = elapsed
    return BenchResult(best_ns, summary[0], summary[1])


def run_any_rays[
    encoding: String, collect_stats: Bool = False
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    rays: List[Rayf32[Frame.WORLD]],
    mut stats: CpuBvhTraversalStats,
) -> Int:
    var hits = 0
    for ray in rays:
        if trace_any_encoded[encoding, collect_stats](
            bvh, encoded, ray, stats
        ).is_occluded():
            hits += 1
    return hits


def benchmark_any[
    encoding: String
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    rays: List[Rayf32[Frame.WORLD]],
) -> BenchResult:
    var unused = CpuBvhTraversalStats()
    var hits = run_any_rays[encoding](bvh, encoded, rays, unused)
    var best_ns = Int.MAX
    for _ in range(TIMING_REPEATS):
        var t0 = perf_counter_ns()
        hits = run_any_rays[encoding](bvh, encoded, rays, unused)
        var elapsed = Int(perf_counter_ns() - t0)
        if elapsed < best_ns:
            best_ns = elapsed
    return BenchResult(best_ns, Float64(hits), hits)


def validate_any[
    encoding: String
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    rays: List[Rayf32[Frame.WORLD]],
) -> Int:
    var mismatches = 0
    var unused = CpuBvhTraversalStats()
    for ray in rays:
        var expected = trace_any_encoded["f32"](bvh, encoded, ray, unused)
        var actual = trace_any_encoded[encoding](bvh, encoded, ray, unused)
        if expected.is_occluded() != actual.is_occluded():
            mismatches += 1
    return mismatches


def validate[
    encoding: String
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    rays: List[Rayf32[Frame.WORLD]],
) -> Tuple[Int, Int, Int, Float32]:
    var hit_mismatches = 0
    var prim_mismatches = 0
    var t_mismatches = 0
    var max_t_error = Float32(0.0)
    var unused = CpuBvhTraversalStats()
    for ray in rays:
        var expected = trace_encoded["f32"](bvh, encoded, ray, unused)
        var actual = trace_encoded[encoding](bvh, encoded, ray, unused)
        if expected.is_hit() != actual.is_hit():
            hit_mismatches += 1
        elif expected.is_hit():
            if expected.prim != actual.prim:
                prim_mismatches += 1
            var error = abs(expected.t - actual.t)
            if error > max_t_error:
                max_t_error = error
            if error > 1.0e-4:
                t_mismatches += 1
    return (hit_mismatches, prim_mismatches, t_mismatches, max_t_error)


def print_encoding[
    encoding: String
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    rays: List[Rayf32[Frame.WORLD]],
    baseline: BenchResult,
) raises:
    var result = benchmark[encoding](bvh, encoded, rays)
    var validation = validate[encoding](bvh, encoded, rays)
    print(
        encoding,
        ", ",
        round(ns_to_mrays_per_s(result.ns, len(rays)), 3),
        ", ",
        round(Float64(baseline.ns) / Float64(result.ns), 3),
        ", ",
        result.hits,
        ", ",
        round(result.checksum - baseline.checksum, 6),
        ", ",
        validation[0],
        ", ",
        validation[1],
        ", ",
        validation[2],
        ", ",
        validation[3],
    )


def print_baseline(result: BenchResult, ray_count: Int) raises:
    print(
        "f32, ",
        round(ns_to_mrays_per_s(result.ns, ray_count), 3),
        ", 1.0, ",
        result.hits,
        ", 0.0, 0, 0, 0, 0.0",
    )


def print_counters[
    encoding: String
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    var stats = CpuBvhTraversalStats()
    _ = run_rays[encoding, collect_stats=True](bvh, encoded, rays, stats)
    var shifts_per_push = 0.0
    if stats.stack_pushes > 0:
        shifts_per_push = Float64(stats.stack_insertion_shifts) / Float64(
            stats.stack_pushes
        )
    print(
        encoding,
        ", ",
        round(Float64(stats.internal_nodes) / Float64(stats.rays), 3),
        ", ",
        round(Float64(stats.aabb_hit_lanes) / Float64(stats.rays), 3),
        ", ",
        round(Float64(stats.leaf_blocks) / Float64(stats.rays), 3),
        ", ",
        round(Float64(stats.stack_pushes) / Float64(stats.rays), 3),
        ", ",
        round(shifts_per_push, 3),
    )


def run_case(
    label: String,
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    print(t"\n{label}: {len(rays)} rays")
    print(
        "encoding, MRay/s, vs_f32, hits, checksum_delta, hit_mismatch,"
        " prim_mismatch, t_mismatch, max_t_error"
    )
    var baseline = benchmark["f32"](bvh, encoded, rays)
    print_baseline(baseline, len(rays))
    print_encoding["f16"](bvh, encoded, rays, baseline)
    print_encoding["global_u16"](bvh, encoded, rays, baseline)
    print_encoding["relative_u16"](bvh, encoded, rays, baseline)
    print_encoding["relative_u8"](bvh, encoded, rays, baseline)
    print_encoding["hybrid_u8"](bvh, encoded, rays, baseline)
    print(
        "counters: encoding, nodes/ray, AABB hits/ray, leaves/ray,"
        " pushes/ray, shifts/push"
    )
    print_counters["f32"](bvh, encoded, rays)
    print_counters["f16"](bvh, encoded, rays)
    print_counters["global_u16"](bvh, encoded, rays)
    print_counters["relative_u16"](bvh, encoded, rays)
    print_counters["relative_u8"](bvh, encoded, rays)
    print_counters["hybrid_u8"](bvh, encoded, rays)


def print_any_encoding[
    encoding: String
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    rays: List[Rayf32[Frame.WORLD]],
    baseline: BenchResult,
) raises:
    var result = benchmark_any[encoding](bvh, encoded, rays)
    print(
        encoding,
        ", ",
        round(ns_to_mrays_per_s(result.ns, len(rays)), 3),
        ", ",
        round(Float64(baseline.ns) / Float64(result.ns), 3),
        ", ",
        result.hits,
        ", ",
        validate_any[encoding](bvh, encoded, rays),
    )


def print_any_counters[
    encoding: String
](
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    var stats = CpuBvhTraversalStats()
    _ = run_any_rays[encoding, collect_stats=True](bvh, encoded, rays, stats)
    print(
        encoding,
        ", ",
        round(Float64(stats.internal_nodes) / Float64(stats.rays), 3),
        ", ",
        round(Float64(stats.aabb_hit_lanes) / Float64(stats.rays), 3),
        ", ",
        round(Float64(stats.leaf_blocks) / Float64(stats.rays), 3),
        ", ",
        round(Float64(stats.stack_pushes) / Float64(stats.rays), 3),
    )


def run_any_case(
    label: String,
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH],
    encoded: EncodedTrees,
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    print(t"\n{label}: {len(rays)} rays")
    print("encoding, MRay/s, vs_f32, hits, hit_mismatch")
    var baseline = benchmark_any["f32"](bvh, encoded, rays)
    print(
        "f32, ",
        round(ns_to_mrays_per_s(baseline.ns, len(rays)), 3),
        ", 1.0, ",
        baseline.hits,
        ", 0",
    )
    print_any_encoding["f16"](bvh, encoded, rays, baseline)
    print_any_encoding["global_u16"](bvh, encoded, rays, baseline)
    print_any_encoding["relative_u16"](bvh, encoded, rays, baseline)
    print_any_encoding["relative_u8"](bvh, encoded, rays, baseline)
    print_any_encoding["hybrid_u8"](bvh, encoded, rays, baseline)
    print(
        "counters: encoding, nodes/ray, AABB hits/ray, leaves/ray, pushes/ray"
    )
    print_any_counters["f32"](bvh, encoded, rays)
    print_any_counters["f16"](bvh, encoded, rays)
    print_any_counters["global_u16"](bvh, encoded, rays)
    print_any_counters["relative_u16"](bvh, encoded, rays)
    print_any_counters["relative_u8"](bvh, encoded, rays)
    print_any_counters["hybrid_u8"](bvh, encoded, rays)


def print_sizes(
    bvh: TriangleBvh[Frame.WORLD, WIDTH, WIDTH], encoded: EncodedTrees
) raises:
    var current_bytes = (
        len(bvh.tree.nodes) * size_of[WideBvhNode[Frame.WORLD, WIDTH]]()
        + len(bvh.tree.child_masks) * 4
    )
    print(t"nodes: {len(bvh.tree.nodes)}, current bytes: {current_bytes}")
    print(
        t"  f16 bytes: {encoded.bytes['f16']() + len(bvh.tree.child_masks) * 4}"
    )
    print(
        t"  global_u16 bytes:"
        t" {encoded.bytes['global_u16']() + len(bvh.tree.child_masks) * 4}"
    )
    print(t"  relative_u16 bytes: {encoded.bytes['relative_u16']()}")
    print(t"  relative_u8 bytes: {encoded.bytes['relative_u8']()}")
    print(t"  hybrid_u8 bytes: {encoded.bytes['hybrid_u8']()}")


def main() raises:
    print("CPU BVH16 node-encoding experiment")
    print(
        t"sizes: f32={size_of[WideBvhNode[Frame.WORLD, WIDTH]]()},"
        t" f16={size_of[Float16Node]()}, gu16={size_of[GlobalUInt16Node]()},"
        t" ru16={size_of[RelativeUInt16Node]()},"
        t" ru8={size_of[RelativeUInt8Node]()},"
        t" hybrid_leaf={size_of[CompactLeafNode]()}"
    )

    var dragon_vertices = pack_obj_triangles[Frame.WORLD](OBJ_PATH)
    var dragon_bounds = compute_bounds(dragon_vertices)
    var dragon_camera = make_camera_rays_and_params(
        dragon_bounds, 512, 288, 1, 0.2
    )
    var dragon_rays = dragon_camera[0].copy()
    var dragon_bvh = TriangleBvh[Frame.WORLD, WIDTH, WIDTH].__init__["sah"](
        dragon_vertices
    )
    var dragon_encoded = EncodedTrees(dragon_bvh)
    var dragon_hit_rays = select_and_repeat_hit_rays(dragon_bvh, dragon_rays)
    var dragon_permuted = permute_rays(dragon_hit_rays)
    print("\nDragon storage")
    print_sizes(dragon_bvh, dragon_encoded)
    run_case("Dragon natural coherent", dragon_bvh, dragon_encoded, dragon_rays)
    run_case(
        "Dragon high-hit coherent", dragon_bvh, dragon_encoded, dragon_hit_rays
    )
    run_case(
        "Dragon high-hit permuted", dragon_bvh, dragon_encoded, dragon_permuted
    )
    run_any_case(
        "Dragon high-hit any coherent",
        dragon_bvh,
        dragon_encoded,
        dragon_hit_rays,
    )
    run_any_case(
        "Dragon high-hit any permuted",
        dragon_bvh,
        dragon_encoded,
        dragon_permuted,
    )

    var grid_bvh = TriangleBvh[Frame.WORLD, WIDTH, WIDTH].__init__["sah"](
        make_grid_triangles()
    )
    var grid_encoded = EncodedTrees(grid_bvh)
    var grid_rays = make_hit_and_miss_rays()
    print("\nGrid storage")
    print_sizes(grid_bvh, grid_encoded)
    run_case("Regular grid", grid_bvh, grid_encoded, grid_rays)

    var depth_bvh = TriangleBvh[Frame.WORLD, WIDTH, WIDTH].__init__["sah"](
        make_depth_overlap_triangles()
    )
    var depth_encoded = EncodedTrees(depth_bvh)
    var depth_rays = make_depth_overlap_rays()
    print("\nLayered-overlap storage")
    print_sizes(depth_bvh, depth_encoded)
    run_case("Layered overlap", depth_bvh, depth_encoded, depth_rays)
