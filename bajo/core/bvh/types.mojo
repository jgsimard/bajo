from std.algorithm import parallelize
from std.math import abs, min, max, clamp
from std.utils.numerics import max_finite, min_finite

from bajo.core.aabb import AABB
from bajo.core.vec import Vec3f32

comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()
comptime INV_3 = Float32(1.0 / 3.0)
comptime BVH_BINS = 16


@fieldwise_init
struct Intersection(TrivialRegisterPassable):
    var t: Float32
    var u: Float32
    var v: Float32
    var inst: UInt32
    var prim: UInt32

    @always_inline
    def __init__(out self):
        self.t = f32_max
        self.u = 0.0
        self.v = 0.0
        self.inst = 0
        self.prim = 0


@fieldwise_init
struct Ray(Copyable):
    var O: Vec3f32
    var mask: UInt32
    var D: Vec3f32
    var rD: Vec3f32
    var hit: Intersection

    def __init__(out self, O: Vec3f32, D: Vec3f32, t_max: Float32 = f32_max):
        self.O = O.copy()
        self.D = D.copy()
        var rDx = clamp(Float32(1.0) / D.x(), f32_min, f32_max)
        var rDy = clamp(Float32(1.0) / D.y(), f32_min, f32_max)
        var rDz = clamp(Float32(1.0) / D.z(), f32_min, f32_max)
        self.rD = Vec3f32(rDx, rDy, rDz)
        self.mask = 0xFFFFFFFF
        self.hit = Intersection()
        self.hit.t = t_max


@fieldwise_init
struct RayFlat(TrivialRegisterPassable):
    var ox: Float32
    var oy: Float32
    var oz: Float32
    var dx: Float32
    var dy: Float32
    var dz: Float32
    var rdx: Float32
    var rdy: Float32
    var rdz: Float32
    var t_max: Float32


@fieldwise_init
struct Hit(TrivialRegisterPassable):
    var t: Float32
    var u: Float32
    var v: Float32
    var prim: UInt32
    var occluded: UInt32


@fieldwise_init
struct BvhNode(Copyable):
    var aabb: AABB
    var left_first: UInt32
    var tri_count: UInt32

    @always_inline
    def __init__(out self):
        self.aabb = AABB.invalid()
        self.left_first = 0
        self.tri_count = 0

    @always_inline
    def is_leaf(self) -> Bool:
        return self.tri_count > 0

    @always_inline
    def surface_area(self) -> Float32:
        return self.aabb.surface_area()


@fieldwise_init
struct Bin(Copyable):
    var bounds: AABB
    var tri_count: UInt32

    def __init__(out self):
        self.bounds = AABB.invalid()
        self.tri_count = 0


@fieldwise_init
struct Fragment(Copyable):
    """Cached primitive bounds used by the builder.

    prim_indices stores indices into this Fragment array. `prim_idx` stores the
    original triangle id, so traversal still reports the mesh primitive id.
    """

    var bmin: Vec3f32
    var prim_idx: UInt32
    var bmax: Vec3f32
    var _pad: UInt32

    @always_inline
    def __init__(out self):
        self.bmin = Vec3f32(f32_max)
        self.prim_idx = 0
        self.bmax = Vec3f32(f32_min)
        self._pad = 0

    @always_inline
    def __init__(
        out self, prim_idx: UInt32, v0: Vec3f32, v1: Vec3f32, v2: Vec3f32
    ):
        self.bmin = vmin(v0, v1, v2)
        self.prim_idx = prim_idx
        self.bmax = vmax(v0, v1, v2)
        self._pad = 0

    @always_inline
    def center_axis(self, axis: Int) -> Float32:
        return (self.bmin[axis] + self.bmax[axis]) * 0.5

    @always_inline
    def grow_into(self, mut aabb: AABB):
        aabb._min = vmin(aabb._min, self.bmin)
        aabb._max = vmax(aabb._max, self.bmax)


@fieldwise_init
struct SplitResult(Copyable):
    """Result of the binned SAH sweep.

    `bin`, `bin_min`, and `bin_scale` describe the exact binning rule used
    during evaluation. Partitioning by these fields keeps the partition in sync
    with the cost and child bounds computed by `_sah`.
    """

    var axis: Int
    var bin: Int
    var pos: Float32
    var cost: Float32
    var bin_min: Float32
    var bin_scale: Float32
    var left_bounds: AABB
    var right_bounds: AABB

    @always_inline
    def __init__(out self):
        self.axis = -1
        self.bin = -1
        self.pos = 0.0
        self.cost = f32_max
        self.bin_min = 0.0
        self.bin_scale = 0.0
        self.left_bounds = AABB.invalid()
        self.right_bounds = AABB.invalid()

    @always_inline
    def valid(self) -> Bool:
        return self.axis >= 0 and self.bin >= 0


@fieldwise_init
struct MortonPrim(TrivialRegisterPassable):
    """Morton-code / fragment-index pair used by the CPU LBVH builder."""

    var code: UInt32
    var frag_idx: UInt32

    @always_inline
    def __init__(out self):
        self.code = 0
        self.frag_idx = 0
