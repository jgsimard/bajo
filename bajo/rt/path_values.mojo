"""Layout-independent path values shared by CPU SIMD and GPU scalar stages."""

from bajo.bvh.constants import f32_max
from bajo.core import Point3, Vec3
from bajo.rt.rays import RT_RAY_T_MIN


struct PathFloatField:
    comptime OX = 0
    comptime OY = 1
    comptime OZ = 2
    comptime T_MIN = 3
    comptime DX = 4
    comptime DY = 5
    comptime DZ = 6
    comptime T_MAX = 7
    comptime TX = 8
    comptime TY = 9
    comptime TZ = 10
    comptime BSDF_PDF = 11
    # GPU storage encodes the medium word as bits in one extra float plane.
    comptime MEDIUM = 12
    comptime PLANES = 13
    comptime VALUE_PLANES = 12


trait PathValues(Copyable):
    comptime WIDTH: SIMDLength

    def float_field[field: Int](self) -> SIMD[.float32, Self.WIDTH]:
        ...

    def ids(self) -> SIMD[.uint32, Self.WIDTH]:
        ...

    def media(self) -> SIMD[.uint32, Self.WIDTH]:
        ...

    def delta_events(self) -> SIMD[.bool, Self.WIDTH]:
        ...


@fieldwise_init
struct PathContinuation[length: SIMDLength](
    PathValues, TrivialRegisterPassable
):
    """Register values; no packet allocation or physical storage layout."""

    comptime WIDTH = Self.length
    var path_ids: SIMD[.uint32, Self.length]
    var origin: Point3[.float32, .WORLD, Self.length]
    var direction: Vec3[.float32, .WORLD, Self.length]
    var throughput: Vec3[.float32, .WORLD, Self.length]
    var pdf: SIMD[.float32, Self.length]
    var delta: SIMD[.bool, Self.length]
    var medium_ids: SIMD[.uint32, Self.length]

    @always_inline
    def ids(self) -> SIMD[.uint32, Self.WIDTH]:
        return self.path_ids

    @always_inline
    def media(self) -> SIMD[.uint32, Self.WIDTH]:
        return self.medium_ids

    @always_inline
    def delta_events(self) -> SIMD[.bool, Self.WIDTH]:
        return self.delta

    @always_inline
    def float_field[field: Int](self) -> SIMD[.float32, Self.WIDTH]:
        comptime __match field:
            case PathFloatField.OX:
                return self.origin.x
            case PathFloatField.OY:
                return self.origin.y
            case PathFloatField.OZ:
                return self.origin.z
            case PathFloatField.T_MIN:
                return SIMD[.float32, Self.WIDTH](RT_RAY_T_MIN)
            case PathFloatField.DX:
                return self.direction.x
            case PathFloatField.DY:
                return self.direction.y
            case PathFloatField.DZ:
                return self.direction.z
            case PathFloatField.T_MAX:
                return SIMD[.float32, Self.WIDTH](f32_max)
            case PathFloatField.TX:
                return self.throughput.x
            case PathFloatField.TY:
                return self.throughput.y
            case PathFloatField.TZ:
                return self.throughput.z
            case PathFloatField.BSDF_PDF:
                return self.pdf
            case _:
                comptime assert False, "not a floating-point path value"
