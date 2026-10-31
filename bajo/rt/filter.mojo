"""Reconstruction filters and PBRT's tabulated Gaussian importance sampler.

Algorithm reference: pbrt-v4 filters.h/.cpp and util/sampling.h (Apache-2.0).
The Gaussian table uses the authored radius at int(32 * radius) cells per axis.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import abs, exp, pi, sqrt
from std.utils.numerics import isfinite


struct ReconstructionFilter(
    Copyable, DevicePassable, TrivialRegisterPassable, Writable
):
    comptime BOX = UInt32(0)
    comptime GAUSSIAN = UInt32(1)
    comptime TRIANGLE = UInt32(2)
    var kind: UInt32
    var x_radius: Float32
    var y_radius: Float32
    var sigma: Float32

    def __init__(
        out self,
        kind: UInt32 = 0,
        x_radius: Float32 = 0.5,
        y_radius: Float32 = 0.5,
        sigma: Float32 = 0.5,
    ):
        self.kind = kind
        self.x_radius = x_radius
        self.y_radius = y_radius
        self.sigma = sigma

    @staticmethod
    def gaussian(
        x_radius: Float32 = 1.5, y_radius: Float32 = 1.5, sigma: Float32 = 0.5
    ) -> Self:
        return Self(Self.GAUSSIAN, x_radius, y_radius, sigma)

    @staticmethod
    def triangle(x_radius: Float32 = 2, y_radius: Float32 = 2) -> Self:
        return Self(Self.TRIANGLE, x_radius, y_radius)

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "ReconstructionFilter"

    def matches(self, other: Self) -> Bool:
        return (
            self.kind == other.kind
            and self.x_radius == other.x_radius
            and self.y_radius == other.y_radius
            and self.sigma == other.sigma
        )

    def validate(self) raises:
        if self.kind > Self.TRIANGLE:
            raise Error("unsupported reconstruction filter")
        if (
            not isfinite(self.x_radius)
            or not isfinite(self.y_radius)
            or self.x_radius <= 0
            or self.y_radius <= 0
        ):
            raise Error("filter radii must be finite and positive")
        if self.kind == Self.GAUSSIAN:
            if not isfinite(self.sigma) or self.sigma <= 0:
                raise Error("Gaussian sigma must be finite and positive")
            if self.x_radius < 1.0 / 32.0 or self.y_radius < 1.0 / 32.0:
                raise Error(
                    "Gaussian radii must provide at least one PBRT table cell"
                )
            if Float64(self.x_radius) * 32 > Float64(Int32.MAX) or Float64(
                self.y_radius
            ) * 32 > Float64(Int32.MAX):
                raise Error("Gaussian table exceeds 32-bit dimensions")

    def evaluate(self, x: Float32, y: Float32) -> Float32:
        if self.kind == Self.BOX:
            return Float32(
                1 if abs(x) <= self.x_radius and abs(y) <= self.y_radius else 0
            )
        if self.kind == Self.TRIANGLE:
            return max(Float32(0), self.x_radius - abs(x)) * max(
                Float32(0), self.y_radius - abs(y)
            )
        var norm = Float32(1) / (self.sigma * sqrt(Float32(2 * pi)))
        var denominator = Float32(2) * self.sigma * self.sigma
        var gx = norm * (
            exp(-x * x / denominator)
            - exp(-self.x_radius * self.x_radius / denominator)
        )
        var gy = norm * (
            exp(-y * y / denominator)
            - exp(-self.y_radius * self.y_radius / denominator)
        )
        return max(Float32(0), gx) * max(Float32(0), gy)


@fieldwise_init
struct FilterSample[width: SIMDLength](TrivialRegisterPassable):
    # Film positions relative to the lower corner of the requested pixel.
    var pixel_u: SIMD[.float32, Self.width]
    var pixel_v: SIMD[.float32, Self.width]
    var weight: SIMD[.float32, Self.width]


struct FilterView(Copyable, DevicePassable, ImplicitlyCopyable, Writable):
    var config: ReconstructionFilter
    var data: Optional[ImmPointer[Float32, ImmUntrackedOrigin]]
    var nx: Int32
    var ny: Int32

    def __init__(
        out self,
        config: ReconstructionFilter = ReconstructionFilter(),
        data: Optional[ImmPointer[Float32, ImmUntrackedOrigin]] = Optional[
            ImmPointer[Float32, ImmUntrackedOrigin]
        ](),
        nx: Int32 = 0,
        ny: Int32 = 0,
    ):
        self.config = config
        self.data = data
        self.nx = nx
        self.ny = ny

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "FilterView"

    @always_inline
    def _cell(self, cdf_base: Int, count: Int, u: Float32) -> Int:
        var data = self.data.value()
        var lo = 0
        var hi = count
        while lo + 1 < hi:
            var mid = (lo + hi) // 2
            if data[unsafe_offset=cdf_base + mid] <= u:
                lo = mid
            else:
                hi = mid
        return lo

    @always_inline
    def _position(
        self, base: Int, cell: Int, count: Int, u: Float32, radius: Float32
    ) -> Float32:
        var data = self.data.value()
        var low = data[unsafe_offset=base + cell]
        var delta = data[unsafe_offset=base + cell + 1] - low
        var t = (
            Float32(cell) + (u - low) / (delta if delta > 0 else Float32(1))
        ) / Float32(count)
        return 0.5 + (1 - t) * (-radius) + t * radius

    @always_inline
    def sample[
        width: SIMDLength
    ](self, u: SIMD[.float32, width], v: SIMD[.float32, width]) -> FilterSample[
        width
    ]:
        var rx = self.config.x_radius
        var ry = self.config.y_radius
        if self.config.kind == ReconstructionFilter.BOX:
            return FilterSample[width](
                u if rx == 0.5 else 0.5 + (1 - u) * (-rx) + u * rx,
                v if ry == 0.5 else 0.5 + (1 - v) * (-ry) + v * ry,
                1,
            )
        if self.config.kind == ReconstructionFilter.TRIANGLE:
            var x = u.lt(0.5).select(sqrt(2 * u) - 1, 1 - sqrt(2 * (1 - u)))
            var y = v.lt(0.5).select(sqrt(2 * v) - 1, 1 - sqrt(2 * (1 - v)))
            return FilterSample[width](0.5 + rx * x, 0.5 + ry * y, 1)
        debug_assert["safe"](
            self.nx > 0 and self.ny > 0,
            "Gaussian sampler tables must be prepared before use",
        )
        var result = FilterSample[width](0, 0, 0)
        var nx = Int(self.nx)
        var ny = Int(self.ny)
        var rows = 1 + ny + 1
        var conditional = rows + ny
        var values = conditional + ny * (nx + 1)
        var data = self.data.value()
        var total = data[unsafe_offset=0]
        for lane in range(Int(width)):
            var y = self._cell(1, ny, v[lane])
            var base = conditional + y * (nx + 1)
            var x = self._cell(base, nx, u[lane])
            result.pixel_u[lane] = self._position(base, x, nx, u[lane], rx)
            result.pixel_v[lane] = self._position(1, y, ny, v[lane], ry)
            var f = data[unsafe_offset=values + y * nx + x]
            var row = data[unsafe_offset=rows + y]
            var pdf = (f / row) * (
                row / total
            ) if row > 0 and total > 0 else Float32(0)
            result.weight[lane] = f / pdf if pdf > 0 else Float32(0)
        return result


struct FilterSampler:
    var config: ReconstructionFilter
    var data: List[Float32]
    var nx: Int32
    var ny: Int32

    def __init__(out self, config: ReconstructionFilter) raises:
        config.validate()
        self.config = config
        self.nx = 0
        self.ny = 0
        self.data = List[Float32](length=1, fill=0)
        if config.kind != ReconstructionFilter.GAUSSIAN:
            return
        var nx = Int(32 * config.x_radius)
        var ny = Int(32 * config.y_radius)
        self.nx = Int32(nx)
        self.ny = Int32(ny)
        var rows = ny + 2
        var conditional = rows + ny
        var values = conditional + ny * (nx + 1)
        self.data.resize(values + nx * ny, fill=0)
        for y in range(ny):
            var ty = (Float32(y) + 0.5) / Float32(ny)
            var py = (1 - ty) * (-config.y_radius) + ty * config.y_radius
            var base = conditional + y * (nx + 1)
            for x in range(nx):
                var tx = (Float32(x) + 0.5) / Float32(nx)
                var px = (1 - tx) * (-config.x_radius) + tx * config.x_radius
                var f = config.evaluate(px, py)
                self.data[values + y * nx + x] = f
                self.data[base + x + 1] = self.data[base + x] + f * (
                    2 * config.x_radius
                ) / Float32(nx)
            var integral = self.data[base + nx]
            self.data[rows + y] = integral
            for x in range(1, nx + 1):
                self.data[base + x] = self.data[
                    base + x
                ] / integral if integral > 0 else Float32(x) / Float32(nx)
            self.data[1 + y + 1] = self.data[1 + y] + integral * (
                2 * config.y_radius
            ) / Float32(ny)
        self.data[0] = self.data[1 + ny]
        if not isfinite(self.data[0]) or self.data[0] <= 0:
            raise Error("Gaussian filter has no finite positive table mass")
        for y in range(1, ny + 1):
            self.data[1 + y] /= self.data[0]

    def view(self) -> FilterView:
        """Borrow tables without tracking their lifetime; retain this owner."""
        return FilterView(
            self.config,
            Optional(
                self.data.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()
            ),
            self.nx,
            self.ny,
        )
