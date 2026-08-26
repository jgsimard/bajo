from std.math import cos, floor, pi, sin, sqrt, cbrt
from std.random import Random

from bajo.core import Vec3f32, dot, Frame


@fieldwise_init
struct Sampler(Equatable, TrivialRegisterPassable, Writable):
    """Runtime-selectable sample sequence used by the ray tracers."""

    var value: UInt32
    comptime INDEPENDENT = Self(0)
    comptime HALTON = Self(1)
    comptime R2 = Self(2)
    comptime OWEN_SOBOL = Self(3)
    comptime SZ = Self(4)
    comptime STBN = Self(5)

    def is_valid(self) -> Bool:
        return self in (
            Sampler.INDEPENDENT,
            Sampler.HALTON,
            Sampler.R2,
            Sampler.OWEN_SOBOL,
            Sampler.SZ,
            Sampler.STBN,
        )


@always_inline
def _reverse_bits32(value: UInt32) -> UInt32:
    var bits = value
    bits = ((bits & UInt32(0x55555555)) << UInt32(1)) | (
        (bits >> UInt32(1)) & UInt32(0x55555555)
    )
    bits = ((bits & UInt32(0x33333333)) << UInt32(2)) | (
        (bits >> UInt32(2)) & UInt32(0x33333333)
    )
    bits = ((bits & UInt32(0x0F0F0F0F)) << UInt32(4)) | (
        (bits >> UInt32(4)) & UInt32(0x0F0F0F0F)
    )
    bits = ((bits & UInt32(0x00FF00FF)) << UInt32(8)) | (
        (bits >> UInt32(8)) & UInt32(0x00FF00FF)
    )
    return (bits << UInt32(16)) | (bits >> UInt32(16))


@always_inline
def _mix32(value: UInt32) -> UInt32:
    var bits = value
    bits ^= bits >> UInt32(16)
    bits *= UInt32(0x7FEB352D)
    bits ^= bits >> UInt32(15)
    bits *= UInt32(0x846CA68B)
    bits ^= bits >> UInt32(16)
    return bits


@always_inline
def _owen_scramble(value: UInt32, seed: UInt32) -> UInt32:
    """Burley's fixed-cost hash-based Owen tree permutation."""
    var bits = _reverse_bits32(value)
    bits ^= bits * UInt32(0x3D20ADEA)
    bits += seed
    bits *= (seed >> UInt32(16)) | UInt32(1)
    bits ^= bits * UInt32(0x05526C56)
    bits ^= bits * UInt32(0x53A22864)
    return _reverse_bits32(bits)


@always_inline
def _u32_to_unit_float(value: UInt32) -> Float32:
    # Taking the high 24 bits gives every Float32 result an exact [0, 1) value.
    return Float32(value >> UInt32(8)) * Float32(5.960464477539063e-8)


@always_inline
def _sobol_bits(index: UInt32, dimension: Int) -> UInt32:
    """First four Joe-Kuo Sobol dimensions, generated without lookup tables."""
    var dim = dimension % 4
    var result = UInt32(0)
    if dim == 0:
        return _reverse_bits32(index)

    if dim == 1:
        var direction = UInt32(0x80000000)
        for bit in range(32):
            if (index & (UInt32(1) << UInt32(bit))) != 0:
                result ^= direction
            direction ^= direction >> UInt32(1)
        return result

    if dim == 2:
        # Primitive polynomial x^2 + x + 1, initial values (1, 3).
        var previous_2 = UInt32(0x80000000)
        var previous_1 = UInt32(0xC0000000)
        for bit in range(32):
            var direction = previous_2
            if bit == 1:
                direction = previous_1
            elif bit >= 2:
                direction = previous_2 ^ (previous_2 >> UInt32(2)) ^ previous_1
                previous_2 = previous_1
                previous_1 = direction
            if (index & (UInt32(1) << UInt32(bit))) != 0:
                result ^= direction
        return result

    # Primitive polynomial x^3 + x + 1, initial values (1, 3, 1).
    var previous_3 = UInt32(0x80000000)
    var previous_2 = UInt32(0xC0000000)
    var previous_1 = UInt32(0x20000000)
    for bit in range(32):
        var direction = previous_3
        if bit == 1:
            direction = previous_2
        elif bit == 2:
            direction = previous_1
        elif bit >= 3:
            direction = previous_3 ^ (previous_3 >> UInt32(3)) ^ previous_2
            previous_3 = previous_2
            previous_2 = previous_1
            previous_1 = direction
        if (index & (UInt32(1) << UInt32(bit))) != 0:
            result ^= direction
    return result


@always_inline
def _gf4_multiply(a: UInt32, b: UInt32) -> UInt32:
    """Multiply two polynomial-basis GF(4) elements, x^2 = x + 1."""
    var a0 = a & UInt32(1)
    var a1 = (a >> UInt32(1)) & UInt32(1)
    var b0 = b & UInt32(1)
    var b1 = (b >> UInt32(1)) & UInt32(1)
    var high = a1 * b1
    var c0 = (a0 * b0) ^ high
    var c1 = (a0 * b1) ^ (a1 * b0) ^ high
    return c0 | (c1 << UInt32(1))


@always_inline
def _sz_bits(index: UInt32, dimension: Int) -> UInt32:
    """Four-dimensional binary SZ (0, 4)-sequence from GF(4) Pascal nets."""
    var dim = dimension % 4
    var symbol = UInt32(dim)
    # The finite-field alphabet is {0, 1, x, x + 1}.
    var result = UInt32(0)
    for column in range(16):
        var input_digit = (index >> UInt32(2 * column)) & UInt32(3)
        if input_digit == 0:
            continue
        var block = UInt32(1)
        for offset in range(column + 1):
            var row = column - offset
            # Lucas' theorem: C(column, row) is odd iff row is a bit subset.
            if (row & column) == row and block != 0:
                var output_digit = _gf4_multiply(block, input_digit)
                result ^= (output_digit & UInt32(1)) << UInt32(31 - 2 * row)
                result ^= ((output_digit >> UInt32(1)) & UInt32(1)) << UInt32(
                    30 - 2 * row
                )
            block = _gf4_multiply(block, symbol)
    return result


@always_inline
def _stbn(
    sample_index: UInt64,
    dimension: Int,
    pixel_id: UInt32,
    image_width: UInt32,
    stage: UInt32,
    seed: UInt32,
) -> Float32:
    """Compact procedural space-time sampler for interactive accumulation."""
    var width = max(image_width, UInt32(1))
    var dim = dimension % 4
    var x = pixel_id % width + UInt32(17 * dim)
    var y = pixel_id / width + UInt32(29 * dim)
    # Interleaved-gradient spatial ordering has strong local stratification.
    var spatial = Float32(0.06711056) * Float32(x)
    spatial += Float32(0.00583715) * Float32(y)
    spatial -= floor(spatial)
    spatial *= Float32(52.9829189)
    spatial -= floor(spatial)

    var temporal_step = Float32(0.6180339887498949)
    if dim == 1:
        temporal_step = Float32(0.7548776662466927)
    elif dim == 2:
        temporal_step = Float32(0.5698402909980532)
    elif dim == 3:
        temporal_step = Float32(0.4142135623730950)
    var stage_shift = _u32_to_unit_float(
        _mix32(stage ^ seed ^ UInt32(0x9E3779B9))
    )
    var value = spatial + stage_shift
    value += Float32(sample_index) * temporal_step
    return value - floor(value)


@always_inline
def _radical_inverse(index: UInt64, base: UInt64) -> Float32:
    var value = Float32(0.0)
    var inverse = Float32(1.0) / Float32(base)
    var factor = inverse
    var digits = index
    while digits > 0:
        value += Float32(digits % base) * factor
        digits /= base
        factor *= inverse
    return value


@always_inline
def _halton(index: UInt64, dimension: Int) -> Float32:
    var base = UInt64(2)
    var dim = dimension % 8
    if dim == 1:
        base = 3
    elif dim == 2:
        base = 5
    elif dim == 3:
        base = 7
    elif dim == 4:
        base = 11
    elif dim == 5:
        base = 13
    elif dim == 6:
        base = 17
    elif dim == 7:
        base = 19
    return _radical_inverse(index + UInt64(1), base)


@always_inline
def _r2(index: UInt64, dimension: Int) -> Float32:
    # Irrational additive recurrences for the renderer's short staged vectors.
    var alpha = Float32(0.7548776662466927)
    var dim = dimension % 4
    if dim == 1:
        alpha = Float32(0.5698402909980532)
    elif dim == 2:
        alpha = Float32(0.4301597090019468)
    elif dim == 3:
        alpha = Float32(0.2451223337533073)
    var value = Float32(index + UInt64(1)) * alpha
    return value - floor(value)


struct Rng:
    var _rng: Random[10]
    var _buffer: SIMD[.float32, 4]
    var _consumed: Int
    var _sampler: Sampler
    var _sample_index: UInt64
    var _dimension: Int
    var _scramble_seed: UInt32
    var _global_seed: UInt32
    var _pixel_id: UInt32
    var _image_width: UInt32
    var _stage: UInt32

    def __init__(
        out self,
        seed: UInt64,
        id: UInt64,
        sampler: Sampler = .INDEPENDENT,
        sample_index: UInt64 = 0,
        pixel_id: UInt32 = 0,
        image_width: UInt32 = 1,
        stage: UInt32 = 0,
    ):
        self._rng = Random[10](seed=seed, subsequence=id)
        self._buffer = self._rng.step_uniform()
        self._consumed = 0
        self._sampler = sampler
        self._sample_index = sample_index
        self._dimension = 0
        self._scramble_seed = _mix32(
            UInt32(seed)
            ^ UInt32(seed >> UInt64(32))
            ^ UInt32(id)
            ^ UInt32(id >> UInt64(32))
        )
        self._global_seed = _mix32(UInt32(seed) ^ UInt32(seed >> UInt64(32)))
        self._pixel_id = pixel_id
        self._image_width = image_width
        self._stage = stage

    def f32(
        mut self, lower_bound: Float32 = 0, upper_bound: Float32 = 1
    ) -> Float32:
        if self._consumed >= 4:
            self._buffer = self._rng.step_uniform()
            self._consumed = 0
        var shift = self._buffer[self._consumed]
        self._consumed += 1
        var val = shift
        if self._sampler == .HALTON:
            val = _halton(self._sample_index, self._dimension) + shift
            val -= floor(val)
        elif self._sampler == .R2:
            val = _r2(self._sample_index, self._dimension) + shift
            val -= floor(val)
        elif self._sampler == .OWEN_SOBOL or self._sampler == .SZ:
            var scramble = _mix32(
                self._scramble_seed
                ^ UInt32(self._dimension) * UInt32(0x9E3779B9)
            )
            var bits = _sobol_bits(UInt32(self._sample_index), self._dimension)
            if self._sampler == .SZ:
                bits = _sz_bits(UInt32(self._sample_index), self._dimension)
            val = _u32_to_unit_float(_owen_scramble(bits, scramble))
        elif self._sampler == .STBN:
            val = _stbn(
                self._sample_index,
                self._dimension,
                self._pixel_id,
                self._image_width,
                self._stage,
                self._global_seed,
            )
        self._dimension += 1
        return val * (upper_bound - lower_bound) + lower_bound

    def vec3f32[
        frame: Frame
    ](mut self, lower_bound: Float32 = 0, upper_bound: Float32 = 1) -> Vec3f32[
        frame
    ]:
        var scale = upper_bound - lower_bound
        var r0 = self.f32() * scale + lower_bound
        var r1 = self.f32() * scale + lower_bound
        var r2 = self.f32() * scale + lower_bound
        return Vec3f32[frame](r0, r1, r2)


def random_unit_vector[frame: Frame](mut rng: Rng) -> Vec3f32[frame]:
    var u = rng.f32()
    var v = rng.f32()
    var theta = 2.0 * pi * u
    var z = 1.0 - 2.0 * v
    var r = sqrt(1.0 - z * z)
    return Vec3f32[frame](r * cos(theta), r * sin(theta), z)


def random_on_hemisphere[
    frame: Frame
](mut rng: Rng, normal: Vec3f32[frame]) -> Vec3f32[frame]:
    var on_unit_sphere = random_unit_vector[frame](rng)
    var sign = dot(on_unit_sphere, normal).lt(0.0).select(Float32(-1.0), 1.0)
    return sign * on_unit_sphere


def random_in_unit_disk[frame: Frame](mut rng: Rng) -> Vec3f32[frame]:
    var u = rng.f32()
    var v = rng.f32()
    var theta = 2.0 * pi * u
    var r = sqrt(v)
    return Vec3f32[frame](r * cos(theta), r * sin(theta), 0.0)


def random_in_unit_sphere[frame: Frame](mut rng: Rng) -> Vec3f32[frame]:
    var u = rng.f32()
    var r = cbrt(u)
    return random_unit_vector[frame](rng) * r
