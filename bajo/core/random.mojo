from std.bit import bit_reverse
from std.builtin.globals import global_constant
from std.math import cos, floor, pi, sin, sqrt, cbrt
from std.random import Random

from bajo.core import Vec3f32, dot, Frame


struct Sampler(EnumLike, Equatable, TrivialRegisterPassable, Writable):
    """Runtime-selectable sample sequence used by the ray tracers."""

    var value: UInt32
    comptime INDEPENDENT = Self.__init__[0]()
    comptime HALTON = Self.__init__[1]()
    comptime R2 = Self.__init__[2]()
    comptime OWEN_SOBOL = Self.__init__[3]()
    comptime SZ = Self.__init__[4]()
    comptime STBN = Self.__init__[5]()
    comptime _enum_case_names = ParameterList.of[
        "INDEPENDENT".value,
        "HALTON".value,
        "R2".value,
        "OWEN_SOBOL".value,
        "SZ".value,
        "STBN".value,
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
    ].values

    def __init__[value: Int](out self):
        """Construct a sampler from a compile-time sequence ID."""
        comptime assert 0 <= value <= 5
        self.value = UInt32(value)

    def __init__(out self, value: Int) raises:
        """Construct a sampler from an external sequence ID."""
        if value < 0 or value > 5:
            raise Error("unknown sampler: expected a sequence ID from 0 to 5")
        self.value = UInt32(value)

    def __init__(out self, value: UInt32) raises:
        """Construct a sampler from a 32-bit sequence ID."""
        self = Self(Int(value))

    def __init__(out self, *, unsafe_from_value: UInt32):
        """Reconstruct a sampler from an already-validated sequence ID."""
        self.value = unsafe_from_value

    def _get_enum_discriminant(self) -> Int:
        return Int(self.value)

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        while True:
            pass

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
    var bits = bit_reverse(value)
    bits ^= bits * UInt32(0x3D20ADEA)
    bits += seed
    bits *= (seed >> UInt32(16)) | UInt32(1)
    bits ^= bits * UInt32(0x05526C56)
    bits ^= bits * UInt32(0x53A22864)
    return bit_reverse(bits)


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
        return bit_reverse(index)

    if dim == 1:
        var direction = UInt32(0x80000000)
        comptime for bit in range(32):
            if (index & (UInt32(1) << UInt32(bit))) != 0:
                result ^= direction
            direction ^= direction >> UInt32(1)
        return result

    if dim == 2:
        # Primitive polynomial x^2 + x + 1, initial values (1, 3).
        var previous_2 = UInt32(0x80000000)
        var previous_1 = UInt32(0xC0000000)
        comptime for bit in range(32):
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
    comptime for bit in range(32):
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
    __match dim:
        case 1:
            temporal_step = Float32(0.7548776662466927)
        case 2:
            temporal_step = Float32(0.5698402909980532)
        case 3:
            temporal_step = Float32(0.4142135623730950)
        case _:
            pass
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


# Distinct Halton dimensions for camera, BSDF, lighting, material and roulette
# streams. Reusing base two at every bounce creates correlated Fresnel choices
# that persist as sample count increases (visible in absorbing glass).
# Use an array via global_constant below: runtime indexing of a large SIMD
# value emits a complete per-thread table copy on the GPU for every draw.
# fmt: off
comptime _HALTON_PRIMES: Array[UInt32, 1024] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
    41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
    97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
    227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359,
    367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433,
    439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
    509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593,
    599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
    661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743,
    751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827,
    829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911,
    919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997,
    1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
    1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,
    1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249,
    1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321,
    1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439,
    1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511,
    1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601,
    1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693,
    1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783,
    1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877,
    1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987,
    1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069,
    2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143,
    2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267,
    2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347,
    2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423,
    2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543,
    2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657,
    2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713,
    2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801,
    2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903,
    2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011,
    3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119,
    3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221,
    3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323,
    3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413,
    3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527,
    3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607,
    3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697,
    3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797,
    3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907,
    3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003,
    4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093,
    4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211,
    4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283,
    4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409,
    4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513,
    4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621,
    4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721,
    4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813,
    4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937,
    4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011,
    5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113,
    5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233,
    5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351,
    5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443,
    5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531,
    5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653,
    5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743,
    5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849,
    5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939,
    5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073,
    6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173,
    6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271,
    6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359,
    6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473,
    6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581,
    6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701,
    6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803,
    6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907,
    6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997,
    7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121,
    7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229,
    7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349,
    7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487,
    7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561,
    7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669,
    7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757,
    7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879,
    7883, 7901, 7907, 7919, 7927, 7933, 7937, 7949, 7951, 7963, 7993, 8009,
    8011, 8017, 8039, 8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111,
    8117, 8123, 8147, 8161,
]
# fmt: on


@always_inline
def _halton(index: UInt64, dimension: Int) -> Float32:
    ref primes = global_constant[_HALTON_PRIMES]()
    return _radical_inverse(index + UInt64(1), UInt64(primes[dimension % 1024]))


@always_inline
def _r2(index: UInt64, dimension: Int) -> Float32:
    # Irrational additive recurrences for the renderer's short staged vectors.
    var alpha = Float32(0.7548776662466927)
    var dim = dimension % 4
    __match dim:
        case 1:
            alpha = Float32(0.5698402909980532)
        case 2:
            alpha = Float32(0.4301597090019468)
        case 3:
            alpha = Float32(0.2451223337533073)
        case _:
            pass
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
        __match self._sampler:
            case .INDEPENDENT:
                pass
            case .HALTON:
                # Eight dimensions per bounce, with disjoint stage domains.
                var domain = self._stage >> UInt32(30)
                var offset = Int(0)
                __match domain:
                    case 1:
                        offset = 6  # roulette
                    case 2:
                        offset = 2  # direct-light sampling
                    case 3:
                        offset = 5  # material selection
                    case _:
                        pass
                var dimension = (
                    Int(self._stage & UInt32(0x3FFFFFFF)) * 8
                    + offset
                    + self._dimension
                )
                val = _halton(self._sample_index, dimension) + shift
                val -= floor(val)
            case .R2:
                val = _r2(self._sample_index, self._dimension) + shift
                val -= floor(val)
            case .OWEN_SOBOL | .SZ:
                var scramble = _mix32(
                    self._scramble_seed
                    ^ UInt32(self._dimension) * UInt32(0x9E3779B9)
                )
                var bits = _sobol_bits(
                    UInt32(self._sample_index), self._dimension
                )
                if self._sampler == .SZ:
                    bits = _sz_bits(UInt32(self._sample_index), self._dimension)
                val = _u32_to_unit_float(_owen_scramble(bits, scramble))
            case .STBN:
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
