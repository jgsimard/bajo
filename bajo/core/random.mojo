from math import acos, asin, atan2, clamp, cos, fma, pi, sin, sqrt, tan
from random import random_float64, Random

from bajo.core.vec import Vec3f32, dot


struct PhiloxRNG:
    var _rng: Random[10]
    var _buffer: SIMD[DType.float32, 4]
    var _consumed: Int

    fn __init__(out self, seed: UInt64, id: UInt64):
        self._rng = Random[10](seed=seed, subsequence=id)
        self._buffer = self._rng.step_uniform()
        self._consumed = 0

    fn next_f32(mut self) -> Float32:
        if self._consumed >= 4:
            self._buffer = self._rng.step_uniform()
            self._consumed = 0
        var val = self._buffer[self._consumed]
        self._consumed += 1
        return val


fn random_unit_vector(mut rng: PhiloxRNG) -> Vec3f32:
    var u = rng.next_f32()
    var v = rng.next_f32()
    var theta = 2.0 * pi * u
    var phi = acos(1.0 - 2.0 * v)
    var sin_phi = sin(phi)
    return Vec3f32(sin_phi * cos(theta), sin_phi * sin(theta), cos(phi))


fn random_on_hemisphere(mut rng: PhiloxRNG, normal: Vec3f32) -> Vec3f32:
    var on_unit_sphere = random_unit_vector(rng)
    var sign = Float32(dot(on_unit_sphere, normal) > 0.0)
    return sign * on_unit_sphere


fn random_in_unit_disk(mut rng: PhiloxRNG) -> Vec3f32:
    var u = rng.next_f32()
    var v = rng.next_f32()
    var theta = 2.0 * pi * u
    var r = sqrt(v)
    return Vec3f32(r * cos(theta), r * sin(theta), 0.0)


fn random_in_unit_sphere(mut rng: PhiloxRNG) -> Vec3f32:
    var u = rng.next_f32()
    var r = pow(u, 1.0 / 3.0)
    return random_unit_vector(rng) * r
