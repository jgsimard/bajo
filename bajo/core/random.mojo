from std.math import cos, pi, sin, sqrt, cbrt
from std.random import Random

from bajo.core import Vec3f32, dot


struct Rng(Movable):
    var _rng: Random[10]
    var _buffer: SIMD[DType.float32, 4]
    var _consumed: Int

    def __init__(out self, seed: UInt64, id: UInt64):
        self._rng = Random[10](seed=seed, subsequence=id)
        self._buffer = self._rng.step_uniform()
        self._consumed = 0

    def f32(
        mut self, lower_bound: Float32 = 0, upper_bound: Float32 = 1
    ) -> Float32:
        if self._consumed >= 4:
            self._buffer = self._rng.step_uniform()
            self._consumed = 0
        val = self._buffer[self._consumed]
        self._consumed += 1
        return val * (upper_bound - lower_bound) + lower_bound

    def vec3f32(
        mut self, lower_bound: Float32 = 0, upper_bound: Float32 = 1
    ) -> Vec3f32:
        scale = upper_bound - lower_bound
        r0 = self.f32() * scale + lower_bound
        r1 = self.f32() * scale + lower_bound
        r2 = self.f32() * scale + lower_bound
        return Vec3f32(r0, r1, r2)


def random_unit_vector(mut rng: Rng) -> Vec3f32:
    u = rng.f32()
    v = rng.f32()
    theta = 2.0 * pi * u
    z = 1.0 - 2.0 * v
    r = sqrt(1.0 - z * z)
    return Vec3f32(r * cos(theta), r * sin(theta), z)


def random_on_hemisphere(mut rng: Rng, normal: Vec3f32) -> Vec3f32:
    on_unit_sphere = random_unit_vector(rng)
    sign = dot(on_unit_sphere, normal).lt(0.0).select(Float32(-1.0), 1.0)
    return sign * on_unit_sphere


def random_in_unit_disk(mut rng: Rng) -> Vec3f32:
    u = rng.f32()
    v = rng.f32()
    theta = 2.0 * pi * u
    r = sqrt(v)
    return Vec3f32(r * cos(theta), r * sin(theta), 0.0)


def random_in_unit_sphere(mut rng: Rng) -> Vec3f32:
    u = rng.f32()
    r = cbrt(u)
    return random_unit_vector(rng) * r
