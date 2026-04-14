from std.math import acos, cos, pi, sin, sqrt
from std.random import Random

from bajo.core.vec import Vec3f32, dot
from bajo.core.vec_simd import Vec3f32 as Vec3f32_simd, dot as dot_simd


struct Rng:
    var _rng: Random[10]
    var _buffer: SIMD[DType.float32, 4]
    var _consumed: Int

    def __init__(out self, seed: UInt64, id: UInt64):
        self._rng = Random[10](seed=seed, subsequence=id)
        self._buffer = self._rng.step_uniform()
        self._consumed = 0

    def f32(mut self) -> Float32:
        if self._consumed >= 4:
            self._buffer = self._rng.step_uniform()
            self._consumed = 0
        val = self._buffer[self._consumed]
        self._consumed += 1
        return val

    def vec3f32(
        mut self, lower_bound: Float32 = 0, upper_bound: Float32 = 1
    ) -> Vec3f32:
        r0 = self.f32()
        r1 = self.f32()
        r2 = self.f32()
        out = Vec3f32(r0, r1, r2) * (upper_bound - lower_bound) + lower_bound
        return out^


def random_unit_vector(mut rng: Rng) -> Vec3f32:
    u = rng.f32()
    v = rng.f32()
    theta = 2.0 * pi * u
    phi = acos(1.0 - 2.0 * v)
    sin_phi = sin(phi)
    return Vec3f32(sin_phi * cos(theta), sin_phi * sin(theta), cos(phi))


def random_on_hemisphere(mut rng: Rng, normal: Vec3f32) -> Vec3f32:
    on_unit_sphere = random_unit_vector(rng)
    sign = Float32(dot(on_unit_sphere, normal) > 0.0)
    return sign * on_unit_sphere


def random_in_unit_disk(mut rng: Rng) -> Vec3f32:
    u = rng.f32()
    v = rng.f32()
    theta = 2.0 * pi * u
    r = sqrt(v)
    return Vec3f32(r * cos(theta), r * sin(theta), 0.0)


def random_in_unit_sphere(mut rng: Rng) -> Vec3f32:
    u = rng.f32()
    r = pow(u, 1.0 / 3.0)
    return random_unit_vector(rng) * r


# TODO : clean this up
def random_unit_vector_simd(mut rng: Rng) -> Vec3f32_simd:
    u = rng.f32()
    v = rng.f32()
    theta = 2.0 * pi * u
    phi = acos(1.0 - 2.0 * v)
    sin_phi = sin(phi)
    return Vec3f32_simd(sin_phi * cos(theta), sin_phi * sin(theta), cos(phi))


def random_on_hemisphere_simd(
    mut rng: Rng, normal: Vec3f32_simd
) -> Vec3f32_simd:
    on_unit_sphere = random_unit_vector_simd(rng)
    sign = Float32(dot_simd(on_unit_sphere, normal) > 0.0)
    return sign * on_unit_sphere


def random_in_unit_disk_simd(mut rng: Rng) -> Vec3f32_simd:
    u = rng.f32()
    v = rng.f32()
    theta = 2.0 * pi * u
    r = sqrt(v)
    return Vec3f32_simd(r * cos(theta), r * sin(theta), 0.0)


def random_in_unit_sphere_simd(mut rng: Rng) -> Vec3f32_simd:
    u = rng.f32()
    r = pow(u, 1.0 / 3.0)
    return random_unit_vector_simd(rng) * r
