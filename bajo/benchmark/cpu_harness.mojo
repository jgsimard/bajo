"""Shared result helpers for CPU RT benchmarks."""

from bajo.core import Vec3W


def pixel_checksum(pixels: List[Vec3W]) -> Float64:
    var checksum = 0.0
    for i, p in enumerate(pixels):
        var weight = Float64((i % 251) + 1)
        checksum += weight * (
            Float64(p.x) + 2.0 * Float64(p.y) + 3.0 * Float64(p.z)
        )
    return checksum
