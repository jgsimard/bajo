"""Bounds fixtures and independent topology hashes shared by H-PLOC tests."""

from std.math import abs

from bajo.bvh.constants import LBVH_SENTINEL
from bajo.bvh.cpu.builder.hploc import HplocTopology
from bajo.core import AABB, Point3f32


def _box(center_x: Float32) -> AABB[.WORLD]:
    return AABB[.WORLD](
        Point3f32[.WORLD](center_x - 0.1, -0.1, -0.1),
        Point3f32[.WORLD](center_x + 0.1, 0.1, 0.1),
    )


def _identity_ids(count: Int) -> List[UInt32]:
    return [UInt32(i) for i in range(count)]


def _flatten_bounds(
    bounds: List[AABB[.WORLD]],
) -> List[Float32]:
    var flat = List[Float32](capacity=len(bounds) * AABB.STRIDE)
    for bound in bounds:
        flat.append(bound._min.x)
        flat.append(bound._min.y)
        flat.append(bound._min.z)
        flat.append(bound._max.x)
        flat.append(bound._max.y)
        flat.append(bound._max.z)
    return flat^


def _bounds_match(
    actual: AABB[.WORLD],
    expected: AABB[.WORLD],
    tolerance: Float64 = 1.0e-4,
) -> Bool:
    return (
        abs(Float64(actual._min.x - expected._min.x)) <= tolerance
        and abs(Float64(actual._min.y - expected._min.y)) <= tolerance
        and abs(Float64(actual._min.z - expected._min.z)) <= tolerance
        and abs(Float64(actual._max.x - expected._max.x)) <= tolerance
        and abs(Float64(actual._max.y - expected._max.y)) <= tolerance
        and abs(Float64(actual._max.z - expected._max.z)) <= tolerance
    )


def _rotate_left(value: UInt64, shift: Int) -> UInt64:
    return (value << UInt64(shift)) | (value >> UInt64(64 - shift))


def _leaf_hash(leaf_id: UInt32) -> UInt64:
    return (
        UInt64(leaf_id + 1)
        ^ (UInt64(leaf_id) << 32)
        ^ UInt64(0x9E3779B97F4A7C15)
    )


def _inner_hash(left: UInt64, right: UInt64) -> UInt64:
    return (
        _rotate_left(left, 13)
        ^ _rotate_left(right, 37)
        ^ UInt64(0xD6E8FEB86659FD93)
    )


def _reference_root_hash(reference: HplocTopology[.WORLD]) -> UInt64:
    var hashes = List[UInt64](length=len(reference.nodes), fill=UInt64(0))
    for i, node in enumerate(reference.nodes):
        if node.left == LBVH_SENTINEL:
            hashes[i] = _leaf_hash(node.leaf_id)
        else:
            hashes[i] = _inner_hash(
                hashes[Int(node.left)], hashes[Int(node.right)]
            )
    return hashes[Int(reference.root)]


def _make_triangles(count: Int) -> List[AABB[.WORLD]]:
    var bounds = List[AABB[.WORLD]](capacity=count)
    for i in range(count):
        var x = Float32((i % 19) * 4 - 36)
        var y = Float32(((i / 19) % 14) * 4 - 26)
        var z = Float32((i * 7) % 11)
        var scale = Float32(2.2) if i % 17 == 0 else Float32(0.7)
        bounds.append(
            AABB(
                Point3f32[.WORLD](x - scale, y - scale, z),
                Point3f32[.WORLD](x + scale, y - scale, z),
                Point3f32[.WORLD](x, y + scale, z + 0.1),
            )
        )
    return bounds^
