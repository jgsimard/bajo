"""Cooperative per-query traversal for small GPU mesh-query batches.

Every lane calls the operation with identical query/root inputs in a 32-thread
block. Independent subtrees preserve complete coverage and original face IDs.
The result is broadcast to every lane; callers publish outputs in one lane.
"""

from max.gpu import thread_idx, block_dim
from max.gpu.memory import AddressSpace
from max.gpu.primitives import block, warp
from max.gpu.sync import barrier
from std.memory import stack_allocation
from std.math import min, max
from std.memory import bitcast
from bajo.core import Frame, Point3f32, Rayf32, Vec3f32, dot
from bajo.core.intersect import project_triangle, vector64
from bajo.bvh.constants import EMPTY_LANE, f32_max
from .mesh_query import (
    MeshQueryView,
    MeshPointResult,
    MeshRayResult,
    _load_mesh_triangle,
)


@always_inline
def _shuffle64(value: Float64, lane: UInt32) -> Float64:
    var bits = bitcast[.uint64](value)
    var low = warp.shuffle_idx(UInt32(bits & UInt64(0xFFFFFFFF)), lane)
    var high = warp.shuffle_idx(UInt32(bits >> 32), lane)
    return bitcast[.float64](UInt64(low) | (UInt64(high) << 32))


@always_inline
def _reduce64[furthest: Bool](value: Float64) -> Float64:
    var result = value
    comptime for shift in range(5):
        var other = _shuffle64(
            result, UInt32(thread_idx.x) ^ UInt32(1 << shift)
        )
        comptime if furthest:
            result = max(result, other)
        else:
            result = min(result, other)
    return result


@fieldwise_init
struct BlockMeshQueryView[
    origin: ImmOrigin, frame: Frame, storage_width: SIMDLength = 8
](Copyable):
    var view: MeshQueryView[Self.origin, Self.frame, Self.storage_width]

    @always_inline
    def _root(self, root: UInt32) -> UInt32:
        if (
            block_dim.x != 32
            or block_dim.y != 1
            or block_dim.z != 1
            or Int(root) >= len(self.view.bvh.nodes) // 2
        ):
            return EMPTY_LANE
        var lane = Int(thread_idx.x)
        var node = root
        comptime for depth in range(5):
            if (
                self.view.bvh.nodes.unsafe_ptr().unsafe_load(Int(node) * 2 + 1)
                != EMPTY_LANE
            ):
                # A short subtree belongs to exactly one lane, including counts.
                return (
                    node if (lane & ((1 << (5 - depth)) - 1))
                    == 0 else EMPTY_LANE
                )
            var left = node + 1
            node = (
                self.view.bvh.nodes.unsafe_ptr().unsafe_load(Int(left) * 2) if (
                    lane & (1 << (4 - depth))
                )
                != 0 else left
            )
        return node

    @always_inline
    def _point[
        furthest: Bool
    ](
        self, point: Point3f32[Self.frame], cutoff: Float32, root: UInt32
    ) -> MeshPointResult[Self.frame]:
        if block_dim.x != 32 or block_dim.y != 1 or block_dim.z != 1:
            return MeshPointResult[Self.frame].miss()
        var lane = Int(thread_idx.x)
        var local: MeshPointResult[Self.frame]
        comptime if furthest:
            local = self.view.query_furthest_point_no_sign(
                point, cutoff, self._root(root)
            )
        else:
            local = self.view.query_point_no_sign(
                point, cutoff, self._root(root)
            )
        var distance2 = Float64(-1.0 if furthest else 1e300)
        if local.result:
            comptime if furthest:
                var delta = vector64(local.point) - vector64(point)
                distance2 = dot(delta, delta)
            else:
                var triangle = _load_mesh_triangle[Self.frame](
                    self.view.points, self.view.indices, local.face
                )
                distance2 = project_triangle(
                    point, triangle[0], triangle[1], triangle[2]
                ).distance2
        var best2: Float64
        comptime if furthest:
            best2 = _reduce64[True](distance2)
        else:
            best2 = _reduce64[False](distance2)
        var face = warp.min(
            UInt32(
                local.face if local.result
                and distance2 == best2 else EMPTY_LANE
            )
        )
        var winner = warp.min(
            UInt32(lane if local.result and local.face == face else 32)
        )
        if face == EMPTY_LANE:
            return MeshPointResult[Self.frame].miss()
        return MeshPointResult(
            True,
            face,
            warp.shuffle_idx(local.u, winner),
            warp.shuffle_idx(local.v, winner),
            _shuffle64(local.distance, winner),
            Point3f32[Self.frame](
                warp.shuffle_idx(local.point.x, winner),
                warp.shuffle_idx(local.point.y, winner),
                warp.shuffle_idx(local.point.z, winner),
            ),
            0.0,
        )

    def query_point_no_sign(
        self,
        point: Point3f32[Self.frame],
        max_dist: Float32 = f32_max,
        root: UInt32 = 0,
    ) -> MeshPointResult[Self.frame]:
        return self._point[False](point, max_dist, root)

    def query_furthest_point_no_sign(
        self,
        point: Point3f32[Self.frame],
        min_dist: Float32 = 0.0,
        root: UInt32 = 0,
    ) -> MeshPointResult[Self.frame]:
        return self._point[True](point, min_dist, root)

    def query_ray(
        self, ray: Rayf32[Self.frame], root: UInt32 = 0
    ) -> MeshRayResult[Self.frame]:
        if block_dim.x != 32 or block_dim.y != 1 or block_dim.z != 1:
            return MeshRayResult[Self.frame].miss()
        var lane = Int(thread_idx.x)
        var local = self.view.query_ray(ray, self._root(root))
        var best = _reduce64[False](Float64(local.t if local.result else 1e300))
        var face = warp.min(
            UInt32(
                local.face if local.result and local.t == best else EMPTY_LANE
            )
        )
        var winner = warp.min(
            UInt32(lane if local.result and local.face == face else 32)
        )
        if face == EMPTY_LANE:
            return MeshRayResult[Self.frame].miss()
        return MeshRayResult(
            True,
            face,
            warp.shuffle_idx(local.u, winner),
            warp.shuffle_idx(local.v, winner),
            best,
            Vec3f32[Self.frame](
                warp.shuffle_idx(local.normal.x, winner),
                warp.shuffle_idx(local.normal.y, winner),
                warp.shuffle_idx(local.normal.z, winner),
            ),
            warp.shuffle_idx(UInt32(local.front_face), winner) != 0,
        )

    def query_ray_count_intersections(
        self, ray: Rayf32[Self.frame], root: UInt32 = 0
    ) -> UInt32:
        if block_dim.x != 32 or block_dim.y != 1 or block_dim.z != 1:
            return 0
        return block.sum[block_size=32, broadcast=True](
            self.view.query_ray_count_intersections(ray, self._root(root))
        )

    def query_ray_anyhit(
        self, ray: Rayf32[Self.frame], root: UInt32 = 0
    ) -> Bool:
        if block_dim.x != 32 or block_dim.y != 1 or block_dim.z != 1:
            return False
        return (
            block.sum[block_size=32, broadcast=True](
                UInt32(self.view.query_ray_anyhit(ray, self._root(root)))
            )
            != 0
        )
