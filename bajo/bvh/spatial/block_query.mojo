"""Collective bounds candidates with explicit per-lane and per-batch validity.

All lanes of a one-dimensional block call next together with identical query
inputs/cutoffs. Each lane tests one preorder node; shared-memory compaction emits
up to block_size original IDs. Missed internal nodes advance the next batch
beyond their complete subtree, retaining every untested reachable node. No global output capacity limit or traversal stack is needed.
"""

from max.gpu import thread_idx, block_dim
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.primitives import block
from std.memory import stack_allocation
from std.math import min, max
from std.utils.numerics import isfinite
from bajo.core import Frame
from bajo.bvh.constants import EMPTY_LANE, f32_max
from .query import BoundsQuery


struct BlockBoundsQuery[origin: ImmOrigin, frame: Frame, block_size: Int](
    Copyable
):
    var query: BoundsQuery[Self.origin, Self.frame]
    var batch_valid: Bool

    def __init__(out self, var query: BoundsQuery[Self.origin, Self.frame]):
        comptime assert Self.block_size > 0
        self.query = query^
        self.batch_valid = True

    def next(mut self, mut index: UInt32, max_dist: Float32 = f32_max) -> Bool:
        """Return this lane's validity; batch_valid is uniform across the block.

        A partial final batch has batch_valid=True even in unused lanes. After
        exhaustion batch_valid=False in all lanes. Invalid/exhausted lanes keep
        index unchanged. Call next in every lane until batch_valid is false;
        never branch around this collective on per-lane validity.
        """
        if (
            block_dim.x != Self.block_size
            or block_dim.y != 1
            or block_dim.z != 1
        ):
            self.batch_valid = False
            return False
        var scratch = stack_allocation[
            Self.block_size + 1, UInt32, address_space=AddressSpace.SHARED
        ]()
        var lane = Int(thread_idx.x)
        if self.query.kind == 1 or self.query.kind == 3:
            if not isfinite(max_dist):
                self.batch_valid = False
                return False
            self.query.cutoff = min(self.query.cutoff, max_dist)
        while self.query.valid and self.query.cursor < self.query.end:
            barrier()  # Previous batch must be consumed before overwriting it.
            var node = Int(self.query.cursor) + lane
            var item = EMPTY_LANE
            var resume = UInt32(
                min(
                    Int(self.query.end),
                    Int(self.query.cursor) + Self.block_size,
                )
            )
            if node < Int(self.query.end):
                var local = self.query.copy()
                local.cursor = UInt32(node)
                local.end = UInt32(node + 1)
                _ = local.next(item, max_dist)
                # A rejected node's escape bounds only rejected descendants.
                # Its range starts inside this batch, so jumping to the largest
                # such escape cannot skip an untested reachable node.
                resume = max(resume, local.cursor)
            comptime if Self.block_size >= 32 and (
                Self.block_size & (Self.block_size - 1)
            ) == 0:
                resume = block.max[block_size=Self.block_size, broadcast=True](
                    resume
                )
                var valid = UInt32(item != EMPTY_LANE)
                var rank = block.prefix_sum[
                    block_size=Self.block_size, exclusive=True
                ](valid)
                var count = block.sum[
                    block_size=Self.block_size, broadcast=True
                ](valid)
                if valid != 0:
                    scratch[unsafe_offset=Int(rank) + 1] = item
                if lane == 0:
                    scratch[unsafe_offset=0] = count
                barrier()
            else:
                var resumes = stack_allocation[
                    Self.block_size, UInt32, address_space=AddressSpace.SHARED
                ]()
                resumes[unsafe_offset=lane] = resume
                scratch[unsafe_offset=lane + 1] = item
                barrier()
                if lane == 0:
                    var count = 0
                    for i in range(Self.block_size):
                        resume = max(resume, resumes[unsafe_offset=i])
                        var candidate = scratch[unsafe_offset=i + 1]
                        if candidate != EMPTY_LANE:
                            scratch[unsafe_offset=count + 1] = candidate
                            count += 1
                    scratch[unsafe_offset=0] = UInt32(count)
                    resumes[unsafe_offset=0] = resume
                barrier()
                resume = resumes[unsafe_offset=0]
            self.query.cursor = min(resume, self.query.end)
            var count = Int(scratch[unsafe_offset=0])
            if count > 0:
                self.batch_valid = True
                if lane < count:
                    index = scratch[unsafe_offset=lane + 1]
                    return True
                return False
        self.batch_valid = False
        return False
