"""CPU/GPU serial batch fallback for any bounds candidate iterator."""

from bajo.core import Frame
from bajo.bvh.constants import f32_max
from .query import BoundsQuery
from .policy import BATCH_WIDTH


@fieldwise_init
struct BoundsQueryBatch[origin: ImmOrigin, frame: Frame](Copyable):
    var query: BoundsQuery[Self.origin, Self.frame]

    @always_inline
    def next[
        width: SIMDLength = BATCH_WIDTH
    ](mut self, output: MutSpan[UInt32, _], max_dist: Float32 = f32_max) -> Int:
        """Fill up to output capacity; preserve unused slots and return count.

        Zero capacity returns zero without consuming the iterator. Cutoffs have
        the same monotonic/inclusive behavior as BoundsQuery.next.
        """
        return self.query.next_batch[width](output, max_dist)
