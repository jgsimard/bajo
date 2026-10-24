struct SegmentOffsets(Copyable):
    var offsets: List[UInt32]

    def __init__(out self, offsets: List[UInt32]):
        self = Self(owned_offsets=offsets.copy())

    def __init__(out self, *, var owned_offsets: List[UInt32]):
        """Consume offsets when the caller no longer needs their storage."""
        debug_assert["safe", _use_compiler_assume=True](
            len(owned_offsets) > 0, "segment offsets require a leading zero"
        )
        debug_assert["safe", _use_compiler_assume=True](
            owned_offsets[0] == 0, "segment offsets must start at zero"
        )
        for i in range(1, len(owned_offsets)):
            debug_assert["safe", _use_compiler_assume=True](
                owned_offsets[i - 1] <= owned_offsets[i],
                "segment offsets must be nondecreasing",
            )
        self.offsets = owned_offsets^

    @staticmethod
    def from_counts(counts: List[Int]) -> Self:
        var offsets = List[UInt32](capacity=len(counts) + 1)
        offsets.append(0)
        var total = UInt32(0)
        for count in counts:
            debug_assert["safe", _use_compiler_assume=True](
                count >= 0, "segment counts must be nonnegative"
            )
            debug_assert["safe", _use_compiler_assume=True](
                UInt64(count) <= UInt64(UInt32.MAX - total),
                "segment item count exceeds UInt32 storage",
            )
            total += UInt32(count)
            offsets.append(total)
        return Self(owned_offsets=offsets^)

    @staticmethod
    def single(item_count: Int) -> Self:
        return Self.from_counts([item_count])

    def segment_count(self) -> Int:
        return len(self.offsets) - 1

    def validate(self) raises:
        """Validate mutable offsets when they enter a checked host API."""
        if len(self.offsets) == 0:
            raise Error("segment offsets require a leading zero")
        if self.offsets[0] != 0:
            raise Error("segment offsets must start at zero")
        for i in range(1, len(self.offsets)):
            if self.offsets[i - 1] > self.offsets[i]:
                raise Error("segment offsets must be nondecreasing")

    def item_count(self) -> Int:
        return Int(self.offsets[len(self.offsets) - 1])

    def begin(self, segment_idx: Int) -> UInt32:
        debug_assert["safe", _use_compiler_assume=True](
            segment_idx >= 0 and segment_idx < self.segment_count(),
            "segment index is out of range",
        )
        return self.offsets[segment_idx]

    def end(self, segment_idx: Int) -> UInt32:
        debug_assert["safe", _use_compiler_assume=True](
            segment_idx >= 0 and segment_idx < self.segment_count(),
            "segment index is out of range",
        )
        return self.offsets[segment_idx + 1]

    def count(self, segment_idx: Int) -> UInt32:
        return self.end(segment_idx) - self.begin(segment_idx)
