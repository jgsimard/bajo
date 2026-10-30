"""Alias distributions and multiple-importance sampling weights."""


def build_alias_distribution(
    weights: Span[Float32, _],
    total: Float64,
    mut probabilities: List[Float32],
    mut indices: List[UInt32],
):
    """Shared Walker-Vose construction for finite emitters and image lights.

    Float64 workspace avoids overflow during normalization and cancellation
    while balancing columns. Zero-total distributions stay empty.
    """
    var count = len(weights)
    probabilities = List[Float32]()
    indices = List[UInt32]()
    if count == 0 or total <= 0.0:
        return
    probabilities = List[Float32](length=count, fill=1.0)
    indices = List[UInt32](
        length=count, fill_with=lambda (i: Int) -> UInt32: UInt32(i)
    )
    var scaled = List[Float64](length=count, fill=0.0)
    var small = List[Int](capacity=count)
    var large = List[Int](capacity=count)
    for i in range(count):
        scaled[i] = Float64(weights[i]) / total * Float64(count)
        if scaled[i] < 1.0:
            small.append(i)
        else:
            large.append(i)
    while len(small) > 0 and len(large) > 0:
        var small_idx = small.pop()
        var large_idx = large.pop()
        probabilities[small_idx] = Float32(scaled[small_idx]).clamp(0.0, 1.0)
        indices[small_idx] = UInt32(large_idx)
        scaled[large_idx] = scaled[large_idx] + scaled[small_idx] - 1.0
        if scaled[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)
    # Unpaired columns already have probability one and alias themselves.


@fieldwise_init
struct AliasDraw[length: SIMDLength = 1](TrivialRegisterPassable):
    var column: SIMD[.int64, Self.length]
    var fraction: SIMD[.float32, Self.length]


@always_inline
def draw_alias_column[
    length: SIMDLength = 1
](random_u: SIMD[.float32, length], count: Int) -> AliasDraw[length]:
    var scaled = random_u * Float32(count)
    var column = scaled.cast[.int64]().clamp(0, SIMD[.int64, length](count - 1))
    return AliasDraw[length](column, scaled - column.cast[.float32]())


@always_inline
def resolve_alias_draw[
    length: SIMDLength = 1
](
    draw: AliasDraw[length],
    probability: SIMD[.float32, length],
    alias_index: SIMD[.uint32, length],
) -> SIMD[.int64, length]:
    # Strict comparison prevents a zero-weight column from winning at u=0.
    return draw.fraction.lt(probability).select(
        draw.column, alias_index.cast[.int64]()
    )


@always_inline
def power_heuristic[
    length: SIMDLength = 1
](
    pdf_a: SIMD[.float32, length],
    pdf_b: SIMD[.float32, length],
) -> SIMD[
    .float32, length
]:
    var a2 = pdf_a * pdf_a
    var b2 = pdf_b * pdf_b
    var denominator = a2 + b2
    var nonzero = denominator.gt(0.0)
    return nonzero.select(a2 / nonzero.select(denominator, 1.0), 0.0)
