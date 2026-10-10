"""Inputs and an independent stable reference for single-byte GPU sort passes."""


def _byte_pass_keys[pairs: Bool](size: Int) -> List[UInt32]:
    var keys = List[UInt32](capacity=size)
    for i in range(size):
        var key = (i * 17) ^ (i << 13)
        comptime if not pairs:
            key ^= i >> 5
        keys.append(UInt32(key))
    return keys^


def _stable_byte_order(keys: List[UInt32]) -> List[Int]:
    var offsets = List[Int](length=256, fill=0)
    for key in keys:
        offsets[Int(key & 255)] += 1
    var total = 0
    for digit in range(256):
        var count = offsets[digit]
        offsets[digit] = total
        total += count
    var order = List[Int](length=len(keys), fill=0)
    for i, key in enumerate(keys):
        var digit = Int(key & 255)
        order[offsets[digit]] = i
        offsets[digit] += 1
    return order^
