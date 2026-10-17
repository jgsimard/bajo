"""Differential decode corpus: zlib strategies, hostile headers, and mutations."""

from pathlib import Path
import random
import shutil
import struct
import subprocess
import tempfile
import zlib


ROOT = Path(__file__).resolve().parents[2]


class Bits:
    def __init__(self):
        self.bits = []

    def put(self, value, count):
        self.bits.extend((value >> bit) & 1 for bit in range(count))

    def finish(self):
        result = bytearray((len(self.bits) + 7) // 8)
        for index, bit in enumerate(self.bits):
            result[index // 8] |= bit << (index % 8)
        return bytes(result)


def canonical(lengths):
    counts = [lengths.count(length) for length in range(16)]
    counts[0] = 0
    codes = [0] * 16
    for length in range(1, 16):
        codes[length] = (codes[length - 1] + counts[length - 1]) << 1
    result = []
    for length in lengths:
        if length:
            result.append((int(f"{codes[length]:0{length}b}"[::-1], 2), length))
            codes[length] += 1
        else:
            result.append((0, 0))
    return result


def wrap(raw, expected=b""):
    return b"\x78\x9c" + raw + struct.pack(">I", zlib.adler32(expected))


def adversarial_streams():
    """Hand-encode table extremes that ordinary compressors rarely produce."""
    order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
    streams = []
    # Complete comb-shaped trees put both EOB/length and distance in subtables
    # and exercise the maximum legal 15-bit codeword length.
    lengths = [0] * 258
    for symbol in range(14):
        lengths[symbol] = symbol + 1
    lengths[256] = lengths[257] = 15
    distances = list(range(1, 15)) + [15, 15]
    precode = [4] * 16 + [0] * 3
    pre_codes = canonical(precode)
    bits = Bits()
    bits.put(5, 3)
    bits.put(1, 5)
    bits.put(15, 5)
    bits.put(15, 4)
    for symbol in order:
        bits.put(precode[symbol], 3)
    for length in lengths + distances:
        bits.put(*pre_codes[length])
    lit_codes = canonical(lengths)
    dist_codes = canonical(distances)
    for _ in range(200):
        bits.put(*lit_codes[0])
    # Literal 13 has a 14-bit code and must take the subtable literal path.
    for _ in range(400):
        bits.put(*lit_codes[13])
    bits.put(*lit_codes[257])
    bits.put(*dist_codes[15])
    bits.put(0, 6)  # distance 193, three-byte match
    bits.put(*lit_codes[256])
    expected = bytes(200) + bytes([13]) * 403
    streams.append((wrap(bits.finish(), expected), expected))
    # Short literal codes let the bulk table emit pairs. Include high-bit
    # bytes in either position and enough data to enter and leave the bulk loop.
    lengths = [0] * 257
    for symbol in (0, 128, 255, 256):
        lengths[symbol] = 2
    codes = canonical(lengths)
    bits = Bits()
    bits.put(5, 3)
    bits.put(0, 5)
    bits.put(0, 5)
    bits.put(15, 4)
    for symbol in order:
        bits.put(precode[symbol], 3)
    for length in lengths + [1]:
        bits.put(*pre_codes[length])
    expected = bytes([0, 128, 255, 128, 0, 255, 255, 0]) * 257
    for byte in expected:
        bits.put(*codes[byte])
    bits.put(*codes[256])
    streams.append((wrap(bits.finish(), expected), expected))
    # A maximum-window match across 32 KiB of history.
    fixed_lengths = [8] * 144 + [9] * 112 + [7] * 24 + [8] * 8
    fixed = canonical(fixed_lengths)
    fixed_dist = canonical([5] * 32)
    bits = Bits()
    bits.put(3, 3)
    for _ in range(32768):
        bits.put(*fixed[0])
    bits.put(*fixed[257])
    bits.put(*fixed_dist[29])
    bits.put(8191, 13)
    bits.put(*fixed[256])
    expected = bytes(32771)
    streams.append((wrap(bits.finish(), expected), expected))
    # Reserved length and distance symbols, and a match before any history.
    for length, distance in ((286, 0), (257, 30), (257, 0)):
        bits = Bits()
        bits.put(3, 3)
        bits.put(*fixed[length])
        if length == 257:
            bits.put(*fixed_dist[distance])
        streams.append((wrap(bits.finish()), b""))
    # Invalid precode trees: empty, oversubscribed, incomplete, illegal repeat.
    for pre_lengths, symbol in (
        ([0, 0, 0, 0], None),
        ([1, 1, 1, 1], None),
        ([0, 0, 0, 1], None),
        ([1, 0, 0, 1], 16),
        ([0, 0, 1, 1], 18),
    ):
        bits = Bits()
        bits.put(5, 3)
        bits.put(0, 5)
        bits.put(0, 5)
        bits.put(0, 4)
        for length in pre_lengths:
            bits.put(length, 3)
        if symbol is not None:
            full_pre = [0] * 19
            for sym, length in zip(order, pre_lengths):
                full_pre[sym] = length
            codes = canonical(full_pre)
            bits.put(*codes[symbol])
            if symbol == 16:
                bits.put(0, 2)
            else:
                bits.put(127, 7)
                bits.put(*codes[symbol])
                bits.put(127, 7)
        streams.append((wrap(bits.finish()), b""))
    return streams


def main():
    rng = random.Random(81753)
    datasets = [
        b"", b"x", bytes(range(256)), b"\xff" * 2049,
        rng.randbytes(70001), bytes(range(251)) * 300,
        b"aabacabacaabba" * 5000,
        rng.randbytes(32768) * 3,
    ]
    datasets += [rng.randbytes(n) for n in (2, 7, 31, 32, 33, 257, 5552)]
    streams = []
    for source in datasets:
        for level in (0, 1, 6, 9):
            for strategy in (
                zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED,
                zlib.Z_HUFFMAN_ONLY, zlib.Z_RLE, zlib.Z_FIXED,
            ):
                compressor = zlib.compressobj(level, zlib.DEFLATED, 15, 8, strategy)
                streams.append((compressor.compress(source) + compressor.flush(), source))
    # Cross-block dictionaries and transitions between block types.
    compressor = zlib.compressobj(6)
    chunks = [rng.randbytes(19000), b"abc" * 5000, b"\0" * 30000, rng.randbytes(2000)]
    encoded = b"".join(compressor.compress(c) + compressor.flush(zlib.Z_SYNC_FLUSH) for c in chunks)
    streams.append((encoded + compressor.flush(), b"".join(chunks)))
    adversarial = adversarial_streams()
    for encoded, expected in adversarial[:3]:
        assert zlib.decompress(encoded) == expected
    streams.extend(adversarial)
    cases = [(encoded, len(source) + 17) for encoded, source in streams]
    for encoded, source in streams[::7]:
        if source:
            cases.append((encoded, len(source) - 1))
        for _ in range(6):
            index = rng.randrange(len(encoded))
            changed = bytearray(encoded)
            changed[index] ^= 1 << rng.randrange(8)
            cases.append((bytes(changed), len(source) + 17))
        if len(encoded) < 128:
            cases.extend((encoded[:i], len(source) + 17) for i in range(len(encoded)))
    # Header flag and block-type combinations, including FDICT, invalid CM/CINFO,
    # stored LEN/NLEN disagreement, and truncated Huffman descriptions.
    for cmf in (0, 8, 0x78, 0x88, 0xF8):
        for flg in range(256):
            cases.append((bytes([cmf, flg, 7, 0, 0, 0, 0, 0]), 256))
    for _ in range(1000):
        cases.append((b"\x78\x9c" + rng.randbytes(rng.randrange(1, 70)), 1024))
    with tempfile.TemporaryDirectory(prefix="bajo-deflate-") as temp:
        temp = Path(temp)
        corpus = temp / "corpus.bin"
        with corpus.open("wb") as file:
            for encoded, capacity in cases:
                failed = False
                expected = b""
                try:
                    decoder = zlib.decompressobj()
                    expected = decoder.decompress(encoded, capacity + 1)
                    failed = not decoder.eof or len(expected) > capacity
                except zlib.error:
                    failed = True
                if failed:
                    expected = b""
                file.write(struct.pack("<IIII", failed, capacity, len(expected), len(encoded)))
                file.write(expected)
                file.write(encoded)
        executable = temp / "interop"
        subprocess.run([
            shutil.which("mojo") or str(ROOT / ".pixi/envs/default/bin/mojo"),
            "build", "-O3", "-I", str(ROOT),
            str(ROOT / "test/parser/deflate_interop.mojo"), "-o", str(executable),
        ], check=True, cwd=ROOT)
        subprocess.run([str(executable), str(corpus)], check=True, timeout=60)


if __name__ == "__main__":
    main()
