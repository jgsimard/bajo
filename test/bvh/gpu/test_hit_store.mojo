"""GPU hit stores preserve all words, alignment offsets and guard regions."""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal
from bajo.bvh.gpu.ray_launch import (
    _store_packed_hit,
    _store_packed_hit_vectorized,
)
from bajo.bvh.gpu.utils import upload_list
from bajo.bvh.types import Hit
from bajo.core import Frame


def _store_bits[
    frame: Frame, vectorized: Bool
](
    words: ImmPointer[UInt32, ImmutAnyOrigin],
    output: MutPointer[UInt32, MutAnyOrigin],
    count: Int32,
    offset: Int32,
):
    var index = global_idx.x
    if index >= Int(count):
        return
    var source = Span(
        unsafe_ptr=words.unsafe_bitcast[Float32](),
        length=Int(count) * Hit.STRIDE,
    )
    var destination = output.unsafe_offset(Int(offset)).unsafe_bitcast[
        Float32
    ]()
    var output_index = Int(count) - 1 - index
    var hit = Hit[frame].load(source, index)
    comptime if vectorized:
        _store_packed_hit_vectorized(hit, destination, Int(count), output_index)
    else:
        _store_packed_hit(hit, destination, Int(count), output_index)


def test_packed_hit_store_bits_offsets_and_frames() raises:
    var patterns = [
        UInt32(0),
        UInt32(0x80000000),
        UInt32(0x7F800000),
        UInt32(0xFF800000),
        UInt32(0x7FC01234),
        UInt32(0x7FA01234),
        UInt32(0xFFC01234),
        UInt32(0xFFA01234),
        UInt32(1),
        UInt32(0x80000001),
        UInt32(0xFFFFFFFF),
    ]
    with DeviceContext() as ctx:
        for count in [1, 257, 8193]:
            var words = List[UInt32](capacity=count * Hit.STRIDE)
            for index in range(count * Hit.STRIDE):
                var value = UInt32(index) * UInt32(1664525) + UInt32(1013904223)
                if index < len(patterns) * Hit.STRIDE:
                    value = patterns[
                        (index + index // Hit.STRIDE) % len(patterns)
                    ]
                words.append(value)
            var source = upload_list(ctx, words)
            for alignment_offset in range(4):
                var offset = 4 + alignment_offset
                var length = offset + count * Hit.STRIDE + 4
                var guard = UInt32(0xDEADBEEF)
                var output = ctx.enqueue_create_buffer[.uint32](length)
                comptime for variant in range(3):
                    comptime frame = Frame.LOCAL if variant == 1 else Frame.WORLD
                    comptime vectorized = variant != 2
                    with output.map_to_host() as values:
                        for i in range(length):
                            values[i] = guard
                    ctx.enqueue_function[_store_bits[frame, vectorized]](
                        source,
                        output,
                        Int32(count),
                        Int32(offset),
                        grid_dim=ceildiv(count, 128),
                        block_dim=128,
                    )
                    ctx.synchronize()
                    with output.map_to_host() as values:
                        for index in range(offset):
                            assert_equal(values[index], guard)
                        for index in range(count):
                            var original = count - 1 - index
                            for field in range(Hit.STRIDE):
                                assert_equal(
                                    values[offset + index * Hit.STRIDE + field],
                                    words[original * Hit.STRIDE + field],
                                )
                        for index in range(offset + count * Hit.STRIDE, length):
                            assert_equal(values[index], guard)


def main() raises:
    comptime if not has_accelerator():
        raise "No accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
