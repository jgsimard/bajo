"""Time EXR ZIP stages on the first chunk of a real image.

Build with `mojo build -O3 -I . bench/parser/exr/bench_postprocessing.mojo`
and pass an EXR path. Copy/allocation costs are included where needed to keep
prediction inputs unchanged between benchmark iterations.
"""

from std.benchmark import Unit, keep, run
from std.ffi import OwnedDLHandle
from std.sys.arg import argv

from bajo.parser.mmap import MMap
from bajo.parser.exr import (
    _parse_header,
    _plan_chunks,
    _undo_zip_prediction,
    _undo_zip_interleave,
    _inflate_chunk,
    _decode_scanlines,
)


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: bench_postprocessing <ZIP EXR path>")
    var mapped = MMap(String(args[1]))
    var bytes = mapped.as_bytes_span()
    var header = _parse_header(bytes)
    if header.compression != 2 and header.compression != 3:
        raise Error("benchmark requires ZIP or ZIPS compression")
    var chunks = _plan_chunks(bytes, header)
    ref chunk = chunks[0]
    var payload = bytes[chunk.begin : chunk.end]
    var predicted = List[UInt8](length=chunk.decoded_size, fill=0)
    var result_size = List[UInt](length=1, fill=UInt(chunk.decoded_size))
    var lib = OwnedDLHandle("libz.so.1")
    var status = lib.call["uncompress", Int32](
        predicted.unsafe_ptr(),
        result_size.unsafe_ptr(),
        payload.unsafe_ptr(),
        UInt(len(payload)),
    )
    if status != 0 or result_size[0] != UInt(chunk.decoded_size):
        raise Error("invalid benchmark chunk")
    var seed = predicted.copy()
    _undo_zip_prediction(predicted)
    var pixels = List[Float32](length=3 * header.width() * chunk.lines, fill=0)
    var scanlines = _undo_zip_interleave(predicted)
    var decoded = List[UInt8](length=chunk.decoded_size, fill=0)
    print(
        "chunk_bytes",
        chunk.decoded_size,
        "width",
        header.width(),
        "lines",
        chunk.lines,
    )
    for channel in header.channels:
        print("pixel_type", channel.pixel_type)

    def inflate() raises {imm, mut decoded, mut result_size}:
        result_size[0] = UInt(len(decoded))
        var result = lib.call["uncompress", Int32](
            decoded.unsafe_ptr(),
            result_size.unsafe_ptr(),
            payload.unsafe_ptr(),
            UInt(len(payload)),
        )
        if result != 0:
            raise Error("inflate failed")
        keep(decoded.unsafe_ptr())

    def prediction() {seed}:
        var output = seed.copy()
        _undo_zip_prediction(output)
        keep(output.unsafe_ptr())

    def interleave() {predicted}:
        var output = _undo_zip_interleave(predicted)
        keep(output.unsafe_ptr())

    def decode() raises {imm, mut pixels}:
        _decode_scanlines(Span(pixels), header, scanlines, chunk.lines)
        keep(pixels.unsafe_ptr())

    def combined() raises {imm, mut predicted, mut decoded}:
        _inflate_chunk(payload, Span(predicted), Span(decoded), lib)
        keep(decoded.unsafe_ptr())

    print("zlib_us", run(inflate, max_iters=10).mean(Unit.us))
    print("prediction_copy_us", run(prediction, max_iters=10).mean(Unit.us))
    print("interleave_alloc_us", run(interleave, max_iters=10).mean(Unit.us))
    print("decode_us", run(decode, max_iters=10).mean(Unit.us))
    print("inflate_postprocess_us", run(combined, max_iters=10).mean(Unit.us))
