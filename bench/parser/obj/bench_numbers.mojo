"""Isolate decimal decoding from OBJ counting, allocation, and threading."""

from std.benchmark import run, keep, Unit
from bajo.parser.number import F32ParseResult, parse_f32_at
from bajo.parser.obj.number import parse_obj_f32_at


def decode[optimized: Bool](text: String) raises:
    var bytes = StringSpan(text).as_bytes()
    var pos = 0
    var sum: Float64 = 0.0
    while pos < len(bytes):
        var parsed: F32ParseResult
        comptime if optimized:
            parsed = parse_obj_f32_at(bytes, pos)
        else:
            parsed = parse_f32_at(bytes, pos)
        sum += Float64(parsed.value)
        pos = parsed.pos + 1
    keep(sum)


def main() raises:
    for name, record in [
        ("six_fraction_digits", "-0.037830 0.127940 0.004475 "),
        (
            "long_fraction_digits",
            "-52.046509000000001 -13.828309000000001 -7.9414709999999999 ",
        ),
    ]:
        var text = String()
        for _ in range(10000):
            text += record

        def reference() raises {text}:
            decode[False](text)

        def optimized() raises {text}:
            decode[True](text)

        var before = run(reference, max_iters=20).mean(Unit.us)
        var after = run(optimized, max_iters=20).mean(Unit.us)
        print(t"OBJ_NUMBER_METRIC,{name},{before},{after}")
