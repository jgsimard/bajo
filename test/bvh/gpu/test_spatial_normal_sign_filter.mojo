"""Certifications must equal the original normalized Float64 dot sign."""
from std.memory import bitcast
from std.math import abs
from std.testing import TestSuite, assert_equal, assert_true
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import Point3f32, Vec3, dot
from bajo.core.intersect import triangle_normal, vector64
from bajo.bvh.gpu.utils import upload_list, _device_span
from bajo.bvh.spatial.gpu_geometry import normal_sign_filter

comptime CASES = 65536


@always_inline
def outcome(data: ImmSpan[Float32, _], i: Int) -> UInt32:
    var a = Point3f32[.WORLD](data[i * 15], data[i * 15 + 1], data[i * 15 + 2])
    var b = Point3f32[.WORLD](
        data[i * 15 + 3], data[i * 15 + 4], data[i * 15 + 5]
    )
    var c = Point3f32[.WORLD](
        data[i * 15 + 6], data[i * 15 + 7], data[i * 15 + 8]
    )
    var p = Point3f32[.WORLD](
        data[i * 15 + 9], data[i * 15 + 10], data[i * 15 + 11]
    )
    var q = Point3f32[.WORLD](
        data[i * 15 + 12], data[i * 15 + 13], data[i * 15 + 14]
    )
    var filtered = normal_sign_filter(p, q, a, b, c)
    if filtered == 0:
        return 0
    var normal = triangle_normal(a, b, c)
    var expected = -1 if dot(normal, vector64(p) - vector64(q)) < 0.0 else 1
    return UInt32(1 if filtered == expected else 2)


def check_kernel[
    count: Int
](data: ImmSpan[Float32, ImmutAnyOrigin], flags: MutSpan[UInt32, MutAnyOrigin]):
    var i = Int(global_idx.x)
    if i < len(flags):
        flags[i] = outcome(data, i)


def test_normal_sign_certifications_match_float64_across_extremes() raises:
    var data = List[Float32]()
    var state = UInt32(0xA73B2195)
    for i in range(CASES):
        var coords = Array[Float32, 15](fill=0.0)
        comptime for j in range(15):
            state = state * UInt32(1664525) + UInt32(1013904223)
            var word = (state & UInt32(0x807FFFFF)) | (
                ((state >> 23) % 255) << 23
            )
            coords[j] = bitcast[.float32](word)
            if i % 8 == 1 or i % 8 == 2:
                coords[j] = Float32(Int(state % 200001) - 100000) / 1000.0
            elif i % 8 == 3:
                coords[j] = bitcast[.float32](
                    (state & UInt32(0x807FFFFF)) | (((state >> 23) % 159) << 23)
                )
        if i % 8 == 2:
            # Near a coordinate plane with offsets from subnormal to ordinary.
            coords[2] = 0
            coords[5] = 0
            coords[8] = 0
            coords[14] = 0
            var exponent = (i // 8) % 160
            coords[11] = bitcast[.float32](
                UInt32(exponent << 23) | UInt32(1) | ((state & UInt32(1)) << 31)
            )
        elif i % 8 == 4:
            # Shared origin / cancellation at large translated coordinates.
            var origin = Float32(1000000000)
            coords[0] = origin
            coords[1] = origin
            coords[2] = origin
            coords[3] = origin + 128
            coords[4] = origin + 64
            coords[5] = origin
            coords[6] = origin + 64
            coords[7] = origin
            coords[8] = origin + 128
            coords[12] = origin + 64
            coords[13] = origin + 64
            coords[14] = origin + 64
            coords[9] = coords[12] + Float32(Int(i % 3) - 1) * 64
            coords[10] = coords[13] + Float32(Int((i // 3) % 3) - 1) * 64
            coords[11] = coords[14] + Float32(Int((i // 9) % 3) - 1) * 64
        elif i % 8 == 5:
            # Parallel edges / cancellation: determinant is exactly zero.
            coords[0] = 0
            coords[1] = 0
            coords[2] = 0
            coords[3] = 1
            coords[4] = 1
            coords[5] = 1
            coords[6] = 2
            coords[7] = 2
            coords[8] = 2
        elif i % 8 == 6:
            # Mixed subnormal/large inputs exercise the FTZ allowance.
            comptime for j in range(15):
                coords[j] = bitcast[.float32](
                    UInt32(j + 1) | ((UInt32(i) & 1) << 31)
                )
            coords[4] = 1000000000
            coords[8] = -1000000000
            coords[9] = 1000000000
        elif i % 8 == 7:
            # Offsets on both sides of a sloped plane, near cancellation.
            coords[0] = 0
            coords[1] = 0
            coords[2] = 0
            coords[3] = 1
            coords[4] = 0
            coords[5] = 1
            coords[6] = 0
            coords[7] = 1
            coords[8] = 1
            coords[12] = 0.25
            coords[13] = 0.25
            coords[14] = 0.5
            coords[9] = 0.5
            coords[10] = 0.5
            var offset = bitcast[.float32](
                UInt32(((i // 8) % 127) << 23) | UInt32(1)
            )
            coords[11] = 1.0 + offset * Float32(1 if i % 16 == 7 else -1)
        if i % 31 == 0:
            comptime for axis in range(3):
                coords[3 + axis] = coords[axis]
        if i % 37 == 0:
            comptime for axis in range(3):
                coords[6 + axis] = coords[axis]
        if i % 43 == 0:
            comptime for axis in range(3):
                coords[9 + axis] = coords[12 + axis]
        for value in coords:
            data.append(value)
    var certified = 0
    for i in range(CASES):
        var result = outcome(data, i)
        if result == 2:
            print(t"CPU_MISMATCH\t{i}")
        assert_true(result != 2)
        certified += Int(result == 1)
    assert_true(certified > 1000)
    print(t"CPU_SIGN_FILTER_PASS\t{CASES}\t{certified}")
    with DeviceContext() as ctx:
        var input = upload_list(ctx, data)
        var output = ctx.enqueue_create_buffer[.uint32](CASES)
        ctx.enqueue_function[check_kernel[CASES]](
            _device_span[mut=False](input),
            _device_span[mut=True](output),
            grid_dim=256,
            block_dim=256,
        )
        ctx.synchronize()
        certified = 0
        with output.map_to_host() as flags:
            for i in range(CASES):
                if flags[i] == 2:
                    print(t"GPU_MISMATCH\t{i}")
                assert_true(flags[i] != 2)
                certified += Int(flags[i] == 1)
        assert_true(certified > 1000)
        print(t"GPU_SIGN_FILTER_PASS\t{CASES}\t{certified}")


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
