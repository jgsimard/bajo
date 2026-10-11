"""Compare scalar and TileTensor film resolve on identical sample buffers."""
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from std.testing import assert_equal
from bajo.rt.gpu.common_kernels import gpu_rt_resolve_kernel
from bajo.rt.wavefront_contract import WaveSampleFloatAbi, wavefront_plane_index


def fill_samples(values: Pointer[Float32, MutAnyOrigin], count: Int32):
    var idx = global_idx.x
    if idx < Int(count):
        values[unsafe_offset=idx] = Float32(idx % 997) * Float32(0.001)


def reference_resolve(
    sample_radiance: Pointer[Float32, ImmutAnyOrigin],
    pixels: Pointer[Float32, MutAnyOrigin],
    sample_capacity_i32: Int32,
    pixel_base_i32: Int32,
    pixel_count_i32: Int32,
    samples_per_pixel_i32: Int32,
):
    var pixel_idx = global_idx.x
    var sample_capacity = Int(sample_capacity_i32)
    var pixel_base = Int(pixel_base_i32)
    var pixel_count = Int(pixel_count_i32)
    if pixel_idx >= pixel_count:
        return
    var samples_per_pixel = Int(samples_per_pixel_i32)
    var red = Float32(0.0)
    var green = Float32(0.0)
    var blue = Float32(0.0)
    var sample_begin = pixel_idx * samples_per_pixel
    for sample_idx in range(sample_begin, sample_begin + samples_per_pixel):
        red += sample_radiance[
            unsafe_offset=wavefront_plane_index(
                WaveSampleFloatAbi.R, sample_capacity, sample_idx
            )
        ]
        green += sample_radiance[
            unsafe_offset=wavefront_plane_index(
                WaveSampleFloatAbi.G, sample_capacity, sample_idx
            )
        ]
        blue += sample_radiance[
            unsafe_offset=wavefront_plane_index(
                WaveSampleFloatAbi.B, sample_capacity, sample_idx
            )
        ]
    var scale = Float32(1.0) / Float32(samples_per_pixel)
    var output_idx = pixel_base + pixel_idx
    pixels[unsafe_offset=3 * output_idx + 0] = red * scale
    pixels[unsafe_offset=3 * output_idx + 1] = green * scale
    pixels[unsafe_offset=3 * output_idx + 2] = blue * scale


def main() raises:
    with DeviceContext() as ctx:
        for spp in range(3):
            var samples = 16 if spp == 0 else 64 if spp == 1 else 512
            var capacity = 1 << 20
            var pixels = capacity // samples
            var data = ctx.enqueue_create_buffer[.float32](3 * capacity)
            var before = ctx.enqueue_create_buffer[.float32](3 * pixels)
            var after = ctx.enqueue_create_buffer[.float32](3 * pixels)
            ctx.enqueue_function[fill_samples](
                data,
                Int32(3 * capacity),
                grid_dim=3 * capacity // 128,
                block_dim=128,
            )
            comptime for use_tile in [False, True]:
                comptime kernel = gpu_rt_resolve_kernel if use_tile else reference_resolve
                var output = after if use_tile else before
                ctx.enqueue_function[kernel](
                    data,
                    output,
                    Int32(capacity),
                    Int32(0),
                    Int32(pixels),
                    Int32(samples),
                    grid_dim=(pixels + 63) // 64,
                    block_dim=64,
                )
                ctx.synchronize()
                for trial in range(5):
                    var start = perf_counter_ns()
                    for _ in range(1000):
                        ctx.enqueue_function[kernel](
                            data,
                            output,
                            Int32(capacity),
                            Int32(0),
                            Int32(pixels),
                            Int32(samples),
                            grid_dim=(pixels + 63) // 64,
                            block_dim=64,
                        )
                    ctx.synchronize()
                    print(
                        "RESOLVE",
                        samples,
                        "TILE",
                        use_tile,
                        "TRIAL",
                        trial,
                        "US",
                        Float64(perf_counter_ns() - start) / 1e6,
                    )
            with before.map_to_host() as a, after.map_to_host() as b:
                for i in range(3 * pixels):
                    assert_equal(a[i], b[i])
            print("EXACT", samples)
