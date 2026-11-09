"""Z-Sobol device samples match PBRT references and CPU SIMD streams."""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.builtin.globals import global_constant
from std.memory import bitcast
from std.testing import TestSuite, assert_equal
from bajo.core.random import Rng
from bajo.core.zsobol import (
    zsobol_log2_samples,
    zsobol_morton_index,
    zsobol_sample_index,
)
from bajo.rt.common import path_stage_rng
from bajo.rt.render_types import RenderSettings, SamplingConfig
from test.core.zsobol_reference import ZSOBOL_REFERENCE


def _reference_kernel[
    _device: Bool = True
](output: Pointer[UInt32, MutAnyOrigin]):
    var index = Int(global_idx.x)
    ref cases = global_constant[ZSOBOL_REFERENCE]()
    if index >= len(cases):
        return
    var reference = cases[index]
    var pixel = reference.y * reference.width + reference.x
    var log2_samples = zsobol_log2_samples(reference.samples)
    var digits = (
        zsobol_log2_samples(max(reference.width, reference.height))
        + (log2_samples + 1) // 2
    )
    var morton = zsobol_morton_index(
        pixel, reference.width, SIMD[.uint64, 1](reference.sample), log2_samples
    )
    var permuted = zsobol_sample_index(
        morton, log2_samples, digits, reference.dimension
    )
    var rng = Rng[1](
        UInt64(reference.seed),
        0,
        sampler=.Z_SOBOL,
        sample_index=SIMD[.uint64, 1](reference.sample),
        pixel_id=pixel,
        image_width=reference.width,
        image_height=reference.height,
        sequence_length=reference.samples,
    )
    for _ in range(Int(reference.dimension)):
        _ = rng.f32()
    output[unsafe_offset=4 * index] = UInt32(permuted)
    output[unsafe_offset=4 * index + 1] = UInt32(permuted >> UInt64(32))
    output[unsafe_offset=4 * index + 2] = bitcast[.uint32](rng.f32())
    output[unsafe_offset=4 * index + 3] = bitcast[.uint32](rng.f32())


def _stream_kernel[
    width: SIMDLength, _device: Bool = True
](
    config: SamplingConfig,
    stage: UInt32,
    output: Pointer[Float32, MutAnyOrigin],
):
    var first = Int(global_idx.x) * Int(width)
    if first >= 257:
        return
    var ids = SIMD[.uint32, width](0)
    for lane in range(Int(width)):
        ids[lane] = UInt32(first + lane)
    var rng = path_stage_rng(config, ids, stage)
    for dimension in range(12):
        var values = rng.f32()
        for lane in range(Int(width)):
            if first + lane < 257:
                output[unsafe_offset=(first + lane) * 12 + dimension] = values[
                    lane
                ]


def test_gpu_zsobol_matches_independent_pbrt_references() raises:
    ref cases = global_constant[ZSOBOL_REFERENCE]()
    with DeviceContext() as ctx:
        var output = ctx.enqueue_create_buffer[.uint32](len(cases) * 4)
        ctx.enqueue_function[_reference_kernel[True]](
            output, grid_dim=(len(cases) + 63) // 64, block_dim=64
        )
        with output.map_to_host() as values:
            for index in range(len(cases)):
                var reference = cases[index]
                assert_equal(values[4 * index], UInt32(reference.index))
                assert_equal(
                    values[4 * index + 1], UInt32(reference.index >> UInt64(32))
                )
                assert_equal(values[4 * index + 2], reference.u_bits)
                assert_equal(values[4 * index + 3], reference.v_bits)


def test_gpu_zsobol_packet_streams_match_cpu_across_stages() raises:
    var settings = RenderSettings(9, 73, 7, UInt64(2026), 8, .Z_SOBOL, 101, 256)
    var config = SamplingConfig.from_settings(settings)
    with DeviceContext() as ctx:
        var output = ctx.enqueue_create_buffer[.float32](257 * 12)
        for stage in [
            UInt32(0),
            UInt32(9),
            UInt32(0x40000009),
            UInt32(0x80000009),
            UInt32(0xC0000009),
        ]:
            comptime for width in [1, 4, 16]:
                ctx.enqueue_function[_stream_kernel[width, True]](
                    config,
                    stage,
                    output,
                    grid_dim=(257 + Int(width) * 63) // (Int(width) * 64),
                    block_dim=64,
                )
                with output.map_to_host() as values:
                    for index in range(257):
                        var rng = path_stage_rng(config, UInt32(index), stage)
                        for dimension in range(12):
                            assert_equal(
                                bitcast[.uint32](
                                    values[index * 12 + dimension]
                                ),
                                bitcast[.uint32](rng.f32()),
                            )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
