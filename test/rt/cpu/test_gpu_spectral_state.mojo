"""Host checks of the GPU spectral plane ABI and independent sample updates."""

from std.memory import bitcast
from std.testing import TestSuite, assert_equal
from bajo.core.sampled_spectrum import SampledSpectrum
from bajo.rt.gpu.spectral_state import GpuSpectralStateView


def test_spectral_state_loads_plane_major_wavelengths_and_throughput() raises:
    comptime capacity = 7
    var storage = List[Float32](length=17 * capacity, fill=0)
    for word in range(len(storage)):
        storage[word] = Float32(word)
    var state = GpuSpectralStateView(
        storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        capacity,
        123,
    )
    for index in range(capacity):
        var wavelengths = state.wavelengths(index)
        var throughput = state.load(8, index)
        var radiance = state.load(12, index)
        for channel in range(4):
            assert_equal(
                wavelengths.lambda_nm[channel],
                Float32(channel * capacity + index),
            )
            assert_equal(
                wavelengths.pdf[channel],
                Float32((4 + channel) * capacity + index),
            )
            assert_equal(
                throughput[channel], Float32((8 + channel) * capacity + index)
            )
            assert_equal(
                radiance[channel], Float32((12 + channel) * capacity + index)
            )
    assert_equal(state.sample_base, UInt32(123))
    # The GPU ABI uses an untracked pointer; retain its owner through the reads.
    assert_equal(storage[16 * capacity], Float32(16 * capacity))


def test_spectral_state_updates_preserve_neighbors_eta_and_guards() raises:
    comptime capacity = 7
    # Guard words surround the exact device buffer, including its eta plane.
    var storage = List[Float32](length=17 * capacity + 2, fill=-999)
    var state = GpuSpectralStateView(
        storage.unsafe_ptr()
        .unsafe_offset(1)
        .unsafe_origin_cast[MutUntrackedOrigin](),
        capacity,
        123,
    )
    var throughput = SampledSpectrum[1](-0.0, 1e20, -1e-20, 3.5)
    var radiance = SampledSpectrum[1](2, 4, 8, 16)
    state.store(8, 0, throughput)
    state.store(12, capacity - 1, SampledSpectrum[1](0))
    state.accumulate(capacity - 1, radiance)
    state.accumulate(capacity - 1, radiance * 0.5)
    var loaded = state.load(8, 0)
    for channel in range(4):
        assert_equal(
            bitcast[.uint32](loaded[channel]),
            bitcast[.uint32](throughput[channel]),
        )
    for word in range(len(storage)):
        var expected = Float32(-999)
        for channel in range(4):
            if word == 1 + (8 + channel) * capacity:
                expected = throughput[channel]
            if word == 1 + (12 + channel) * capacity + capacity - 1:
                expected = radiance[channel] * 1.5
        assert_equal(
            bitcast[.uint32](storage[word]), bitcast[.uint32](expected)
        )


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
