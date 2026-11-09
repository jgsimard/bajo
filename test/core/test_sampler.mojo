"""Sampler IDs are checked when they enter the typed sampling API."""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from std.memory import bitcast
from std.builtin.globals import global_constant
from bajo.core.random import Rng, Sampler
from bajo.core.zsobol import (
    zsobol_log2_samples,
    zsobol_morton_index,
    zsobol_sample_index,
)
from test.core.zsobol_reference import ZSOBOL_REFERENCE


def test_sampler_sequence_ids() raises:
    var cases = [
        Sampler.INDEPENDENT,
        Sampler.HALTON,
        Sampler.R2,
        Sampler.OWEN_SOBOL,
        Sampler.SZ,
        Sampler.STBN,
        Sampler.Z_SOBOL,
    ]
    for id in range(len(cases)):
        assert_equal(Sampler(id), cases[id])
        assert_equal(Sampler(UInt32(id)), cases[id])
        assert_equal(Int(cases[id]), id)


def test_sampler_rejects_unknown_sequence_ids() raises:
    for id in [Int.MIN, -1, 7, Int.MAX]:
        with assert_raises():
            _ = Sampler(id)
    for id in [UInt32(7), UInt32.MAX]:
        with assert_raises():
            _ = Sampler(id)


def test_zsobol_matches_independent_pbrt_index_and_sample_references() raises:
    ref cases = global_constant[ZSOBOL_REFERENCE]()
    var covered_high_bits = False
    for index in range(len(cases)):
        var reference = cases[index]
        var log2_samples = zsobol_log2_samples(reference.samples)
        var digits = (
            zsobol_log2_samples(max(reference.width, reference.height))
            + (log2_samples + 1) // 2
        )
        var pixel = reference.y * reference.width + reference.x
        var morton = zsobol_morton_index(
            pixel,
            reference.width,
            SIMD[.uint64, 1](reference.sample),
            log2_samples,
        )
        assert_equal(
            zsobol_sample_index(
                morton, log2_samples, digits, reference.dimension
            ),
            reference.index,
        )
        covered_high_bits |= reference.index > UInt64(UInt32.MAX)
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
        assert_equal(bitcast[.uint32](rng.f32()), reference.u_bits)
        assert_equal(bitcast[.uint32](rng.f32()), reference.v_bits)
    assert_true(covered_high_bits)


def test_zsobol_prefixes_are_stratified_two_dimensional_nets() raises:
    for samples in [4, 16, 256]:
        var occupied_x = List[Bool](length=samples, fill=False)
        var occupied_y = List[Bool](length=samples, fill=False)
        var occupied_cells = List[Bool](length=samples, fill=False)
        var side = 2 if samples == 4 else 4 if samples == 16 else 16
        for sample in range(samples):
            var rng = Rng[1](
                97,
                0,
                sampler=.Z_SOBOL,
                sample_index=UInt64(sample),
                pixel_id=301,
                image_width=32,
                image_height=17,
                sequence_length=UInt32(samples),
            )
            var u = rng.f32()[0]
            var v = rng.f32()[0]
            assert_true(u >= 0 and u < 1 and v >= 0 and v < 1)
            var x = Int(u * Float32(samples))
            var y = Int(v * Float32(samples))
            var cell = Int(u * Float32(side)) + side * Int(v * Float32(side))
            assert_equal(occupied_x[x], False)
            assert_equal(occupied_y[y], False)
            assert_equal(occupied_cells[cell], False)
            occupied_x[x] = True
            occupied_y[y] = True
            occupied_cells[cell] = True


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
