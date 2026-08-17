from std.bit import count_trailing_zeros
from std.gpu import WARP_SIZE, global_idx, lane_id, warp_id
from std.gpu.primitives import warp
from std.math import min
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from max.gpu.host import DeviceContext

from bajo.bvh.gpu.builder.hploc_wave import (
    HPLOC_INVALID_LANE,
    HPLOC_WAVE_SIZE,
    HplocWaveMinimum,
    hploc_wave_ballot,
    hploc_wave_first_lane,
    hploc_wave_is_better,
    hploc_wave_rank,
)
from bajo.bvh.gpu.utils import _device_span


comptime HPLOC_WAVE_TEST_WAVES = 2
comptime HPLOC_WAVE_TEST_THREADS = HPLOC_WAVE_TEST_WAVES * WARP_SIZE


struct HplocShuffleProbe:
    comptime STRIDE = 4
    comptime LANE = 0
    comptime WAVE = 1
    comptime ACTIVE = 2
    comptime SHUFFLE = 3


struct HplocScanProbe:
    comptime STRIDE = 2
    comptime INCLUSIVE_SUM = 0
    comptime EXCLUSIVE_SUM = 1


struct HplocElectionProbe:
    comptime STRIDE = 4
    comptime BALLOT = 0
    comptime RANK = 1
    comptime LEADER = 2
    comptime ELECTED = 3


def hploc_shuffle_probe_kernel[
    block_size: Int,
](output: MutSpan[UInt64, MutAnyOrigin]):
    comptime assert block_size == HPLOC_WAVE_TEST_THREADS
    var thread = global_idx.x
    var lane = lane_id()
    var wave = warp_id()
    var active_count = WARP_SIZE - wave * 5
    var active = lane < active_count

    var value = UInt32(wave * 1000 + lane)
    var source_lane = UInt32((lane * 7 + 3) & (WARP_SIZE - 1))
    var shuffled = warp.shuffle_idx(value, source_lane)

    var base = thread * HplocShuffleProbe.STRIDE
    output.unsafe_get(base + HplocShuffleProbe.LANE) = UInt64(lane)
    output.unsafe_get(base + HplocShuffleProbe.WAVE) = UInt64(wave)
    output.unsafe_get(base + HplocShuffleProbe.ACTIVE) = UInt64(active)
    output.unsafe_get(base + HplocShuffleProbe.SHUFFLE) = UInt64(shuffled)


def hploc_scan_probe_kernel[
    block_size: Int,
](output: MutSpan[UInt32, MutAnyOrigin]):
    comptime assert block_size == HPLOC_WAVE_TEST_THREADS
    var thread = global_idx.x
    var lane = lane_id()
    var wave = warp_id()
    var active_count = WARP_SIZE - wave * 5
    var scan_value = UInt32(lane + 1) if lane < active_count else UInt32(0)
    var inclusive_sum = warp.prefix_sum[exclusive=False](scan_value)
    var base = thread * HplocScanProbe.STRIDE
    output.unsafe_get(base + HplocScanProbe.INCLUSIVE_SUM) = inclusive_sum
    output.unsafe_get(base + HplocScanProbe.EXCLUSIVE_SUM) = (
        inclusive_sum - scan_value
    )


def hploc_reduction_probe_kernel[
    block_size: Int,
](output: MutSpan[UInt32, MutAnyOrigin]):
    comptime assert block_size == HPLOC_WAVE_TEST_THREADS
    var thread = global_idx.x
    var lane = lane_id()
    var wave = warp_id()
    var active_count = WARP_SIZE - wave * 5
    var minimum_cost = (
        UInt32((lane * 13 + wave * 7) % 17) if lane
        < active_count else UInt32.MAX
    )

    comptime for bit in range(count_trailing_zeros(WARP_SIZE)):
        var other_cost = warp.shuffle_xor(minimum_cost, UInt32(1 << bit))
        if other_cost < minimum_cost:
            minimum_cost = other_cost

    output.unsafe_get(thread) = minimum_cost


def hploc_election_probe_kernel[
    block_size: Int,
](output: MutSpan[UInt64, MutAnyOrigin]):
    comptime assert block_size == HPLOC_WAVE_TEST_THREADS
    var thread = global_idx.x
    var lane = lane_id()
    var wave = warp_id()
    var active_count = WARP_SIZE - wave * 5
    var active = lane < active_count
    var cost = UInt32((lane * 13 + wave * 7) % 17)
    var elected = active and cost == 0
    var ballot = hploc_wave_ballot(elected)
    var rank = hploc_wave_rank(ballot, lane)
    var leader = hploc_wave_first_lane(ballot)

    var base = thread * HplocElectionProbe.STRIDE
    output.unsafe_get(base + HplocElectionProbe.BALLOT) = ballot
    output.unsafe_get(base + HplocElectionProbe.RANK) = UInt64(rank)
    output.unsafe_get(base + HplocElectionProbe.LEADER) = UInt64(leader)
    output.unsafe_get(base + HplocElectionProbe.ELECTED) = UInt64(elected)


def _active_count(wave: Int) -> Int:
    return WARP_SIZE - wave * 5


def _minimum_cost(wave: Int, active_count: Int) -> UInt32:
    var minimum = UInt32.MAX
    for lane in range(active_count):
        var cost = UInt32((lane * 13 + wave * 7) % 17)
        if cost < minimum:
            minimum = cost
    return minimum


def _election_ballot(wave: Int, active_count: Int) -> UInt64:
    var ballot = UInt64(0)
    for lane in range(active_count):
        if (lane * 13 + wave * 7) % 17 == 0:
            ballot |= UInt64(1) << UInt64(lane)
    return ballot


def test_hploc_shuffle_scan_and_reduction() raises:
    assert_true(HPLOC_WAVE_SIZE == 32 or HPLOC_WAVE_SIZE == 64)

    with DeviceContext() as ctx:
        var output = ctx.enqueue_create_buffer[DType.uint64](
            HPLOC_WAVE_TEST_THREADS * HplocShuffleProbe.STRIDE
        )
        var scans = ctx.enqueue_create_buffer[DType.uint32](
            HPLOC_WAVE_TEST_THREADS * HplocScanProbe.STRIDE
        )
        var reductions = ctx.enqueue_create_buffer[DType.uint32](
            HPLOC_WAVE_TEST_THREADS
        )
        ctx.enqueue_function[
            hploc_shuffle_probe_kernel[HPLOC_WAVE_TEST_THREADS]
        ](
            _device_span[mut=True](output),
            grid_dim=1,
            block_dim=HPLOC_WAVE_TEST_THREADS,
        )
        ctx.enqueue_function[hploc_scan_probe_kernel[HPLOC_WAVE_TEST_THREADS]](
            _device_span[mut=True](scans),
            grid_dim=1,
            block_dim=HPLOC_WAVE_TEST_THREADS,
        )
        ctx.enqueue_function[
            hploc_reduction_probe_kernel[HPLOC_WAVE_TEST_THREADS]
        ](
            _device_span[mut=True](reductions),
            grid_dim=1,
            block_dim=HPLOC_WAVE_TEST_THREADS,
        )
        ctx.synchronize()

        with output.map_to_host() as result, scans.map_to_host() as scanned, reductions.map_to_host() as reduced:
            for wave in range(HPLOC_WAVE_TEST_WAVES):
                var active_count = _active_count(wave)
                var expected_minimum = UInt64(_minimum_cost(wave, active_count))

                for lane in range(WARP_SIZE):
                    var thread = wave * WARP_SIZE + lane
                    var base = thread * HplocShuffleProbe.STRIDE
                    var scan_base = thread * HplocScanProbe.STRIDE
                    var source_lane = (lane * 7 + 3) & (WARP_SIZE - 1)
                    var clamped_lane = min(lane + 1, active_count)
                    var inclusive = UInt64(
                        clamped_lane * (clamped_lane + 1) / 2
                    )
                    var prior_active = min(lane, active_count)
                    var exclusive = UInt64(
                        prior_active * (prior_active + 1) / 2
                    )

                    assert_equal(
                        result[base + HplocShuffleProbe.LANE], UInt64(lane)
                    )
                    assert_equal(
                        result[base + HplocShuffleProbe.WAVE], UInt64(wave)
                    )
                    assert_equal(
                        result[base + HplocShuffleProbe.ACTIVE],
                        UInt64(lane < active_count),
                    )
                    assert_equal(
                        result[base + HplocShuffleProbe.SHUFFLE],
                        UInt64(wave * 1000 + source_lane),
                    )
                    assert_equal(
                        scanned[scan_base + HplocScanProbe.INCLUSIVE_SUM],
                        UInt32(inclusive),
                    )
                    assert_equal(
                        scanned[scan_base + HplocScanProbe.EXCLUSIVE_SUM],
                        UInt32(exclusive),
                    )
                    assert_equal(reduced[thread], UInt32(expected_minimum))


def test_hploc_ballot_rank_and_leader_election() raises:
    with DeviceContext() as ctx:
        var output = ctx.enqueue_create_buffer[DType.uint64](
            HPLOC_WAVE_TEST_THREADS * HplocElectionProbe.STRIDE
        )
        ctx.enqueue_function[
            hploc_election_probe_kernel[HPLOC_WAVE_TEST_THREADS]
        ](
            _device_span[mut=True](output),
            grid_dim=1,
            block_dim=HPLOC_WAVE_TEST_THREADS,
        )
        ctx.synchronize()

        with output.map_to_host() as result:
            for wave in range(HPLOC_WAVE_TEST_WAVES):
                var active_count = _active_count(wave)
                var ballot = _election_ballot(wave, active_count)
                var leader = UInt64(hploc_wave_first_lane(ballot))

                for lane in range(WARP_SIZE):
                    var thread = wave * WARP_SIZE + lane
                    var base = thread * HplocElectionProbe.STRIDE
                    var elected = (
                        lane < active_count and (lane * 13 + wave * 7) % 17 == 0
                    )
                    var rank = UInt64(0)
                    for prior in range(lane):
                        if (
                            prior < active_count
                            and (prior * 13 + wave * 7) % 17 == 0
                        ):
                            rank += 1

                    assert_equal(
                        result[base + HplocElectionProbe.BALLOT], ballot
                    )
                    assert_equal(result[base + HplocElectionProbe.RANK], rank)
                    assert_equal(
                        result[base + HplocElectionProbe.LEADER], leader
                    )
                    assert_equal(
                        result[base + HplocElectionProbe.ELECTED],
                        UInt64(elected),
                    )


def test_hploc_deterministic_candidate_ordering() raises:
    var current = HplocWaveMinimum(7, 12)
    assert_true(hploc_wave_is_better(HplocWaveMinimum(6, 99), current))
    assert_true(hploc_wave_is_better(HplocWaveMinimum(7, 4), current))
    assert_false(hploc_wave_is_better(HplocWaveMinimum(7, 20), current))
    assert_false(hploc_wave_is_better(HplocWaveMinimum(8, 0), current))
    assert_equal(hploc_wave_first_lane(0), HPLOC_INVALID_LANE)


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
