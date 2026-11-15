"""CPU/device parity for sample cascades, reweighting, filtering and reuse."""
from max.gpu.host import DeviceContext
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_raises,
)
from std.memory import bitcast
from bajo.core.color import Color
from bajo.gpu.buffers import upload_list
from bajo.postprocess import (
    DenoiseStatistics,
    DenoiseSampleBatch,
    AtrousWorkspace,
    AtrousSettings,
    reconstruct_frame,
    atrous_denoise,
)
from bajo.postprocess.gpu import (
    GpuDenoiseGuides,
    GpuDenoiseSampleBatch,
    GpuDenoiseStatistics,
    GpuAtrousWorkspace,
    enqueue_atrous_denoise,
)
from test.postprocess.cpu.test_atrous import flat_guides, sample_fixture


def upload_samples(
    mut ctx: DeviceContext, batch: DenoiseSampleBatch
) raises -> GpuDenoiseSampleBatch:
    var result = GpuDenoiseSampleBatch(ctx, len(batch.colors))
    var words = List[Float32]()
    for color in batch.colors:
        comptime for c in range(3):
            words.append(color[c])
    result.colors = upload_list(ctx, words)
    result.weights = upload_list(ctx, batch.weights)
    result.components = upload_list(ctx, batch.components)
    return result^


def test_cpu_gpu_statistics_reweighting_filter_and_input_preservation() raises:
    with DeviceContext() as ctx:
        for width in [1, 7, 17]:
            var guides = flat_guides(width, 3)
            var gpu_guides = GpuDenoiseGuides.upload(ctx, guides)
            var host = DenoiseStatistics(width, 3)
            var device = GpuDenoiseStatistics(ctx, width, 3)
            var cpu = AtrousWorkspace()
            var gpu = GpuAtrousWorkspace(ctx, width, 3)
            for spp in [1, 2, 13]:
                var batch = sample_fixture(width, 3, spp)
                for i in range(len(batch.colors)):
                    batch.colors[i] = Color(
                        -0.25 + Float32(i % 3) * 0.1, 2 + Float32(i % 2), 0.75
                    )
                    batch.components[i] = UInt32((i // spp) % 3)
                    batch.weights[i] = Float32(1) + Float32(i % 2) * 0.1
                batch.colors[0] = Color(10000)
                var uploaded = upload_samples(ctx, batch)
                host.add(batch)
                device.add(ctx, uploaded, spp)
                for firefly in [False, True]:
                    var frame = reconstruct_frame(host, guides, firefly)
                    for passes in [0, 1, 4]:
                        var settings = AtrousSettings(
                            passes=passes, firefly=firefly
                        )
                        atrous_denoise(frame, guides, cpu, settings)
                        enqueue_atrous_denoise(
                            ctx, device, gpu_guides, gpu, settings
                        )
                        with gpu.ping.map_to_host() as actual:
                            for i in range(width * 3):
                                comptime for c in range(3):
                                    assert_almost_equal(
                                        actual[3 * i + c],
                                        cpu.ping[i][c],
                                        atol=5e-4,
                                        rtol=2e-5,
                                    )
                        with gpu.variance.map_to_host() as actual:
                            for i in range(width * 3):
                                assert_almost_equal(
                                    actual[i],
                                    cpu.variance[i],
                                    atol=2e-3,
                                    rtol=2e-4,
                                )
                with uploaded.colors.map_to_host() as unchanged:
                    for i in range(len(batch.colors)):
                        comptime for c in range(3):
                            assert_equal(
                                bitcast[.uint32](unchanged[i * 3 + c]),
                                bitcast[.uint32](batch.colors[i][c]),
                            )
            assert_equal(host.samples, device.samples)
            assert_equal(host.n_bins, device.n_bins)
            var wrong = GpuDenoiseGuides(ctx, width + 1, 3)
            with assert_raises():
                enqueue_atrous_denoise(ctx, device, wrong, gpu)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
