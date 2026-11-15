"""Reference stages, sample variance, fireflies, and preserved lighting detail."""
from std.testing import (
    TestSuite,
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_raises,
)
from std.memory import bitcast
from bajo.core.color import Color, luminance
from bajo.core import Point3W, Vec3W
from bajo.rt import (
    SceneBuilder,
    CpuScene,
    Camera,
    RenderSettings,
    render_wavefront,
)
from bajo.rt.render_types import RenderColorMode
from bajo.rt.material_types import Environment
from bajo.rt.filter import ReconstructionFilter
from bajo.postprocess import (
    AtrousSettings,
    DenoiseGuides,
    GuideLayout,
    AtrousWorkspace,
    atrous_denoise,
    DenoiseFrame,
    DenoiseStatistics,
    DenoiseSampleBatch,
    reconstruct_frame,
)


def flat_guides(width: Int, height: Int) raises -> DenoiseGuides:
    var result = DenoiseGuides(width, height)
    for i in range(width * height):
        var at = i * GuideLayout.STRIDE
        result.surfaces[i] = 0
        result.fields[at] = Float32(i % width)
        result.fields[at + 1] = Float32(i // width)
        result.fields[at + GuideLayout.NORMAL + 2] = 1
        comptime for c in range(3):
            result.fields[at + GuideLayout.ALBEDO + c] = 0.5
        result.fields[at + GuideLayout.FOOTPRINT] = 1
        result.fields[at + GuideLayout.DEPTH] = 1
        result.fields[at + GuideLayout.DIFFUSE] = 1
    return result^


def sample_fixture(width: Int, height: Int, spp: Int) -> DenoiseSampleBatch:
    var batch = DenoiseSampleBatch(width * height, spp)
    for i in range(len(batch.colors)):
        batch.colors[i] = Color(1)
        batch.components[i] = 1
    return batch^


def test_weighted_sample_variance_and_uneven_batches() raises:
    var guides = flat_guides(2, 1)
    var full = sample_fixture(2, 1, 5)
    var left = sample_fixture(2, 1, 2)
    var right = sample_fixture(2, 1, 3)
    for pixel in range(2):
        for s in range(5):
            var c = Color(
                Float32(s + 1), Float32(2 * s) + 0.5, Float32(pixel) + 0.1
            )
            var w = Float32(0.25 + Float32(s) * 0.5)
            var i = pixel * 5 + s
            full.colors[i] = c
            full.weights[i] = w
            if s < 2:
                left.colors[pixel * 2 + s] = c
                left.weights[pixel * 2 + s] = w
            else:
                right.colors[pixel * 3 + s - 2] = c
                right.weights[pixel * 3 + s - 2] = w
    var stats = DenoiseStatistics(2, 1)
    stats.add(full)
    var split = DenoiseStatistics(2, 1)
    split.add(left)
    split.add(right)
    var frame = reconstruct_frame(stats, guides, False)
    var other = reconstruct_frame(split, guides, False)
    for pixel in range(2):
        var total = Color(0)
        var norm = Float32(0)
        for s in range(5):
            var i = pixel * 5 + s
            total += full.colors[i] * full.weights[i]
            norm += full.weights[i]
        var mean = total / norm
        var variance = Float32(0)
        for s in range(5):
            var i = pixel * 5 + s
            var residual = (
                2 * luminance(full.colors[i] - mean) * full.weights[i]
            )
            variance += residual * residual
        variance *= 5.0 / 4.0 / (norm * norm)
        assert_almost_equal(frame.variance[pixel], variance, atol=2e-5)
        assert_almost_equal(other.variance[pixel], variance, atol=2e-5)
        comptime for c in range(3):
            assert_almost_equal(frame.diffuse[pixel][c], mean[c], atol=2e-6)
            assert_almost_equal(other.diffuse[pixel][c], mean[c], atol=2e-6)
    assert_equal(stats.samples, 5)
    assert_equal(split.samples, 5)


def test_firefly_reweighting_retains_repeated_glossy_highlight() raises:
    var guides = flat_guides(9, 7)
    var batch = sample_fixture(9, 7, 64)
    for i in range(len(batch.components)):
        batch.components[i] = 2
    # A rare path is an outlier; a repeatedly sampled microhighlight is detail.
    batch.colors[31 * 64] = Color(10000)
    for s in range(64):
        batch.colors[15 * 64 + s] = Color(100)
    var stats = DenoiseStatistics(9, 7)
    stats.add(batch)
    var frame = reconstruct_frame(stats, guides)
    var workspace = AtrousWorkspace()
    atrous_denoise(frame, guides, workspace)
    assert_true(workspace.ping[31].x < 2)
    assert_almost_equal(workspace.ping[15].x, Float32(100), atol=1e-4)
    assert_almost_equal(workspace.ping[14].x, Float32(1), atol=2e-6)
    assert_equal(batch.colors[31 * 64].x, Float32(10000))


def test_converged_shadow_and_equal_luminance_glossy_detail() raises:
    var guides = flat_guides(33, 5)
    var frame = DenoiseFrame(165, 64)
    for i in range(165):
        frame.diffuse[i] = Color(Float32(0.05) if i % 33 < 16 else Float32(1))
        frame.glossy[i] = Color(1, 0, 0) if i % 33 == 23 else Color(
            0, 0.2126 / 0.7152, 0
        )
    var workspace = AtrousWorkspace()
    atrous_denoise(frame, guides, workspace)
    for i in range(165):
        var expected = frame.diffuse[i] + frame.glossy[i]
        comptime for c in range(3):
            assert_almost_equal(workspace.ping[i][c], expected[c], atol=2e-5)


def test_texture_demodulation_and_diffuse_noise_reduction() raises:
    var guides = flat_guides(17, 9)
    var frame = DenoiseFrame(153, 16)
    var raw = Float32(0)
    for i in range(153):
        var a = Float32(0.2 if i % 2 else 0.8)
        var noise = Float32(-0.2 if (i // 17 + i % 17) % 2 else 0.2)
        comptime for c in range(3):
            guides.fields[i * GuideLayout.STRIDE + GuideLayout.ALBEDO + c] = a
        frame.diffuse[i] = Color(a * (1 + noise))
        frame.variance[i] = 0.04
        raw += a * a * noise * noise
    var workspace = AtrousWorkspace()
    atrous_denoise(frame, guides, workspace)
    var error = Float32(0)
    for i in range(153):
        var target = guides.fields[i * GuideLayout.STRIDE + GuideLayout.ALBEDO]
        var d = workspace.ping[i].x - target
        error += d * d
    assert_true(error < raw * 0.25)


def test_signed_hdr_passthrough_and_zero_passes() raises:
    var guides = flat_guides(3, 1)
    var batch = sample_fixture(3, 1, 64)
    for i in range(len(batch.colors)):
        batch.colors[i] = Color(-0.25, 12, 0.75)
        if i // 64 == 1:
            batch.colors[i] = Color(-10, -1, -2)
    var stats = DenoiseStatistics(3, 1)
    stats.add(batch)
    var frame = reconstruct_frame(stats, guides, False)
    var workspace = AtrousWorkspace()
    atrous_denoise(frame, guides, workspace, AtrousSettings(passes=0))
    for i in range(3):
        comptime for c in range(3):
            assert_almost_equal(
                workspace.ping[i][c], batch.colors[i * 64][c], atol=2e-6
            )
    assert_true(frame.variance[0] < 1e-9)


def test_reference_one_pass_and_variance_propagation() raises:
    # Constant normal/depth/luminance => Falcor's separable B3 weights only.
    # RGB chroma differs at constant luminance to exercise the reference kernel.
    var guides = flat_guides(5, 5)
    var frame = DenoiseFrame(25, 16)
    for i in range(25):
        frame.diffuse[i] = Color(0.5)
        frame.variance[i] = 0.2
    var workspace = AtrousWorkspace()
    atrous_denoise(frame, guides, workspace, AtrousSettings(passes=1))
    # (sum of squared [1/6, 2/3, 1, 2/3, 1/6])^2 / (8/3)^4.
    var one_d_sq = Float32(1 + 2 * 4.0 / 9.0 + 2.0 / 36.0)
    var expected = 0.2 * one_d_sq * one_d_sq / (Float32(8.0 / 3.0) ** 4)
    assert_almost_equal(workspace.ping[12].x, Float32(0.5), atol=1e-6)
    assert_almost_equal(workspace.variance[12], expected, atol=1e-6)


def test_missing_statistics_and_invalid_dimensions_raise() raises:
    var guides = flat_guides(2, 1)
    var workspace = AtrousWorkspace()
    var empty = DenoiseFrame(2, 0)
    with assert_raises():
        atrous_denoise(empty, guides, workspace)
    var stats = DenoiseStatistics(2, 1)
    with assert_raises():
        _ = reconstruct_frame(stats, guides)
    with assert_raises():
        AtrousSettings(passes=9).validate()


def test_constant_extreme_hdr_has_zero_sample_variance() raises:
    var guides = flat_guides(2, 1)
    var batch = sample_fixture(2, 1, 127)
    for i in range(len(batch.colors)):
        batch.colors[i] = Color(1e20, -1e19, 3e19)
        batch.weights[i] = Float32(0.5) if i % 2 else Float32(1)
    var stats = DenoiseStatistics(2, 1)
    stats.add(batch)
    var frame = reconstruct_frame(stats, guides, False)
    for i in range(2):
        # Relative variance is bounded by Float64 accumulation roundoff, not
        # catastrophic cancellation of uncentered squared HDR moments.
        assert_true(frame.variance[i] / Float32(1e38) < 1e-20)
        assert_almost_equal(frame.diffuse[i].x, Float32(1e20), rtol=1e-6)


def test_sample_sidecar_preserves_raw_transport_bits() raises:
    var builder = SceneBuilder()
    var material = builder.add_lambertian(Color(0.6, 0.4, 0.2))
    builder.add_sphere(Point3W(0), 1, material)
    builder.set_environment(Environment.uniform(Color(2)))
    var world = CpuScene[16, 16](builder^.finish())
    var camera = Camera.from_vfov(
        Point3W(0, 0, 3), Point3W(0), Vec3W(0, 1, 0), 10
    )
    for mode in [RenderColorMode.RGB, RenderColorMode.SPECTRAL]:
        var settings = RenderSettings(3, 2, 5, 2026, 2, color_mode=mode)
        settings.filter = ReconstructionFilter.gaussian()
        var raw = render_wavefront[.MIS, 4](settings, camera, world)
        settings.collect_denoise = True
        var collected = render_wavefront[.MIS, 4](settings, camera, world)
        assert_equal(len(collected.denoise_samples.value().colors), 30)
        for i in range(6):
            comptime for c in range(3):
                assert_equal(
                    bitcast[.uint32](raw.pixels[i][c]),
                    bitcast[.uint32](collected.pixels[i][c]),
                )
        for tag in collected.denoise_samples.value().components:
            assert_equal(tag, UInt32(1))


def test_compact_components_survive_cascade_growth_and_mixed_coverage() raises:
    var guides = flat_guides(2, 1)
    var first = sample_fixture(2, 1, 4)
    for s in range(4):
        first.components[4 + s] = 2
        first.colors[4 + s] = Color(5, 6, 7)
    var stats = DenoiseStatistics(2, 1)
    stats.add(first)
    var prior_bins = stats.n_bins
    var tail = sample_fixture(2, 1, 1)
    # One pixel now sees a passthrough surface; another sees a bright glossy
    # sample. The new brightness range must preserve both existing components.
    tail.components[0] = 0
    tail.colors[0] = Color(2)
    tail.components[1] = 2
    tail.colors[1] = Color(1e6)
    stats.add(tail)
    assert_true(stats.n_bins > prior_bins)
    assert_equal(len(stats.bins), stats.n_bins * 2 * (27 + 4))
    assert_equal(len(stats.meta), 2 * 5)
    var frame = reconstruct_frame(stats, guides, False)
    comptime for c in range(3):
        assert_almost_equal(frame.diffuse[0][c], Float32(0.8), atol=1e-6)
        assert_almost_equal(frame.passthrough[0][c], Float32(0.4), atol=1e-6)
        assert_almost_equal(
            frame.glossy[1][c], (Float32(5 + c) * 4 + 1e6) / 5, atol=0.02
        )
    assert_almost_equal(frame.variance[0], Float32(0.16), atol=1e-6)
    assert_equal(frame.variance[1], Float32(0))


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
