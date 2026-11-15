# Reference-based image reconstruction

The viewer uses two published methods: Zirr, Hanika and Dachsbacher's
[firefly sample reweighting](https://jo.dreggn.org/home/2018_fireflies.pdf)
(CGForum 2018, sections 4–5), followed by the spatial stages of
[Falcor SVGF](https://github.com/NVIDIAGameWorks/Falcor/tree/759aad033ff610fb0d82c74f7e0a508d0096d5f2/Source/RenderPasses/SVGFPass).
The Falcor port is pinned to `759aad033ff610fb0d82c74f7e0a508d0096d5f2`;
its BSD license is retained in [FALCOR_LICENSE.txt](FALCOR_LICENSE.txt).
This is spatial reconstruction of a progressive image, not full temporal SVGF.

## Behavior

Complete camera samples are classified at their actual first resolved material:
Lambertian samples are diffuse; metals and coated materials are glossy.
Backgrounds, dielectric paths and camera-visible emissive surfaces pass through.
A coated material remains one glossy component: there is no approximate split
of its internal layers. An emissive scattering surface conservatively passes
its whole path through, keeping its camera-visible emission intact.

Firefly reweighting applies to diffuse and glossy contributions. It uses the
paper's overlapping base-eight cascades, adjacent-buffer sample counting,
3×3 rejection test, value-based lower bound and conservative κ = κ_min = 1.
The cascade's numerical floor is 1e-6: values below it are retained in the
lowest buffer without rejection. Nonpositive-luminance samples retain signed
RGB in passthrough; a single scalar weight preserves the RGB ratio of each
cascade. This does not independently reject chromatic outliers of ordinary
luminance. The method introduces finite-sample bias; poorly sampled real
lighting can be attenuated along with fireflies.

Only diffuse illumination is spatially filtered. Albedo is removed with
Falcor's 0.001 floor, then restored after filtering. Glossy reflections receive
firefly reweighting **without spatial blur**, so they retain detail and residual
noise. Default SVGF settings are four 5×5 B3 passes at steps 1, 2, 4, 8,
`phi_color=10`, and `phi_normal=128`. Variance is smoothed before evaluating
luminance weights and propagated with squared filter weights each pass.

The adaptations to the reference are explicit:

- Independent camera-sample statistics replace reprojected temporal history;
  variance describes the accumulated mean, including reconstruction weights.
- Falcor's 7×7 moment bootstrap uses camera sample count when fewer than four
  samples are available. No filtered results feed back into accumulation.
- Float64 centered RGB moments and adjacent-cascade cross moments reconstruct
  variance after reweighting without unstable uncentered HDR subtraction.
- Screen-space planar depth derivatives come from **geometric** normals;
  shading normals still guide normal edge stopping. A standard Float32 gamma(8)
  error bound supplies a minimum derivative for ray-hit roundoff on flat planes.
  Boundary kernels normalize
  their in-bounds taps. Background and non-diffuse guides bypass spatial mixing.

Guides use deterministic pixel/lens-center rays, respect opacity and texture
maps, and approximate coverage/depth-of-field boundaries. They contain thirteen
Float32 words: position XYZ, shading normal XYZ, albedo RGB, world footprint,
linear view depth, depth fwidth, and a diffuse-filter eligibility flag.
`UInt32.MAX` marks a miss. No renderer material-ID bit encoding is assumed by
the filter.

## CPU API

```mojo
from bajo.postprocess import DenoiseStatistics, AtrousWorkspace, denoise
from bajo.rt import render_wavefront
from bajo.rt.cpu.denoise_guides import make_cpu_denoise_guides

settings.collect_denoise = True
# RenderResult.denoise_samples contains film-converted complete samples,
# their reconstruction weights and first-hit component classifications.
var result = render_wavefront[.MIS](settings, camera, scene)
var stats = DenoiseStatistics(settings.image_width, settings.image_height)
stats.add(result.denoise_samples.value())
var guides = make_cpu_denoise_guides(scene, camera,
                               settings.image_width, settings.image_height)
var workspace = AtrousWorkspace()
denoise(stats, guides, workspace)
# workspace.ping contains linear RGB output. Retain stats across sample batches.
```

The lower-level `reconstruct_frame(stats, guides, firefly=True)` returns an
owned `DenoiseFrame` with diffuse/glossy/passthrough images, diffuse illumination
variance, and camera sample count. `atrous_denoise(frame, guides, workspace,
settings)` filters that prepared frame. For this lower-level API, firefly
selection belongs to `reconstruct_frame`; `settings.firefly` is consumed by
the higher-level `denoise` function. Zero passes recomposes the prepared
components without spatial filtering. Missing statistics raise; there is no
fallback to the previous spatial-variance-only estimator.

## GPU API and memory

`GpuDenoiseSampleBatch`, `GpuDenoiseStatistics`, `GpuDenoiseGuides`,
`GpuAtrousWorkspace` and `enqueue_atrous_denoise` live in `bajo.postprocess.gpu`.
GPU preview enables complete-sample capture, updates resident statistics, then
enqueues reconstruction/filtering. Only a scalar maximum is read back to size
the cascade. The filter itself adds no synchronization or image readback.

Cascade arrays grow to cover observed finite Float32 luminance rather than
clamping overflow. Each allocated level costs `(27 + 4) × 8 = 248` bytes per
pixel: 27 Float64 diffuse statistics and four glossy count/RGB sums. Glossy
variance is not stored or reconstructed. Metadata costs 40 bytes per pixel.
The temporary complete-sample sidecar costs
20 bytes per camera sample. This correctness baseline uses more memory than
the old filter. Allocations and filter scratch are reused; allocation or
reconstruction failures report an error and publish the retained raw preview.

## Viewer and verification

The existing **Denoise: Off / On** button remains off by default and supports
CPU/GPU RGB/spectral output. Statistics are collected from the first batch even
while off. Toggling or refreshing completed renders does not trace again,
change sample counts, or modify raw beauty sums. Camera/render generations
reset all auxiliary state. Raster and NORMALS bypass filtering; AO samples are retained as passthrough.

```bash
pixi run test postprocess/cpu
pixi run test postprocess/gpu
pixi run test viewer
```

`mojo build -I . scripts/check_denoising.mojo` builds a reproducible quality
check; add `--target-accelerator sm_120` for an RTX 5060 Ti build. Its arguments
are `[cornell|scene.pbrt] [output-prefix] [width] [spp] [reference-spp]`.
It writes raw, denoised and reference PPMs, and reports linear MSE, cascade
memory and postprocess timing. Compare image crops as well as the aggregate
error; MSE alone does not establish detail preservation.
