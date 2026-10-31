# Bistro implementation and verification

Goal: render the checked-in/downloaded PBRT Bistro scene completely on CPU and
GPU, preserving its authored geometry, materials, lights, sampling, and film
settings. Do not replace missing behavior with material substitutions, dropped
parameters, or RGB stand-ins for spectral transport.

Status reviewed: 2026-10-30. Reconstruction filtering and the spectral foundations
described below have passing targeted checks. Spectral transport is not connected.
All three Bistro entrypoints load; full-asset GPU RGB
diagnostics have been produced for each camera. A complete, faithful render at
the authored settings has not been produced or verified. Loading and diagnostic rendering
do not close the missing features below.

## Confirmed remaining gaps

These are implementation requirements, not proposed visual approximations.
The references below identify behavior in the current working tree; passing
older tests does not verify later edits.

| Missing behavior | Evidence and required change |
| --- | --- |
| Wavelength sampling and path state | Four-wavelength arithmetic, visible/uniform sampling, PDFs, and secondary termination have CPU/GPU tests. Neither renderer carries their wavelengths/PDFs through primary rays, continuation, material/shadow queues, compaction, or sample accumulation. Integrate four-component throughput and radiance throughout CPU/GPU transport. |
| Spectral materials and textures | Original-table RGB reconstruction and packed optical-curve evaluation have passing reference tests. GPU scene preparation now retains optical curves and material spectrum IDs. BSDF evaluation still uses RGB projections or scalar IOR. Connect spectral material/texture evaluation at the path's actual wavelengths, including dispersive transmission and wavelength-PDF updates where required; give RGB reconstruction data renderer-owned storage. |
| Spectral lights and environment | Emission and environment evaluation still supply RGB. Implement wavelength-dependent area-light/environment evaluation, photometric normalization, and consistent light-sampling and emissive-hit MIS PDFs. The tested RGB reconstruction upload wrapper still needs renderer ownership/binding and actual light/texture consumers. |
| Texture footprints and filtering | Normal and opacity maps use repeating bilinear lookups; general material/environment lookups still use the existing sampling paths. There is no complete mipmap/footprint pipeline. Propagate derivatives or ray differentials, build mip levels, and implement the required filtering on CPU and GPU. Verify scalar-channel and color/data decoding semantics across every consumer. |
| Authored `zsobol` sampling | All three entrypoints request `Sampler "zsobol"` with 256 samples per pixel. The parser recognizes only `halton` and `sobol`; `zsobol` silently becomes independent sampling. Implement Z-Sobol and its CPU/GPU sequence integration. |
| Spectral film input for sample clamping | `apply_spectral_sample` now has CPU/GPU numerical tests for PDF-weighted CIE integration, exposure, and sensor-space clamping. CPU accumulation and GPU reduction still call the RGB `apply_sample` path. Connect actual complete-path radiance/wavelength/PDF samples and validated light normalization, preserving sensor clamp → output conversion → filter weighting → pixel reduction. |
| Declared EXR output | Film filenames are `bistro_cafe.exr`, `bistro_boulangerie.exr`, and `bistro_vespa.exr`. The parser does not retain the filename, and `image_output.mojo` provides raw Float32 RGB and PPM output, not an EXR writer. Retain output settings and write the requested linear HDR image. |
| Complete-scene verification | All three entrypoints load with 243 opacity bindings and 102 independent emission bindings each. Full-asset RGB diagnostics render all three cameras on GPU at 192×108, 32 spp, depth 8; an earlier Café diagnostic also rendered on CPU. Authored-resolution/settings renders, a complete parameter audit, converged image inspection, and PBRT comparison remain unverified. |

Film ISO is already read and applied through `FilmResponse`: Café and Vespa use
110, and Boulangerie uses 40. Resolution and sample count are also retained;
they are not missing parser features. Spectral film integration remains part of
the wavelength-domain requirement above.

Primary evidence: [scene declarations](assets/bistro/bistro_cafe.pbrt),
[materials](assets/bistro/materials.pbrt),
[PBRT parser](bajo/parser/pbrt/parser.mojo),
[CPU lighting](bajo/rt/cpu/lighting.mojo),
[GPU lighting](bajo/rt/gpu/lighting.mojo), and
[image output](bajo/rt/image_output.mojo).
PBRT's per-sample clamp is defined in
[RGBFilm::AddSample](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/film.h).
PBRT defaults are defined by its
[scene builder](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/scene.cpp),
[Gaussian filter](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/filters.cpp),
and [path integrators](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/cpu/integrators.cpp).

## Required work

- [x] Named brass (CuZn) spectra with original wavelength tables retained in scene data.
- [ ] Spectral material/light evaluation, wavelength transport, and film integration.
  - [x] Verify four-wavelength arithmetic, wavelength sampling/PDFs, and secondary-wavelength termination on CPU/GPU.
  - [x] Verify the original-resolution RGB-to-spectrum evaluator against the original PBRT table, including reflectance, unbounded, and illuminant cases.
  - [x] Pack original optical curves and bind their material IDs in GPU scene preparation; verify device evaluation.
  - [ ] Bind retained optical/reconstruction spectra to material, texture, light, and environment evaluation on CPU/GPU.
  - [ ] Carry wavelength/PDF state and four-component throughput/radiance through every transport queue and contribution.
  - [ ] Implement and verify spectral light normalization, dispersive transmission, MIS consistency, and actual sensor integration.
- [x] Tangent-space normal maps on triangle instances, linear decoding, and CPU/GPU shading.
  - [x] Linear PNG data decoding and separate, unassociated alpha storage.
  - [x] Resolve material normal-map paths and load them with data encoding.
  - [x] Derive UV tangent frames and apply mapped normals on CPU and GPU.
- [x] Alpha textures in closest-hit and shadow traversal, including instances.
  - [x] Preserve unassociated PNG color and a separate alpha plane for PBRT image loading.
  - [x] Retain per-shape scalar/image alpha bindings, including instance-owned bindings.
  - [x] Implement CPU candidate rejection before closest-hit reduction and shadow early exit.
  - [x] Verify GPU wide/compressed BLAS and TLAS rejection, including sphere candidates.
  - [x] Apply opacity to sampled emitters and emissive hits; retain accepted samples' original light PDFs.
  - [x] Verify transparent/opaque PNG emitter regions and mirrored instances on CPU/GPU.
- [x] Inline triangle-mesh UVs and vertex normals, including transforms and winding.
- [x] Correct PBRT column-major Transform and ConcatTransform input.
- [x] Rough dielectric reflection/refraction, exact Fresnel, and radiance transport weights.
- [x] Coated-conductor reflectance textures and correct optical defaults in the current transport path.
- [x] Independent emission and scattering properties on the same surface.
- [ ] Filtered texture sampling, mipmaps, derivatives, and data-image semantics.
- [ ] Z-Sobol sampling with the authored 256 samples per pixel on CPU/GPU.
- [ ] Retain declared film filenames and implement linear HDR EXR output.
- [ ] Complete spectral film sample clamping.
  - [x] Retain the authored limit and implement complete-sample CPU/GPU clamping before averaging.
  - [x] Retain full original CIE/D65 tables and use CIE XYZ sensor coordinates for the RGB transport diagnostic.
  - [x] Implement and numerically verify four-wavelength CIE sensor integration and clamping as a shared CPU/GPU API.
  - [ ] Feed actual wavelength-integrated sensor samples into the clamp.
- [x] PBRT's default Gaussian reconstruction filtering, declared box/triangle/Gaussian filters, and CPU/GPU sampling/weighting.
- [x] PBRT path-depth defaults, MIS selection, and terminal-emission semantics for Bistro's surface transport.
- [x] Fit the complete, original-resolution texture set in available GPU memory without reducing precision or image resolution.
- [ ] Bistro-specific parser, CPU, and GPU regressions.
  - [x] Optical-table retention, matrix input, and inline vertex-attribute regressions.
  - [x] Dielectric transport and CPU/GPU render agreement for PATH, NEE, and MIS.
  - [x] PNG raw samples, alpha, palettes, grayscale, 16-bit samples, and all five filters.
  - [x] Normal maps across four material types, mirrored instances, and front/back views.
  - [x] CPU coplanar-candidate, packet visibility, fractional coverage, and scalar-channel cases.
  - [x] GPU alpha traversal, transparent-canopy MIS rendering, and masked-emitter illumination.
  - [x] Emissive BSDFs, sided emission/light PDFs, and continued scattering on CPU/GPU.
  - [x] Depth-zero primary emission/environment, terminal emission, and scattering limits on CPU/GPU.
  - [x] Gaussian reference samples, reconstruction beyond pixel bounds, film/filter order, and filter changes during GPU graph replay.
  - [ ] General texture filtering and spectral rendering.
- [ ] Load all three Bistro camera scenes without discarded supported parameters.
  - [x] Load Café, Boulangerie, and Vespa geometry, images, and active normal maps.
  - [x] Load all three entrypoints with 243 retained opacity bindings each.
  - [x] Honor shape alpha and retain 102 emission bindings with their original BSDFs per scene.
  - [x] Load all three entrypoints with the implicit depth of 5 and MIS transport selection.
  - [x] Load all three entrypoints with the implicit Gaussian filter: radius 1.5 and sigma 0.5.
  - [ ] Honor remaining authored sampling/film settings and audit all remaining parameters.
- [ ] Render and inspect full-scene output; compare with a PBRT reference.

## Implemented behavior

### Geometry and transforms

Inline attributed triangle meshes now use the existing instance geometry path.
Vertex UVs and normals stay attached to their corners when winding is reversed;
instance shading interpolates the attributes and transforms normals with the
inverse transpose. Both indexed and unindexed input work. Attribute-count
mismatches and invalid indices are rejected. The parser accepts `normal N`,
`normal3 N`, `point2 uv`, `float uv`, and `float st`.

PBRT matrix input is converted from column-major serialization to Bajo's affine
row records. This fixes a second full-load blocker discovered after adding
brass: valid Bistro translations were previously rejected as projective
transforms. Matrix composition and genuine projective rejection have regression
coverage. One older numeric-token fixture used the wrong serialization and has
been corrected.

Sources: [PBRT parser](bajo/parser/pbrt/parser.mojo),
[parser regressions](test/parser/pbrt/test_bistro.mojo), and
[GPU regressions](test/rt/gpu/test_bistro_gpu.mojo).

### Normal maps

Material normal-map filenames resolve relative to the declaring file, including
included material files. The image cache distinguishes color encoding from data
encoding, so the same PNG can supply color and raw normal values without
conflating the two interpretations.

Triangle UVs provide a source-space U derivative. CPU hit assembly transforms
that tangent with the instance; GPU preparation stores per-triangle tangents,
and GPU hit assembly transforms them into world space. Both shading paths use
the same tangent-frame normal conversion and bilinear, repeating texture lookup.
Geometric normals remain separate for medium boundaries and surface orientation.

GPU regressions check analytic mapped normals and CPU/GPU agreement for diffuse,
coated diffuse, coated conductor, and dielectric materials, with both mirrored
and ordinary instance transforms and front/back views. All 16 combinations pass.
This covers Bistro's triangle-instance path. Sphere UV/tangent mapping and a
full-scene visual comparison are not covered by those tests.

Sources: [texture math](bajo/rt/texture_math.mojo),
[CPU hit assembly](bajo/rt/cpu/scene.mojo),
[GPU hit assembly](bajo/rt/gpu/scene_trace.mojo), and
[GPU normal-map cases](test/rt/gpu/test_bistro_gpu.mojo).

### Optical spectra

The original PBRT Al, Au, Cu, and CuZn eta/k wavelength tables are included with
their license and source checksum. A piecewise-linear spectrum evaluator
supports scalar and SIMD wavelengths. Imported named tables are shared in
`SurfaceStore.optical_spectra`; conductor and dielectric materials retain
spectral indices, including authored dielectric wavelength curves.

The current rendering path still uses its RGB compatibility projection. Retaining
the tables and accepting brass does **not** complete spectral rendering. The
spectral foundations below have targeted tests; spectral material/light
evaluation, transport, device-table integration, and film input remain required.

Sources: [spectral values](bajo/core/spectrum.mojo),
[original optical tables](bajo/parser/pbrt/optical_data.mojo),
[table generator](scripts/generate_optical_spectra.py), and
[PBRT license](bajo/parser/pbrt/LICENSE-pbrt.txt).

### Spectral foundations and optical device data

Seven CPU and three GPU spectral regressions pass. They verify the foundations
below, including actual GPU scene preparation and device execution. No spectral
Bistro render has been produced; full transport integration remains open.

| Verified foundation | Integration still missing |
| --- | --- |
| `SampledSpectrum` and `SampledWavelengths`: four components per path, visible/uniform sampling, PDFs, and lane-specific idempotent secondary termination | Primary generation and all CPU/GPU transport queues must retain this state |
| Dense original CIE lookup, zero-PDF handling, and PDF-weighted XYZ sensor integration | Actual spectral path contributions must reach the sensor estimator |
| Resolution-64 RGB reconstruction using the unchanged original generator and table; albedo/unbounded/illuminant evaluation matches independent C++ references | Renderer ownership and binding to actual material/light/texture consumers |
| `GpuRgbSpectrumTable`, checked against every uploaded Float32 word and independent spectral references; scalar and SIMD device evaluation | Integrate this owner into render resources and dispatches |
| Packed original optical curves, binary-search interpolation, GPU scene-owned curve buffers and conductor/dielectric spectrum IDs | BSDFs must consume these curves at path wavelengths instead of RGB/scalar projections |
| `FilmResponse.apply_spectral_sample`: CIE integration, exposure, sensor clamp, then RGB output; negative output RGB is preserved | Validated per-light photometric scaling and calls from actual complete-path reduction |

The generated RGB reconstruction asset contains 64 nodes and 2,359,296 Float32
coefficients in a 9,437,456-byte binary file. The manifest records the generator
command and SHA-256 hashes. The independent C++ reference generator uses the
original generated C++ table, not Bajo's binary or evaluator. Cases cover every
maximum-channel axis, ties, low values, black/white/gray, and all three spectrum
roles. D65 normalization and dense spectral-to-RGB round trips also pass. These
checks verify the reconstruction model, not full spectral transport.

The existing CPU accumulation and GPU resolve code still call `apply_sample`
with RGB radiance. Spectral CPU/GPU modules are now registered in the RT runner.
Tests use scalar, four-lane, and sixteen-lane execution and 257 device cases;
prepared-scene checks evaluate retained brass eta/k and wavelength-dependent
dielectric curves plus constant fallbacks. This does not make the current RGB
renderer a spectral renderer.

Sources: [sampled spectra and wavelengths](bajo/core/sampled_spectrum.mojo),
[RGB reconstruction evaluator](bajo/core/rgb_spectrum.mojo),
[packed optical curves](bajo/core/spectrum_tables.mojo),
[table manifest](bajo/core/data/srgb_spectra.json),
[original generator](scripts/pbrt_rgb2spec_opt.cpp),
[binary importer](scripts/generate_rgb_spectra.py),
[independent C++ reference generator](scripts/generate_spectral_references.py),
[GPU upload wrapper](bajo/rt/gpu/spectra.mojo),
[GPU material ownership](bajo/rt/gpu/materials.mojo),
[CPU spectral tests](test/rt/cpu/test_spectra.mojo),
[GPU spectral tests](test/rt/gpu/test_spectra_gpu.mojo),
[film response](bajo/rt/film.mojo),
[CPU sample reduction](bajo/rt/cpu/wavefront/__init__.mojo),
[GPU sample reduction](bajo/rt/gpu/common_kernels.mojo), and
[RT test registration](test/rt/run_tests.mojo).

### Dielectrics

Standalone dielectric materials now retain roughness and use the shared GGX
dielectric interface implementation for reflection and transmission. CPU and GPU
paths use exact Fresnel, matching BSDF/PDF evaluation, and the radiance eta-squared
transmission factor. Direct-light evaluation supports rough dielectric
transmission as well as reflection. Matching-index interfaces remain delta
transmission, and smooth total internal reflection retains its RNG behavior.

Analytic tests cover 60-degree air/glass Fresnel reflectance, entering/leaving
radiance weights, total internal reflection, matching indices, rough sample/PDF
consistency, and masked packet material gathers. A previous test expected unit
transmission weight; its expectation now reflects the physical eta-squared
factor.

Sources: [shared shading](bajo/rt/shading.mojo),
[CPU transport tests](test/rt/cpu/test_dielectric.mojo), and
[GPU render tests](test/rt/gpu/test_bistro_gpu.mojo).

### PNG data textures

An explicit data-decoding path keeps raw normalized samples without sRGB
conversion or alpha compositing. It handles noninterlaced grayscale, RGB,
indexed, grayscale-alpha, and RGBA input at the valid 1/2/4/8/16-bit depths,
including palette transparency. Image textures can retain a separate alpha
plane. Path and memory asset loaders expose `read_data_texture`.

The default `parse_png` API keeps its prior compositing behavior. PBRT path and
memory asset loaders now request preserved alpha and unassociated color;
normal-map bindings use the linear data path. Shape-alpha bindings use the
retained image channels. Interlaced data PNGs currently raise an explicit error;
the inspected Bistro normal assets are noninterlaced.
Color encoding/profile semantics for the complete filtered texture pipeline
still need review.

Sources: [data decoder](bajo/parser/png/data.mojo),
[asset loaders](bajo/parser/asset_loader.mojo), and
[PNG tests](test/parser/png/test_png.mojo).

### Lossless GPU texture storage

GPU image descriptors now identify raw Float32 RGB or a lossless palette format.
When an image contains at most 256 distinct channel bit patterns and the palette
saves memory, its palette stores the original Float32 values and each texel
stores three 8-bit indices in one word. There is no sample quantization,
transfer-function recomputation, or resolution reduction. Images that exceed
the palette capacity roll back to their original Float32 storage.

Nearest and bilinear material lookup, normal-map lookup, environment radiance,
and environment light sampling decode the selected format. Host/device tests
recover every original texel bit for full sRGB tables, arbitrary Float32 values,
signed zero, HDR values, and mixed raw/paletted images. Filtering and addressing
are checked against the CPU image implementation, including negative/repeated
UVs and clamped environment UVs.

Bistro's 263 images use 3,670,288,624 bytes of GPU RGB storage, down from
10,984,956,144 bytes. 242 images are paletted and 21 retain raw Float32 storage.
The complete scene now prepares and renders on the available RTX 5060 Ti.
This resolves the observed texture allocation failure; a general out-of-core
texture system for scenes larger than device capacity is not implemented.

Sources: [lossless packer](bajo/rt/gpu/texture_storage.mojo),
[device texture access](bajo/rt/gpu/textures.mojo),
[host tests](test/rt/cpu/test_texture_storage.mojo),
[device tests](test/rt/gpu/test_texture_storage_gpu.mojo), and
[repeatable GPU diagnostic](scripts/check_bistro_gpu.mojo).

### Coated-conductor reflectance and defaults

The parser retains constant and image-textured `reflectance`, including
`MASTER_Glass_Exterior`. Material records distinguish this parameterization
from explicit eta/k. CPU scalar/SIMD and GPU BSDF setup sample reflectance at
the hit UV, derive eta=1 and k=2 sqrt(R)/sqrt(1-R), and divide both optical
constants by the coating's IOR. Reflectance is capped at 0.9999 for this
conversion, matching PBRT's finite white limit. Reflectance and explicit eta/k
are mutually exclusive. Absent eta or k defaults independently to the original
copper spectrum, which remains retained for future spectral transport.

This implements the parameter rules in
[PBRT's coated-conductor material](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/materials.cpp).
The current transport is still RGB; wavelength-dependent evaluation remains
an explicit separate requirement.

The new render test exposed an existing matching-index interface bug in layered
MIS. Sampling treated eta=1 as delta transmission while value/PDF evaluation
could still treat it as rough. Value/PDF evaluation and layered MIS now share
the same delta-interface condition. The regression verifies that changing
interface roughness leaves the layered value/PDF unchanged when eta=1.

Six host tests pass, covering reflectance recovery, the white limit, conflicting
parameters, copper table retention, scalar/packet optics, parser bindings, and
the matching-index case. The new GPU test checks PATH, NEE, and MIS with both
unit and 1.5 interface IOR; its CPU/GPU comparisons pass after the delta-PDF fix.
All three asset-backed loads also verify the single authored textured
coated-conductor material.

Sources: [optical conversion](bajo/core/optics.mojo),
[material authoring](bajo/rt/material_types.mojo),
[CPU setup](bajo/rt/cpu/bsdf.mojo),
[GPU setup](bajo/rt/gpu/textures.mojo),
[host regressions](test/rt/cpu/test_coated_reflectance.mojo), and
[GPU regressions](test/rt/gpu/test_bistro_gpu.mojo).

### Independent emission and scattering

Area-light declarations now attach emission to each shape or instance instead
of replacing its selected material. Bindings retain radiance and sidedness,
follow attribute scopes, and are validated independently of material storage.
The scattering material keeps its reflectance, normal maps, roughness, and
other parameters. Programmatic emission-only materials remain supported.

CPU hit packets and GPU traversal evaluate emitted light before routing the
same hit through its original BSDF. Emission uses geometric orientation and
opacity, separately from mapped shading normals. Light records store radiance
directly rather than assuming that every emitter has an emission-only material.
Two-sided emitters use both hemispheres, with matching power-selection weights
and sampling/hit PDFs. Secondary emission follows the existing NEE/MIS rules
while reflected/transmitted paths continue through their BSDF queues.

The analytic diffuse-plane regression verifies emitted radiance plus reflected
constant environment illumination, on both sides and for one-/two-sided
emission. CPU/GPU comparisons cover PATH, NEE, and MIS for diffuse, coated
diffuse, coated conductor, and dielectric emitters. Additional tests check
back-facing two-sided illumination, attribute scoping, geometry-kind binding
identity, packet merging, scalar alpha traversal, and invalid records.

All three Bistro loads retain 102 independent emission bindings and no
emission-only replacement materials. Their 20,638 triangle light records retain
the original BSDF IDs. Full-asset GPU diagnostics after this change complete at
192×108, 32 spp, depth 8. Those diagnostics predate the depth, film, and
reconstruction-filter corrections below. Spectral rendering, texture filtering,
authored sampling/output, and final verification remain open.

Sources: [emission bindings](bajo/rt/emission.mojo),
[scene/light construction](bajo/rt/scene_data.mojo),
[GPU emission lookup](bajo/rt/gpu/emission.mojo),
[CPU regressions](test/rt/cpu/test_emissive_bsdf.mojo), and
[GPU regressions](test/rt/gpu/test_emissive_bsdf_gpu.mojo).

### Path depth and terminal emission

PBRT defaults to `volpath` when Bistro omits `Integrator`. Its maximum depth
defaults to 5; both PBRT `path` and `volpath` use combined light/BSDF sampling.
The parser now selects Bajo's MIS transport and depth 5 for the implicit
integrator and explicit `path`/`volpath` declarations without `maxdepth`.
Explicit limits remain intact. Programmatic `RenderSettings` retain their
existing default of 8.

Both renderers now count scattering events rather than tracing iterations.
After the last allowed scattering event, a terminal ray still evaluates surface
emission or the environment with its throughput, medium attenuation, and
emissive-hit MIS weight. That vertex starts neither a direct-light sample nor
another BSDF event. A limit of zero evaluates directly visible emission and
environment illumination while leaving nonemissive surface hits black.

An analytic regression uses two emitting, matching-index transmitting planes
and a constant environment. Limits zero, one, and two expose exactly the first
emitter, both emitters, and both emitters plus the environment, respectively.
Separate diffuse and primary-miss cases check that depth zero cannot acquire
reflected illumination and still sees the environment. All cases pass for
PATH, NEE, and MIS on CPU (scalar and packet) and GPU (chunked and full-capacity).
All three asset-backed Bistro loads also assert depth 5 and MIS selection.

This closes the depth/terminal-emission gap for Bistro, which declares no
participating media. It does not implement general volumetric scattering or
close the remaining spectral film or texture-filtering requirements. Earlier full-asset
diagnostics below predate the depth correction and are intermediate evidence.

Sources: [PBRT defaults](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/scene.cpp),
[PBRT path/volume integrators](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/cpu/integrators.cpp),
[parser](bajo/parser/pbrt/parser.mojo),
[CPU transport](bajo/rt/cpu/wavefront/packet.mojo),
[GPU scheduler](bajo/rt/gpu/resources.mojo),
[GPU transport](bajo/rt/gpu/scene_trace.mojo), and
[analytic tests](test/rt/cpu/test_emissive_bsdf.mojo).

### Film sample accumulation and clamping

The parser retains `maxcomponentvalue`, including Bistro's limit of 20. The
default is positive infinity; negative and NaN limits are rejected. Film ISO is
applied to each complete sample before its sensor-space maximum is tested.
Clamping scales all channels together, preserving their ratios. The linear
output response is then applied before pixel reduction, with no second film
application after reduction.

CPU workers retain complete path radiance in a bounded, reusable buffer per
chunk. All direct-light, emissive-hit, and environment contributions accumulate
by sample ID. Each worker owns complete pixels and reduces their processed
samples in sample order. GPU sample radiance already has independent storage;
its scalar and four-sample reduction paths now process film response/clamping
before summing the pixel. The old post-reduction GPU film launch is removed.

PBRT's default CIE sensor measures XYZ. Complete original CIE 1931 values at
every integer wavelength from 360 through 830 nm and the original D65 table are
retained with their license and source checksum. The RGB diagnostic derives
the sRGB-to-XYZ matrix from those tables and the declared sRGB primaries, rather
than clamping the largest output RGB component. An independent high-precision
calculation supplies matrix regression values; every copied table entry was
also checked against the original PBRT source.

The new tests distinguish complete-path clamping from clamping each light
contribution, and sample clamping from clamping an averaged pixel. They cover
exposure order, channel ratios, the CIE sensor, invalid limits, scalar/packet
CPU rendering, odd/even sample counts, and scalar/vector GPU reduction with
different chunk capacities. Prepared GPU graph replay also tracks changes to
the clamp limit and sensor matrix. All three asset-backed Bistro loads assert
the retained limit of 20.

This implements the sample storage and clamp operation for current RGB
transport. It does **not** complete spectral film support: future spectral
transport must integrate its sampled wavelengths into sensor coordinates
before this nonlinear operation. The spectral film API has numerical CPU/GPU
coverage but remains disconnected from transport. Reconstruction filtering is implemented and
verified separately below; texture-footprint filtering remains unfinished.

Sources: [film response](bajo/rt/film.mojo),
[original CIE/D65 tables](bajo/core/cie_data.mojo),
[colorimetry](bajo/core/colorimetry.mojo),
[table generator](scripts/generate_cie_tables.py),
[CPU film tests](test/rt/cpu/test_film.mojo), and
[GPU film tests](test/rt/gpu/test_film_gpu.mojo).
Reference order: [PBRT RGBFilm::AddSample](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/film.h).

### Reconstruction filters

Bistro omits `PixelFilter`; the parser now supplies PBRT's Gaussian default
with radius 1.5 on each axis and sigma 0.5. Explicit Gaussian, box, and triangle
declarations retain their parameters. Programmatic `RenderSettings` keep the
existing half-pixel box default. Other filter names raise an explicit error;
Mitchell and sinc filters are not implemented and are not requested by Bistro.

Gaussian sampling follows PBRT's tabulated importance sampler: evaluate the
filter at cell centers on a grid with `int(32 * radius)` cells per axis, build
the conditional and marginal CDFs, and sample continuously within the selected
cell. The default grid is 48×48. Rectangular radii and custom sigma work without
a fixed table-size cap. Box and triangle filters use their analytic samplers.
Invalid dimensions, sigma, and empty/nonfinite Gaussian distributions fail
explicitly.

Primary rays use the filter's complete footprint, including film positions
outside the nominal pixel. CPU and GPU share the camera sample routine. Each
sample retains a separate filter weight; film conversion/clamping precedes
weighting, and the final pixel divides its weighted sum by the sum of weights.
GPU sample storage now includes a fourth plane for the weight. CPU filter
tables remain alive until all workers finish; device tables persist with the
render target and update when filter settings change. Graph replay captures
both settings and table addresses.

Reference tests check Gaussian sample positions and table mass, rectangular
grids, analytic filters, packet/scalar agreement, and invalid parameters. A
geometric-edge scene places its light outside the nominal pixel: the half-pixel
box sees zero light, while Gaussian reconstruction matches the independently
integrated reference coverage. Film tests verify that weighting follows
clamping. Device sample tests, CPU/GPU rendering comparisons, changing-filter
graph replay, and the expanded sample-storage contract pass. All three Bistro
loads assert the default filter parameters.

This closes Bistro's reconstruction-filter gap. Texture mipmaps and surface
footprints remain separate requirements; full spectral rendering and authored
sampling/output verification are also unfinished.

Sources: [filter authoring and sampler](bajo/rt/filter.mojo),
[camera samples](bajo/rt/camera_sampling.mojo),
[GPU table ownership](bajo/rt/gpu/filter.mojo),
[CPU regressions](test/rt/cpu/test_filter.mojo), and
[GPU regressions](test/rt/gpu/test_filter_gpu.mojo).
Reference: [PBRT filter definitions](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/filters.h),
[table construction](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/filters.cpp),
and [piecewise-constant sampling](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/sampling.h).

### Shape opacity

The parser stores scalar and image alpha per primitive or instance, independent
of shared mesh and material ownership. CPU traversal evaluates alpha for each
candidate before reducing the nearest hit or taking a shadow early exit. A
rejected candidate does not advance the ray origin or minimum distance, so an
opaque coplanar candidate remains eligible. Fractional alpha uses deterministic
stochastic acceptance shared by primary and shadow traversal.

Scalar image evaluation uses a nontrivial alpha channel when present and the RGB
average otherwise. Opacity uses repeating bilinear interpolation and retained UV
scales. GPU preparation uploads compact scalar images and binding tables;
filtered wide/compressed triangle BLAS, instance TLAS, sphere traversal, primary
rays, and shadow rays have implementations in the working tree.

CPU/GPU light sampling now interpolates each emitter triangle's UVs and tests
opacity at the sampled point. UV corners follow geometry corners when mirrored
instances require a winding correction. Rejected samples contribute nothing;
accepted samples keep the original unmasked selection/area PDF. Emissive hits
also apply position-stable stochastic alpha. Emissive-hit PDF evaluation uses
the geometric normal. This follows the separation between geometric alpha and
emitted-radiance masking in
[PBRT's diffuse area light](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/lights.h)
and [light sampling](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/lights.cpp).

Six GPU tests pass: coplanar flat triangles, textured coincident triangles in one
mesh, fractional coverage matching CPU decisions for identical rays, coplanar
spheres, masked-emitter MIS renders, and a fully transparent canopy that
preserves the baseline MIS render. They cover wide/compressed triangle BLAS,
TLAS2 and wide TLAS, and wide/compact sphere trees. GPU test kernel arguments
now use a fixed-width count, and filtered any-hit callbacks set the required
shadow-hit marker. The initial compilation and shadow-visibility failures are
resolved. A sphere fixture also needed preparation in its requested BVH format.

CPU tests cover emitter coverage/PDFs and emission masking in addition to
closest-hit/shadow rejection. All three Bistro scenes load with 243 opacity
bindings each. Mipmapped footprint filtering and a full-scene PBRT comparison
remain required separately.

Sources: [opacity records](bajo/rt/opacity.mojo),
[CPU traversal](bajo/rt/cpu/opacity.mojo),
[GPU traversal](bajo/rt/gpu/opacity.mojo),
[light records](bajo/rt/lighting_types.mojo),
[asset-backed load check](scripts/check_bistro_load.mojo),
[CPU tests](test/rt/cpu/test_opacity.mojo), and
[GPU tests](test/rt/gpu/test_opacity_gpu.mojo).

## Scene and runtime evidence

- Assets contain 132 materials (123 coated diffuse, 6 dielectric, and 3 coated
  conductor) and 131 active normal-map declarations (the
  Vespa headlight normal-map line is commented out), 243 alpha-masked shapes,
  850 PLY mesh declarations, 741 inline attributed triangle meshes, and
  102 area-light declarations attached to named scattering materials.
- The initial load failed on `metal-CuZn-eta`, followed by a matrix-layout error.
  Both blockers are resolved.
- `pixi run mojo -I . examples/viewer_scene.mojo --pbrt assets/bistro/bistro_cafe.pbrt`
  completed with 1,591 meshes and 1,591 instances and camera position
  `[-11.7, 2.5, 6.5]`. This inspected scene data without preparing a CPU BVH.
- That load still reported independent sampling for the authored `zsobol`, and
  it preceded the normal-map implementation. Successful ingestion alone is not
  evidence of complete scene support.
- A subsequent parser probe loaded Café, Boulangerie, and Vespa in sequence with
  all active normal maps enabled. Each contains 1,591 meshes, 1,591 instances,
  131 normal-map bindings, 263 images, and two retained brass spectra. The probe
  was run with `pixi run mojo -I . /tmp/bajo_inspect_bistro.mojo` and the three
  scene paths; the temporary probe is not a committed regression fixture.
- An asset-backed check after opacity integration loaded all three entrypoints
  successfully. Each retains 243 opacity bindings, alongside the same 1,591
  meshes/instances, 131 normal-map bindings, 263 images, and two spectra. Each
  contains 2,832,101 source triangles and 20,638 triangle light records. Run
  `pixi run mojo -I . scripts/check_bistro_load.mojo` to repeat the assertions.
  Remaining material and sampling/film parameters listed above are unfinished.
  No render with complete authored semantics/settings or PBRT reference
  comparison is verified yet.
- GPU access requires running outside the sandbox. An escalated `nvidia-smi`
  identifies an RTX 5060 Ti with 16 GiB VRAM. After reconstruction-filter
  integration, all 71 GPU RT tests across sixteen modules passed in two runs on
  that device, including actual rendering, CPU/GPU pixel comparisons, sample
  storage, primary grid caps, queue ordering, and prepared graph replay.
- A CPU diagnostic rendered Café's complete 2,832,101-triangle geometry with
  alpha enabled using
  `pixi run mojo -I . examples/pbrt_scene.mojo assets/bistro/bistro_cafe.pbrt /tmp/bajo_bistro_cafe_alpha_cpu.ppm 96 4 4`.
  It completed with finite pixels. The tiny 96×54 image was inspected after a
  format-only PNG conversion; 4 spp is too noisy to assess material fidelity
  or claim a finished render. The overrides are diagnostic settings, not the
  authored target.
- The corresponding full-asset GPU probe failed while allocating its
  10,984,956,144-byte RGB texture buffer, before rendering. Its allocator
  reported an 8.25 GiB chunk limit and the direct allocation also failed.
  A subsequent device query showed 9,281 MiB free out of 16,311 MiB; other
  applications occupied part of VRAM. This is an observed capacity failure,
  not evidence that Bistro can never fit on a 16 GiB device. The subsequent
  lossless storage implementation resolves this failure.
- The full-asset GPU diagnostic now completes with original texture samples and
  resolution. `pixi run mojo -I . scripts/check_bistro_gpu.mojo assets/bistro/bistro_cafe.pbrt /tmp/bajo_bistro_cafe_lossless_gpu.ppm 192 32 8`
  prepares every mesh/instance, reports the 3,670,288,624-byte texture allocation,
  renders finite RGB pixels, and writes PPM plus raw Float32 RGB. The image was
  inspected after a format-only PNG conversion. Buildings, pavement, foliage,
  and the camera composition are visible; Monte Carlo noise remains substantial.
  This is a diagnostic with sampling/resolution/depth overrides and the remaining
  semantic gaps listed above, not a completed faithful reference render.
- After coated-conductor reflectance integration, all three entrypoints also
  completed 192×108, 32 spp, depth-8 full-asset GPU diagnostics through
  `scripts/check_bistro_gpu.mojo`, producing finite pixels. Their filenames are
  `/tmp/bajo_bistro_cafe_coated_gpu.ppm`,
  `/tmp/bajo_bistro_boulangerie_coated_gpu.ppm`, and
  `/tmp/bajo_bistro_vespa_coated_gpu.ppm`, with companion `.ppm.rgb` buffers.
  These were intermediate diagnostics before independent emission/BSDF support.
  The later eta=1 layered-MIS fix does not affect Bistro's
  authored coated-conductor interface IORs, but final rendering verification
  must still use the final implementation.
- After independent emission/BSDF integration, all three full-asset GPU
  diagnostics again completed at 192×108, 32 spp, depth 8, with finite pixels
  and the same lossless texture storage. Outputs are
  `/tmp/bajo_bistro_cafe_emissive_gpu.ppm`,
  `/tmp/bajo_bistro_boulangerie_emissive_gpu.ppm`, and
  `/tmp/bajo_bistro_vespa_emissive_gpu.ppm`, each with a companion `.ppm.rgb`.
- After depth and per-sample film integration, Café completed a full-asset GPU
  diagnostic at 96×54, 4 spp, depth 5 with the authored sample limit of 20.
  `pixi run mojo -I . scripts/check_bistro_gpu.mojo assets/bistro/bistro_cafe.pbrt /tmp/bajo_bistro_cafe_film_gpu.ppm 96 4 5`
  wrote finite pixels to PPM and a companion `.ppm.rgb` buffer. The PPM was
  inspected after a format-only PNG conversion; the image is very noisy and
  too small to assess material fidelity. This remains an RGB diagnostic with
  explicit overrides, not an authored-settings or spectral reference render.
- After reconstruction-filter integration, the same Café diagnostic completed
  with the actual default Gaussian filter at 96×54, 4 spp, depth 5, preserving
  all original geometry and texture samples. Outputs are
  `/tmp/bajo_bistro_cafe_filter_gpu.ppm` and its `.ppm.rgb` buffer. The pixels
  are finite and nonzero; its format-only PNG was inspected. It remains too
  small and noisy to establish material fidelity or reference agreement.

## Verified checks

The latest recorded CPU/PBRT and GPU runs below include independent emission/BSDF
support, the depth correction, per-sample film processing, and reconstruction
filtering. They predate the new spectral helpers and subsequent `FilmResponse`
changes, so they are historical evidence, not a passing result for the current
working tree. Earlier rows record narrower checks from their respective
integration stages; their test counts describe those runs, not the current
suite size. No renderer tests were rerun for this documentation update.

| Command | Result | Scope |
| --- | --- | --- |
| `pixi run bash scripts/run_tests.sh test/parser/png/test_png.mojo test/parser/pbrt` | 56 tests passed across 5 modules | 10 Bistro, 6 BMW, 5 Crown, 21 existing PBRT, 14 PNG tests |
| `pixi run bash scripts/run_tests.sh test/parser/pbrt/test_bistro.mojo test/rt/cpu/test_dielectric.mojo` | 16 tests passed across 2 modules | Includes normal-map path/encoding, tangent orientation, and bilinear scalar/SIMD checks |
| `pixi run bash scripts/run_tests.sh test/rt/cpu/test_instance_shading_simd.mojo test/rt/cpu/test_gpu_material_layout.mojo test/rt/cpu/test_crown.mojo test/rt/cpu/test_cpu.mojo` | 60 tests passed across 4 modules | Existing instance, material ABI, Crown, and CPU renderer behavior after normal-map integration |
| `pixi run bash scripts/run_tests.sh test/rt/gpu/test_bistro_gpu.mojo` | 3 tests passed on the RTX 5060 Ti | Rough-glass PATH/NEE/MIS agreement, inline-mesh normals, and 16 normal-map material/transform/view combinations |
| `pixi run bash scripts/run_tests.sh test/rt/cpu/test_opacity.mojo test/parser/png/test_png.mojo test/parser/pbrt/test_bistro.mojo` | 29 tests passed across 3 modules after opacity integration | 5 CPU opacity, 14 PNG, and 10 Bistro parser tests |
| `pixi run bash scripts/run_tests.sh test/rt/cpu/test_opacity.mojo test/rt/cpu/test_instance_shading_simd.mojo test/rt/cpu/test_cpu.mojo test/parser/pbrt test/parser/png/test_png.mojo` | 113 tests passed across 8 modules after emitter integration | 9 CPU opacity/emitter, 2 instance packet, 46 CPU renderer, and 56 parser/PNG tests |
| `pixi run bash scripts/run_tests.sh test/rt/gpu/test_opacity_gpu.mojo test/rt/gpu/test_bistro_gpu.mojo` | 9 GPU tests passed after opacity/emitter integration | 6 opacity tests plus rough dielectric, inline normals, and all 16 normal-map combinations |
| `pixi run bash scripts/run_tests.sh test/bvh/gpu/test_triangle_shadow.mojo test/bvh/gpu/test_tlas_visibility.mojo test/bvh/gpu/test_cwbvh8_node_tasks.mojo` | 12 GPU tests passed | Existing triangle shadow masks, 10 TLAS visibility/closest-hit cases, and compressed-node slab boundaries |
| `pixi run mojo -I . scripts/check_bistro_load.mojo` | All 3 scenes passed after reconstruction-filter integration | Geometry/image/normal-map/spectrum counts, 243 image opacity bindings and 102 emission bindings per scene, original BSDFs on all 20,638 light records, resolution, sample count, depth 5, MIS selection, film sample limit 20, and Gaussian radius 1.5 / sigma 0.5 |
| `pixi run bash scripts/run_tests.sh test/rt/cpu/test_texture_storage.mojo test/rt/cpu/test_gpu_material_layout.mojo test/rt/cpu/test_cpu.mojo` | 55 tests passed after lossless storage integration | Exact texel reconstruction, mixed storage/addressing/filtering, material layout, and CPU renderer regressions |
| `pixi run bash scripts/run_tests.sh test/rt/gpu/test_texture_storage_gpu.mojo test/rt/gpu/test_bistro_gpu.mojo test/rt/gpu/test_opacity_gpu.mojo` | 10 GPU tests passed after lossless storage integration | Bit-exact device texels, material/environment addressing/filtering, all earlier Bistro shader and opacity cases |
| `pixi run bash scripts/run_tests.sh test/rt/cpu/test_coated_reflectance.mojo test/rt/cpu/test_coated_simd.mojo test/rt/cpu/test_dielectric.mojo test/rt/cpu/test_bmw_materials.mojo test/rt/cpu/test_coated_diffuse.mojo` | 25 tests passed after matching-index fix | Reflectance/defaults, scalar/SIMD coating, dielectric transport, conductor optics, and matching-index layered value/PDF invariance |
| `pixi run bash scripts/run_tests.sh test/rt/gpu/test_bistro_gpu.mojo test/rt/gpu/test_texture_storage_gpu.mojo test/rt/gpu/test_opacity_gpu.mojo test/rt/gpu/test_coated_shading_gpu.mojo test/rt/gpu/test_bmw_materials_gpu.mojo` | 14 GPU tests passed after final matching-index corrections | Reflectance PATH/NEE/MIS at two interface IORs, earlier Bistro shader/opacity cases, exact texture decoding, coating sample parity, and mixed-material/chunk/batch invariance |
| `pixi run bash scripts/run_tests.sh test/rt/cpu` | All 125 tests passed across 17 modules after lossless storage, coated reflectance, and matching-index fixes | Complete CPU RT subsystem, including material ABI, packets, media, sampling, continuation, and the new feature regressions |
| `pixi run bash scripts/run_tests.sh test/rt/cpu test/parser/pbrt` | All 172 tests passed across 22 modules after independent emission/BSDF integration | 130 CPU RT tests and 42 PBRT parser tests; includes emission plus reflection, sided light PDFs, scope/validation, packet merging, and geometry binding identity |
| `pixi run bash scripts/run_tests.sh test/rt/gpu/test_emissive_bsdf_gpu.mojo test/rt/gpu/test_opacity_gpu.mojo test/rt/gpu/test_bistro_gpu.mojo test/rt/gpu/test_gpu.mojo test/rt/gpu/test_bmw_materials_gpu.mojo test/rt/gpu/test_coated_shading_gpu.mojo test/rt/gpu/test_texture_storage_gpu.mojo` | All 43 tests passed across 7 modules after independent emission/BSDF integration | 4 independent-emission, 6 opacity, 4 Bistro, 25 general GPU, 1 BMW, 2 coating, and 1 lossless-texture test |
| `pixi run bash scripts/run_tests.sh test/rt/gpu/test_emissive_bsdf_gpu.mojo test/rt/gpu/test_gpu.mojo` | All 30 tests passed across 2 modules after depth correction | Analytic terminal emission at depths 0/1/2, no terminal scattering/direct-light sample, chunk invariance, independent emission, and all general GPU rendering regressions |
| `pixi run bash scripts/run_tests.sh test/rt/cpu test/parser/pbrt` | All 174 tests passed across 22 modules after depth correction | 131 CPU RT tests and 43 PBRT parser tests; includes explicit/default limits, MIS selection, terminal emission, packet widths, all existing transport, and material/opacity regressions |
| `pixi run bash scripts/run_tests.sh test/rt/gpu/test_bistro_gpu.mojo test/rt/gpu/test_opacity_gpu.mojo test/rt/gpu/test_texture_storage_gpu.mojo test/rt/gpu/test_coated_shading_gpu.mojo test/rt/gpu/test_bmw_materials_gpu.mojo` | All 14 tests passed across 5 modules after depth correction | Bistro normal maps, dielectric/coated-conductor transport, alpha traversal and emitters, exact texture storage, layered shading, and mixed-material CPU/GPU agreement |
| `pixi run bash scripts/run_tests.sh test/rt/cpu test/parser/pbrt` | All 178 tests passed across 23 modules after film integration | 135 CPU RT tests and 43 PBRT parser tests; complete-sample film clamping, independent CIE matrix values, all existing transport/material/opacity tests, and authored/default settings |
| `pixi run bash scripts/run_tests.sh test/rt/gpu/test_film_gpu.mojo test/rt/gpu/test_crown_gpu.mojo test/rt/gpu/test_emissive_bsdf_gpu.mojo test/rt/gpu/test_gpu.mojo test/rt/gpu/test_bistro_gpu.mojo test/rt/gpu/test_opacity_gpu.mojo test/rt/gpu/test_texture_storage_gpu.mojo test/rt/gpu/test_coated_shading_gpu.mojo test/rt/gpu/test_bmw_materials_gpu.mojo test/rt/gpu/test_target_capacity_gpu.mojo` | All 53 tests passed across 10 modules after film integration | Complete-sample scalar/vector reduction, chunk capacities, ISO/media response, all earlier Bistro material/opacity/emission cases, general rendering, and graph capture/replay with changing film limits and sensor matrices |
| `pixi run bash scripts/run_tests.sh test/rt/cpu/test_filter.mojo test/rt/cpu/test_film.mojo test/parser/pbrt/test_pbrt.mojo` | All 31 tests passed across 3 modules after reconstruction-filter integration | 5 filter, 4 film, and 22 PBRT parser tests; includes reference Gaussian sampling, nonlocal reconstruction, host owner lifetime, and clamp-before-filter behavior |
| `pixi run bash scripts/run_tests.sh test/rt/gpu/test_filter_gpu.mojo test/rt/gpu/test_film_gpu.mojo test/rt/gpu/test_target_capacity_gpu.mojo test/rt/gpu/test_wavefront_contract_gpu.mojo` | All 9 tests passed across 4 modules after reconstruction-filter integration | Device sampler agreement, edge reconstruction, film reduction, changing-filter graph replay, and four-plane sample storage |
| `pixi run bash scripts/run_tests.sh test/rt/gpu/test_bistro_gpu.mojo test/rt/gpu/test_bmw_materials_gpu.mojo test/rt/gpu/test_coated_shading_gpu.mojo test/rt/gpu/test_crown_gpu.mojo test/rt/gpu/test_cwbvh_call_gpu.mojo test/rt/gpu/test_delta_metal_gpu.mojo test/rt/gpu/test_emissive_bsdf_gpu.mojo test/rt/gpu/test_gpu.mojo test/rt/gpu/test_opacity_gpu.mojo test/rt/gpu/test_primary_launch_gpu.mojo test/rt/gpu/test_ray_order_gpu.mojo test/rt/gpu/test_texture_storage_gpu.mojo` | All 62 tests passed across 12 modules after reconstruction-filter integration | Every remaining GPU RT module: all materials, opacity and emitters, media/ISO, traversal call boundaries, delta reflection, all integrators, exact texture storage, primary grid caps, and queue ordering |
| `pixi run bash scripts/run_tests.sh test/rt/cpu test/parser/pbrt` | All 183 tests passed across 24 modules after reconstruction-filter integration | Complete CPU RT subsystem (140 tests) and PBRT parser (43 tests), including filter ownership, sampling/weighting, all prior features, packets, scheduling, and material ABI checks |

Additional targeted BMW, Crown, coated-diffuse, and SIMD sampling modules passed
during dielectric implementation. A repository-wide host/GPU suite has not
been run; the complete GPU RT subsystem has passed after filter integration.
The data-decoder unused-assignment warning has been fixed. GPU Bistro shader
regressions now pass after opacity traversal and emitter integration. One
normal-map fixture initially assumed that mirroring reverses the authored
outward normal; that expectation was corrected to match the inverse-transpose
orientation convention, and explicit back-face views now test the actual flip.
All eight generated optical tables were also compared numerically with their
original PBRT source arrays; every wavelength and value matched.

## Next implementation work

1. Connect spectral material/texture/light evaluation and four-wavelength path
   state throughout CPU/GPU transport. Feed complete spectral samples to the
   CIE sensor before clamping, output conversion, filter weighting, and reduction.
   Validate per-light photometric normalization with independent path references.
2. Add texture footprints, mipmaps, and the required material/environment
   filtering; verify color versus scalar/data semantics for each consumer.
3. Implement Z-Sobol, retain film filenames, and write the declared EXR outputs.
4. Audit every authored parameter and implicit default, then render all three
   cameras on CPU/GPU at 1920×1080 and 256 spp with the implicit depth of 5 and
   Gaussian radius 1.5 / sigma 0.5. Preserve each camera's lens settings, ISO,
   sample clamp, and environment settings. Inspect converged linear HDR output
   and compare with PBRT before claiming completion.

| Entrypoint | Declared output | ISO | Camera settings to preserve |
| --- | --- | --- | --- |
| `bistro_cafe.pbrt` | `bistro_cafe.exr` | 110 | FOV 53° and authored transform |
| `bistro_boulangerie.pbrt` | `bistro_boulangerie.exr` | 40 | FOV 60° and authored transform |
| `bistro_vespa.pbrt` | `bistro_vespa.exr` | 110 | FOV 45°, focal distance 4.5, lens radius 0.02, and authored transform |

All three declare `maxcomponentvalue = 20`, `Sampler "zsobol"` with 256 spp,
and an infinite light using `textures/sky.exr` at scale 40 with its authored
rotations. These settings must survive the final parser-to-renderer audit.
