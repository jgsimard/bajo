# Bistro implementation and verification

Goal: render the checked-in/downloaded PBRT Bistro scene completely on CPU and
GPU, preserving its authored geometry, materials, lights, sampling, and film
settings. Do not replace missing behavior with material substitutions, dropped
parameters, or RGB stand-ins for spectral transport.

Bistro support is partial. RGB and opt-in spectral CPU/GPU transport, physical
CIE/Canon sensors, and reconstruction filters have regression coverage. Texture
footprints, EXR output, and full authored-settings fidelity remain unverified.
See [spectral rendering](#end-to-end-spectral-rendering) for usage and scope.

## Confirmed remaining gaps

These are implementation requirements, not proposed visual approximations.
The source links identify the remaining integration points. Passing component
regressions does not establish complete-scene fidelity.

| Missing behavior | Evidence and required change |
| --- | --- |
| Texture footprints and filtering | Normal and opacity maps use repeating bilinear lookups; general material/environment lookups still use the existing sampling paths. There is no complete mipmap/footprint pipeline. Propagate derivatives or ray differentials, build mip levels, and implement the required filtering on CPU and GPU. Verify scalar-channel and color/data decoding semantics across every consumer. |
| Declared EXR output | Film filenames are `bistro_cafe.exr`, `bistro_boulangerie.exr`, and `bistro_vespa.exr`. The parser does not retain the filename, and `image_output.mojo` provides raw Float32 RGB and PPM output, not an EXR writer. Retain output settings and write the requested linear HDR image. |
| Complete-scene verification | All three entrypoints load with 243 opacity bindings and 102 independent emission bindings each. Full-asset RGB diagnostics render all three cameras on GPU at 192×108, 32 spp, depth 8; an earlier Café diagnostic also rendered on CPU. Authored-resolution/settings renders, a complete parameter audit, converged image inspection, and PBRT comparison remain unverified. |

Film ISO is already read and applied through `FilmResponse`: Café and Vespa use
110, and Boulangerie uses 40. Resolution and sample count are also retained;
they are not missing parser features. Spectral sensor integration is implemented
in the opt-in spectral mode.

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
- [x] Spectral material/light evaluation, wavelength transport, and film integration.
  - [x] Verify four-wavelength arithmetic, wavelength sampling/PDFs, and secondary-wavelength termination on CPU/GPU.
  - [x] Verify the original-resolution RGB-to-spectrum evaluator against the original PBRT table, including reflectance, unbounded, and illuminant cases.
  - [x] Pack original optical curves and bind their material IDs in GPU scene preparation; verify device evaluation.
  - [x] Bind retained optical/reconstruction spectra to material, texture, light, and environment evaluation on CPU/GPU.
  - [x] Carry wavelength/PDF state and four-component throughput/radiance through every transport queue and contribution.
  - [x] Implement and verify spectral light normalization, dispersive transmission, MIS consistency, and actual sensor integration.
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
- [x] Z-Sobol sampling with the authored 256 samples per pixel on CPU/GPU.
- [ ] Retain declared film filenames and implement linear HDR EXR output.
- [x] Complete spectral film sample clamping.
  - [x] Retain the authored limit and implement complete-sample CPU/GPU clamping before averaging.
  - [x] Retain full original CIE/D65 tables and use CIE XYZ sensor coordinates for the RGB transport diagnostic.
  - [x] Implement and numerically verify four-wavelength CIE sensor integration and clamping as a shared CPU/GPU API.
  - [x] Feed actual wavelength-integrated sensor samples into the clamp.
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
  - [x] Spectral rendering and physical CIE/Canon integration.
  - [ ] General texture filtering.
- [ ] Load all three Bistro camera scenes without discarded supported parameters.
  - [x] Load Café, Boulangerie, and Vespa geometry, images, and active normal maps.
  - [x] Load all three entrypoints with 243 retained opacity bindings each.
  - [x] Honor shape alpha and retain 102 emission bindings with their original BSDFs per scene.
  - [x] Load all three entrypoints with the implicit depth of 5 and MIS transport selection.
  - [x] Load all three entrypoints with the implicit Gaussian filter: radius 1.5 and sigma 0.5.
  - [ ] Honor declared film output settings and audit all remaining parameters.
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

RGB remains the compatibility default. Opt-in spectral transport samples these
curves at the camera sample's actual wavelengths on CPU and GPU. Dielectric
IOR curves select a primary-wavelength direction and terminate secondary
wavelengths exactly once.
Sources: [spectral values](bajo/core/spectrum.mojo),
[original optical tables](bajo/parser/pbrt/optical_data.mojo),
[table generator](scripts/generate_optical_spectra.py), and
[PBRT license](bajo/parser/pbrt/LICENSE-pbrt.txt).

### End-to-end spectral rendering

`RenderSettings.color_mode = .SPECTRAL` and `--color-mode spectral` select actual
four-wavelength transport. Scalar/RGB/curve/blackbody sources and texture products
survive authoring and preparation. Lazy renderer-owned reconstruction resources
feed spectral BSDFs, emission, NEE/MIS, and homogeneous absorption. Complete
sample radiance reaches CIE 1931 or measured Canon sensor integration before
exposure, common sensor-channel clamping, output conversion, and filtering.
RGB remains the default with its original streams and queue layouts.

For programmatic renders, set `settings.color_mode = .SPECTRAL`. Low-level GPU
callers must enable scene resources with `scene.enable_spectral(ctx)` before
`enqueue_render_gpu`; convenience entrypoints and the prepared renderer handle
this automatically. Retain GPU owners until asynchronous work completes.
Changing color mode resets prepared progressive accumulation and graph capture.

`settings.film.sensor_kind` selects `.CIE1931` or `.CANON_EOS_5D_MKIV`.
PBRT sensor and white-balance declarations are retained. Use
`film.set_output_transform(red, green, blue)` for user output transforms,
independently of Canon calibration. Output remains linear RGB, including
negative values. Coating interfaces currently require a constant float IOR;
unsupported spectral declarations fail explicitly.

The reconstruction table is loaded once per owning spectral renderer/workspace.
Compile with an absolute import path for working-directory-independent package
lookup, bundle `srgb_spectra.bin` beside a portable Linux executable, or set
`BAJO_SPECTRUM_TABLE`. Missing resources fail before tracing.

Independent original-source references cover reconstruction, wavelength PDFs,
physical sensor calibration, and integration. Additional transport tests cover
sample IDs, scalar/SIMD execution, chunks, progressive offsets, graph replay,
mode changes, and absorption with dispersion. Controlled eight-seed PBRT image
comparisons retain the agreed error tolerance; the analytical dispersive sphere
provides an independent reference where PBRT's volumetric estimator differs.

Regression coverage lives in the
[CPU transport tests](test/rt/cpu/test_spectral_transport.mojo) and
[GPU transport tests](test/rt/gpu/test_spectral_transport_gpu.mojo).
The [sensor tables](bajo/core/sensor_data.mojo) retain the measured data's source
hashes. Reference values in the CPU transport tests were computed independently
from the original PBRT sources using NumPy.

At 192×108, 32 spp, depth five and Z-Sobol, all three Bistro cameras passed
CPU/GPU integration checks. Median warmed spectral render time was 1.09–1.18×
RGB on a Ryzen 7 9700X and 1.61–1.69× on an RTX 5060 Ti. Those measurements do
not establish performance for other scenes or complete Bistro fidelity.

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
the palette capacity retain their original Float32 storage. Raw images copy
directly into pinned upload buffers; packing does not flatten them into an
additional RGB array.

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
copper spectrum, evaluated at the sampled wavelengths in spectral mode.

This implements the parameter rules in
[PBRT's coated-conductor material](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/materials.cpp).
RGB uses its compatibility projection; spectral mode derives these parameters
from sampled reflectance or the retained eta/k curves.

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
authored output, and final verification remain open.

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
close the remaining texture-filtering requirements. Earlier full-asset
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

RGB and spectral transport both reduce complete camera samples. Spectral mode
integrates the actual wavelengths against the selected physical sensor first;
it does not reuse the RGB Canon approximation. Reconstruction filtering is
implemented separately below; texture-footprint filtering remains unfinished.

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
footprints remain separate requirements; authored-resolution convergence and
output verification are also unfinished.

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
rays, and shadow rays use the same opacity acceptance rule.

CPU/GPU light sampling now interpolates each emitter triangle's UVs and tests
opacity at the sampled point. UV corners follow geometry corners when mirrored
instances require a winding correction. Rejected samples contribute nothing;
accepted samples keep the original unmasked selection/area PDF. Emissive hits
also apply position-stable stochastic alpha. Emissive-hit PDF evaluation uses
the geometric normal. This follows the separation between geometric alpha and
emitted-radiance masking in
[PBRT's diffuse area light](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/lights.h)
and [light sampling](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/lights.cpp).

GPU regressions cover coplanar flat triangles, textured coincident triangles in
one mesh, fractional coverage matching CPU decisions for identical rays,
coplanar spheres, masked-emitter MIS renders, and a fully transparent canopy
that preserves the baseline MIS render. They exercise wide/compressed triangle
BLAS, TLAS4 and wide-leaf TLAS, and wide/compact sphere trees.

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

### Z-Sobol sampling

The PBRT parser now selects `Sampler.Z_SOBOL` for `Sampler "zsobol"`, retains
its sample count, and honors the signed 32-bit `seed` parameter (default 0).
The native viewer exposes the same sampler through `--sampler zsobol` and P.
Existing sampler IDs remain unchanged; Z-Sobol is ID 6.

CPU SIMD and GPU code use PBRT's hierarchical Morton-index permutation and
Fast Owen scrambling. Adjacent random dimensions share a two-dimensional net;
path-stage salts distinguish camera, scattering, light, material, and roulette
streams. Full image width/height and sequence length are retained across batches,
including portrait images and nonzero sample offsets. Non-power-of-two sequence
lengths take a prefix of the next power-of-two domain without changing the
requested number of samples. Pixel/sample indices are limited to 52 bits.

The default `fastowen` randomization is implemented; other explicitly requested
randomizations are rejected. The renderer keeps its existing path-stage dimension
allocation, so identical whole-image sample order to PBRT is not claimed.
Independent scalar C++ fixtures cover 108 index/sample pairs, including odd
sample exponents, high index bits, and signed seed bit patterns. CPU/GPU tests
check exact reference bits, two-dimensional stratification, SIMD agreement,
partial batches, rendered-image agreement, and sample-batch invariance.

Sources: [Z-Sobol permutation](bajo/core/zsobol.mojo),
[shared sampler integration](bajo/core/random.mojo),
[independent C++ reference generator](scripts/generate_zsobol_references.py),
[CPU references and stratification](test/core/test_sampler.mojo), and
[GPU references and streams](test/rt/gpu/test_zsobol_gpu.mojo).

## Verification

Run the parser and CPU rendering regressions with:

```sh
pixi run bash scripts/run_tests.sh test/parser test/rt/cpu
```

The Bistro-specific GPU checks cover material and normal-map agreement,
filtered opacity traversal, independent emission, film/filter processing,
lossless texture storage, and retained render targets:

```sh
pixi run bash scripts/run_tests.sh \
  test/rt/gpu/test_bistro_gpu.mojo \
  test/rt/gpu/test_opacity_gpu.mojo \
  test/rt/gpu/test_emissive_bsdf_gpu.mojo \
  test/rt/gpu/test_film_gpu.mojo \
  test/rt/gpu/test_filter_gpu.mojo \
  test/rt/gpu/test_texture_storage_gpu.mojo \
  test/rt/gpu/test_target_capacity_gpu.mojo
pixi run bash scripts/run_tests.sh test/rt/cpu/test_spectra.mojo test/rt/gpu/test_spectra_gpu.mojo
```

Once the original Bistro assets are available under `assets/bistro`, check all
three entrypoints and generate a small GPU diagnostic with:

```sh
pixi run mojo -I . scripts/check_bistro_load.mojo assets/bistro
pixi run mojo -I . scripts/check_bistro_gpu.mojo assets/bistro/bistro_cafe.pbrt /tmp/bistro_cafe.ppm 96 4 4
```

The load check verifies retained geometry, texture/normal-map/opacity/emission
bindings, optical curves, and supported scene settings. The GPU diagnostic
reports texture storage and retained optical curves, checks finite output, and
writes a PPM plus a raw linear RGB buffer. Its resolution, sample count, and
depth are explicit diagnostic overrides. Append `--color-mode spectral` to the
existing GPU diagnostic to select spectral transport. These checks do not
establish agreement with PBRT at the full authored settings.

## Next implementation work

1. Add texture footprints, mipmaps, and the required material/environment
   filtering; verify color versus scalar/data semantics for each consumer.
2. Retain film filenames and write the declared EXR outputs.
3. Audit every authored parameter and implicit default, then render all three
   cameras on CPU/GPU at 1920×1080 and 256 spp with the implicit depth of 5 and
   Gaussian radius 1.5 / sigma 0.5. Preserve each camera's lens settings, ISO,
   sample clamp, and environment settings. Inspect converged linear HDR output
   and compare with PBRT before claiming complete Bistro support.

| Entrypoint | Declared output | ISO | Camera settings to preserve |
| --- | --- | --- | --- |
| `bistro_cafe.pbrt` | `bistro_cafe.exr` | 110 | FOV 53° and authored transform |
| `bistro_boulangerie.pbrt` | `bistro_boulangerie.exr` | 40 | FOV 60° and authored transform |
| `bistro_vespa.pbrt` | `bistro_vespa.exr` | 110 | FOV 45°, focal distance 4.5, lens radius 0.02, and authored transform |

All three declare `maxcomponentvalue = 20`, `Sampler "zsobol"` with 256 spp,
and an infinite light using `textures/sky.exr` at scale 40 with its authored
rotations. These settings must survive the final parser-to-renderer audit.
