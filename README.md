# bajo

`bajo = low (...) level = batch mojo`

Bajo is a work-in-progress, pure-Mojo engine for high-throughput geometry and rendering workloads. The long-term goal is to support many concurrent reinforcement-learning, spatial-computing, and physics environments on a GPU.

The current implementation is focused on the foundations needed for that goal: math and geometry primitives, OBJ/MTL and PBRT scene loading, CPU and GPU BVHs, GPU sorting, and CPU/GPU path tracing. Particle simulation, fluids, rigid bodies, and batched environments are planned work. See the detailed [roadmap](roadmap.md) for the current status of individual components.

## Requirements

- Linux x86-64
- [Pixi](https://pixi.sh/)
- A supported accelerator for GPU tests, GPU examples, and GPU benchmarks

Pixi provides the pinned nightly Mojo, MAX, and Python dependencies declared in
[`pixi.toml`](pixi.toml):

```bash
pixi install
```

For the native raylib window bindings, install the pinned [raylib 6.2 release](https://github.com/raysan5/raylib/releases/tag/6.2):

```bash
pixi run install_raylib
```

This verifies the release archive's SHA-256 checksum and installs it in
`.pixi/raylib/`, upgrading older installations. Pass
`.pixi/raylib/lib/libraylib.so` to `bajo.viewer.raylib.Window`.

Examples and asset-dependent benchmarks download their meshes on first use:

```bash
pixi run download_assets
```


## Interactive renderer

The viewer runs in Mojo with raylib 6.2 for the window, input, and raster rendering.
`pixi run viewer` installs raylib locally when needed and opens the RTIAW scene:

```bash
pixi run viewer
pixi run viewer --scene cornell --bounds both
pixi run viewer --scene cornell --mode trace --width 320 --max-spp 32 --samples-per-batch 4
pixi run viewer --pbrt examples/scenes/pbrt_showcase.pbrt
```

`Tab` switches between raylib rasterization and progressive Bajo ray tracing.
Both modes share the loaded geometry, instance transforms, camera, and field of
view. Raster geometry is uploaded once per scene; shared meshes stay shared.
Raster materials use approximate colors and directional lighting, including
unlit emitters; glass, textures, shadows, and indirect lighting are not reproduced.
Ray tracing supports `PATH`, `NEE`, `MIS`, `NORMALS`, and `AO`, and all six samplers.
A native background thread handles tracing and acceleration preparation while
raylib continues processing input. Camera/settings changes reset accumulation;
completed frames from an older camera are discarded. Closing waits for the
current render batch to finish.

| Control | Action |
| --- | --- |
| `W/S`, `A/D`, `Q/E` | Move forward/back, sideways, down/up |
| Left mouse drag; Shift | Look around; move faster |
| `R`; `F`; `Esc` | Reset camera; fullscreen; close |
| `Tab`; `G` | Raster/trace; CPU/GPU tracing backend |
| `1`–`5`; `P` | PATH/NEE/MIS/NORMALS/AO; cycle sampler |
| `C` | Next built-in scene (current name appears in the overlay) |
| `O` / **Open scene** | Browse and load a PBRT file |
| `B` | Cycle bounds off / objects / CPU BVH / both |
| `[` / `]` | Decrease/increase displayed BVH depth |
| `+` / `-`; `,` / `.` | Adjust maximum samples; path depth |
| `J` / `K`; batch **− / +** buttons | Smaller/larger samples per batch; increasing past the sample budget selects **all** |

Set `--samples-per-batch N` (alias `--batch-spp`) to a positive sample count, or
`-1` to render all remaining samples in one batch. The default is 1. The overlay
shows the selected batch size and completed batch count; `J` / `K` and the batch
**− / +** buttons adjust it without clearing accumulated samples. Each batch is
averaged in linear color, with its sample count as the weight; the last batch is
clamped to the remaining budget. For example, 10 total samples at 4 per batch
renders batches of 4, 4, and 2. With a fixed batch size, moving-camera previews
are capped by `--moving-spp`; `-1` always uses the full remaining budget.

Load a custom scene without restarting by clicking **Open scene**, pressing `O`,
or dropping a `.pbrt` / `.pbrt-v4` file onto the window. In the browser, double-click
folders/files or select a row and press `Enter` / **Open**. Use `Backspace` / **Up**
for the parent folder, the mouse wheel or arrow keys to browse, and `Ctrl+V` to
open a copied file or directory path. `Esc` / **Cancel** closes the browser.
The loaded filename appears in the overlay. A failed load reports the error
and keeps the current scene; a successful load resets its camera and accumulation.
The raster/trace mode and selected algorithm stay as selected.

Bounding boxes appear in raster mode: object bounds in yellow and CPU BVH
bounds in cyan. The BVH overlay includes top-level instance bounds and transformed
mesh BLAS bounds, and is labeled CPU BVH even when GPU tracing is selected.
Flat world triangles have individual triangle bounds because scene data does not
retain source object groups. Debug snapshots cap object boxes at 10,000 and BVH
boxes at 20,000 to keep large scenes usable.

For GPU tracing on the default `sm_120` target:

```bash
pixi run viewer_gpu --scene cornell --mode trace --backend gpu
# Other GPU targets:
BAJO_GPU_ARCH=sm_89 pixi run viewer_gpu --mode trace --backend gpu
```

The launcher builds and caches a standalone executable before opening the window.
Source or compiler changes rebuild it; `pixi run clean` removes the cache. This
avoids a render-worker stall under `mojo run`. A raster preview stays visible
until the first traced frame is ready. Use `pixi run viewer --help` for
scene names, render options, CPU builder/traversal selection, and library path.
For graphical smoke checks, `--hidden --frames 4 --screenshot /tmp/viewer.png`
renders a few frames and exits; a working display/OpenGL context is still needed.
The previous Python viewer remains available as `pixi run viewer_legacy`.

Run the scene-switch and file-loading regressions with a working display/OpenGL context:

```bash
BAJO_TEST_VIEWER_GRAPHICS=1 pixi run test viewer/test_viewer_graphics.py
```

The native core can also be exercised without a window:

```bash
pixi run viewer_scene --scene cornell
pixi run viewer_scene --scene cornell --prepare-cpu
pixi run viewer_scene --pbrt examples/scenes/pbrt_showcase.pbrt
```

`bajo.viewer` exposes shared camera/render state and PBRT loading that retains the
authored camera and settings. CPU preparation moves the geometry into `CpuScene`;
camera, integrator, and render-setting changes retain prepared acceleration.
CPU and GPU tracing prepare their own acceleration structures lazily.


## Tests

Run the host-only test suites without an accelerator:

```bash
pixi run test_host
```

Run the GPU test suites:

```bash
pixi run test_gpu
```

Tests focus on public API behavior, invalid input, CPU/GPU agreement and
numerical regressions. SIMD and launch configurations use representative
cases rather than exhaustive matrices. Internal tree layouts and diagnostic
counters are not separate contracts.

The host and GPU tasks include the relevant Mojo and Python tests. Run the
Python checks separately with `pixi run test_python_host` or
`pixi run test_python_gpu`; the GPU task enables the viewer GPU integration tests.

The test runner accepts a suite path when a smaller run is useful, for
example `pixi run test core` or `pixi run test bvh/cpu`. With no suite argument,
`pixi run test` runs all Mojo and Python tests and requires a supported accelerator.

Each subsystem (`core`, `parser`, `bvh`, `rt`, `sort`, `gpu`, and `viewer`) has
one `run_tests.mojo` entry point that runs its test modules in one process and
reports failures after all selected modules finish. Test modules expose
`run_tests()`; add new modules to their subsystem runner. The shell runner checks
that every discovered test is registered. Suite and individual-file filters
select modules at compile time, so host-only runs do not compile GPU tests.
For example, `pixi run test core/test_aabb.mojo` runs just the AABB tests.

## Examples

The main examples are available as Pixi tasks:

```bash
pixi run example_rtiaw        # small path-tracing example
pixi run example_cornell      # Cornell box: PATH, NEE, and MIS
pixi run example_mis_showcase  # compare PATH, NEE, and MIS
pixi run example_pbrt         # render the checked-in PBRT scene
pixi run example_lbvh         # CPU/GPU instanced OBJ normal render
```

Most examples write a PPM image in the repository root. The LBVH example needs
the downloaded OBJ assets and produces CPU and GPU outputs. Its GPU path uses
H-PLOC CWBVH8 BLASes with one triangle per leaf and an LBVH2 TLAS with one
instance per leaf, with direct traversal for camera rays. The source files in
[`examples/`](examples/) are also intended to be runnable and easy to modify.

## Benchmarks

`bench_smoke` runs a synthetic CPU BVH benchmark and a small CPU MIS render;
it needs no downloaded assets or accelerator. The render compares scalar,
native SIMD and parallel SIMD using a warmup, three timed repeats and a
checksum. For a larger run or another scene:

```bash
pixi run mojo -I . bench/rt/bench_cpu_rt.mojo mixed 640 360 4 5
```

Parser benchmarks measure complete loaders, with digests outside the timed
region. For example:

```bash
pixi run mojo -I . bench/parser/bench_loaders.mojo png path/to/image.png
```

Available loaders are `obj`, `ply`, `png`, `exr` and `pbrt`; `bench_obj` and
`bench_ply` provide shortcuts using downloaded assets.

Common tasks (larger benchmarks are opt-in):

```bash
pixi run bench_smoke
pixi run bench_bvh_smoke
pixi run bench_bvh_cpu_dragon
pixi run bench_bvh_gpu_sweep
pixi run bench_rt_cpu
pixi run bench_rt_gpu
pixi run bench_rt_cpu_gpu
pixi run bench_sort_mojo
```

GPU benchmarks require a supported accelerator. The benchmarks in this repository
exercise Bajo's Mojo implementations. Generated results under `bench/results/`
are ignored by Git.

## Repository layout

- [`bajo/core`](bajo/core/) — vectors, matrices, quaternions, transforms, rays, and intersections
- [`bajo/parser`](bajo/parser/) — pure-Mojo OBJ/MTL/PBRT loading
- [`bajo/bvh`](bajo/bvh/) — CPU/GPU BVH construction and traversal
- [`bajo/rt`](bajo/rt/) — CPU and GPU ray-tracing and shading pipelines
- [`bajo/sort`](bajo/sort/) — CPU and GPU sorting implementations
- [`test`](test/) — Mojo tests grouped by subsystem
- [`bench`](bench/) — performance and diagnostic benchmarks

All repeatable commands and dependencies are defined in [`pixi.toml`](pixi.toml).

## References

See [`sources.md`](sources.md) for the papers and other references used by the
project.
