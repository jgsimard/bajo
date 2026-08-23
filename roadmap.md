# Bajo roadmap

The long-term goal is a pure-Mojo engine for high-throughput geometry, rendering, spatial computing, and batched GPU simulation workloads.

The current rendering and acceleration-structure foundations are strong enough to serve as a stable base. The next major phase is GPU particle simulation, starting with a reusable hash-grid substrate.

### Legend

- ✅ Done and tested
- 🚧 In progress / partially working
- ⬜ Planned

## Current foundation

### Math and geometry primitives

- ✅ `Vec`, `Mat`, `Quat`, `AABB`, rays, and intersection helpers
- ✅ transforms and coordinate-frame helpers
- ✅ CPU and GPU geometry data layouts
- 🚧 Complete and simplify the transform API
- 🚧 Expand numerical and edge-case test coverage

### Scene input

- ✅ Pure-Mojo OBJ/MTL loading
  - ✅ vertex positions, normals, texture coordinates, and faces
  - ✅ lines, groups, objects, materials, and texture references
  - ✅ negative indices and triangulation
  - ✅ mmap loading
- ✅ PBRT text-scene parsing and loading
- 🚧 Parser polish and diagnostics
- ⬜ Parallel OBJ/MTL parsing

### GPU sorting

- ✅ Bitonic sort
- ✅ Radix sort
- ✅ Onesweep
- ✅ Benchmark coverage
- 🚧 API and workspace polish
- 🚧 Segmented sort

### BVH and spatial acceleration

- ✅ CPU BVH construction and traversal
- ✅ GPU LBVH construction
- ✅ GPU H-PLOC construction
- ✅ Wide and compressed GPU layouts
- ✅ CPU and GPU TLAS/BLAS instance traversal
- ✅ Sphere and triangle geometry support
- ✅ Correctness, quality, diagnostic, and performance benchmarks
- 🚧 Documentation and API stabilization

BVH work is now in maintenance mode. New changes should be driven by a
measured regression or by a concrete simulation use case.

### Ray tracing and rendering

- ✅ CPU depth-first and wavefront renderers
- ✅ GPU wavefront path tracing
- ✅ PATH, NEE, MIS, AO, and normal-rendering modes
- ✅ Sphere, triangle, mixed, and instanced scenes
- ✅ Persistent GPU render targets and asynchronous submission
- ✅ CPU/GPU comparison benchmarks
- 🚧 API stabilization, regression tests, and renderer polish

Rendering is also in maintenance mode while simulation infrastructure is built.

## Next major phase: GPU simulation substrate

### Particle data and execution model

- ⬜ Add a `bajo/sim` module
- ⬜ Define structure-of-arrays particle storage
- ⬜ Add persistent device buffers and double-buffered state
- ⬜ Add timestep, reset, and deterministic-seed APIs
- ⬜ Provide a CPU reference implementation
- ⬜ Add CPU/GPU parity tests
- ⬜ Add step-time and throughput benchmarks

### GPU hash grid

- ⬜ Particle buffer foundation
- ⬜ Dense uniform-grid descriptor
- ⬜ Particle-to-cell key generation
- ⬜ Sort particles by cell ID using the existing GPU sort primitives
- ⬜ Build cell start/end ranges
- ⬜ Implement 27-cell neighbor queries
- ⬜ Validate neighbor results against a CPU reference
- ⬜ Benchmark build, query, and end-to-end step throughput

The first end-to-end milestone is a GPU particle system that can integrate
many particles, rebuild the grid, and answer neighbor queries every step.

## Simulation workloads

### Basic particle simulation

- ⬜ Gravity integration
- ⬜ Boundary, plane, and box collision
- ⬜ Sphere-sphere contact forces
- ⬜ Particle pile example
- ⬜ GPU particle visualization
- ⬜ Deterministic replay/checksum validation

### Batched simulation environments

- ⬜ Environment IDs and contiguous environment ranges
- ⬜ No cross-environment collisions
- ⬜ Reset kernels and randomized initialization
- ⬜ Action, observation, and reward buffers
- ⬜ RL-style toy environment
- ⬜ Throughput measured in environment-steps per second


### SPH fluid simulation

- ⬜ Basic SPH particle data
- ⬜ Density pass
- ⬜ Pressure pass
- ⬜ Viscosity pass
- ⬜ Force accumulation
- ⬜ Boundary collision
- ⬜ Dam-break example
- ⬜ SPH benchmarks

### Mesh and particle coupling

- ⬜ Particle-vs-plane collision
- ⬜ Particle-vs-triangle-mesh queries
- ⬜ Particle-vs-instanced-mesh queries
- ⬜ Mesh height queries
- ⬜ Sphere/capsule-vs-mesh queries

Reuse the existing BVH and TLAS infrastructure where it provides a clear
benefit, without reopening general BVH optimization work prematurely.

### Rigid-body simulation

- ⬜ Rigid-body data model
- ⬜ Sphere, plane, box, and capsule shapes
- ⬜ Rigid-body integration
- ⬜ Broadphase pair generation
- ⬜ Narrowphase contact generation
- ⬜ Contact solver
- ⬜ Rigid-body examples and benchmarks
