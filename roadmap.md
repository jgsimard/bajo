### Legend
- ✅ Done and tested
- 🚧 In progress / partially working
- ⬜ Planned

### 🚧 Math primitives
- ✅ [Vec](bajo/core/vec.mojo)
- ✅ [Mat](bajo/core/mat.mojo)
- ✅ [Quat](bajo/core/quat.mojo)
- ✅ [AABB](bajo/core/aabb.mojo)
- ✅ ray / triangle intersection helpers
- ✅ transform helpers
- ⬜ complete transform API
- ⬜ stronger tests

### 🚧 Pure-Mojo `.obj` / `.mtl` parser
- ✅ vertex positions
- ✅ normals
- ✅ texture coordinates
- ✅ faces
- ✅ lines
- ✅ groups / objects
- ✅ materials
- ✅ texture references
- ✅ negative indices
- ✅ triangulation
- ✅ mmap loading
- ⬜ parallel parser
- 🚧 polish 

### 🚧 GPU sort
- ✅ [Bitonic Sort](bajo/sort/gpu/bitonic_sort.mojo)
- ✅ [Radix Sort](bajo/sort/gpu/radix_sort.mojo)
- ✅ [Onesweep](bajo/sort/gpu/onesweep.mojo)
- 🚧 [benchmark](bajo/sort/gpu/README.md)
- ⬜ Segmented sort
- 🚧 polish 

###  🚧 Gpu BVH
- ✅ morton codes
- ✅ build kernel
- ✅ LBVH
    - ✅ [raycast example](examples/lbvh_normals.mojo)
- 🚧 TLAS
    - 🚧 on cpu
    - 🚧 on gpu
    - 🚧 multiple instance raycast example
- ⬜ H-PLOC BVH
- 🚧 polish : Mojo only support scalar GPU buffers for now :(

### GPU Hash Grid
- ⬜ Particle buffer foundation
- ⬜ Basic particle integration kernels
- ⬜ Dense uniform grid descriptor
- ⬜ Particle-to-cell key generation
- ⬜ Sort particles by cell id
- ⬜ Build cell start/end ranges
- ⬜ 27-cell neighbor query
- ⬜ CPU/GPU validation
- ⬜ Hash grid benchmarks

### Particle Simulation
- ⬜ Gravity particle example
- ⬜ Ground and box collision
- ⬜ DEM sphere collision
- ⬜ Neighbor-based contact forces
- ⬜ Particle pile example
- ⬜ Particle visualization
- ⬜ Particle simulation benchmarks

### SPH Fluid Simulation
- ⬜ Basic SPH particle data
- ⬜ Density pass
- ⬜ Pressure pass
- ⬜ Viscosity pass
- ⬜ Force accumulation
- ⬜ Boundary collision
- ⬜ Dam-break example
- ⬜ SPH benchmarks

### Mesh / Particle Coupling
- ⬜ Particle vs plane collision
- ⬜ Particle vs triangle mesh approximation
- ⬜ Particle vs instanced mesh
- ⬜ Mesh height queries
- ⬜ Sphere / capsule vs mesh queries

### Rigid Body Simulation
- ⬜ Rigid body data model
- ⬜ Sphere, plane, box, capsule shapes
- ⬜ Rigid body integration
- ⬜ Broadphase pair generation
- ⬜ Narrowphase contact generation
- ⬜ Contact solver
- ⬜ Rigid body examples

### Batched Simulation Environments
- ⬜ Environment ids
- ⬜ No cross-environment collisions
- ⬜ Reset kernels
- ⬜ Random initialization
- ⬜ Observation buffers
- ⬜ Action buffers
- ⬜ Reward buffers
- ⬜ RL-style toy examples