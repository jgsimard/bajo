# bajo
bajo = low (... level) = batch mojo

Bajo is a work-in-progress, pure-Mojo simulation engine designed for Reinforcement Learning (RL) environments, spatial computing, and physics simulations. The end goal is to run thousands of environments concurrently with high throughput on a single GPU, similar to frameworks like [NVIDIA Warp](https://github.com/NVIDIA/warp) and the [Madrona Engine](https://madrona-engine.github.io/).

Because Mojo is still relatively new, I am currently building out the foundational GPU primitives required for physics simulations from scratch.


## Roadmap

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

## Literature & References
### Mojo
- Mojo community

### Engines & Environments
- [NVIDIA Warp](https://github.com/NVIDIA/warp) a very cool c++/cuda/python framework for accelerated simulation, data generation and spatial computing
- [Madrona Engine](https://madrona-engine.github.io/) research game engine designed specifically for creating learning environments that execute with extremely high throughput on a single GPU
- [Flecs](https://github.com/SanderMertens/flecs) fast ECS by Sander Mertens
- [PufferLib](https://puffer.ai/) : super fast RL environement library

### BVH & Spatial Computing
- [H-PLOC](https://gpuopen.com/download/HPLOC.pdf) : Modern GPU BVH that is a single kernel call, almost SAH quality with build time 15% slower then LBVH
- [tinybvh](https://github.com/jbikker/tinybvh/tree/main)
- [cuBQL](https://github.com/NVIDIA/cuBQL)
- https://github.com/ToruNiina/lbvh
- [KittenGpuLBVH](https://github.com/jerry060599/KittenGpuLBVH)
- [NexusBVH](https://github.com/StokastX/NexusBVH/)

### GPU
- [Mojo 🔥 GPU Puzzles](https://puzzles.modular.com/introduction.html)
- Courses by Kayvon Fatahlian
    - [Visual Computing Systems](https://gfxcourses.stanford.edu/cs348k/spring25)
    - [Parallel Computing](https://gfxcourses.stanford.edu/cs149/fall24)
- [GPU Programming Primitives for Computer Graphics](https://gpu-primitives-course.github.io/)
- https://github.com/b0nes164/GPUSorting
- https://github.com/IlyaGrebnov/libcusort
- https://ajdillhoff.github.io/courses/cse5373/
- AMD Opengpu
    - https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf
    - https://github.com/amdadvtech/Orochi/tree/course/Course

### Graphics
- [Ray Tracing Gems Series](https://www.realtimerendering.com/raytracinggems/)
- [Ray Tracing in One Weekend Series](https://raytracing.github.io/)
- [Graphics Gems Series](https://www.realtimerendering.com/resources/GraphicsGems/)
- [HIPRT](https://gpuopen.com/hiprt/): HIP RT is a ray tracing library for HIP
- [Yocto/GL](https://github.com/xelatihy/yocto-gl) Tiny C++ Libraries for Data-Driven Physically-based Graphics 
- [Rendering Algorithms](https://cs87-dartmouth.github.io/Fall2025/) course by Wojciech Jarosz
- Physically Based Modeling ONLINE SIGGRAPH 2001 COURSE NOTES https://graphics.pixar.com/pbm2001/

### ECS
- [ECS overwatch](https://www.youtube.com/watch?v=W3aieHjyNvw)
- [madrona](https://madrona-engine.github.io/#faq)
- https://www.richardlord.net/blog/ecs/what-is-an-entity-framework
- zig game engine & graphics toolkit
    - https://machengine.org/
    - https://devlog.hexops.com/categories/build-an-ecs/
    - https://github.com/hexops/mach/blob/main/src/module.zig
- https://github.com/skypjack/entt/wiki
- https://skypjack.github.io/entt/md_docs_md_entity.html
- https://gist.github.com/dakom/82551fff5d2b843cbe1601bbaff2acbf <
- https://github.com/hexops/mach/issues/127
- https://github.com/abeimler/ecs_benchmark
- https://github.com/empyreanx/pico_headers
- Sander Mertens
    - Entity Component System FAQ : https://github.com/SanderMertens/ecs-faq
    - https://ajmmertens.medium.com/building-an-ecs-1-where-are-my-entities-and-components-63d07c7da742
    - https://ajmmertens.medium.com/building-an-ecs-2-archetypes-and-vectorization-fe21690805f9
    - https://ajmmertens.medium.com/doing-a-lot-with-a-little-ecs-identifiers-25a72bd2647
    - https://ajmmertens.medium.com/building-an-ecs-storage-in-pictures-642b8bfd6e04
    - https://github.com/SanderMertens/ecs-faq?tab=readme-ov-file#resources


### Math
- [Numerical Linear Algebra with Julia](https://epubs.siam.org/doi/book/10.1137/1.9781611976557)
- [Algorithms for Sparse Linear Systems](https://link.springer.com/book/10.1007/978-3-031-25820-6)

### Physics
- [Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html)
- [Physics from scratch](https://www.youtube.com/playlist?list=PLwMZtAEBQ8ZywWPf6twbspmYzGg0Fr2DJ)


### Robotics
- https://github.com/calderpg/common_robotics_utilities
- https://royfeatherstone.org/spatial/index.html#spatial-software

