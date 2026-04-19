# bajo
bajo = low (... level) = batch mojo

Bajo is a work-in-progress, pure-Mojo simulation engine designed for Reinforcement Learning (RL) environments, spatial computing, and physics simulations. The end goal is to run thousands of environments concurrently with high throughput on a single GPU, similar to frameworks like [NVIDIA Warp](https://github.com/NVIDIA/warp) and the [Madrona Engine](https://madrona-engine.github.io/).

Because Mojo is still relatively new, I am currently building out the foundational GPU primitives required for physics simulations from scratch.


## Roadmap
- [x] RTIAW
- [x] Math primitives (vec, mat, quat)
- [x] Intersections (AABB)
- [x] GPU radix sort : [onesweep](bajo/sort/gpu/onesweep.mojo)
- [ ] LBVH
- [ ] H-PLOC BVH.
- [ ] Implement a pure-Mojo `.obj` / `.mtl` file parser for 3D meshes
- [ ] Segmented GPU sort
- [ ] GPU Ray Tracer (first milestone)
- [ ] Hash Grid
- [ ] Fluid Simulation (second milestone)
- [ ] rigid-body
- [ ] physics integrator
- [ ] constraint solver


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

### GPU
- [Mojo 🔥 GPU Puzzles](https://puzzles.modular.com/introduction.html)
- [Visual Computing Systems](https://gfxcourses.stanford.edu/cs348k/spring25), [Parallel Computing](https://gfxcourses.stanford.edu/cs149/fall24): Courses by Kayvon Fatahlian
- [GPU Programming Primitives for Computer Graphics](https://gpu-primitives-course.github.io/)
- https://github.com/b0nes164/GPUSorting
- https://github.com/IlyaGrebnov/libcusort
- https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf
- https://github.com/amdadvtech/Orochi/tree/course/Course
- https://ajdillhoff.github.io/courses/cse5373/


### Graphics
- [Ray Tracing Gems Series](https://www.realtimerendering.com/raytracinggems/)
- [Ray Tracing in One Weekend Series](https://raytracing.github.io/)
- [Graphics Gems Series](https://www.realtimerendering.com/resources/GraphicsGems/)
- [HIPRT](https://gpuopen.com/hiprt/): HIP RT is a ray tracing library for HIP
- [Yocto/GL](https://github.com/xelatihy/yocto-gl) Tiny C++ Libraries for Data-Driven Physically-based Graphics 
- [Rendering Algorithms](https://cs87-dartmouth.github.io/Fall2025/) course by Wojciech Jarosz

### Math
- [Numerical Linear Algebra with Julia](https://epubs.siam.org/doi/book/10.1137/1.9781611976557)
- [Algorithms for Sparse Linear Systems](https://link.springer.com/book/10.1007/978-3-031-25820-6)


