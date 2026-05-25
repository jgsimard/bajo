# bajo
bajo = low (... level) = batch mojo

Bajo is a work-in-progress, pure-Mojo simulation engine designed for Reinforcement Learning (RL) environments, spatial computing, and physics simulations. The end goal is to run thousands of environments concurrently with high throughput on a single GPU, similar to frameworks like [NVIDIA Warp](https://github.com/NVIDIA/warp) and the [Madrona Engine](https://madrona-engine.github.io/).

Because Mojo is still relatively new, I am currently building out the foundational GPU primitives required for physics simulations from scratch.


## Pixi tasks

Common commands:

```bash
pixi run test              # run all tests
pixi run bench_all         # run all benchmarks
```

Examples:
```
pixi run example_lbvh
```
Runs the GPU LBVH normal-rendering example. It should produce the following image
![lbvh example](renders/example_tlas_lbvh_normals.png)

## Roadmap
for a detailed version see [roadmap](roadmap.md)

1. Math primitives (Vec, Mat, Quat, Ray, Hit, missing Spatial)
2. [obj parser](bajo/obj/) (single threaded done)
3. [GPU sort](bajo/sort/gpu/README.md) (Bitonic, Radix, Onesweep, but only uints, missing segmented)
4. [CPU BVH](bajo/bvh/cpu/)  (Bounds, Triangle, Sphere, Tlas with 1 blas only)
5. [GPU BVH](bajo/bvh/gpu/)  (Bounds, Triangle, Sphere, Tlas with 1 blas only)
6. GPU Hash Grid (not started)
7. Particle Simulation (not started)
8. SPH Fluid Simulation (not started)
9. Mesh / Particle Coupling (not started)
10. Rigid Body Simulation (not started)
11. Batched Simulation Environments (not started)

## Literature & References
see [sources](sources.md)

## Things I wish to be added to mojo
1. struct gpu buffers, not just dtype buffers
2. Parametric Traits (to simplyfy typed bvh)
3. enums
