# CPU/GPU ray-tracing comparison

## 2026-09-27 — identical long workload

Command: `pixi run bench_rt_cpu_gpu`

- Scene: Cornell triangle scene; NEE-64 uses the same 64-light receiver scene
  as the GPU optimization benchmark.
- Toolchain: Mojo `1.2.0.dev2026092705 (3d1f3942)` on an NVIDIA GeForce RTX
  5060 Ti.
- Workload: 1024x1024, 8 samples/pixel, depth 8, median of 9.
- CPU: packet-width 16 wavefront renderer with 1,024-path parallel chunks.
  AO uses the optimized tiled depth-first renderer because CPU wavefront AO is
  not implemented.
- GPU: node8/leaf4 triangle BVH, 256K path working set, 64-thread blocks.
- `GPU device` ends with device-resident pixels. `GPU host` additionally
  includes synchronization, status checking, allocation of the host color
  list, and pixel download. `CPU total` already ends with host-resident pixels.
- Scene/BVH construction is excluded for both. CPU total is the complete public
  CPU render call, including per-call output/RNG initialization; GPU timings use
  the persistent target API and exclude its one-time allocation.

| Case | CPU render median ms | CPU total median ms | GPU device median ms | GPU host median ms | GPU device speedup | GPU host speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PATH | 388.940 | 389.550 | 11.688 | 14.278 | 33.276x | 27.283x |
| AO | 156.323 | 156.328 | 2.513 | 5.106 | 62.210x | 30.617x |
| NEE | 1,860.302 | 1,860.963 | 29.365 | 32.047 | 63.351x | 58.070x |
| MIS | 1,850.050 | 1,850.840 | 29.894 | 32.620 | 61.886x | 56.739x |
| NEE-64 | 425.275 | 425.853 | 8.046 | 10.731 | 52.857x | 39.684x |

Detailed timing and throughput:

| Case | CPU total min..max ms | CPU Msample/s | GPU device min..max ms | GPU Msample/s | GPU host min..max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| PATH | 386.011–397.403 | 21.534 | 11.550–12.225 | 717.695 | 14.134–15.037 |
| AO | 152.091–162.246 | 53.660 | 2.507–3.017 | 3,338.334 | 5.084–5.602 |
| NEE | 1,853.275–1,886.763 | 4.508 | 29.021–29.649 | 285.665 | 31.640–32.242 |
| MIS | 1,848.708–1,881.374 | 4.532 | 29.452–30.950 | 280.607 | 32.057–33.537 |
| NEE-64 | 415.906–441.272 | 19.698 | 7.996–8.554 | 1,042.603 | 10.628–11.152 |

Checksums are deterministic within each backend. CPU packet traversal and GPU
traversal can reassociate floating-point operations, so cross-backend checksums
are expected to be close rather than bit-identical:

| Case | CPU checksum | GPU checksum | Absolute delta | Relative delta |
| --- | ---: | ---: | ---: | ---: |
| PATH | 845,426,382.297 | 845,427,648.343 | 1,266.046 | 1.498 ppm |
| AO | 349,672,972.311 | 349,673,226.576 | 254.265 | 0.727 ppm |
| NEE | 845,384,427.619 | 845,385,006.865 | 579.246 | 0.685 ppm |
| MIS | 845,726,179.646 | 845,726,782.144 | 602.498 | 0.712 ppm |
| NEE-64 | 1,153,380,092.641 | 1,153,379,541.937 | 550.704 | 0.477 ppm |
