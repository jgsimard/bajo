# CPU/GPU ray-tracing comparison

## 2026-08-15 — identical long workload

Command: `pixi run bench_rt_cpu_gpu`

- Scene: Cornell triangle scene; NEE-64 uses the same 64-light receiver scene
  as the GPU optimization benchmark.
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
| PATH | 226.544 | 227.103 | 10.366 | 13.147 | 21.855x | 17.274x |
| AO | 122.341 | 146.533 | 2.958 | 5.735 | 41.365x | 25.551x |
| NEE | 480.628 | 481.335 | 19.773 | 22.477 | 24.308x | 21.415x |
| MIS | 481.941 | 482.713 | 21.140 | 23.827 | 22.797x | 20.259x |
| NEE-64 | 109.589 | 110.198 | 7.114 | 9.780 | 15.404x | 11.268x |

Detailed timing and throughput:

| Case | CPU total min..max ms | CPU Msample/s | GPU device min..max ms | GPU Msample/s | GPU host min..max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| PATH | 226.527–229.186 | 36.937 | 10.309–10.598 | 809.254 | 12.999–13.596 |
| AO | 143.045–155.652 | 57.247 | 2.888–3.313 | 2,836.268 | 5.615–6.197 |
| NEE | 480.816–495.857 | 17.428 | 19.471–20.131 | 424.252 | 22.318–22.808 |
| MIS | 481.333–507.348 | 17.378 | 20.710–21.190 | 396.806 | 23.462–24.051 |
| NEE-64 | 108.727–113.943 | 76.123 | 6.786–7.333 | 1,179.110 | 9.454–10.024 |

Checksums are deterministic within each backend. CPU packet traversal and GPU
traversal can reassociate floating-point operations, so cross-backend checksums
are expected to be close rather than bit-identical:

| Case | CPU checksum | GPU checksum | Absolute delta | Relative delta |
| --- | ---: | ---: | ---: | ---: |
| PATH | 845,426,382.297 | 845,427,648.343 | 1,266.046 | 1.498 ppm |
| AO | 349,688,158.505 | 349,673,226.576 | 14,931.929 | 42.701 ppm |
| NEE | 845,675,131.674 | 845,675,358.661 | 226.986 | 0.268 ppm |
| MIS | 845,675,728.422 | 845,675,956.289 | 227.867 | 0.269 ppm |
| NEE-64 | 1,153,954,017.287 | 1,153,953,988.576 | 28.711 | 0.025 ppm |
