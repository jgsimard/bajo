# CPU/GPU ray-tracing comparison

2026-09-27 · `pixi run bench_rt_cpu_gpu` · Mojo `1.2.0.dev2026092705 (3d1f3942)` · RTX 5060 Ti.

Cornell triangles (NEE-64 uses the 64-light receiver): 1024×1024, 8 spp, depth 8, median of nine runs. CPU: packet16 wavefront, 1K chunks; AO uses tiled depth-first. GPU: node8/leaf4, 256K working set, 64-thread blocks.

CPU total and GPU host end with host pixels; GPU device ends with device pixels. Both exclude scene/BVH builds. CPU includes per-call output/RNG setup; GPU excludes one-time persistent-target allocation.

| Case | CPU render median ms | CPU total median ms | GPU device median ms | GPU host median ms | GPU device speedup | GPU host speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PATH | 388.940 | 389.550 | 11.688 | 14.278 | 33.276x | 27.283x |
| AO | 156.323 | 156.328 | 2.513 | 5.106 | 62.210x | 30.617x |
| NEE | 1,860.302 | 1,860.963 | 29.365 | 32.047 | 63.351x | 58.070x |
| MIS | 1,850.050 | 1,850.840 | 29.894 | 32.620 | 61.886x | 56.739x |
| NEE-64 | 425.275 | 425.853 | 8.046 | 10.731 | 52.857x | 39.684x |

Checksums are deterministic within each backend. Cross-backend differences are 0.477–1.498 ppm due to floating-point reassociation.
