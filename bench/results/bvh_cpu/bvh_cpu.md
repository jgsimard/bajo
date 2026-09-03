# CPU BVH benchmark results

- **Date:** 2026-09-02T22:21:41-04:00
- **CPU:** AMD Ryzen 7 9700X 8-Core Processor
- **System:** Linux-7.0.0-30-generic-x86_64-with-glibc2.43
- **Mojo:** `Mojo 1.1.0.dev2026090205 (db60a5b8)`
- **Embree:** `Embree 4.4.0`; compiler: `g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0`
- **TinyBVH:** `TinyBVH 1.8.0`; compiler: `Ubuntu clang version 21.1.8 (6ubuntu1)`
- **C++ flags:** Embree harness `-O3 -DNDEBUG -march=native`; TinyBVH harness additionally `-ffast-math -mavx2 -mfma`
- **Build thread modes:** `1` and `all`
- **All-thread affinity:** `0-15` (16 logical CPUs)
- **Traversal:** one calling thread; timings use the `threads=1` run
- **Raw data:** CSV/TXT retain both build-thread runs
- **Build timing:** median of five builds after one untimed warm-up per configuration
- **Correctness gate:** triangle/instance/ray counts must match; hit counts must agree within 50 ppm (minimum two boundary rays)
- **Interpretation:** negative `Bajo vs competitor` means Bajo is slower; positive means faster

## Where Bajo still needs work

Traversal deficits larger than 2%:

| Workload | Bajo traversal | Bajo MRay/s | Fastest competitor | Competitor traversal | Competitor MRay/s | Bajo vs competitor (%) |
| --- | --- | --- | --- | --- | --- | --- |

Build deficits larger than 2% (lower time is better):

| Geometry | Build threads | Bajo build ms | Fastest competitor | Competitor build ms | Bajo vs competitor (%) |
| --- | --- | --- | --- | --- | --- |

These are the optimization queue: the most negative rows are the largest measured deficits on this machine. Differences within 2% are treated as parity.

## Best traversal per implementation

| Workload | Implementation | Build | Layout | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | vs Bajo (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dragon camera any-hit | bajo | sah | bvh16 | coh-packet16 | 16 | 48.182 | 8.667 | 5.981 | 98.611 | 0.000 |
| Dragon camera any-hit | embree | high | native | inc-packet16 | 16 | 263.207 | 27.788 | 6.340 | 93.036 | -5.654 |
| Dragon camera any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 66.954 | 31.988 | 11.410 | 51.694 | -47.578 |
| Dragon camera closest-hit | bajo | sah | bvh16 | adaptive-16-8-4-scalar | 16 | 48.182 | 8.667 | 7.503 | 78.609 | 0.000 |
| Dragon camera closest-hit | embree | medium | native | coh-packet8 | 8 | 48.107 | 6.209 | 8.813 | 66.924 | -14.865 |
| Dragon camera closest-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 66.723 | 34.207 | 14.258 | 41.369 | -47.374 |
| Instanced Dragon any-hit | bajo | sah | bvh4 | packet16 | 16 | 49.518 | 8.808 | 1.243 | 118.667 | 0.000 |
| Instanced Dragon any-hit | embree | medium | native | inc-packet16 | 16 | 46.804 | 6.018 | 1.454 | 101.429 | -14.526 |
| Instanced Dragon any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 67.899 | 31.925 | 2.656 | 55.520 | -53.214 |
| Instanced Dragon closest-hit | bajo | sah | bvh4 | packet16 | 16 | 49.518 | 8.808 | 1.793 | 82.241 | 0.000 |
| Instanced Dragon closest-hit | embree | high | native | inc-packet8 | 8 | 263.128 | 28.595 | 1.871 | 78.805 | -4.178 |
| Instanced Dragon closest-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 66.760 | 31.720 | 3.117 | 47.304 | -42.481 |
| Dragon shuffled any-hit | bajo | sah | bvh16 | scalar1 | 1 | 48.182 | 8.667 | 9.573 | 61.611 | 0.000 |
| Dragon shuffled any-hit | embree | high | native | inc-packet16 | 16 | 262.380 | 28.598 | 9.959 | 59.222 | -3.878 |
| Dragon shuffled any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 66.504 | 31.999 | 14.365 | 41.061 | -33.354 |
| Dragon shuffled closest-hit | embree | medium | native | inc-packet8 | 8 | 46.665 | 5.865 | 14.297 | 41.255 | 0.289 |
| Dragon shuffled closest-hit | bajo | sah | bvh16 | scalar1 | 1 | 48.182 | 8.667 | 14.339 | 41.136 | 0.000 |
| Dragon shuffled closest-hit | tinybvh | high | bvh8 | scalar1 | 1 | 368.133 | 70.038 | 16.927 | 34.845 | -15.293 |
| Regular-grid any-hit | bajo | hploc | bvh16 | scalar1 | 1 | 2.194 | 0.962 | 4.500 | 58.250 | 0.000 |
| Regular-grid any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 10.655 | 4.481 | 5.953 | 44.033 | -24.407 |
| Regular-grid any-hit | embree | high | native | inc-packet16 | 16 | 12.922 | 1.840 | 6.052 | 43.316 | -25.638 |
| Regular-grid closest-hit | bajo | hploc | bvh16 | adaptive-16-8-scalar | 16 | 2.194 | 0.962 | 5.001 | 52.419 | 0.000 |
| Regular-grid closest-hit | tinybvh | high | bvh8 | scalar1 | 1 | 31.949 | 6.727 | 10.095 | 25.969 | -50.459 |
| Regular-grid closest-hit | embree | high | native | inc-packet8 | 8 | 12.895 | 1.753 | 10.377 | 25.261 | -51.809 |
| Flattened triangle grid any-hit | bajo | sah | bvh16 | coh-packet16 | 16 | 0.009 | 0.009 | 0.482 | 306.143 | 0.000 |
| Flattened triangle grid any-hit | embree | high | native | coh-packet16 | 16 | 0.017 | 0.024 | 0.501 | 294.548 | -3.787 |
| Flattened triangle grid closest-hit | bajo | sah | bvh16 | coh-packet16 | 16 | 0.009 | 0.009 | 0.662 | 222.743 | 0.000 |
| Flattened triangle grid closest-hit | embree | high | native | coh-packet8 | 8 | 0.016 | 0.034 | 0.700 | 210.714 | -5.400 |
| Instanced triangle any-hit | bajo | lbvh | bvh4 | packet16 | 16 | 0.008 | 0.008 | 0.387 | 381.224 | 0.000 |
| Instanced triangle any-hit | embree | medium | native | inc-packet16 | 16 | 0.038 | 0.063 | 0.574 | 256.833 | -32.629 |
| Instanced triangle closest-hit | bajo | lbvh | bvh4 | packet16 | 16 | 0.008 | 0.008 | 0.683 | 215.956 | 0.000 |
| Instanced triangle closest-hit | embree | medium | native | coh-packet8 | 8 | 0.038 | 0.058 | 0.814 | 181.168 | -16.109 |

## Best scalar traversal per implementation

This removes packet-width advantages and is the fairest direct comparison with TinyBVH's scalar API.

| Workload | Implementation | Build | Layout | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | vs Bajo (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dragon camera any-hit | bajo | sah | bvh16 | scalar1 | 1 | 48.182 | 8.667 | 7.473 | 78.925 | 0.000 |
| Dragon camera any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 66.954 | 31.988 | 11.410 | 51.694 | -34.502 |
| Dragon camera any-hit | embree | medium | native | scalar1 | 1 | 46.384 | 5.756 | 15.584 | 37.847 | -52.047 |
| Dragon camera closest-hit | bajo | sah | bvh16 | scalar1 | 1 | 48.182 | 8.667 | 9.160 | 64.393 | 0.000 |
| Dragon camera closest-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 66.723 | 34.207 | 14.258 | 41.369 | -35.755 |
| Dragon camera closest-hit | embree | medium | native | scalar1 | 1 | 48.107 | 6.209 | 18.226 | 32.362 | -49.743 |
| Instanced Dragon any-hit | bajo | sah | bvh4 | scalar1 | 1 | 49.518 | 8.808 | 2.189 | 67.367 | 0.000 |
| Instanced Dragon any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 67.899 | 31.925 | 2.656 | 55.520 | -17.586 |
| Instanced Dragon any-hit | embree | high | native | scalar1 | 1 | 262.510 | 28.012 | 3.703 | 39.816 | -40.897 |
| Instanced Dragon closest-hit | bajo | sah | bvh4 | scalar1 | 1 | 49.518 | 8.808 | 2.616 | 56.365 | 0.000 |
| Instanced Dragon closest-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 66.760 | 31.720 | 3.117 | 47.304 | -16.076 |
| Instanced Dragon closest-hit | embree | high | native | scalar1 | 1 | 263.128 | 28.595 | 4.104 | 35.926 | -36.262 |
| Dragon shuffled any-hit | bajo | sah | bvh16 | scalar1 | 1 | 48.182 | 8.667 | 9.573 | 61.611 | 0.000 |
| Dragon shuffled any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 66.504 | 31.999 | 14.365 | 41.061 | -33.354 |
| Dragon shuffled any-hit | embree | medium | native | scalar1 | 1 | 46.498 | 5.693 | 16.991 | 34.715 | -43.655 |
| Dragon shuffled closest-hit | bajo | sah | bvh16 | scalar1 | 1 | 48.182 | 8.667 | 14.339 | 41.136 | 0.000 |
| Dragon shuffled closest-hit | tinybvh | high | bvh8 | scalar1 | 1 | 368.133 | 70.038 | 16.927 | 34.845 | -15.293 |
| Dragon shuffled closest-hit | embree | high | native | scalar1 | 1 | 263.190 | 28.047 | 19.786 | 29.810 | -27.533 |
| Regular-grid any-hit | bajo | hploc | bvh16 | scalar1 | 1 | 2.194 | 0.962 | 4.500 | 58.250 | 0.000 |
| Regular-grid any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 10.655 | 4.481 | 5.953 | 44.033 | -24.407 |
| Regular-grid any-hit | embree | high | native | scalar1 | 1 | 12.922 | 1.840 | 13.030 | 20.118 | -65.463 |
| Regular-grid closest-hit | bajo | hploc | bvh16 | scalar1 | 1 | 2.194 | 0.962 | 7.323 | 35.797 | 0.000 |
| Regular-grid closest-hit | tinybvh | high | bvh8 | scalar1 | 1 | 31.949 | 6.727 | 10.095 | 25.969 | -27.455 |
| Regular-grid closest-hit | embree | medium | native | scalar1 | 1 | 11.299 | 1.676 | 16.931 | 15.483 | -56.748 |
| Flattened triangle grid any-hit | bajo | sah | bvh16 | scalar1 | 1 | 0.006 | 0.005 | 1.183 | 124.624 | 0.000 |
| Flattened triangle grid any-hit | embree | high | native | scalar1 | 1 | 0.017 | 0.024 | 2.964 | 49.744 | -60.085 |
| Flattened triangle grid closest-hit | bajo | sah | bvh16 | scalar1 | 1 | 0.006 | 0.005 | 1.990 | 74.083 | 0.000 |
| Flattened triangle grid closest-hit | embree | high | native | scalar1 | 1 | 0.016 | 0.034 | 3.364 | 43.836 | -40.829 |
| Instanced triangle any-hit | bajo | lbvh | bvh4 | scalar1 | 1 | 0.008 | 0.008 | 1.892 | 77.948 | 0.000 |
| Instanced triangle any-hit | embree | high | native | scalar1 | 1 | 0.037 | 0.064 | 3.379 | 43.640 | -44.014 |
| Instanced triangle closest-hit | bajo | lbvh | bvh4 | scalar1 | 1 | 0.008 | 0.008 | 2.506 | 58.831 | 0.000 |
| Instanced triangle closest-hit | embree | medium | native | scalar1 | 1 | 0.038 | 0.058 | 3.856 | 38.245 | -34.992 |

## Fastest build per geometry and implementation

| Geometry | Implementation | Build | Layout | Build ms (1) | Build ms (all) |
| --- | --- | --- | --- | --- | --- |
| dragon | bajo | lbvh | bvh16 | 8.678 | 3.562 |
| dragon | tinybvh | sah | bvh2 | 40.427 | 4.744 |
| dragon | embree | medium | native | 46.498 | 5.693 |
| dragon-instances | bajo | lbvh | bvh4 | 9.468 | 3.513 |
| dragon-instances | tinybvh | sah | bvh2 | 40.488 | 5.080 |
| dragon-instances | embree | medium | native | 46.804 | 6.018 |
| grid | bajo | lbvh | bvh16 | 1.346 | 0.503 |
| grid | tinybvh | sah | bvh2 | 8.198 | 1.186 |
| grid | embree | medium | native | 10.560 | 1.406 |
| triangle-grid | bajo | sah | bvh16 | 0.006 | 0.005 |
| triangle-grid | embree | medium | native | 0.011 | 0.015 |
| triangle-instances | bajo | lbvh | bvh4 | 0.008 | 0.008 |
| triangle-instances | embree | high | native | 0.037 | 0.058 |

## Coverage

The matrix covers synthetic and real mesh geometry, closest-hit and early-exit any-hit, coherent camera ordering and the same rays shuffled to remove neighboring-ray coherence, plus an instance-heavy BLAS/TLAS scene (one reused BLAS, a 12x9 translated-instance grid). A one-triangle BLAS and its flattened 108-triangle equivalent isolate instance continuation from BLAS complexity. Traversal is single-calling-thread; build is measured with one CPU and all available CPUs. The core traversal suites report the best of eight timed repetitions after one warmup. Instance diagnostics report the median of eight repetitions, each averaged across eight timed traversal batches, to resolve small performance differences.

## Regular-grid closest-hit

Triangles per BLAS: 65536; instances: 1; rays: 262144; query: closest; ray order: structured.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh2 | 2 | 2 | scalar1 | 1 | 4.556 | 1.899 | 17.508 | 14.973 |  | 32767 | 6443188224.000 |
| bajo | hploc | bvh4 | 4 | 4 | scalar1 | 1 | 4.618 | 1.593 | 10.385 | 25.242 |  | 5461 | 6443188224.000 |
| bajo | hploc | bvh8 | 8 | 8 | scalar1 | 1 | 4.141 | 1.566 | 10.952 | 23.935 |  | 1641 | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 2.194 | 0.962 | 5.141 | 50.991 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 2.194 | 0.962 | 5.001 | 52.419 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet16 | 16 | 2.194 | 0.962 | 5.765 | 45.471 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet4 | 4 | 2.194 | 0.962 | 11.738 | 22.332 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet8 | 8 | 2.194 | 0.962 | 7.123 | 36.802 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 2.194 | 0.962 | 5.951 | 44.052 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 2.194 | 0.962 | 13.509 | 19.405 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 2.194 | 0.962 | 7.684 | 34.117 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 2.194 | 0.962 | 7.323 | 35.797 |  | 273 | 6443188224.000 |
| bajo | lbvh | bvh2 | 2 | 2 | scalar1 | 1 | 2.149 | 1.353 | 18.999 | 13.797 |  | 32767 | 6443188224.000 |
| bajo | lbvh | bvh4 | 4 | 4 | scalar1 | 1 | 1.660 | 0.940 | 11.274 | 23.253 |  | 5461 | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | adaptive-16-8-4-scalar | 16 | 1.496 | 0.766 | 5.711 | 45.900 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | adaptive-16-8-scalar | 16 | 1.496 | 0.766 | 5.616 | 46.682 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | coh-packet16 | 16 | 1.496 | 0.766 | 5.954 | 44.031 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | coh-packet4 | 4 | 1.496 | 0.766 | 9.480 | 27.654 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | coh-packet8 | 8 | 1.496 | 0.766 | 7.134 | 36.747 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | packet16 | 16 | 1.496 | 0.766 | 6.411 | 40.889 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | packet4 | 4 | 1.496 | 0.766 | 11.013 | 23.804 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | packet8 | 8 | 1.496 | 0.766 | 7.840 | 33.439 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | scalar1 | 1 | 1.496 | 0.766 | 11.729 | 22.350 |  | 4681 | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 1.346 | 0.503 | 7.762 | 33.771 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 1.346 | 0.503 | 7.460 | 35.139 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet16 | 16 | 1.346 | 0.503 | 8.419 | 31.137 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet4 | 4 | 1.346 | 0.503 | 12.731 | 20.592 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet8 | 8 | 1.346 | 0.503 | 9.754 | 26.876 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 1.346 | 0.503 | 8.535 | 30.715 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 1.346 | 0.503 | 14.385 | 18.224 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 1.346 | 0.503 | 10.457 | 25.069 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 1.346 | 0.503 | 7.464 | 35.119 |  | 273 | 6443188224.000 |
| bajo | median | bvh2 | 2 | 2 | scalar1 | 1 | 8.511 | 2.235 | 19.190 | 13.660 |  | 32767 | 6443188224.000 |
| bajo | median | bvh4 | 4 | 4 | scalar1 | 1 | 8.301 | 2.486 | 11.346 | 23.105 |  | 5461 | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | adaptive-16-8-4-scalar | 16 | 7.632 | 2.289 | 5.528 | 47.420 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | adaptive-16-8-scalar | 16 | 7.632 | 2.289 | 5.469 | 47.932 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | coh-packet16 | 16 | 7.632 | 2.289 | 6.155 | 42.591 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | coh-packet4 | 4 | 7.632 | 2.289 | 9.642 | 27.188 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | coh-packet8 | 8 | 7.632 | 2.289 | 7.069 | 37.083 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | packet16 | 16 | 7.632 | 2.289 | 6.571 | 39.894 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | packet4 | 4 | 7.632 | 2.289 | 11.788 | 22.239 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | packet8 | 8 | 7.632 | 2.289 | 7.985 | 32.830 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | scalar1 | 1 | 7.632 | 2.289 | 11.643 | 22.515 |  | 1193 | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 7.135 | 2.112 | 7.448 | 35.197 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 7.135 | 2.112 | 7.368 | 35.581 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet16 | 16 | 7.135 | 2.112 | 8.008 | 32.734 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet4 | 4 | 7.135 | 2.112 | 12.307 | 21.300 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet8 | 8 | 7.135 | 2.112 | 9.332 | 28.092 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | packet16 | 16 | 7.135 | 2.112 | 8.182 | 32.037 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | packet4 | 4 | 7.135 | 2.112 | 13.831 | 18.953 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | packet8 | 8 | 7.135 | 2.112 | 9.968 | 26.300 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | scalar1 | 1 | 7.135 | 2.112 | 7.549 | 34.724 |  | 273 | 6443188224.000 |
| bajo | sah | bvh2 | 2 | 2 | scalar1 | 1 | 13.857 | 2.869 | 18.379 | 14.263 |  | 32767 | 6443188224.000 |
| bajo | sah | bvh4 | 4 | 4 | scalar1 | 1 | 12.043 | 2.382 | 9.874 | 26.549 |  | 5461 | 6443188224.000 |
| bajo | sah | bvh8 | 8 | 8 | scalar1 | 1 | 10.302 | 2.289 | 10.001 | 26.211 |  | 1273 | 6443188224.000 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 9.135 | 1.892 | 8.317 | 31.520 |  | 273 | 6443188224.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 12.895 | 1.753 | 13.204 | 19.854 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 12.895 | 1.753 | 14.246 | 18.401 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 12.895 | 1.753 | 12.943 | 20.254 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 12.895 | 1.753 | 10.862 | 24.135 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 12.895 | 1.753 | 11.443 | 22.908 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 12.895 | 1.753 | 10.377 | 25.261 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | scalar1 | 1 | 12.895 | 1.753 | 16.942 | 15.473 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 11.299 | 1.676 | 13.215 | 19.837 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 11.299 | 1.676 | 14.146 | 18.531 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 11.299 | 1.676 | 12.961 | 20.226 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 11.299 | 1.676 | 10.833 | 24.199 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 11.299 | 1.676 | 11.515 | 22.766 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 11.299 | 1.676 | 10.464 | 25.052 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | scalar1 | 1 | 11.299 | 1.676 | 16.931 | 15.483 | 196608 |  | 6443188224.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 29.241 | 4.025 | 18.270 | 14.348 | 196608 |  | 6443188224.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 31.261 | 6.282 | 12.129 | 21.612 | 196608 |  | 6443188223.985 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 31.949 | 6.727 | 10.095 | 25.969 | 196608 |  | 6443188223.985 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 8.126 | 1.197 | 18.313 | 14.315 | 196608 |  | 6443188224.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 9.466 | 3.043 | 12.105 | 21.656 | 196608 |  | 6443188223.985 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 10.793 | 4.189 | 10.124 | 25.895 | 196608 |  | 6443188223.985 |

## Regular-grid any-hit

Triangles per BLAS: 65536; instances: 1; rays: 262144; query: any; ray order: structured.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet16 | 16 | 2.194 | 0.962 | 4.860 | 53.937 | 196608 |  | 196608.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 2.194 | 0.962 | 5.126 | 51.142 | 196608 |  | 196608.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 2.194 | 0.962 | 12.075 | 21.709 | 196608 |  | 196608.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 2.194 | 0.962 | 6.856 | 38.236 | 196608 |  | 196608.000 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 2.194 | 0.962 | 4.500 | 58.250 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet16 | 16 | 1.346 | 0.503 | 6.237 | 42.029 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 1.346 | 0.503 | 6.470 | 40.516 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 1.346 | 0.503 | 12.585 | 20.830 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 1.346 | 0.503 | 8.310 | 31.546 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 1.346 | 0.503 | 4.582 | 57.217 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet16 | 16 | 7.135 | 2.112 | 7.165 | 36.588 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | packet16 | 16 | 7.135 | 2.112 | 7.497 | 34.965 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | packet4 | 4 | 7.135 | 2.112 | 13.537 | 19.365 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | packet8 | 8 | 7.135 | 2.112 | 9.163 | 28.610 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | scalar1 | 1 | 7.135 | 2.112 | 4.551 | 57.597 | 196608 |  | 196608.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 12.922 | 1.840 | 11.482 | 22.831 | 196608 |  | 196608.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 12.922 | 1.840 | 13.757 | 19.056 | 196608 |  | 196608.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 12.922 | 1.840 | 12.294 | 21.323 | 196608 |  | 196608.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 12.922 | 1.840 | 6.052 | 43.316 | 196608 |  | 196608.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 12.922 | 1.840 | 7.706 | 34.018 | 196608 |  | 196608.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 12.922 | 1.840 | 6.325 | 41.448 | 196608 |  | 196608.000 |
| embree | high | native |  |  | scalar1 | 1 | 12.922 | 1.840 | 13.030 | 20.118 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 10.560 | 1.406 | 11.432 | 22.930 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 10.560 | 1.406 | 13.754 | 19.060 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 10.560 | 1.406 | 12.395 | 21.149 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 10.560 | 1.406 | 6.059 | 43.268 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 10.560 | 1.406 | 7.741 | 33.864 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 10.560 | 1.406 | 6.359 | 41.225 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | scalar1 | 1 | 10.560 | 1.406 | 13.042 | 20.100 | 196608 |  | 196608.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 29.232 | 4.119 | 17.355 | 15.105 | 196608 |  | 196608.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 31.140 | 5.918 | 7.846 | 33.412 | 196608 |  | 196608.000 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 32.268 | 6.327 | 5.968 | 43.928 | 196608 |  | 196608.000 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 8.198 | 1.186 | 17.477 | 14.999 | 196608 |  | 196608.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 9.543 | 3.131 | 7.810 | 33.566 | 196608 |  | 196608.000 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 10.655 | 4.481 | 5.953 | 44.033 | 196608 |  | 196608.000 |

## Dragon camera closest-hit

Triangles per BLAS: 249882; instances: 1; rays: 589824; query: closest; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 19.596 | 7.452 | 7.579 | 77.824 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 19.596 | 7.452 | 7.729 | 76.313 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet16 | 16 | 19.596 | 7.452 | 9.311 | 63.348 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet4 | 4 | 19.596 | 7.452 | 11.589 | 50.896 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet8 | 8 | 19.596 | 7.452 | 10.042 | 58.736 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 19.596 | 7.452 | 11.356 | 51.938 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 19.596 | 7.452 | 15.867 | 37.172 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 19.596 | 7.452 | 12.838 | 45.944 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 19.596 | 7.452 | 10.226 | 57.678 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 8.678 | 3.562 | 7.915 | 74.520 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 8.678 | 3.562 | 8.044 | 73.325 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet16 | 16 | 8.678 | 3.562 | 9.921 | 59.453 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet4 | 4 | 8.678 | 3.562 | 12.824 | 45.995 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet8 | 8 | 8.678 | 3.562 | 10.859 | 54.314 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 8.678 | 3.562 | 12.695 | 46.462 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 8.678 | 3.562 | 16.256 | 36.284 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 8.678 | 3.562 | 13.195 | 44.702 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 8.678 | 3.562 | 10.140 | 58.168 | 71597 |  | 7943562615.175 |
| bajo | median | bvh16 | 16 | 16 | scalar1 | 1 | 38.509 | 10.984 | 10.101 | 58.390 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 48.182 | 8.667 | 7.503 | 78.609 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 48.182 | 8.667 | 7.574 | 77.875 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet16 | 16 | 48.182 | 8.667 | 9.167 | 64.342 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet4 | 4 | 48.182 | 8.667 | 12.486 | 47.237 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet8 | 8 | 48.182 | 8.667 | 10.970 | 53.769 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 48.182 | 8.667 | 10.953 | 53.851 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 48.182 | 8.667 | 14.840 | 39.747 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 48.182 | 8.667 | 12.108 | 48.714 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 48.182 | 8.667 | 9.160 | 64.393 | 71597 |  | 7943562615.175 |
| embree | high | native |  |  | coh-packet16 | 16 | 262.240 | 29.375 | 9.530 | 61.889 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | coh-packet4 | 4 | 262.240 | 29.375 | 11.296 | 52.216 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | coh-packet8 | 8 | 262.240 | 29.375 | 8.889 | 66.357 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet16 | 16 | 262.240 | 29.375 | 11.944 | 49.380 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet4 | 4 | 262.240 | 29.375 | 12.820 | 46.010 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet8 | 8 | 262.240 | 29.375 | 11.640 | 50.671 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | scalar1 | 1 | 262.240 | 29.375 | 18.602 | 31.707 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet16 | 16 | 48.107 | 6.209 | 9.508 | 62.031 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet4 | 4 | 48.107 | 6.209 | 11.403 | 51.724 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet8 | 8 | 48.107 | 6.209 | 8.813 | 66.924 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet16 | 16 | 48.107 | 6.209 | 11.954 | 49.339 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet4 | 4 | 48.107 | 6.209 | 12.764 | 46.210 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet8 | 8 | 48.107 | 6.209 | 11.645 | 50.651 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | scalar1 | 1 | 48.107 | 6.209 | 18.226 | 32.362 | 71598 |  | 7943741995.515 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 351.721 | 46.841 | 24.872 | 23.715 | 71599 |  | 7943796499.200 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 365.083 | 65.292 | 16.823 | 35.061 | 71599 |  | 7943796499.445 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 369.372 | 70.293 | 14.355 | 41.089 | 71599 |  | 7943796499.439 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.580 | 4.971 | 24.810 | 23.774 | 71599 |  | 7943796499.200 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 61.583 | 25.486 | 16.764 | 35.183 | 71599 |  | 7943796499.445 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 66.723 | 34.207 | 14.258 | 41.369 | 71599 |  | 7943796499.439 |

## Dragon camera any-hit

Triangles per BLAS: 249882; instances: 1; rays: 589824; query: any; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet16 | 16 | 19.596 | 7.452 | 6.257 | 94.269 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 19.596 | 7.452 | 8.192 | 72.003 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 19.596 | 7.452 | 12.149 | 48.548 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 19.596 | 7.452 | 10.224 | 57.692 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 19.596 | 7.452 | 8.148 | 72.385 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet16 | 16 | 8.678 | 3.562 | 6.422 | 91.848 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 8.678 | 3.562 | 8.378 | 70.398 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 8.678 | 3.562 | 12.810 | 46.046 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 8.678 | 3.562 | 10.448 | 56.452 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 8.678 | 3.562 | 8.646 | 68.219 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet16 | 16 | 48.182 | 8.667 | 5.981 | 98.611 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 48.182 | 8.667 | 8.081 | 72.985 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 48.182 | 8.667 | 11.544 | 51.094 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 48.182 | 8.667 | 9.775 | 60.342 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 48.182 | 8.667 | 7.473 | 78.925 | 71597 |  | 71597.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 263.207 | 27.788 | 7.124 | 82.790 | 71598 |  | 71598.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 263.207 | 27.788 | 10.185 | 57.912 | 71598 |  | 71598.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 263.207 | 27.788 | 8.676 | 67.985 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 263.207 | 27.788 | 6.340 | 93.036 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 263.207 | 27.788 | 8.988 | 65.621 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 263.207 | 27.788 | 6.784 | 86.940 | 71598 |  | 71598.000 |
| embree | high | native |  |  | scalar1 | 1 | 263.207 | 27.788 | 15.776 | 37.387 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 46.384 | 5.756 | 7.150 | 82.491 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 46.384 | 5.756 | 10.240 | 57.600 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 46.384 | 5.756 | 8.090 | 72.910 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 46.384 | 5.756 | 6.448 | 91.469 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 46.384 | 5.756 | 9.075 | 64.996 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 46.384 | 5.756 | 6.770 | 87.128 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | scalar1 | 1 | 46.384 | 5.756 | 15.584 | 37.847 | 71598 |  | 71598.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 352.281 | 46.750 | 19.116 | 30.856 | 71599 |  | 71599.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 364.908 | 65.672 | 13.419 | 43.955 | 71599 |  | 71599.000 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 369.397 | 69.840 | 11.433 | 51.591 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.390 | 5.056 | 18.977 | 31.082 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 60.940 | 26.926 | 13.475 | 43.771 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 66.954 | 31.988 | 11.410 | 51.694 | 71599 |  | 71599.000 |

## Dragon shuffled closest-hit

Triangles per BLAS: 249882; instances: 1; rays: 589824; query: closest; ray order: shuffled-camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 19.596 | 7.452 | 32.644 | 18.068 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 19.596 | 7.452 | 17.013 | 34.669 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 19.596 | 7.452 | 23.787 | 24.796 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 19.596 | 7.452 | 21.737 | 27.134 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 19.596 | 7.452 | 28.965 | 20.364 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 19.596 | 7.452 | 15.524 | 37.994 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 8.678 | 3.562 | 34.366 | 17.163 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 8.678 | 3.562 | 17.550 | 33.609 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 8.678 | 3.562 | 24.001 | 24.575 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 8.678 | 3.562 | 25.163 | 23.440 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 8.678 | 3.562 | 28.975 | 20.356 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 8.678 | 3.562 | 16.013 | 36.834 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 48.182 | 8.667 | 29.373 | 20.081 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 48.182 | 8.667 | 15.955 | 36.968 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 48.182 | 8.667 | 22.886 | 25.772 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 48.182 | 8.667 | 20.652 | 28.561 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 48.182 | 8.667 | 27.732 | 21.269 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 48.182 | 8.667 | 14.339 | 41.136 | 71597 |  | 7943562615.175 |
| embree | high | native |  |  | coh-packet16 | 16 | 263.190 | 28.047 | 31.054 | 18.994 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | coh-packet4 | 4 | 263.190 | 28.047 | 31.397 | 18.786 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | coh-packet8 | 8 | 263.190 | 28.047 | 32.114 | 18.367 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet16 | 16 | 263.190 | 28.047 | 14.636 | 40.301 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet4 | 4 | 263.190 | 28.047 | 15.495 | 38.066 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet8 | 8 | 263.190 | 28.047 | 14.305 | 41.231 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | scalar1 | 1 | 263.190 | 28.047 | 19.786 | 29.810 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet16 | 16 | 46.665 | 5.865 | 31.248 | 18.876 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet4 | 4 | 46.665 | 5.865 | 31.508 | 18.720 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet8 | 8 | 46.665 | 5.865 | 32.254 | 18.287 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet16 | 16 | 46.665 | 5.865 | 14.679 | 40.180 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet4 | 4 | 46.665 | 5.865 | 15.392 | 38.320 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet8 | 8 | 46.665 | 5.865 | 14.297 | 41.255 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | scalar1 | 1 | 46.665 | 5.865 | 19.801 | 29.788 | 71598 |  | 7943741995.515 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 351.792 | 46.464 | 31.828 | 18.531 | 71599 |  | 7943796499.200 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 364.714 | 65.848 | 19.984 | 29.515 | 71599 |  | 7943796499.445 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 368.133 | 70.038 | 16.927 | 34.845 | 71599 |  | 7943796499.439 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.218 | 5.186 | 31.791 | 18.553 | 71599 |  | 7943796499.200 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 61.008 | 25.532 | 19.961 | 29.549 | 71599 |  | 7943796499.445 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 66.616 | 31.741 | 17.028 | 34.638 | 71599 |  | 7943796499.439 |

## Dragon shuffled any-hit

Triangles per BLAS: 249882; instances: 1; rays: 589824; query: any; ray order: shuffled-camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 19.596 | 7.452 | 18.470 | 31.935 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 19.596 | 7.452 | 16.877 | 34.949 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 19.596 | 7.452 | 23.765 | 24.819 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 19.596 | 7.452 | 10.467 | 56.350 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 8.678 | 3.562 | 18.817 | 31.345 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 8.678 | 3.562 | 20.438 | 28.859 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 8.678 | 3.562 | 23.723 | 24.862 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 8.678 | 3.562 | 11.203 | 52.648 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 48.182 | 8.667 | 18.147 | 32.502 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 48.182 | 8.667 | 16.218 | 36.369 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 48.182 | 8.667 | 23.209 | 25.414 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 48.182 | 8.667 | 9.573 | 61.611 | 71597 |  | 71597.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 262.380 | 28.598 | 23.343 | 25.268 | 71598 |  | 71598.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 262.380 | 28.598 | 25.876 | 22.794 | 71598 |  | 71598.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 262.380 | 28.598 | 25.384 | 23.236 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 262.380 | 28.598 | 9.959 | 59.222 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 262.380 | 28.598 | 12.092 | 48.776 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 262.380 | 28.598 | 10.531 | 56.006 | 71598 |  | 71598.000 |
| embree | high | native |  |  | scalar1 | 1 | 262.380 | 28.598 | 17.073 | 34.548 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 46.498 | 5.693 | 23.322 | 25.290 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 46.498 | 5.693 | 25.906 | 22.768 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 46.498 | 5.693 | 25.390 | 23.231 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 46.498 | 5.693 | 10.013 | 58.906 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 46.498 | 5.693 | 12.068 | 48.874 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 46.498 | 5.693 | 10.547 | 55.922 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | scalar1 | 1 | 46.498 | 5.693 | 16.991 | 34.715 | 71598 |  | 71598.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 351.998 | 46.534 | 26.091 | 22.606 | 71599 |  | 71599.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 365.231 | 65.882 | 17.045 | 34.605 | 71599 |  | 71599.000 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 371.468 | 70.318 | 14.378 | 41.023 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.427 | 4.744 | 26.280 | 22.444 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 61.096 | 26.603 | 16.934 | 34.830 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 66.504 | 31.999 | 14.365 | 41.061 | 71599 |  | 71599.000 |

## Instanced Dragon closest-hit

Triangles per BLAS: 249882; instances: 108; rays: 147456; query: closest; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh4 | 4 | 1 | packet16 | 16 | 19.564 | 7.698 | 1.881 | 78.374 | 8250 |  | 1022292747.551 |
| bajo | hploc | bvh4 | 4 | 1 | packet4 | 4 | 19.564 | 7.698 | 2.343 | 62.943 | 8250 |  | 1022292747.551 |
| bajo | hploc | bvh4 | 4 | 1 | packet8 | 8 | 19.564 | 7.698 | 1.972 | 74.771 | 8250 |  | 1022292747.551 |
| bajo | hploc | bvh4 | 4 | 1 | scalar1 | 1 | 19.564 | 7.698 | 2.653 | 55.579 | 8250 |  | 1022292747.551 |
| bajo | lbvh | bvh4 | 4 | 1 | packet16 | 16 | 9.468 | 3.513 | 1.948 | 75.699 | 8250 |  | 1022292856.202 |
| bajo | lbvh | bvh4 | 4 | 1 | packet4 | 4 | 9.468 | 3.513 | 2.405 | 61.323 | 8250 |  | 1022292856.202 |
| bajo | lbvh | bvh4 | 4 | 1 | packet8 | 8 | 9.468 | 3.513 | 2.054 | 71.789 | 8250 |  | 1022292856.202 |
| bajo | lbvh | bvh4 | 4 | 1 | scalar1 | 1 | 9.468 | 3.513 | 2.712 | 54.370 | 8250 |  | 1022292856.202 |
| bajo | sah | bvh4 | 4 | 1 | packet16 | 16 | 49.518 | 8.808 | 1.793 | 82.241 | 8250 |  | 1022292845.456 |
| bajo | sah | bvh4 | 4 | 1 | packet4 | 4 | 49.518 | 8.808 | 2.273 | 64.870 | 8250 |  | 1022292845.456 |
| bajo | sah | bvh4 | 4 | 1 | packet8 | 8 | 49.518 | 8.808 | 1.894 | 77.852 | 8250 |  | 1022292845.456 |
| bajo | sah | bvh4 | 4 | 1 | scalar1 | 1 | 49.518 | 8.808 | 2.616 | 56.365 | 8250 |  | 1022292845.456 |
| embree | high | native |  |  | coh-packet16 | 16 | 263.128 | 28.595 | 3.964 | 37.195 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | coh-packet4 | 4 | 263.128 | 28.595 | 3.790 | 38.910 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | coh-packet8 | 8 | 263.128 | 28.595 | 3.504 | 42.084 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | inc-packet16 | 16 | 263.128 | 28.595 | 2.081 | 70.848 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | inc-packet4 | 4 | 263.128 | 28.595 | 2.334 | 63.168 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | inc-packet8 | 8 | 263.128 | 28.595 | 1.871 | 78.805 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | scalar1 | 1 | 263.128 | 28.595 | 4.104 | 35.926 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | coh-packet16 | 16 | 47.781 | 6.140 | 3.962 | 37.222 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | coh-packet4 | 4 | 47.781 | 6.140 | 3.763 | 39.186 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | coh-packet8 | 8 | 47.781 | 6.140 | 3.494 | 42.199 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | inc-packet16 | 16 | 47.781 | 6.140 | 2.093 | 70.453 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | inc-packet4 | 4 | 47.781 | 6.140 | 2.369 | 62.253 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | inc-packet8 | 8 | 47.781 | 6.140 | 1.885 | 78.233 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | scalar1 | 1 | 47.781 | 6.140 | 4.115 | 35.836 | 8256 |  | 1024176197.799 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 353.124 | 46.769 | 5.095 | 28.940 | 8256 |  | 1024143130.934 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 365.131 | 66.163 | 3.487 | 42.293 | 8256 |  | 1024092270.370 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 370.773 | 70.147 | 3.135 | 47.030 | 8256 |  | 1024092270.381 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.426 | 5.132 | 5.267 | 27.998 | 8256 |  | 1024143130.934 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 61.042 | 25.361 | 3.467 | 42.531 | 8256 |  | 1024092270.370 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 66.760 | 31.720 | 3.117 | 47.304 | 8256 |  | 1024092270.381 |

## Instanced Dragon any-hit

Triangles per BLAS: 249882; instances: 108; rays: 147456; query: any; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh4 | 4 | 1 | packet16 | 16 | 19.564 | 7.698 | 1.315 | 112.154 | 8250 |  | 8250.000 |
| bajo | hploc | bvh4 | 4 | 1 | packet4 | 4 | 19.564 | 7.698 | 1.776 | 83.045 | 8250 |  | 8250.000 |
| bajo | hploc | bvh4 | 4 | 1 | packet8 | 8 | 19.564 | 7.698 | 1.461 | 100.938 | 8250 |  | 8250.000 |
| bajo | hploc | bvh4 | 4 | 1 | scalar1 | 1 | 19.564 | 7.698 | 2.209 | 66.750 | 8250 |  | 8250.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet16 | 16 | 9.468 | 3.513 | 1.416 | 104.107 | 8250 |  | 8250.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet4 | 4 | 9.468 | 3.513 | 1.817 | 81.157 | 8250 |  | 8250.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet8 | 8 | 9.468 | 3.513 | 1.541 | 95.699 | 8250 |  | 8250.000 |
| bajo | lbvh | bvh4 | 4 | 1 | scalar1 | 1 | 9.468 | 3.513 | 2.294 | 64.284 | 8250 |  | 8250.000 |
| bajo | sah | bvh4 | 4 | 1 | packet16 | 16 | 49.518 | 8.808 | 1.243 | 118.667 | 8250 |  | 8250.000 |
| bajo | sah | bvh4 | 4 | 1 | packet4 | 4 | 49.518 | 8.808 | 1.677 | 87.951 | 8250 |  | 8250.000 |
| bajo | sah | bvh4 | 4 | 1 | packet8 | 8 | 49.518 | 8.808 | 1.369 | 107.709 | 8250 |  | 8250.000 |
| bajo | sah | bvh4 | 4 | 1 | scalar1 | 1 | 49.518 | 8.808 | 2.189 | 67.367 | 8250 |  | 8250.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 262.510 | 28.012 | 2.899 | 50.857 | 8256 |  | 8256.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 262.510 | 28.012 | 3.189 | 46.245 | 8256 |  | 8256.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 262.510 | 28.012 | 2.818 | 52.321 | 8256 |  | 8256.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 262.510 | 28.012 | 1.456 | 101.304 | 8256 |  | 8256.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 262.510 | 28.012 | 2.066 | 71.389 | 8256 |  | 8256.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 262.510 | 28.012 | 1.571 | 93.854 | 8256 |  | 8256.000 |
| embree | high | native |  |  | scalar1 | 1 | 262.510 | 28.012 | 3.703 | 39.816 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 46.804 | 6.018 | 2.897 | 50.903 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 46.804 | 6.018 | 3.193 | 46.179 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 46.804 | 6.018 | 2.835 | 52.017 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 46.804 | 6.018 | 1.454 | 101.429 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 46.804 | 6.018 | 2.054 | 71.776 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 46.804 | 6.018 | 1.576 | 93.542 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | scalar1 | 1 | 46.804 | 6.018 | 3.708 | 39.762 | 8256 |  | 8256.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 352.272 | 46.569 | 4.181 | 35.267 | 8256 |  | 8256.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 368.595 | 65.107 | 2.999 | 49.175 | 8256 |  | 8256.000 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 373.344 | 69.601 | 2.661 | 55.420 | 8256 |  | 8256.000 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.488 | 5.080 | 4.320 | 34.134 | 8256 |  | 8256.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 61.686 | 27.331 | 2.948 | 50.018 | 8256 |  | 8256.000 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 67.899 | 31.925 | 2.656 | 55.520 | 8256 |  | 8256.000 |

## Instanced triangle closest-hit

Triangles per BLAS: 1; instances: 108; rays: 147456; query: closest; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh4 | 4 | 1 | packet16 | 16 | 0.014 | 0.013 | 0.712 | 207.188 | 20416 |  | 1177505.908 |
| bajo | hploc | bvh4 | 4 | 1 | packet4 | 4 | 0.014 | 0.013 | 1.164 | 126.646 | 20416 |  | 1177505.908 |
| bajo | hploc | bvh4 | 4 | 1 | packet8 | 8 | 0.014 | 0.013 | 0.788 | 187.174 | 20416 |  | 1177505.908 |
| bajo | hploc | bvh4 | 4 | 1 | scalar1 | 1 | 0.014 | 0.013 | 2.560 | 57.604 | 20416 |  | 1177505.908 |
| bajo | lbvh | bvh4 | 4 | 1 | packet16 | 16 | 0.008 | 0.008 | 0.683 | 215.956 | 20416 |  | 1177505.908 |
| bajo | lbvh | bvh4 | 4 | 1 | packet4 | 4 | 0.008 | 0.008 | 1.078 | 136.850 | 20416 |  | 1177505.908 |
| bajo | lbvh | bvh4 | 4 | 1 | packet8 | 8 | 0.008 | 0.008 | 0.751 | 196.464 | 20416 |  | 1177505.908 |
| bajo | lbvh | bvh4 | 4 | 1 | scalar1 | 1 | 0.008 | 0.008 | 2.506 | 58.831 | 20416 |  | 1177505.908 |
| bajo | sah | bvh4 | 4 | 1 | packet16 | 16 | 0.023 | 0.023 | 0.683 | 215.744 | 20416 |  | 1177505.908 |
| bajo | sah | bvh4 | 4 | 1 | packet4 | 4 | 0.023 | 0.023 | 1.089 | 135.449 | 20416 |  | 1177505.908 |
| bajo | sah | bvh4 | 4 | 1 | packet8 | 8 | 0.023 | 0.023 | 0.778 | 189.475 | 20416 |  | 1177505.908 |
| bajo | sah | bvh4 | 4 | 1 | scalar1 | 1 | 0.023 | 0.023 | 2.512 | 58.700 | 20416 |  | 1177505.908 |
| embree | high | native |  |  | coh-packet16 | 16 | 0.037 | 0.058 | 1.005 | 146.663 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | coh-packet4 | 4 | 0.037 | 0.058 | 1.293 | 114.077 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | coh-packet8 | 8 | 0.037 | 0.058 | 0.816 | 180.779 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | inc-packet16 | 16 | 0.037 | 0.058 | 1.130 | 130.537 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | inc-packet4 | 4 | 0.037 | 0.058 | 1.378 | 107.009 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | inc-packet8 | 8 | 0.037 | 0.058 | 0.972 | 151.779 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | scalar1 | 1 | 0.037 | 0.058 | 3.859 | 38.211 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | coh-packet16 | 16 | 0.038 | 0.058 | 1.008 | 146.215 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | coh-packet4 | 4 | 0.038 | 0.058 | 1.291 | 114.241 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | coh-packet8 | 8 | 0.038 | 0.058 | 0.814 | 181.168 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | inc-packet16 | 16 | 0.038 | 0.058 | 1.131 | 130.433 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | inc-packet4 | 4 | 0.038 | 0.058 | 1.381 | 106.774 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | inc-packet8 | 8 | 0.038 | 0.058 | 0.968 | 152.345 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | scalar1 | 1 | 0.038 | 0.058 | 3.856 | 38.245 | 20416 |  | 1177505.898 |

## Instanced triangle any-hit

Triangles per BLAS: 1; instances: 108; rays: 147456; query: any; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh4 | 4 | 1 | packet16 | 16 | 0.014 | 0.013 | 0.420 | 351.267 | 20416 |  | 20416.000 |
| bajo | hploc | bvh4 | 4 | 1 | packet4 | 4 | 0.014 | 0.013 | 0.935 | 157.696 | 20416 |  | 20416.000 |
| bajo | hploc | bvh4 | 4 | 1 | packet8 | 8 | 0.014 | 0.013 | 0.551 | 267.768 | 20416 |  | 20416.000 |
| bajo | hploc | bvh4 | 4 | 1 | scalar1 | 1 | 0.014 | 0.013 | 1.916 | 76.941 | 20416 |  | 20416.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet16 | 16 | 0.008 | 0.008 | 0.387 | 381.224 | 20416 |  | 20416.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet4 | 4 | 0.008 | 0.008 | 0.847 | 174.094 | 20416 |  | 20416.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet8 | 8 | 0.008 | 0.008 | 0.496 | 297.205 | 20416 |  | 20416.000 |
| bajo | lbvh | bvh4 | 4 | 1 | scalar1 | 1 | 0.008 | 0.008 | 1.892 | 77.948 | 20416 |  | 20416.000 |
| bajo | sah | bvh4 | 4 | 1 | packet16 | 16 | 0.023 | 0.023 | 0.392 | 376.412 | 20416 |  | 20416.000 |
| bajo | sah | bvh4 | 4 | 1 | packet4 | 4 | 0.023 | 0.023 | 0.850 | 173.467 | 20416 |  | 20416.000 |
| bajo | sah | bvh4 | 4 | 1 | packet8 | 8 | 0.023 | 0.023 | 0.496 | 297.030 | 20416 |  | 20416.000 |
| bajo | sah | bvh4 | 4 | 1 | scalar1 | 1 | 0.023 | 0.023 | 1.895 | 77.830 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 0.037 | 0.064 | 0.670 | 219.984 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 0.037 | 0.064 | 1.294 | 113.997 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 0.037 | 0.064 | 0.801 | 183.982 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 0.037 | 0.064 | 0.577 | 255.771 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 0.037 | 0.064 | 1.144 | 128.904 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 0.037 | 0.064 | 0.656 | 224.715 | 20416 |  | 20416.000 |
| embree | high | native |  |  | scalar1 | 1 | 0.037 | 0.064 | 3.379 | 43.640 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 0.038 | 0.063 | 0.675 | 218.312 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 0.038 | 0.063 | 1.295 | 113.863 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 0.038 | 0.063 | 0.798 | 184.782 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 0.038 | 0.063 | 0.574 | 256.833 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 0.038 | 0.063 | 1.149 | 128.292 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 0.038 | 0.063 | 0.655 | 225.146 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | scalar1 | 1 | 0.038 | 0.063 | 3.402 | 43.340 | 20416 |  | 20416.000 |

## Flattened triangle grid closest-hit

Triangles per BLAS: 108; instances: 1; rays: 147456; query: closest; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | sah | bvh16 | 16 | 8 | coh-packet16 | 16 | 0.009 | 0.009 | 0.662 | 222.743 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet16 | 16 | 0.006 | 0.005 | 0.864 | 170.614 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet4 | 4 | 0.006 | 0.005 | 1.906 | 77.351 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet8 | 8 | 0.006 | 0.005 | 1.292 | 114.134 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 0.006 | 0.005 | 0.752 | 196.195 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 0.006 | 0.005 | 1.766 | 83.481 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 0.006 | 0.005 | 1.140 | 129.364 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 0.006 | 0.005 | 1.990 | 74.083 | 20416 |  | 2281401.908 |
| embree | high | native |  |  | coh-packet16 | 16 | 0.016 | 0.034 | 0.898 | 164.199 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | coh-packet4 | 4 | 0.016 | 0.034 | 1.152 | 127.963 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | coh-packet8 | 8 | 0.016 | 0.034 | 0.700 | 210.714 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | inc-packet16 | 16 | 0.016 | 0.034 | 1.507 | 97.815 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | inc-packet4 | 4 | 0.016 | 0.034 | 1.612 | 91.484 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | inc-packet8 | 8 | 0.016 | 0.034 | 1.398 | 105.454 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | scalar1 | 1 | 0.016 | 0.034 | 3.364 | 43.836 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | coh-packet16 | 16 | 0.012 | 0.016 | 0.893 | 165.118 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | coh-packet4 | 4 | 0.012 | 0.016 | 1.150 | 128.261 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | coh-packet8 | 8 | 0.012 | 0.016 | 0.700 | 210.538 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | inc-packet16 | 16 | 0.012 | 0.016 | 1.508 | 97.753 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | inc-packet4 | 4 | 0.012 | 0.016 | 1.621 | 90.955 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | inc-packet8 | 8 | 0.012 | 0.016 | 1.402 | 105.203 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | scalar1 | 1 | 0.012 | 0.016 | 3.366 | 43.805 | 20416 |  | 2281401.897 |

## Flattened triangle grid any-hit

Triangles per BLAS: 108; instances: 1; rays: 147456; query: any; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | sah | bvh16 | 16 | 8 | coh-packet16 | 16 | 0.009 | 0.009 | 0.482 | 306.143 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet16 | 16 | 0.006 | 0.005 | 0.608 | 242.692 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet4 | 4 | 0.006 | 0.005 | 1.410 | 104.613 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet8 | 8 | 0.006 | 0.005 | 0.844 | 174.650 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 0.006 | 0.005 | 0.582 | 253.471 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 0.006 | 0.005 | 1.486 | 99.236 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 0.006 | 0.005 | 0.834 | 176.767 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 0.006 | 0.005 | 1.183 | 124.624 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 0.017 | 0.024 | 0.501 | 294.548 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 0.017 | 0.024 | 1.088 | 135.531 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 0.017 | 0.024 | 0.649 | 227.036 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 0.017 | 0.024 | 0.517 | 285.462 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 0.017 | 0.024 | 0.992 | 148.653 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 0.017 | 0.024 | 0.593 | 248.709 | 20416 |  | 20416.000 |
| embree | high | native |  |  | scalar1 | 1 | 0.017 | 0.024 | 2.964 | 49.744 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 0.011 | 0.015 | 0.505 | 292.256 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 0.011 | 0.015 | 1.109 | 132.945 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 0.011 | 0.015 | 0.649 | 227.184 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 0.011 | 0.015 | 0.531 | 277.822 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 0.011 | 0.015 | 0.992 | 148.576 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 0.011 | 0.015 | 0.591 | 249.388 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | scalar1 | 1 | 0.011 | 0.015 | 2.965 | 49.729 | 20416 |  | 20416.000 |
