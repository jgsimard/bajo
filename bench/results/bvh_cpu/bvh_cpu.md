# CPU BVH benchmark results

- **Date:** 2026-09-02T18:22:24-04:00
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
| Dragon camera any-hit | bajo | sah | bvh16 | coh-packet16 | 16 | 58.302 | 17.591 | 6.004 | 98.231 | 0.000 |
| Dragon camera any-hit | embree | medium | native | inc-packet16 | 16 | 49.326 | 6.032 | 6.477 | 91.071 | -7.289 |
| Dragon camera any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 67.620 | 34.490 | 11.490 | 51.335 | -47.741 |
| Dragon camera closest-hit | bajo | sah | bvh16 | adaptive-16-8-4-scalar | 16 | 58.302 | 17.591 | 7.566 | 77.953 | 0.000 |
| Dragon camera closest-hit | embree | high | native | coh-packet8 | 8 | 264.886 | 28.606 | 8.996 | 65.567 | -15.889 |
| Dragon camera closest-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 68.282 | 32.670 | 14.349 | 41.105 | -47.270 |
| Instanced Dragon any-hit | bajo | sah | bvh4 | packet16 | 16 | 58.605 | 16.955 | 1.230 | 119.900 | 0.000 |
| Instanced Dragon any-hit | embree | high | native | inc-packet16 | 16 | 261.560 | 29.718 | 1.452 | 101.580 | -15.279 |
| Instanced Dragon any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 67.791 | 35.499 | 2.664 | 55.356 | -53.832 |
| Instanced Dragon closest-hit | bajo | sah | bvh4 | packet16 | 16 | 58.605 | 16.955 | 1.759 | 83.821 | 0.000 |
| Instanced Dragon closest-hit | embree | high | native | inc-packet8 | 8 | 264.751 | 28.560 | 1.879 | 78.482 | -6.370 |
| Instanced Dragon closest-hit | tinybvh | high | bvh8 | scalar1 | 1 | 369.049 | 73.298 | 3.139 | 46.980 | -43.952 |
| Dragon shuffled any-hit | bajo | sah | bvh16 | scalar1 | 1 | 58.302 | 17.591 | 9.537 | 61.843 | 0.000 |
| Dragon shuffled any-hit | embree | high | native | inc-packet16 | 16 | 264.278 | 28.990 | 10.039 | 58.755 | -4.993 |
| Dragon shuffled any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 66.956 | 34.343 | 14.592 | 40.420 | -34.641 |
| Dragon shuffled closest-hit | bajo | sah | bvh16 | scalar1 | 1 | 58.302 | 17.591 | 14.274 | 41.321 | 0.000 |
| Dragon shuffled closest-hit | embree | high | native | inc-packet8 | 8 | 263.428 | 28.187 | 14.444 | 40.835 | -1.176 |
| Dragon shuffled closest-hit | tinybvh | high | bvh8 | scalar1 | 1 | 368.183 | 72.321 | 17.140 | 34.413 | -16.718 |
| Regular-grid any-hit | bajo | hploc | bvh16 | scalar1 | 1 | 2.285 | 1.301 | 4.500 | 58.253 | 0.000 |
| Regular-grid any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 10.698 | 4.994 | 5.951 | 44.048 | -24.385 |
| Regular-grid any-hit | embree | medium | native | inc-packet16 | 16 | 10.714 | 1.489 | 6.018 | 43.560 | -25.223 |
| Regular-grid closest-hit | bajo | hploc | bvh16 | adaptive-16-8-scalar | 16 | 2.285 | 1.301 | 4.948 | 52.983 | 0.000 |
| Regular-grid closest-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 10.872 | 4.515 | 10.166 | 25.786 | -51.332 |
| Regular-grid closest-hit | embree | high | native | inc-packet8 | 8 | 13.011 | 1.910 | 10.434 | 25.125 | -52.579 |
| Flattened triangle grid any-hit | bajo | sah | bvh16 | coh-packet16 | 16 | 0.017 | 0.017 | 0.482 | 305.661 | 0.000 |
| Flattened triangle grid any-hit | embree | medium | native | inc-packet16 | 16 | 0.011 | 0.016 | 0.490 | 300.715 | -1.618 |
| Flattened triangle grid closest-hit | bajo | sah | bvh16 | coh-packet16 | 16 | 0.017 | 0.017 | 0.663 | 222.305 | 0.000 |
| Flattened triangle grid closest-hit | embree | high | native | coh-packet8 | 8 | 0.017 | 0.027 | 0.699 | 211.028 | -5.073 |
| Instanced triangle any-hit | bajo | lbvh | bvh4 | packet16 | 16 | 0.009 | 0.009 | 0.391 | 377.030 | 0.000 |
| Instanced triangle any-hit | embree | medium | native | inc-packet16 | 16 | 0.037 | 0.058 | 0.569 | 259.190 | -31.255 |
| Instanced triangle closest-hit | bajo | lbvh | bvh4 | packet16 | 16 | 0.009 | 0.009 | 0.685 | 215.236 | 0.000 |
| Instanced triangle closest-hit | embree | medium | native | coh-packet8 | 8 | 0.037 | 0.062 | 0.815 | 180.955 | -15.927 |

## Best scalar traversal per implementation

This removes packet-width advantages and is the fairest direct comparison with TinyBVH's scalar API.

| Workload | Implementation | Build | Layout | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | vs Bajo (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dragon camera any-hit | bajo | sah | bvh16 | scalar1 | 1 | 58.302 | 17.591 | 7.609 | 77.521 | 0.000 |
| Dragon camera any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 67.620 | 34.490 | 11.490 | 51.335 | -33.779 |
| Dragon camera any-hit | embree | medium | native | scalar1 | 1 | 49.326 | 6.032 | 16.011 | 36.838 | -52.480 |
| Dragon camera closest-hit | bajo | sah | bvh16 | scalar1 | 1 | 58.302 | 17.591 | 9.309 | 63.359 | 0.000 |
| Dragon camera closest-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 68.282 | 32.670 | 14.349 | 41.105 | -35.124 |
| Dragon camera closest-hit | embree | high | native | scalar1 | 1 | 264.886 | 28.606 | 18.525 | 31.839 | -49.748 |
| Instanced Dragon any-hit | bajo | sah | bvh4 | scalar1 | 1 | 58.605 | 16.955 | 2.150 | 68.597 | 0.000 |
| Instanced Dragon any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 67.791 | 35.499 | 2.664 | 55.356 | -19.303 |
| Instanced Dragon any-hit | embree | medium | native | scalar1 | 1 | 47.129 | 6.123 | 3.703 | 39.817 | -41.955 |
| Instanced Dragon closest-hit | bajo | sah | bvh4 | scalar1 | 1 | 58.605 | 16.955 | 2.534 | 58.181 | 0.000 |
| Instanced Dragon closest-hit | tinybvh | high | bvh8 | scalar1 | 1 | 369.049 | 73.298 | 3.139 | 46.980 | -19.252 |
| Instanced Dragon closest-hit | embree | medium | native | scalar1 | 1 | 49.458 | 6.346 | 4.141 | 35.608 | -38.798 |
| Dragon shuffled any-hit | bajo | sah | bvh16 | scalar1 | 1 | 58.302 | 17.591 | 9.537 | 61.843 | 0.000 |
| Dragon shuffled any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 66.956 | 34.343 | 14.592 | 40.420 | -34.641 |
| Dragon shuffled any-hit | embree | medium | native | scalar1 | 1 | 47.995 | 5.919 | 17.201 | 34.290 | -44.553 |
| Dragon shuffled closest-hit | bajo | sah | bvh16 | scalar1 | 1 | 58.302 | 17.591 | 14.274 | 41.321 | 0.000 |
| Dragon shuffled closest-hit | tinybvh | high | bvh8 | scalar1 | 1 | 368.183 | 72.321 | 17.140 | 34.413 | -16.718 |
| Dragon shuffled closest-hit | embree | medium | native | scalar1 | 1 | 48.320 | 5.965 | 20.071 | 29.387 | -28.881 |
| Regular-grid any-hit | bajo | hploc | bvh16 | scalar1 | 1 | 2.285 | 1.301 | 4.500 | 58.253 | 0.000 |
| Regular-grid any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 10.698 | 4.994 | 5.951 | 44.048 | -24.385 |
| Regular-grid any-hit | embree | high | native | scalar1 | 1 | 13.013 | 1.894 | 13.070 | 20.058 | -65.567 |
| Regular-grid closest-hit | bajo | hploc | bvh16 | scalar1 | 1 | 2.285 | 1.301 | 7.357 | 35.630 | 0.000 |
| Regular-grid closest-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 10.872 | 4.515 | 10.166 | 25.786 | -27.628 |
| Regular-grid closest-hit | embree | high | native | scalar1 | 1 | 13.011 | 1.910 | 16.988 | 15.431 | -56.691 |
| Flattened triangle grid any-hit | bajo | sah | bvh16 | scalar1 | 1 | 0.008 | 0.008 | 1.186 | 124.279 | 0.000 |
| Flattened triangle grid any-hit | embree | medium | native | scalar1 | 1 | 0.011 | 0.016 | 2.962 | 49.776 | -59.948 |
| Flattened triangle grid closest-hit | bajo | sah | bvh16 | scalar1 | 1 | 0.008 | 0.008 | 2.011 | 73.331 | 0.000 |
| Flattened triangle grid closest-hit | embree | high | native | scalar1 | 1 | 0.017 | 0.027 | 3.366 | 43.813 | -40.253 |
| Instanced triangle any-hit | bajo | lbvh | bvh4 | scalar1 | 1 | 0.009 | 0.009 | 1.895 | 77.830 | 0.000 |
| Instanced triangle any-hit | embree | medium | native | scalar1 | 1 | 0.037 | 0.058 | 3.381 | 43.612 | -43.965 |
| Instanced triangle closest-hit | bajo | lbvh | bvh4 | scalar1 | 1 | 0.009 | 0.009 | 2.519 | 58.535 | 0.000 |
| Instanced triangle closest-hit | embree | high | native | scalar1 | 1 | 0.037 | 0.049 | 3.866 | 38.146 | -34.832 |

## Fastest build per geometry and implementation

| Geometry | Implementation | Build | Layout | Build ms (1) | Build ms (all) |
| --- | --- | --- | --- | --- | --- |
| dragon | bajo | lbvh | bvh16 | 8.440 | 3.591 |
| dragon | tinybvh | sah | bvh2 | 40.500 | 4.948 |
| dragon | embree | medium | native | 47.995 | 5.919 |
| dragon-instances | bajo | lbvh | bvh4 | 8.888 | 3.835 |
| dragon-instances | tinybvh | sah | bvh2 | 40.471 | 5.212 |
| dragon-instances | embree | medium | native | 47.129 | 6.123 |
| grid | bajo | lbvh | bvh16 | 1.318 | 0.601 |
| grid | embree | medium | native | 10.714 | 1.489 |
| grid | tinybvh | sah | bvh2 | 8.256 | 1.514 |
| triangle-grid | bajo | sah | bvh16 | 0.008 | 0.008 |
| triangle-grid | embree | medium | native | 0.012 | 0.015 |
| triangle-instances | bajo | lbvh | bvh4 | 0.009 | 0.009 |
| triangle-instances | embree | high | native | 0.037 | 0.049 |

## Coverage

The matrix covers synthetic and real mesh geometry, closest-hit and early-exit any-hit, coherent camera ordering and the same rays shuffled to remove neighboring-ray coherence, plus an instance-heavy BLAS/TLAS scene (one reused BLAS, a 12x9 translated-instance grid). A one-triangle BLAS and its flattened 108-triangle equivalent isolate instance continuation from BLAS complexity. Traversal is single-calling-thread; build is measured with one CPU and all available CPUs. The core traversal suites report the best of eight timed repetitions after one warmup. Instance diagnostics report the median of eight repetitions, each averaged across eight timed traversal batches, to resolve small performance differences.

## Regular-grid closest-hit

Triangles per BLAS: 65536; instances: 1; rays: 262144; query: closest; ray order: structured.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh2 | 2 | 2 | scalar1 | 1 | 4.474 | 1.979 | 17.457 | 15.017 |  | 32767 | 6443188224.000 |
| bajo | hploc | bvh4 | 4 | 4 | scalar1 | 1 | 4.581 | 1.953 | 10.335 | 25.365 |  | 5461 | 6443188224.000 |
| bajo | hploc | bvh8 | 8 | 8 | scalar1 | 1 | 4.173 | 1.812 | 10.994 | 23.844 |  | 1641 | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 2.285 | 1.301 | 4.964 | 52.812 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 2.285 | 1.301 | 4.948 | 52.983 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet16 | 16 | 2.285 | 1.301 | 5.746 | 45.625 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet4 | 4 | 2.285 | 1.301 | 11.780 | 22.253 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet8 | 8 | 2.285 | 1.301 | 7.106 | 36.889 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 2.285 | 1.301 | 5.967 | 43.931 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 2.285 | 1.301 | 13.469 | 19.463 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 2.285 | 1.301 | 7.730 | 33.913 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 2.285 | 1.301 | 7.357 | 35.630 |  | 273 | 6443188224.000 |
| bajo | lbvh | bvh2 | 2 | 2 | scalar1 | 1 | 2.073 | 1.167 | 19.067 | 13.749 |  | 32767 | 6443188224.000 |
| bajo | lbvh | bvh4 | 4 | 4 | scalar1 | 1 | 1.652 | 0.925 | 11.298 | 23.202 |  | 5461 | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | adaptive-16-8-4-scalar | 16 | 1.495 | 0.758 | 5.702 | 45.974 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | adaptive-16-8-scalar | 16 | 1.495 | 0.758 | 5.532 | 47.389 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | coh-packet16 | 16 | 1.495 | 0.758 | 6.125 | 42.796 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | coh-packet4 | 4 | 1.495 | 0.758 | 9.428 | 27.805 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | coh-packet8 | 8 | 1.495 | 0.758 | 7.591 | 34.533 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | packet16 | 16 | 1.495 | 0.758 | 6.421 | 40.827 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | packet4 | 4 | 1.495 | 0.758 | 10.965 | 23.907 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | packet8 | 8 | 1.495 | 0.758 | 7.927 | 33.069 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | scalar1 | 1 | 1.495 | 0.758 | 11.869 | 22.087 |  | 4681 | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 1.318 | 0.601 | 7.307 | 35.876 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 1.318 | 0.601 | 7.311 | 35.858 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet16 | 16 | 1.318 | 0.601 | 8.023 | 32.673 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet4 | 4 | 1.318 | 0.601 | 12.298 | 21.316 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet8 | 8 | 1.318 | 0.601 | 9.358 | 28.011 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 1.318 | 0.601 | 8.168 | 32.095 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 1.318 | 0.601 | 13.919 | 18.834 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 1.318 | 0.601 | 10.118 | 25.910 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 1.318 | 0.601 | 7.490 | 34.999 |  | 273 | 6443188224.000 |
| bajo | median | bvh2 | 2 | 2 | scalar1 | 1 | 8.469 | 2.619 | 19.060 | 13.754 |  | 32767 | 6443188224.000 |
| bajo | median | bvh4 | 4 | 4 | scalar1 | 1 | 8.327 | 2.697 | 11.311 | 23.177 |  | 5461 | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | adaptive-16-8-4-scalar | 16 | 7.600 | 2.628 | 5.536 | 47.353 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | adaptive-16-8-scalar | 16 | 7.600 | 2.628 | 5.512 | 47.558 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | coh-packet16 | 16 | 7.600 | 2.628 | 6.294 | 41.647 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | coh-packet4 | 4 | 7.600 | 2.628 | 9.678 | 27.086 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | coh-packet8 | 8 | 7.600 | 2.628 | 7.397 | 35.439 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | packet16 | 16 | 7.600 | 2.628 | 6.613 | 39.643 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | packet4 | 4 | 7.600 | 2.628 | 11.698 | 22.409 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | packet8 | 8 | 7.600 | 2.628 | 7.944 | 32.997 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | scalar1 | 1 | 7.600 | 2.628 | 11.546 | 22.705 |  | 1193 | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 7.111 | 2.385 | 7.358 | 35.627 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 7.111 | 2.385 | 7.306 | 35.882 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet16 | 16 | 7.111 | 2.385 | 7.997 | 32.781 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet4 | 4 | 7.111 | 2.385 | 12.169 | 21.543 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet8 | 8 | 7.111 | 2.385 | 9.275 | 28.263 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | packet16 | 16 | 7.111 | 2.385 | 8.220 | 31.892 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | packet4 | 4 | 7.111 | 2.385 | 14.008 | 18.714 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | packet8 | 8 | 7.111 | 2.385 | 10.021 | 26.159 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | scalar1 | 1 | 7.111 | 2.385 | 7.472 | 35.083 |  | 273 | 6443188224.000 |
| bajo | sah | bvh2 | 2 | 2 | scalar1 | 1 | 20.392 | 4.668 | 17.922 | 14.627 |  | 32767 | 6443188224.000 |
| bajo | sah | bvh4 | 4 | 4 | scalar1 | 1 | 15.426 | 3.963 | 9.784 | 26.793 |  | 5461 | 6443188224.000 |
| bajo | sah | bvh8 | 8 | 8 | scalar1 | 1 | 12.006 | 3.897 | 10.035 | 26.123 |  | 1273 | 6443188224.000 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 10.058 | 2.993 | 8.332 | 31.463 |  | 273 | 6443188224.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 13.011 | 1.910 | 15.249 | 17.191 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 13.011 | 1.910 | 14.230 | 18.422 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 13.011 | 1.910 | 13.008 | 20.152 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 13.011 | 1.910 | 11.028 | 23.771 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 13.011 | 1.910 | 11.508 | 22.779 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 13.011 | 1.910 | 10.434 | 25.125 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | scalar1 | 1 | 13.011 | 1.910 | 16.988 | 15.431 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 11.267 | 1.496 | 13.247 | 19.788 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 11.267 | 1.496 | 14.794 | 17.719 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 11.267 | 1.496 | 13.005 | 20.157 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 11.267 | 1.496 | 10.885 | 24.082 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 11.267 | 1.496 | 11.873 | 22.078 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 11.267 | 1.496 | 10.684 | 24.536 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | scalar1 | 1 | 11.267 | 1.496 | 17.127 | 15.306 | 196608 |  | 6443188224.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 29.602 | 4.374 | 19.083 | 13.737 | 196608 |  | 6443188224.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 31.260 | 6.046 | 12.091 | 21.681 | 196608 |  | 6443188223.985 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 32.233 | 7.365 | 10.181 | 25.748 | 196608 |  | 6443188223.985 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 8.256 | 1.514 | 18.565 | 14.120 | 196608 |  | 6443188224.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 9.725 | 3.159 | 12.123 | 21.625 | 196608 |  | 6443188223.985 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 10.872 | 4.515 | 10.166 | 25.786 | 196608 |  | 6443188223.985 |

## Regular-grid any-hit

Triangles per BLAS: 65536; instances: 1; rays: 262144; query: any; ray order: structured.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet16 | 16 | 2.285 | 1.301 | 4.878 | 53.744 | 196608 |  | 196608.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 2.285 | 1.301 | 5.119 | 51.212 | 196608 |  | 196608.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 2.285 | 1.301 | 12.019 | 21.810 | 196608 |  | 196608.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 2.285 | 1.301 | 6.854 | 38.249 | 196608 |  | 196608.000 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 2.285 | 1.301 | 4.500 | 58.253 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet16 | 16 | 1.318 | 0.601 | 6.241 | 42.001 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 1.318 | 0.601 | 6.485 | 40.420 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 1.318 | 0.601 | 12.670 | 20.690 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 1.318 | 0.601 | 8.376 | 31.297 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 1.318 | 0.601 | 4.545 | 57.672 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet16 | 16 | 7.111 | 2.385 | 7.074 | 37.058 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | packet16 | 16 | 7.111 | 2.385 | 7.512 | 34.895 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | packet4 | 4 | 7.111 | 2.385 | 13.385 | 19.585 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | packet8 | 8 | 7.111 | 2.385 | 9.208 | 28.468 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | scalar1 | 1 | 7.111 | 2.385 | 4.537 | 57.780 | 196608 |  | 196608.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 13.013 | 1.894 | 11.458 | 22.878 | 196608 |  | 196608.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 13.013 | 1.894 | 13.899 | 18.861 | 196608 |  | 196608.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 13.013 | 1.894 | 12.333 | 21.255 | 196608 |  | 196608.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 13.013 | 1.894 | 6.069 | 43.196 | 196608 |  | 196608.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 13.013 | 1.894 | 7.926 | 33.074 | 196608 |  | 196608.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 13.013 | 1.894 | 6.353 | 41.265 | 196608 |  | 196608.000 |
| embree | high | native |  |  | scalar1 | 1 | 13.013 | 1.894 | 13.070 | 20.058 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 10.714 | 1.489 | 11.441 | 22.913 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 10.714 | 1.489 | 13.926 | 18.824 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 10.714 | 1.489 | 12.321 | 21.275 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 10.714 | 1.489 | 6.018 | 43.560 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 10.714 | 1.489 | 7.753 | 33.814 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 10.714 | 1.489 | 6.421 | 40.827 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | scalar1 | 1 | 10.714 | 1.489 | 13.077 | 20.046 | 196608 |  | 196608.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 29.320 | 4.272 | 17.061 | 15.366 | 196608 |  | 196608.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 31.075 | 6.016 | 7.775 | 33.715 | 196608 |  | 196608.000 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 31.941 | 6.612 | 5.960 | 43.981 | 196608 |  | 196608.000 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 8.180 | 1.580 | 17.138 | 15.296 | 196608 |  | 196608.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 9.652 | 3.532 | 7.801 | 33.605 | 196608 |  | 196608.000 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 10.698 | 4.994 | 5.951 | 44.048 | 196608 |  | 196608.000 |

## Dragon camera closest-hit

Triangles per BLAS: 249882; instances: 1; rays: 589824; query: closest; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 20.124 | 9.606 | 7.617 | 77.438 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 20.124 | 9.606 | 7.649 | 77.113 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet16 | 16 | 20.124 | 9.606 | 9.414 | 62.657 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet4 | 4 | 20.124 | 9.606 | 11.992 | 49.185 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet8 | 8 | 20.124 | 9.606 | 10.171 | 57.988 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 20.124 | 9.606 | 11.369 | 51.881 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 20.124 | 9.606 | 15.973 | 36.927 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 20.124 | 9.606 | 13.205 | 44.665 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 20.124 | 9.606 | 9.981 | 59.093 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 8.440 | 3.591 | 7.797 | 75.650 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 8.440 | 3.591 | 7.753 | 76.077 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet16 | 16 | 8.440 | 3.591 | 9.523 | 61.937 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet4 | 4 | 8.440 | 3.591 | 12.287 | 48.003 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet8 | 8 | 8.440 | 3.591 | 10.435 | 56.522 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 8.440 | 3.591 | 11.788 | 50.036 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 8.440 | 3.591 | 16.662 | 35.400 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 8.440 | 3.591 | 13.300 | 44.347 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 8.440 | 3.591 | 10.439 | 56.503 | 71597 |  | 7943562615.175 |
| bajo | median | bvh16 | 16 | 16 | scalar1 | 1 | 38.553 | 11.863 | 10.145 | 58.139 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 58.302 | 17.591 | 7.566 | 77.953 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 58.302 | 17.591 | 7.568 | 77.933 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet16 | 16 | 58.302 | 17.591 | 9.309 | 63.358 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet4 | 4 | 58.302 | 17.591 | 12.532 | 47.067 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet8 | 8 | 58.302 | 17.591 | 10.838 | 54.421 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 58.302 | 17.591 | 11.268 | 52.344 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 58.302 | 17.591 | 15.069 | 39.140 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 58.302 | 17.591 | 12.309 | 47.918 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 58.302 | 17.591 | 9.309 | 63.359 | 71597 |  | 7943562615.175 |
| embree | high | native |  |  | coh-packet16 | 16 | 264.886 | 28.606 | 9.900 | 59.579 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | coh-packet4 | 4 | 264.886 | 28.606 | 11.704 | 50.396 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | coh-packet8 | 8 | 264.886 | 28.606 | 8.996 | 65.567 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet16 | 16 | 264.886 | 28.606 | 12.265 | 48.090 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet4 | 4 | 264.886 | 28.606 | 13.299 | 44.350 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet8 | 8 | 264.886 | 28.606 | 11.914 | 49.506 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | scalar1 | 1 | 264.886 | 28.606 | 18.525 | 31.839 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet16 | 16 | 50.183 | 6.405 | 9.752 | 60.485 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet4 | 4 | 50.183 | 6.405 | 11.807 | 49.957 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet8 | 8 | 50.183 | 6.405 | 9.288 | 63.507 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet16 | 16 | 50.183 | 6.405 | 12.298 | 47.963 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet4 | 4 | 50.183 | 6.405 | 14.543 | 40.558 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet8 | 8 | 50.183 | 6.405 | 11.985 | 49.213 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | scalar1 | 1 | 50.183 | 6.405 | 19.103 | 30.876 | 71598 |  | 7943741995.515 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 353.441 | 48.139 | 25.062 | 23.535 | 71599 |  | 7943796499.200 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 374.405 | 66.836 | 17.298 | 34.098 | 71599 |  | 7943796499.445 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 372.191 | 73.896 | 14.504 | 40.666 | 71599 |  | 7943796499.439 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.500 | 4.948 | 24.907 | 23.681 | 71599 |  | 7943796499.200 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 61.603 | 31.248 | 16.764 | 35.183 | 71599 |  | 7943796499.445 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 68.282 | 32.670 | 14.349 | 41.105 | 71599 |  | 7943796499.439 |

## Dragon camera any-hit

Triangles per BLAS: 249882; instances: 1; rays: 589824; query: any; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet16 | 16 | 20.124 | 9.606 | 6.369 | 92.607 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 20.124 | 9.606 | 8.663 | 68.089 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 20.124 | 9.606 | 12.431 | 47.450 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 20.124 | 9.606 | 10.369 | 56.884 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 20.124 | 9.606 | 8.185 | 72.061 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet16 | 16 | 8.440 | 3.591 | 6.691 | 88.147 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 8.440 | 3.591 | 8.681 | 67.948 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 8.440 | 3.591 | 13.198 | 44.690 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 8.440 | 3.591 | 10.969 | 53.772 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 8.440 | 3.591 | 8.649 | 68.194 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet16 | 16 | 58.302 | 17.591 | 6.004 | 98.231 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 58.302 | 17.591 | 8.426 | 70.002 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 58.302 | 17.591 | 11.942 | 49.392 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 58.302 | 17.591 | 10.023 | 58.849 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 58.302 | 17.591 | 7.609 | 77.521 | 71597 |  | 71597.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 265.636 | 28.463 | 7.302 | 80.775 | 71598 |  | 71598.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 265.636 | 28.463 | 10.831 | 54.458 | 71598 |  | 71598.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 265.636 | 28.463 | 8.216 | 71.786 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 265.636 | 28.463 | 6.520 | 90.461 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 265.636 | 28.463 | 9.916 | 59.481 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 265.636 | 28.463 | 7.009 | 84.156 | 71598 |  | 71598.000 |
| embree | high | native |  |  | scalar1 | 1 | 265.636 | 28.463 | 16.333 | 36.112 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 49.326 | 6.032 | 7.637 | 77.235 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 49.326 | 6.032 | 10.502 | 56.164 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 49.326 | 6.032 | 8.134 | 72.516 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 49.326 | 6.032 | 6.477 | 91.071 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 49.326 | 6.032 | 9.197 | 64.132 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 49.326 | 6.032 | 7.001 | 84.249 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | scalar1 | 1 | 49.326 | 6.032 | 16.011 | 36.838 | 71598 |  | 71598.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 351.802 | 48.335 | 19.463 | 30.305 | 71599 |  | 71599.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 366.555 | 68.659 | 13.537 | 43.570 | 71599 |  | 71599.000 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 369.038 | 76.571 | 11.509 | 51.249 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.504 | 6.283 | 20.360 | 28.969 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 62.063 | 27.976 | 14.064 | 41.940 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 67.620 | 34.490 | 11.490 | 51.335 | 71599 |  | 71599.000 |

## Dragon shuffled closest-hit

Triangles per BLAS: 249882; instances: 1; rays: 589824; query: closest; ray order: shuffled-camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 20.124 | 9.606 | 32.867 | 17.946 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 20.124 | 9.606 | 16.934 | 34.830 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 20.124 | 9.606 | 24.073 | 24.501 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 20.124 | 9.606 | 22.092 | 26.698 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 20.124 | 9.606 | 29.060 | 20.297 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 20.124 | 9.606 | 15.440 | 38.202 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 8.440 | 3.591 | 34.543 | 17.075 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 8.440 | 3.591 | 17.428 | 33.843 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 8.440 | 3.591 | 24.364 | 24.209 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 8.440 | 3.591 | 25.574 | 23.063 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 8.440 | 3.591 | 29.014 | 20.329 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 8.440 | 3.591 | 15.994 | 36.877 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 58.302 | 17.591 | 29.140 | 20.241 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 58.302 | 17.591 | 15.846 | 37.223 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 58.302 | 17.591 | 23.397 | 25.210 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 58.302 | 17.591 | 20.668 | 28.538 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 58.302 | 17.591 | 27.657 | 21.326 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 58.302 | 17.591 | 14.274 | 41.321 | 71597 |  | 7943562615.175 |
| embree | high | native |  |  | coh-packet16 | 16 | 263.428 | 28.187 | 31.627 | 18.650 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | coh-packet4 | 4 | 263.428 | 28.187 | 31.763 | 18.570 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | coh-packet8 | 8 | 263.428 | 28.187 | 32.725 | 18.024 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet16 | 16 | 263.428 | 28.187 | 14.909 | 39.562 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet4 | 4 | 263.428 | 28.187 | 15.685 | 37.603 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet8 | 8 | 263.428 | 28.187 | 14.444 | 40.835 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | scalar1 | 1 | 263.428 | 28.187 | 20.117 | 29.319 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet16 | 16 | 48.320 | 5.965 | 31.543 | 18.699 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet4 | 4 | 48.320 | 5.965 | 31.784 | 18.557 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet8 | 8 | 48.320 | 5.965 | 33.085 | 17.828 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet16 | 16 | 48.320 | 5.965 | 15.036 | 39.229 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet4 | 4 | 48.320 | 5.965 | 15.745 | 37.460 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet8 | 8 | 48.320 | 5.965 | 14.480 | 40.735 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | scalar1 | 1 | 48.320 | 5.965 | 20.071 | 29.387 | 71598 |  | 7943741995.515 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 356.403 | 47.315 | 32.240 | 18.295 | 71599 |  | 7943796499.200 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 369.259 | 68.339 | 24.512 | 24.063 | 71599 |  | 7943796499.445 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 368.183 | 72.321 | 17.140 | 34.413 | 71599 |  | 7943796499.439 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.312 | 5.167 | 32.340 | 18.238 | 71599 |  | 7943796499.200 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 61.619 | 28.607 | 20.069 | 29.390 | 71599 |  | 7943796499.445 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 66.756 | 33.942 | 17.319 | 34.056 | 71599 |  | 7943796499.439 |

## Dragon shuffled any-hit

Triangles per BLAS: 249882; instances: 1; rays: 589824; query: any; ray order: shuffled-camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 20.124 | 9.606 | 18.836 | 31.313 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 20.124 | 9.606 | 17.446 | 33.809 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 20.124 | 9.606 | 24.216 | 24.357 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 20.124 | 9.606 | 10.583 | 55.735 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 8.440 | 3.591 | 18.935 | 31.150 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 8.440 | 3.591 | 20.590 | 28.646 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 8.440 | 3.591 | 23.952 | 24.625 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 8.440 | 3.591 | 11.169 | 52.808 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 58.302 | 17.591 | 18.280 | 32.267 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 58.302 | 17.591 | 16.534 | 35.672 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 58.302 | 17.591 | 23.524 | 25.073 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 58.302 | 17.591 | 9.537 | 61.843 | 71597 |  | 71597.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 264.278 | 28.990 | 23.755 | 24.829 | 71598 |  | 71598.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 264.278 | 28.990 | 26.726 | 22.069 | 71598 |  | 71598.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 264.278 | 28.990 | 25.929 | 22.748 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 264.278 | 28.990 | 10.039 | 58.755 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 264.278 | 28.990 | 12.864 | 45.851 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 264.278 | 28.990 | 10.820 | 54.514 | 71598 |  | 71598.000 |
| embree | high | native |  |  | scalar1 | 1 | 264.278 | 28.990 | 17.371 | 33.955 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 47.995 | 5.919 | 23.538 | 25.058 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 47.995 | 5.919 | 26.343 | 22.390 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 47.995 | 5.919 | 26.030 | 22.660 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 47.995 | 5.919 | 10.066 | 58.594 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 47.995 | 5.919 | 12.267 | 48.081 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 47.995 | 5.919 | 10.690 | 55.173 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | scalar1 | 1 | 47.995 | 5.919 | 17.201 | 34.290 | 71598 |  | 71598.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 350.956 | 47.817 | 26.416 | 22.328 | 71599 |  | 71599.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 364.170 | 67.630 | 17.025 | 34.644 | 71599 |  | 71599.000 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 368.773 | 72.098 | 14.851 | 39.716 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.634 | 5.124 | 26.567 | 22.201 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 64.855 | 27.758 | 17.031 | 34.633 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 66.956 | 34.343 | 14.592 | 40.420 | 71599 |  | 71599.000 |

## Instanced Dragon closest-hit

Triangles per BLAS: 249882; instances: 108; rays: 147456; query: closest; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh4 | 4 | 1 | packet16 | 16 | 20.251 | 8.990 | 1.889 | 78.041 | 8250 |  | 1022292747.551 |
| bajo | hploc | bvh4 | 4 | 1 | packet4 | 4 | 20.251 | 8.990 | 2.330 | 63.290 | 8250 |  | 1022292747.551 |
| bajo | hploc | bvh4 | 4 | 1 | packet8 | 8 | 20.251 | 8.990 | 1.987 | 74.207 | 8250 |  | 1022292747.551 |
| bajo | hploc | bvh4 | 4 | 1 | scalar1 | 1 | 20.251 | 8.990 | 2.624 | 56.200 | 8250 |  | 1022292747.551 |
| bajo | lbvh | bvh4 | 4 | 1 | packet16 | 16 | 8.888 | 3.835 | 1.939 | 76.048 | 8250 |  | 1022292856.202 |
| bajo | lbvh | bvh4 | 4 | 1 | packet4 | 4 | 8.888 | 3.835 | 2.341 | 62.992 | 8250 |  | 1022292856.202 |
| bajo | lbvh | bvh4 | 4 | 1 | packet8 | 8 | 8.888 | 3.835 | 2.034 | 72.494 | 8250 |  | 1022292856.202 |
| bajo | lbvh | bvh4 | 4 | 1 | scalar1 | 1 | 8.888 | 3.835 | 2.686 | 54.888 | 8250 |  | 1022292856.202 |
| bajo | sah | bvh4 | 4 | 1 | packet16 | 16 | 58.605 | 16.955 | 1.759 | 83.821 | 8250 |  | 1022292856.202 |
| bajo | sah | bvh4 | 4 | 1 | packet4 | 4 | 58.605 | 16.955 | 2.172 | 67.897 | 8250 |  | 1022292856.202 |
| bajo | sah | bvh4 | 4 | 1 | packet8 | 8 | 58.605 | 16.955 | 1.865 | 79.072 | 8250 |  | 1022292856.202 |
| bajo | sah | bvh4 | 4 | 1 | scalar1 | 1 | 58.605 | 16.955 | 2.534 | 58.181 | 8250 |  | 1022292856.202 |
| embree | high | native |  |  | coh-packet16 | 16 | 264.751 | 28.560 | 3.970 | 37.143 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | coh-packet4 | 4 | 264.751 | 28.560 | 3.795 | 38.852 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | coh-packet8 | 8 | 264.751 | 28.560 | 3.504 | 42.077 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | inc-packet16 | 16 | 264.751 | 28.560 | 2.079 | 70.923 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | inc-packet4 | 4 | 264.751 | 28.560 | 2.342 | 62.953 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | inc-packet8 | 8 | 264.751 | 28.560 | 1.879 | 78.482 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | scalar1 | 1 | 264.751 | 28.560 | 4.144 | 35.583 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | coh-packet16 | 16 | 49.458 | 6.346 | 4.014 | 36.734 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | coh-packet4 | 4 | 49.458 | 6.346 | 3.789 | 38.917 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | coh-packet8 | 8 | 49.458 | 6.346 | 3.512 | 41.985 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | inc-packet16 | 16 | 49.458 | 6.346 | 2.128 | 69.302 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | inc-packet4 | 4 | 49.458 | 6.346 | 2.359 | 62.496 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | inc-packet8 | 8 | 49.458 | 6.346 | 1.883 | 78.307 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | scalar1 | 1 | 49.458 | 6.346 | 4.141 | 35.608 | 8256 |  | 1024176197.799 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 351.558 | 48.117 | 5.103 | 28.895 | 8256 |  | 1024143130.934 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 364.612 | 68.078 | 3.493 | 42.216 | 8256 |  | 1024092270.370 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 369.049 | 73.298 | 3.139 | 46.980 | 8256 |  | 1024092270.381 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.471 | 5.212 | 5.250 | 28.087 | 8256 |  | 1024143130.934 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 61.737 | 28.851 | 3.484 | 42.323 | 8256 |  | 1024092270.370 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 67.059 | 34.040 | 3.152 | 46.780 | 8256 |  | 1024092270.381 |

## Instanced Dragon any-hit

Triangles per BLAS: 249882; instances: 108; rays: 147456; query: any; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh4 | 4 | 1 | packet16 | 16 | 20.251 | 8.990 | 1.335 | 110.484 | 8250 |  | 8250.000 |
| bajo | hploc | bvh4 | 4 | 1 | packet4 | 4 | 20.251 | 8.990 | 1.800 | 81.924 | 8250 |  | 8250.000 |
| bajo | hploc | bvh4 | 4 | 1 | packet8 | 8 | 20.251 | 8.990 | 1.474 | 100.017 | 8250 |  | 8250.000 |
| bajo | hploc | bvh4 | 4 | 1 | scalar1 | 1 | 20.251 | 8.990 | 2.213 | 66.626 | 8250 |  | 8250.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet16 | 16 | 8.888 | 3.835 | 1.402 | 105.174 | 8250 |  | 8250.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet4 | 4 | 8.888 | 3.835 | 1.846 | 79.865 | 8250 |  | 8250.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet8 | 8 | 8.888 | 3.835 | 1.534 | 96.154 | 8250 |  | 8250.000 |
| bajo | lbvh | bvh4 | 4 | 1 | scalar1 | 1 | 8.888 | 3.835 | 2.297 | 64.181 | 8250 |  | 8250.000 |
| bajo | sah | bvh4 | 4 | 1 | packet16 | 16 | 58.605 | 16.955 | 1.230 | 119.900 | 8250 |  | 8250.000 |
| bajo | sah | bvh4 | 4 | 1 | packet4 | 4 | 58.605 | 16.955 | 1.693 | 87.101 | 8250 |  | 8250.000 |
| bajo | sah | bvh4 | 4 | 1 | packet8 | 8 | 58.605 | 16.955 | 1.374 | 107.342 | 8250 |  | 8250.000 |
| bajo | sah | bvh4 | 4 | 1 | scalar1 | 1 | 58.605 | 16.955 | 2.150 | 68.597 | 8250 |  | 8250.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 261.560 | 29.718 | 2.903 | 50.802 | 8256 |  | 8256.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 261.560 | 29.718 | 3.301 | 44.673 | 8256 |  | 8256.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 261.560 | 29.718 | 2.825 | 52.200 | 8256 |  | 8256.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 261.560 | 29.718 | 1.452 | 101.580 | 8256 |  | 8256.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 261.560 | 29.718 | 2.229 | 66.146 | 8256 |  | 8256.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 261.560 | 29.718 | 1.575 | 93.639 | 8256 |  | 8256.000 |
| embree | high | native |  |  | scalar1 | 1 | 261.560 | 29.718 | 3.890 | 37.911 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 47.129 | 6.123 | 2.899 | 50.866 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 47.129 | 6.123 | 3.214 | 45.880 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 47.129 | 6.123 | 2.825 | 52.199 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 47.129 | 6.123 | 1.452 | 101.550 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 47.129 | 6.123 | 2.077 | 71.006 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 47.129 | 6.123 | 1.581 | 93.273 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | scalar1 | 1 | 47.129 | 6.123 | 3.703 | 39.817 | 8256 |  | 8256.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 354.658 | 49.180 | 4.155 | 35.489 | 8256 |  | 8256.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 365.794 | 69.044 | 2.947 | 50.044 | 8256 |  | 8256.000 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 369.751 | 74.629 | 2.685 | 54.929 | 8256 |  | 8256.000 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.683 | 5.312 | 4.206 | 35.061 | 8256 |  | 8256.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 61.944 | 29.897 | 2.951 | 49.960 | 8256 |  | 8256.000 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 67.791 | 35.499 | 2.664 | 55.356 | 8256 |  | 8256.000 |

## Instanced triangle closest-hit

Triangles per BLAS: 1; instances: 108; rays: 147456; query: closest; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh4 | 4 | 1 | packet16 | 16 | 0.017 | 0.015 | 0.712 | 207.037 | 20416 |  | 1177505.908 |
| bajo | hploc | bvh4 | 4 | 1 | packet4 | 4 | 0.017 | 0.015 | 1.168 | 126.248 | 20416 |  | 1177505.908 |
| bajo | hploc | bvh4 | 4 | 1 | packet8 | 8 | 0.017 | 0.015 | 0.789 | 186.804 | 20416 |  | 1177505.908 |
| bajo | hploc | bvh4 | 4 | 1 | scalar1 | 1 | 0.017 | 0.015 | 2.592 | 56.887 | 20416 |  | 1177505.908 |
| bajo | lbvh | bvh4 | 4 | 1 | packet16 | 16 | 0.009 | 0.009 | 0.685 | 215.236 | 20416 |  | 1177505.908 |
| bajo | lbvh | bvh4 | 4 | 1 | packet4 | 4 | 0.009 | 0.009 | 1.080 | 136.523 | 20416 |  | 1177505.908 |
| bajo | lbvh | bvh4 | 4 | 1 | packet8 | 8 | 0.009 | 0.009 | 0.751 | 196.282 | 20416 |  | 1177505.908 |
| bajo | lbvh | bvh4 | 4 | 1 | scalar1 | 1 | 0.009 | 0.009 | 2.519 | 58.535 | 20416 |  | 1177505.908 |
| bajo | sah | bvh4 | 4 | 1 | packet16 | 16 | 0.045 | 0.045 | 0.690 | 213.802 | 20416 |  | 1177505.908 |
| bajo | sah | bvh4 | 4 | 1 | packet4 | 4 | 0.045 | 0.045 | 1.084 | 136.042 | 20416 |  | 1177505.908 |
| bajo | sah | bvh4 | 4 | 1 | packet8 | 8 | 0.045 | 0.045 | 0.777 | 189.814 | 20416 |  | 1177505.908 |
| bajo | sah | bvh4 | 4 | 1 | scalar1 | 1 | 0.045 | 0.045 | 2.525 | 58.410 | 20416 |  | 1177505.908 |
| embree | high | native |  |  | coh-packet16 | 16 | 0.037 | 0.049 | 1.007 | 146.476 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | coh-packet4 | 4 | 0.037 | 0.049 | 1.308 | 112.696 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | coh-packet8 | 8 | 0.037 | 0.049 | 0.816 | 180.736 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | inc-packet16 | 16 | 0.037 | 0.049 | 1.137 | 129.704 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | inc-packet4 | 4 | 0.037 | 0.049 | 1.384 | 106.563 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | inc-packet8 | 8 | 0.037 | 0.049 | 0.967 | 152.529 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | scalar1 | 1 | 0.037 | 0.049 | 3.866 | 38.146 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | coh-packet16 | 16 | 0.037 | 0.062 | 1.005 | 146.726 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | coh-packet4 | 4 | 0.037 | 0.062 | 1.322 | 111.545 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | coh-packet8 | 8 | 0.037 | 0.062 | 0.815 | 180.955 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | inc-packet16 | 16 | 0.037 | 0.062 | 1.130 | 130.438 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | inc-packet4 | 4 | 0.037 | 0.062 | 1.384 | 106.581 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | inc-packet8 | 8 | 0.037 | 0.062 | 0.965 | 152.881 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | scalar1 | 1 | 0.037 | 0.062 | 3.873 | 38.072 | 20416 |  | 1177505.898 |

## Instanced triangle any-hit

Triangles per BLAS: 1; instances: 108; rays: 147456; query: any; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh4 | 4 | 1 | packet16 | 16 | 0.017 | 0.015 | 0.420 | 350.765 | 20416 |  | 20416.000 |
| bajo | hploc | bvh4 | 4 | 1 | packet4 | 4 | 0.017 | 0.015 | 0.950 | 155.200 | 20416 |  | 20416.000 |
| bajo | hploc | bvh4 | 4 | 1 | packet8 | 8 | 0.017 | 0.015 | 0.550 | 267.960 | 20416 |  | 20416.000 |
| bajo | hploc | bvh4 | 4 | 1 | scalar1 | 1 | 0.017 | 0.015 | 1.927 | 76.517 | 20416 |  | 20416.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet16 | 16 | 0.009 | 0.009 | 0.391 | 377.030 | 20416 |  | 20416.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet4 | 4 | 0.009 | 0.009 | 0.854 | 172.648 | 20416 |  | 20416.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet8 | 8 | 0.009 | 0.009 | 0.497 | 296.837 | 20416 |  | 20416.000 |
| bajo | lbvh | bvh4 | 4 | 1 | scalar1 | 1 | 0.009 | 0.009 | 1.895 | 77.830 | 20416 |  | 20416.000 |
| bajo | sah | bvh4 | 4 | 1 | packet16 | 16 | 0.045 | 0.045 | 0.499 | 295.456 | 20416 |  | 20416.000 |
| bajo | sah | bvh4 | 4 | 1 | packet4 | 4 | 0.045 | 0.045 | 0.865 | 170.509 | 20416 |  | 20416.000 |
| bajo | sah | bvh4 | 4 | 1 | packet8 | 8 | 0.045 | 0.045 | 0.756 | 195.050 | 20416 |  | 20416.000 |
| bajo | sah | bvh4 | 4 | 1 | scalar1 | 1 | 0.045 | 0.045 | 2.714 | 54.339 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 0.037 | 0.060 | 0.675 | 218.413 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 0.037 | 0.060 | 1.334 | 110.539 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 0.037 | 0.060 | 0.805 | 183.265 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 0.037 | 0.060 | 0.575 | 256.604 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 0.037 | 0.060 | 1.246 | 118.312 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 0.037 | 0.060 | 0.655 | 225.180 | 20416 |  | 20416.000 |
| embree | high | native |  |  | scalar1 | 1 | 0.037 | 0.060 | 3.381 | 43.611 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 0.037 | 0.058 | 0.670 | 220.044 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 0.037 | 0.058 | 1.294 | 113.962 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 0.037 | 0.058 | 0.796 | 185.143 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 0.037 | 0.058 | 0.569 | 259.190 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 0.037 | 0.058 | 1.146 | 128.684 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 0.037 | 0.058 | 0.653 | 225.665 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | scalar1 | 1 | 0.037 | 0.058 | 3.381 | 43.612 | 20416 |  | 20416.000 |

## Flattened triangle grid closest-hit

Triangles per BLAS: 108; instances: 1; rays: 147456; query: closest; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | sah | bvh16 | 16 | 8 | coh-packet16 | 16 | 0.017 | 0.017 | 0.663 | 222.305 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet16 | 16 | 0.008 | 0.008 | 0.861 | 171.330 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet4 | 4 | 0.008 | 0.008 | 1.889 | 78.066 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet8 | 8 | 0.008 | 0.008 | 1.282 | 114.983 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 0.008 | 0.008 | 0.757 | 194.740 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 0.008 | 0.008 | 1.805 | 81.715 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 0.008 | 0.008 | 1.124 | 131.146 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 0.008 | 0.008 | 2.011 | 73.331 | 20416 |  | 2281401.908 |
| embree | high | native |  |  | coh-packet16 | 16 | 0.017 | 0.027 | 0.893 | 165.107 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | coh-packet4 | 4 | 0.017 | 0.027 | 1.154 | 127.809 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | coh-packet8 | 8 | 0.017 | 0.027 | 0.699 | 211.028 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | inc-packet16 | 16 | 0.017 | 0.027 | 1.516 | 97.254 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | inc-packet4 | 4 | 0.017 | 0.027 | 1.674 | 88.087 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | inc-packet8 | 8 | 0.017 | 0.027 | 1.410 | 104.609 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | scalar1 | 1 | 0.017 | 0.027 | 3.366 | 43.813 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | coh-packet16 | 16 | 0.012 | 0.015 | 0.889 | 165.786 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | coh-packet4 | 4 | 0.012 | 0.015 | 1.151 | 128.079 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | coh-packet8 | 8 | 0.012 | 0.015 | 0.699 | 210.910 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | inc-packet16 | 16 | 0.012 | 0.015 | 1.525 | 96.693 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | inc-packet4 | 4 | 0.012 | 0.015 | 1.673 | 88.141 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | inc-packet8 | 8 | 0.012 | 0.015 | 1.407 | 104.815 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | scalar1 | 1 | 0.012 | 0.015 | 3.366 | 43.812 | 20416 |  | 2281401.897 |

## Flattened triangle grid any-hit

Triangles per BLAS: 108; instances: 1; rays: 147456; query: any; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | sah | bvh16 | 16 | 8 | coh-packet16 | 16 | 0.017 | 0.017 | 0.482 | 305.661 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet16 | 16 | 0.008 | 0.008 | 0.608 | 242.329 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet4 | 4 | 0.008 | 0.008 | 1.411 | 104.472 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet8 | 8 | 0.008 | 0.008 | 0.846 | 174.347 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 0.008 | 0.008 | 0.582 | 253.184 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 0.008 | 0.008 | 1.470 | 100.316 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 0.008 | 0.008 | 0.839 | 175.672 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 0.008 | 0.008 | 1.186 | 124.279 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 0.017 | 0.028 | 0.495 | 297.880 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 0.017 | 0.028 | 1.123 | 131.334 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 0.017 | 0.028 | 0.649 | 227.068 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 0.017 | 0.028 | 0.492 | 299.664 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 0.017 | 0.028 | 1.112 | 132.636 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 0.017 | 0.028 | 0.593 | 248.770 | 20416 |  | 20416.000 |
| embree | high | native |  |  | scalar1 | 1 | 0.017 | 0.028 | 2.990 | 49.313 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 0.011 | 0.016 | 0.497 | 296.954 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 0.011 | 0.016 | 1.088 | 135.587 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 0.011 | 0.016 | 0.651 | 226.664 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 0.011 | 0.016 | 0.490 | 300.715 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 0.011 | 0.016 | 0.993 | 148.476 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 0.011 | 0.016 | 0.592 | 248.902 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | scalar1 | 1 | 0.011 | 0.016 | 2.962 | 49.776 | 20416 |  | 20416.000 |
