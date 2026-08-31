# CPU BVH benchmark results

- **Date:** 2026-08-31T18:34:20-04:00
- **CPU:** AMD Ryzen 7 9700X 8-Core Processor
- **System:** Linux-7.0.0-30-generic-x86_64-with-glibc2.43
- **Mojo:** `Mojo 1.1.0.dev2026083005 (ffc874b9)`
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
| Instanced triangle any-hit | packet16 | 150.014 | embree | inc-packet16 | 259.297 | -42.146 |
| Instanced triangle closest-hit | packet16 | 106.649 | embree | coh-packet8 | 181.045 | -41.093 |
| Flattened triangle grid any-hit | packet16 | 252.523 | embree | inc-packet16 | 299.581 | -15.708 |
| Flattened triangle grid closest-hit | packet16 | 194.764 | embree | coh-packet8 | 208.793 | -6.719 |

Build deficits larger than 2% (lower time is better):

| Geometry | Build threads | Bajo build ms | Fastest competitor | Competitor build ms | Bajo vs competitor (%) |
| --- | --- | --- | --- | --- | --- |

These are the optimization queue: the most negative rows are the largest measured deficits on this machine. Differences within 2% are treated as parity.

## Best traversal per implementation

| Workload | Implementation | Build | Layout | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | vs Bajo (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dragon camera any-hit | bajo | sah | bvh16 | coh-packet16 | 16 | 59.486 | 19.624 | 6.053 | 97.451 | 0.000 |
| Dragon camera any-hit | embree | medium | native | inc-packet16 | 16 | 47.612 | 6.194 | 6.476 | 91.078 | -6.540 |
| Dragon camera any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 67.954 | 32.546 | 11.608 | 50.812 | -47.859 |
| Dragon camera closest-hit | bajo | sah | bvh16 | adaptive-16-8-4-scalar | 16 | 59.486 | 19.624 | 7.545 | 78.171 | 0.000 |
| Dragon camera closest-hit | embree | high | native | coh-packet8 | 8 | 263.790 | 28.810 | 9.053 | 65.153 | -16.653 |
| Dragon camera closest-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 68.498 | 32.296 | 14.711 | 40.093 | -48.711 |
| Instanced Dragon any-hit | bajo | sah | bvh4 | packet16 | 16 | 59.451 | 17.649 | 1.222 | 120.621 | 0.000 |
| Instanced Dragon any-hit | embree | high | native | inc-packet16 | 16 | 261.612 | 29.585 | 1.444 | 102.148 | -15.315 |
| Instanced Dragon any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 67.699 | 31.935 | 2.721 | 54.191 | -55.073 |
| Instanced Dragon closest-hit | bajo | sah | bvh4 | packet16 | 16 | 59.451 | 17.649 | 1.760 | 83.798 | 0.000 |
| Instanced Dragon closest-hit | embree | medium | native | inc-packet8 | 8 | 49.306 | 6.010 | 1.905 | 77.410 | -7.623 |
| Instanced Dragon closest-hit | tinybvh | high | bvh8 | scalar1 | 1 | 373.683 | 70.802 | 3.195 | 46.148 | -44.929 |
| Dragon shuffled any-hit | bajo | sah | bvh16 | scalar1 | 1 | 59.486 | 19.624 | 9.534 | 61.867 | 0.000 |
| Dragon shuffled any-hit | embree | medium | native | inc-packet16 | 16 | 50.410 | 5.918 | 10.043 | 58.729 | -5.072 |
| Dragon shuffled any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 69.166 | 32.718 | 15.143 | 38.949 | -37.044 |
| Dragon shuffled closest-hit | bajo | sah | bvh16 | scalar1 | 1 | 59.486 | 19.624 | 14.327 | 41.167 | 0.000 |
| Dragon shuffled closest-hit | embree | medium | native | inc-packet8 | 8 | 47.876 | 5.872 | 14.404 | 40.949 | -0.530 |
| Dragon shuffled closest-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 68.369 | 32.420 | 17.897 | 32.956 | -19.946 |
| Regular-grid any-hit | bajo | hploc | bvh16 | scalar1 | 1 | 2.524 | 1.460 | 4.527 | 57.905 | 0.000 |
| Regular-grid any-hit | embree | high | native | inc-packet16 | 16 | 12.999 | 1.614 | 6.023 | 43.523 | -24.837 |
| Regular-grid any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 12.876 | 4.798 | 6.436 | 40.730 | -29.661 |
| Regular-grid closest-hit | bajo | hploc | bvh16 | adaptive-16-8-4-scalar | 16 | 2.524 | 1.460 | 4.997 | 52.458 | 0.000 |
| Regular-grid closest-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 11.272 | 4.071 | 10.218 | 25.656 | -51.092 |
| Regular-grid closest-hit | embree | medium | native | inc-packet8 | 8 | 11.152 | 1.612 | 10.479 | 25.016 | -52.312 |
| Flattened triangle grid any-hit | embree | medium | native | inc-packet16 | 16 | 0.012 | 0.015 | 0.492 | 299.581 | 18.635 |
| Flattened triangle grid any-hit | bajo | sah | bvh16 | packet16 | 16 | 0.008 | 0.008 | 0.584 | 252.523 | 0.000 |
| Flattened triangle grid closest-hit | embree | high | native | coh-packet8 | 8 | 0.017 | 0.025 | 0.706 | 208.793 | 7.203 |
| Flattened triangle grid closest-hit | bajo | sah | bvh16 | packet16 | 16 | 0.008 | 0.008 | 0.757 | 194.764 | 0.000 |
| Instanced triangle any-hit | embree | medium | native | inc-packet16 | 16 | 0.038 | 0.060 | 0.569 | 259.297 | 72.849 |
| Instanced triangle any-hit | bajo | lbvh | bvh4 | packet16 | 16 | 0.011 | 0.011 | 0.983 | 150.014 | 0.000 |
| Instanced triangle closest-hit | embree | high | native | coh-packet8 | 8 | 0.038 | 0.064 | 0.814 | 181.045 | 69.758 |
| Instanced triangle closest-hit | bajo | lbvh | bvh4 | packet16 | 16 | 0.011 | 0.011 | 1.383 | 106.649 | 0.000 |

## Best scalar traversal per implementation

This removes packet-width advantages and is the fairest direct comparison with TinyBVH's scalar API.

| Workload | Implementation | Build | Layout | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | vs Bajo (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dragon camera any-hit | bajo | sah | bvh16 | scalar1 | 1 | 59.486 | 19.624 | 7.561 | 78.014 | 0.000 |
| Dragon camera any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 67.954 | 32.546 | 11.608 | 50.812 | -34.868 |
| Dragon camera any-hit | embree | medium | native | scalar1 | 1 | 47.612 | 6.194 | 15.646 | 37.699 | -51.677 |
| Dragon camera closest-hit | bajo | sah | bvh16 | scalar1 | 1 | 59.486 | 19.624 | 9.340 | 63.152 | 0.000 |
| Dragon camera closest-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 68.498 | 32.296 | 14.711 | 40.093 | -36.513 |
| Dragon camera closest-hit | embree | medium | native | scalar1 | 1 | 48.695 | 6.374 | 18.484 | 31.910 | -49.471 |
| Instanced Dragon any-hit | bajo | sah | bvh4 | scalar1 | 1 | 59.451 | 17.649 | 2.167 | 68.051 | 0.000 |
| Instanced Dragon any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 67.699 | 31.935 | 2.721 | 54.191 | -20.367 |
| Instanced Dragon any-hit | embree | high | native | scalar1 | 1 | 261.612 | 29.585 | 3.718 | 39.655 | -41.728 |
| Instanced Dragon closest-hit | bajo | sah | bvh4 | scalar1 | 1 | 59.451 | 17.649 | 2.541 | 58.035 | 0.000 |
| Instanced Dragon closest-hit | tinybvh | high | bvh8 | scalar1 | 1 | 373.683 | 70.802 | 3.195 | 46.148 | -20.482 |
| Instanced Dragon closest-hit | embree | medium | native | scalar1 | 1 | 49.306 | 6.010 | 4.119 | 35.795 | -38.322 |
| Dragon shuffled any-hit | bajo | sah | bvh16 | scalar1 | 1 | 59.486 | 19.624 | 9.534 | 61.867 | 0.000 |
| Dragon shuffled any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 69.166 | 32.718 | 15.143 | 38.949 | -37.044 |
| Dragon shuffled any-hit | embree | high | native | scalar1 | 1 | 263.586 | 27.870 | 17.270 | 34.153 | -44.796 |
| Dragon shuffled closest-hit | bajo | sah | bvh16 | scalar1 | 1 | 59.486 | 19.624 | 14.327 | 41.167 | 0.000 |
| Dragon shuffled closest-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 68.369 | 32.420 | 17.897 | 32.956 | -19.946 |
| Dragon shuffled closest-hit | embree | high | native | scalar1 | 1 | 262.127 | 29.376 | 19.922 | 29.607 | -28.081 |
| Regular-grid any-hit | bajo | hploc | bvh16 | scalar1 | 1 | 2.524 | 1.460 | 4.527 | 57.905 | 0.000 |
| Regular-grid any-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 12.876 | 4.798 | 6.436 | 40.730 | -29.661 |
| Regular-grid any-hit | embree | high | native | scalar1 | 1 | 12.999 | 1.614 | 13.052 | 20.085 | -65.314 |
| Regular-grid closest-hit | bajo | hploc | bvh16 | scalar1 | 1 | 2.524 | 1.460 | 7.381 | 35.516 | 0.000 |
| Regular-grid closest-hit | tinybvh | sah | bvh8 | scalar1 | 1 | 11.272 | 4.071 | 10.218 | 25.656 | -27.762 |
| Regular-grid closest-hit | embree | medium | native | scalar1 | 1 | 11.152 | 1.612 | 16.960 | 15.457 | -56.479 |
| Flattened triangle grid any-hit | bajo | sah | bvh16 | scalar1 | 1 | 0.008 | 0.008 | 1.189 | 123.976 | 0.000 |
| Flattened triangle grid any-hit | embree | medium | native | scalar1 | 1 | 0.012 | 0.015 | 2.961 | 49.801 | -59.830 |
| Flattened triangle grid closest-hit | bajo | sah | bvh16 | scalar1 | 1 | 0.008 | 0.008 | 1.962 | 75.152 | 0.000 |
| Flattened triangle grid closest-hit | embree | high | native | scalar1 | 1 | 0.017 | 0.025 | 3.369 | 43.763 | -41.767 |
| Instanced triangle any-hit | bajo | sah | bvh4 | scalar1 | 1 | 0.050 | 0.048 | 1.908 | 77.285 | 0.000 |
| Instanced triangle any-hit | embree | high | native | scalar1 | 1 | 0.037 | 0.061 | 3.385 | 43.566 | -43.629 |
| Instanced triangle closest-hit | bajo | lbvh | bvh4 | scalar1 | 1 | 0.011 | 0.011 | 2.525 | 58.404 | 0.000 |
| Instanced triangle closest-hit | embree | medium | native | scalar1 | 1 | 0.037 | 0.057 | 3.848 | 38.316 | -34.395 |

## Fastest build per geometry and implementation

| Geometry | Implementation | Build | Layout | Build ms (1) | Build ms (all) |
| --- | --- | --- | --- | --- | --- |
| dragon | bajo | lbvh | bvh16 | 11.926 | 4.122 |
| dragon | tinybvh | sah | bvh2 | 40.667 | 4.790 |
| dragon | embree | medium | native | 47.876 | 5.872 |
| dragon-instances | bajo | lbvh | bvh4 | 13.305 | 4.139 |
| dragon-instances | tinybvh | sah | bvh2 | 41.618 | 4.882 |
| dragon-instances | embree | medium | native | 49.306 | 6.010 |
| grid | bajo | lbvh | bvh16 | 1.401 | 0.638 |
| grid | tinybvh | sah | bvh2 | 8.191 | 1.224 |
| grid | embree | medium | native | 10.615 | 1.372 |
| triangle-grid | bajo | sah | bvh16 | 0.008 | 0.008 |
| triangle-grid | embree | medium | native | 0.012 | 0.014 |
| triangle-instances | bajo | lbvh | bvh4 | 0.011 | 0.011 |
| triangle-instances | embree | medium | native | 0.037 | 0.057 |

## Coverage

The matrix covers synthetic and real mesh geometry, closest-hit and early-exit any-hit, coherent camera ordering and the same rays shuffled to remove neighboring-ray coherence, plus an instance-heavy BLAS/TLAS scene (one reused BLAS, a 12x9 translated-instance grid). A one-triangle BLAS and its flattened 108-triangle equivalent isolate instance continuation from BLAS complexity. Traversal is single-calling-thread; build is measured with one CPU and all available CPUs. The core traversal suites report the best of eight timed repetitions after one warmup. Instance diagnostics report the median of eight repetitions, each averaged across eight timed traversal batches, to resolve small performance differences.

## Regular-grid closest-hit

Triangles per BLAS: 65536; instances: 1; rays: 262144; query: closest; ray order: structured.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh2 | 2 | 2 | scalar1 | 1 | 5.490 | 2.440 | 17.154 | 15.282 |  | 32767 | 6443188224.000 |
| bajo | hploc | bvh4 | 4 | 4 | scalar1 | 1 | 5.539 | 2.229 | 10.239 | 25.603 |  | 5461 | 6443188224.000 |
| bajo | hploc | bvh8 | 8 | 8 | scalar1 | 1 | 4.982 | 2.224 | 10.980 | 23.874 |  | 1641 | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 2.524 | 1.460 | 4.997 | 52.458 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 2.524 | 1.460 | 5.017 | 52.256 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet16 | 16 | 2.524 | 1.460 | 5.915 | 44.319 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet4 | 4 | 2.524 | 1.460 | 11.865 | 22.094 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet8 | 8 | 2.524 | 1.460 | 7.347 | 35.682 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 2.524 | 1.460 | 6.446 | 40.667 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 2.524 | 1.460 | 14.928 | 17.561 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 2.524 | 1.460 | 8.307 | 31.557 | 196608 |  | 6443188224.000 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 2.524 | 1.460 | 7.381 | 35.516 |  | 273 | 6443188224.000 |
| bajo | lbvh | bvh2 | 2 | 2 | scalar1 | 1 | 2.396 | 1.459 | 18.855 | 13.903 |  | 32767 | 6443188224.000 |
| bajo | lbvh | bvh4 | 4 | 4 | scalar1 | 1 | 1.809 | 1.011 | 11.219 | 23.366 |  | 5461 | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | adaptive-16-8-4-scalar | 16 | 1.626 | 0.842 | 5.785 | 45.318 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | adaptive-16-8-scalar | 16 | 1.626 | 0.842 | 5.684 | 46.121 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | coh-packet16 | 16 | 1.626 | 0.842 | 5.993 | 43.741 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | coh-packet4 | 4 | 1.626 | 0.842 | 9.951 | 26.343 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | coh-packet8 | 8 | 1.626 | 0.842 | 7.304 | 35.890 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | packet16 | 16 | 1.626 | 0.842 | 6.657 | 39.376 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | packet4 | 4 | 1.626 | 0.842 | 11.420 | 22.954 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | packet8 | 8 | 1.626 | 0.842 | 8.001 | 32.764 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | scalar1 | 1 | 1.626 | 0.842 | 11.912 | 22.007 |  | 4681 | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 1.401 | 0.638 | 7.603 | 34.480 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 1.401 | 0.638 | 7.727 | 33.927 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet16 | 16 | 1.401 | 0.638 | 8.422 | 31.126 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet4 | 4 | 1.401 | 0.638 | 12.525 | 20.930 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet8 | 8 | 1.401 | 0.638 | 9.841 | 26.638 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 1.401 | 0.638 | 8.265 | 31.718 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 1.401 | 0.638 | 14.293 | 18.341 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 1.401 | 0.638 | 10.379 | 25.257 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 1.401 | 0.638 | 7.567 | 34.643 |  | 273 | 6443188224.000 |
| bajo | median | bvh2 | 2 | 2 | scalar1 | 1 | 8.761 | 2.741 | 18.825 | 13.925 |  | 32767 | 6443188224.000 |
| bajo | median | bvh4 | 4 | 4 | scalar1 | 1 | 8.694 | 2.665 | 11.231 | 23.342 |  | 5461 | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | adaptive-16-8-4-scalar | 16 | 7.832 | 2.572 | 5.696 | 46.026 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | adaptive-16-8-scalar | 16 | 7.832 | 2.572 | 5.657 | 46.341 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | coh-packet16 | 16 | 7.832 | 2.572 | 6.287 | 41.697 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | coh-packet4 | 4 | 7.832 | 2.572 | 9.955 | 26.333 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | coh-packet8 | 8 | 7.832 | 2.572 | 7.252 | 36.147 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | packet16 | 16 | 7.832 | 2.572 | 6.865 | 38.188 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | packet4 | 4 | 7.832 | 2.572 | 11.813 | 22.190 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | packet8 | 8 | 7.832 | 2.572 | 7.927 | 33.071 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | scalar1 | 1 | 7.832 | 2.572 | 11.545 | 22.706 |  | 1193 | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 7.280 | 2.292 | 7.377 | 35.534 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 7.280 | 2.292 | 7.352 | 35.658 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet16 | 16 | 7.280 | 2.292 | 8.010 | 32.727 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet4 | 4 | 7.280 | 2.292 | 12.191 | 21.504 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet8 | 8 | 7.280 | 2.292 | 9.362 | 28.002 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | packet16 | 16 | 7.280 | 2.292 | 8.274 | 31.683 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | packet4 | 4 | 7.280 | 2.292 | 14.002 | 18.722 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | packet8 | 8 | 7.280 | 2.292 | 10.162 | 25.796 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | scalar1 | 1 | 7.280 | 2.292 | 7.412 | 35.368 |  | 273 | 6443188224.000 |
| bajo | sah | bvh2 | 2 | 2 | scalar1 | 1 | 19.404 | 5.872 | 17.471 | 15.004 |  | 32767 | 6443188224.000 |
| bajo | sah | bvh4 | 4 | 4 | scalar1 | 1 | 15.204 | 4.081 | 9.845 | 26.626 |  | 5461 | 6443188224.000 |
| bajo | sah | bvh8 | 8 | 8 | scalar1 | 1 | 11.911 | 3.610 | 10.029 | 26.139 |  | 1273 | 6443188224.000 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 10.018 | 3.118 | 8.457 | 30.996 |  | 273 | 6443188224.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 12.959 | 1.785 | 13.351 | 19.634 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 12.959 | 1.785 | 14.234 | 18.417 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 12.959 | 1.785 | 13.040 | 20.102 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 12.959 | 1.785 | 10.924 | 23.997 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 12.959 | 1.785 | 11.666 | 22.471 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 12.959 | 1.785 | 10.529 | 24.897 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | scalar1 | 1 | 12.959 | 1.785 | 17.287 | 15.164 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 11.152 | 1.612 | 13.226 | 19.820 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 11.152 | 1.612 | 14.134 | 18.546 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 11.152 | 1.612 | 13.040 | 20.103 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 11.152 | 1.612 | 10.879 | 24.097 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 11.152 | 1.612 | 11.444 | 22.906 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 11.152 | 1.612 | 10.479 | 25.016 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | scalar1 | 1 | 11.152 | 1.612 | 16.960 | 15.457 | 196608 |  | 6443188224.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 29.558 | 4.340 | 18.367 | 14.273 | 196608 |  | 6443188224.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 31.787 | 6.094 | 12.319 | 21.280 | 196608 |  | 6443188223.985 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 33.419 | 7.068 | 10.322 | 25.397 | 196608 |  | 6443188223.985 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 8.191 | 1.224 | 18.546 | 14.135 | 196608 |  | 6443188224.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 10.064 | 3.157 | 12.143 | 21.588 | 196608 |  | 6443188223.985 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 11.272 | 4.071 | 10.218 | 25.656 | 196608 |  | 6443188223.985 |

## Regular-grid any-hit

Triangles per BLAS: 65536; instances: 1; rays: 262144; query: any; ray order: structured.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet16 | 16 | 2.524 | 1.460 | 4.881 | 53.705 | 196608 |  | 196608.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 2.524 | 1.460 | 5.167 | 50.735 | 196608 |  | 196608.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 2.524 | 1.460 | 12.080 | 21.700 | 196608 |  | 196608.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 2.524 | 1.460 | 6.861 | 38.206 | 196608 |  | 196608.000 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 2.524 | 1.460 | 4.527 | 57.905 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet16 | 16 | 1.401 | 0.638 | 6.271 | 41.800 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 1.401 | 0.638 | 6.967 | 37.626 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 1.401 | 0.638 | 12.930 | 20.274 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 1.401 | 0.638 | 8.641 | 30.338 | 196608 |  | 196608.000 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 1.401 | 0.638 | 4.609 | 56.876 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet16 | 16 | 7.280 | 2.292 | 7.107 | 36.886 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | packet16 | 16 | 7.280 | 2.292 | 7.546 | 34.741 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | packet4 | 4 | 7.280 | 2.292 | 13.476 | 19.453 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | packet8 | 8 | 7.280 | 2.292 | 9.244 | 28.359 | 196608 |  | 196608.000 |
| bajo | median | bvh16 | 16 | 16 | scalar1 | 1 | 7.280 | 2.292 | 4.597 | 57.027 | 196608 |  | 196608.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 12.999 | 1.614 | 11.510 | 22.776 | 196608 |  | 196608.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 12.999 | 1.614 | 13.836 | 18.947 | 196608 |  | 196608.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 12.999 | 1.614 | 12.377 | 21.179 | 196608 |  | 196608.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 12.999 | 1.614 | 6.023 | 43.523 | 196608 |  | 196608.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 12.999 | 1.614 | 7.817 | 33.535 | 196608 |  | 196608.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 12.999 | 1.614 | 6.364 | 41.189 | 196608 |  | 196608.000 |
| embree | high | native |  |  | scalar1 | 1 | 12.999 | 1.614 | 13.052 | 20.085 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 10.615 | 1.372 | 11.522 | 22.752 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 10.615 | 1.372 | 13.836 | 18.947 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 10.615 | 1.372 | 12.364 | 21.203 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 10.615 | 1.372 | 6.093 | 43.021 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 10.615 | 1.372 | 7.821 | 33.516 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 10.615 | 1.372 | 6.336 | 41.377 | 196608 |  | 196608.000 |
| embree | medium | native |  |  | scalar1 | 1 | 10.615 | 1.372 | 13.054 | 20.082 | 196608 |  | 196608.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 32.629 | 4.147 | 17.414 | 15.054 | 196608 |  | 196608.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 31.936 | 5.981 | 7.818 | 33.532 | 196608 |  | 196608.000 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 43.900 | 6.573 | 9.367 | 27.985 | 196608 |  | 196608.000 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 8.214 | 1.627 | 17.260 | 15.188 | 196608 |  | 196608.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 9.929 | 3.046 | 7.850 | 33.392 | 196608 |  | 196608.000 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 12.876 | 4.798 | 6.436 | 40.730 | 196608 |  | 196608.000 |

## Dragon camera closest-hit

Triangles per BLAS: 249882; instances: 1; rays: 589824; query: closest; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 24.281 | 11.626 | 7.660 | 77.005 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 24.281 | 11.626 | 7.769 | 75.921 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet16 | 16 | 24.281 | 11.626 | 9.562 | 61.685 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet4 | 4 | 24.281 | 11.626 | 11.904 | 49.548 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet8 | 8 | 24.281 | 11.626 | 10.111 | 58.337 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 24.281 | 11.626 | 11.393 | 51.773 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 24.281 | 11.626 | 16.213 | 36.379 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 24.281 | 11.626 | 13.022 | 45.296 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 24.281 | 11.626 | 10.118 | 58.292 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 11.926 | 4.122 | 7.756 | 76.050 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 11.926 | 4.122 | 7.902 | 74.640 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet16 | 16 | 11.926 | 4.122 | 9.583 | 61.550 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet4 | 4 | 11.926 | 4.122 | 12.586 | 46.864 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet8 | 8 | 11.926 | 4.122 | 10.594 | 55.675 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 11.926 | 4.122 | 12.131 | 48.621 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 11.926 | 4.122 | 16.896 | 34.910 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 11.926 | 4.122 | 14.011 | 42.098 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 11.926 | 4.122 | 10.465 | 56.362 | 71597 |  | 7943562615.175 |
| bajo | median | bvh16 | 16 | 16 | scalar1 | 1 | 40.029 | 12.147 | 10.157 | 58.074 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 59.486 | 19.624 | 7.545 | 78.171 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 59.486 | 19.624 | 7.586 | 77.751 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet16 | 16 | 59.486 | 19.624 | 9.372 | 62.938 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet4 | 4 | 59.486 | 19.624 | 12.750 | 46.262 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet8 | 8 | 59.486 | 19.624 | 11.074 | 53.263 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 59.486 | 19.624 | 11.159 | 52.859 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 59.486 | 19.624 | 15.273 | 38.620 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 59.486 | 19.624 | 12.722 | 46.361 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 59.486 | 19.624 | 9.340 | 63.152 | 71597 |  | 7943562615.175 |
| embree | high | native |  |  | coh-packet16 | 16 | 263.790 | 28.810 | 9.715 | 60.712 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | coh-packet4 | 4 | 263.790 | 28.810 | 11.609 | 50.809 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | coh-packet8 | 8 | 263.790 | 28.810 | 9.053 | 65.153 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet16 | 16 | 263.790 | 28.810 | 12.193 | 48.375 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet4 | 4 | 263.790 | 28.810 | 13.064 | 45.148 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet8 | 8 | 263.790 | 28.810 | 11.943 | 49.386 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | scalar1 | 1 | 263.790 | 28.810 | 18.493 | 31.894 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet16 | 16 | 48.695 | 6.374 | 9.688 | 60.884 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet4 | 4 | 48.695 | 6.374 | 11.575 | 50.958 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet8 | 8 | 48.695 | 6.374 | 9.091 | 64.879 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet16 | 16 | 48.695 | 6.374 | 12.165 | 48.486 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet4 | 4 | 48.695 | 6.374 | 13.038 | 45.238 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet8 | 8 | 48.695 | 6.374 | 11.904 | 49.548 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | scalar1 | 1 | 48.695 | 6.374 | 18.484 | 31.910 | 71598 |  | 7943741995.515 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 358.928 | 48.339 | 25.667 | 22.980 | 71599 |  | 7943796499.200 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 370.711 | 64.525 | 17.135 | 34.422 | 71599 |  | 7943796499.445 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 376.856 | 70.376 | 14.816 | 39.810 | 71599 |  | 7943796499.439 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 46.259 | 4.985 | 26.138 | 22.565 | 71599 |  | 7943796499.200 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 66.015 | 26.500 | 17.736 | 33.255 | 71599 |  | 7943796499.445 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 68.498 | 32.296 | 14.711 | 40.093 | 71599 |  | 7943796499.439 |

## Dragon camera any-hit

Triangles per BLAS: 249882; instances: 1; rays: 589824; query: any; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | coh-packet16 | 16 | 24.281 | 11.626 | 6.432 | 91.707 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 24.281 | 11.626 | 8.683 | 67.932 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 24.281 | 11.626 | 12.542 | 47.026 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 24.281 | 11.626 | 10.569 | 55.807 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 24.281 | 11.626 | 8.388 | 70.315 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet16 | 16 | 11.926 | 4.122 | 6.774 | 87.077 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 11.926 | 4.122 | 8.740 | 67.489 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 11.926 | 4.122 | 13.079 | 45.098 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 11.926 | 4.122 | 10.983 | 53.703 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 11.926 | 4.122 | 8.896 | 66.300 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet16 | 16 | 59.486 | 19.624 | 6.053 | 97.451 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 59.486 | 19.624 | 8.334 | 70.775 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 59.486 | 19.624 | 12.148 | 48.552 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 59.486 | 19.624 | 10.342 | 57.031 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 59.486 | 19.624 | 7.561 | 78.014 | 71597 |  | 71597.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 261.629 | 28.656 | 7.474 | 78.920 | 71598 |  | 71598.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 261.629 | 28.656 | 10.436 | 56.519 | 71598 |  | 71598.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 261.629 | 28.656 | 8.228 | 71.687 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 261.629 | 28.656 | 7.119 | 82.854 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 261.629 | 28.656 | 9.372 | 62.934 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 261.629 | 28.656 | 7.071 | 83.420 | 71598 |  | 71598.000 |
| embree | high | native |  |  | scalar1 | 1 | 261.629 | 28.656 | 16.194 | 36.423 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 47.612 | 6.194 | 7.300 | 80.795 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 47.612 | 6.194 | 10.596 | 55.665 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 47.612 | 6.194 | 8.106 | 72.763 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 47.612 | 6.194 | 6.476 | 91.078 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 47.612 | 6.194 | 9.200 | 64.113 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 47.612 | 6.194 | 6.974 | 84.570 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | scalar1 | 1 | 47.612 | 6.194 | 15.646 | 37.699 | 71598 |  | 71598.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 360.214 | 46.968 | 19.821 | 29.758 | 71599 |  | 71599.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 376.148 | 65.815 | 13.666 | 43.159 | 71599 |  | 71599.000 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 375.810 | 70.662 | 11.824 | 49.884 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.682 | 5.151 | 19.924 | 29.603 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 62.405 | 26.870 | 13.776 | 42.814 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 67.954 | 32.546 | 11.608 | 50.812 | 71599 |  | 71599.000 |

## Dragon shuffled closest-hit

Triangles per BLAS: 249882; instances: 1; rays: 589824; query: closest; ray order: shuffled-camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 24.281 | 11.626 | 33.089 | 17.825 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 24.281 | 11.626 | 17.307 | 34.080 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 24.281 | 11.626 | 24.171 | 24.402 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 24.281 | 11.626 | 22.539 | 26.169 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 24.281 | 11.626 | 29.297 | 20.132 | 71597 |  | 7943562615.175 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 24.281 | 11.626 | 15.648 | 37.694 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 11.926 | 4.122 | 34.991 | 16.857 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 11.926 | 4.122 | 17.803 | 33.131 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 11.926 | 4.122 | 24.950 | 23.640 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 11.926 | 4.122 | 25.918 | 22.757 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 11.926 | 4.122 | 29.427 | 20.044 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 11.926 | 4.122 | 16.191 | 36.429 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | adaptive-16-8-4-scalar | 16 | 59.486 | 19.624 | 29.285 | 20.141 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | adaptive-16-8-scalar | 16 | 59.486 | 19.624 | 16.074 | 36.695 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 59.486 | 19.624 | 23.265 | 25.352 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 59.486 | 19.624 | 21.037 | 28.037 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 59.486 | 19.624 | 27.972 | 21.087 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 59.486 | 19.624 | 14.327 | 41.167 | 71597 |  | 7943562615.175 |
| embree | high | native |  |  | coh-packet16 | 16 | 262.127 | 29.376 | 33.222 | 17.754 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | coh-packet4 | 4 | 262.127 | 29.376 | 33.082 | 17.829 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | coh-packet8 | 8 | 262.127 | 29.376 | 36.181 | 16.302 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet16 | 16 | 262.127 | 29.376 | 16.591 | 35.551 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet4 | 4 | 262.127 | 29.376 | 15.563 | 37.900 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet8 | 8 | 262.127 | 29.376 | 16.291 | 36.206 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | scalar1 | 1 | 262.127 | 29.376 | 19.922 | 29.607 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet16 | 16 | 47.876 | 5.872 | 31.184 | 18.915 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet4 | 4 | 47.876 | 5.872 | 31.535 | 18.704 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet8 | 8 | 47.876 | 5.872 | 32.459 | 18.171 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet16 | 16 | 47.876 | 5.872 | 14.758 | 39.967 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet4 | 4 | 47.876 | 5.872 | 15.536 | 37.966 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet8 | 8 | 47.876 | 5.872 | 14.404 | 40.949 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | scalar1 | 1 | 47.876 | 5.872 | 19.926 | 29.601 | 71598 |  | 7943741995.515 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 357.649 | 48.098 | 33.849 | 17.425 | 71599 |  | 7943796499.200 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 376.777 | 65.981 | 21.363 | 27.610 | 71599 |  | 7943796499.445 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 378.094 | 70.412 | 17.907 | 32.938 | 71599 |  | 7943796499.439 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.636 | 5.216 | 35.143 | 16.783 | 71599 |  | 7943796499.200 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 62.270 | 26.294 | 20.680 | 28.521 | 71599 |  | 7943796499.445 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 68.369 | 32.420 | 17.897 | 32.956 | 71599 |  | 7943796499.439 |

## Dragon shuffled any-hit

Triangles per BLAS: 249882; instances: 1; rays: 589824; query: any; ray order: shuffled-camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh16 | 16 | 16 | packet16 | 16 | 24.281 | 11.626 | 18.801 | 31.372 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet4 | 4 | 24.281 | 11.626 | 17.372 | 33.953 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | packet8 | 8 | 24.281 | 11.626 | 24.374 | 24.199 | 71597 |  | 71597.000 |
| bajo | hploc | bvh16 | 16 | 16 | scalar1 | 1 | 24.281 | 11.626 | 10.457 | 56.407 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 11.926 | 4.122 | 18.913 | 31.187 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 11.926 | 4.122 | 20.982 | 28.111 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 11.926 | 4.122 | 24.168 | 24.405 | 71597 |  | 71597.000 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 11.926 | 4.122 | 11.148 | 52.909 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 59.486 | 19.624 | 18.423 | 32.016 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 59.486 | 19.624 | 16.640 | 35.446 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 59.486 | 19.624 | 23.480 | 25.120 | 71597 |  | 71597.000 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 59.486 | 19.624 | 9.534 | 61.867 | 71597 |  | 71597.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 263.586 | 27.870 | 26.340 | 22.393 | 71598 |  | 71598.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 263.586 | 27.870 | 26.134 | 22.569 | 71598 |  | 71598.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 263.586 | 27.870 | 25.874 | 22.796 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 263.586 | 27.870 | 10.063 | 58.614 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 263.586 | 27.870 | 12.125 | 48.647 | 71598 |  | 71598.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 263.586 | 27.870 | 10.706 | 55.092 | 71598 |  | 71598.000 |
| embree | high | native |  |  | scalar1 | 1 | 263.586 | 27.870 | 17.270 | 34.153 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 50.410 | 5.918 | 23.563 | 25.032 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 50.410 | 5.918 | 26.953 | 21.884 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 50.410 | 5.918 | 25.860 | 22.808 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 50.410 | 5.918 | 10.043 | 58.729 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 50.410 | 5.918 | 12.940 | 45.580 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 50.410 | 5.918 | 10.586 | 55.717 | 71598 |  | 71598.000 |
| embree | medium | native |  |  | scalar1 | 1 | 50.410 | 5.918 | 17.823 | 33.093 | 71598 |  | 71598.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 360.678 | 46.561 | 28.283 | 20.855 | 71599 |  | 71599.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 374.760 | 66.695 | 17.733 | 33.261 | 71599 |  | 71599.000 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 383.168 | 71.231 | 15.223 | 38.746 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.667 | 4.790 | 27.901 | 21.140 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 62.803 | 26.104 | 17.585 | 33.542 | 71599 |  | 71599.000 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 69.166 | 32.718 | 15.143 | 38.949 | 71599 |  | 71599.000 |

## Instanced Dragon closest-hit

Triangles per BLAS: 249882; instances: 108; rays: 147456; query: closest; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh4 | 4 | 1 | packet16 | 16 | 24.679 | 10.518 | 1.874 | 78.667 | 8250 |  | 1022292747.551 |
| bajo | hploc | bvh4 | 4 | 1 | packet4 | 4 | 24.679 | 10.518 | 2.308 | 63.880 | 8250 |  | 1022292747.551 |
| bajo | hploc | bvh4 | 4 | 1 | packet8 | 8 | 24.679 | 10.518 | 1.988 | 74.166 | 8250 |  | 1022292747.551 |
| bajo | hploc | bvh4 | 4 | 1 | scalar1 | 1 | 24.679 | 10.518 | 2.622 | 56.231 | 8250 |  | 1022292747.551 |
| bajo | lbvh | bvh4 | 4 | 1 | packet16 | 16 | 13.305 | 4.139 | 1.957 | 75.334 | 8250 |  | 1022292856.202 |
| bajo | lbvh | bvh4 | 4 | 1 | packet4 | 4 | 13.305 | 4.139 | 2.383 | 61.882 | 8250 |  | 1022292856.202 |
| bajo | lbvh | bvh4 | 4 | 1 | packet8 | 8 | 13.305 | 4.139 | 2.053 | 71.809 | 8250 |  | 1022292856.202 |
| bajo | lbvh | bvh4 | 4 | 1 | scalar1 | 1 | 13.305 | 4.139 | 2.726 | 54.092 | 8250 |  | 1022292856.202 |
| bajo | sah | bvh4 | 4 | 1 | packet16 | 16 | 59.451 | 17.649 | 1.760 | 83.798 | 8250 |  | 1022292856.202 |
| bajo | sah | bvh4 | 4 | 1 | packet4 | 4 | 59.451 | 17.649 | 2.177 | 67.742 | 8250 |  | 1022292856.202 |
| bajo | sah | bvh4 | 4 | 1 | packet8 | 8 | 59.451 | 17.649 | 1.852 | 79.608 | 8250 |  | 1022292856.202 |
| bajo | sah | bvh4 | 4 | 1 | scalar1 | 1 | 59.451 | 17.649 | 2.541 | 58.035 | 8250 |  | 1022292856.202 |
| embree | high | native |  |  | coh-packet16 | 16 | 265.191 | 28.814 | 4.319 | 34.142 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | coh-packet4 | 4 | 265.191 | 28.814 | 4.381 | 33.657 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | coh-packet8 | 8 | 265.191 | 28.814 | 3.717 | 39.670 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | inc-packet16 | 16 | 265.191 | 28.814 | 2.415 | 61.060 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | inc-packet4 | 4 | 265.191 | 28.814 | 2.417 | 60.998 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | inc-packet8 | 8 | 265.191 | 28.814 | 2.021 | 72.966 | 8256 |  | 1024176197.799 |
| embree | high | native |  |  | scalar1 | 1 | 265.191 | 28.814 | 4.579 | 32.203 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | coh-packet16 | 16 | 49.306 | 6.010 | 3.974 | 37.101 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | coh-packet4 | 4 | 49.306 | 6.010 | 3.836 | 38.442 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | coh-packet8 | 8 | 49.306 | 6.010 | 3.515 | 41.945 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | inc-packet16 | 16 | 49.306 | 6.010 | 2.128 | 69.299 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | inc-packet4 | 4 | 49.306 | 6.010 | 2.355 | 62.621 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | inc-packet8 | 8 | 49.306 | 6.010 | 1.905 | 77.410 | 8256 |  | 1024176197.799 |
| embree | medium | native |  |  | scalar1 | 1 | 49.306 | 6.010 | 4.119 | 35.795 | 8256 |  | 1024176197.799 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 361.439 | 47.052 | 5.759 | 25.604 | 8256 |  | 1024143130.934 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 375.527 | 66.937 | 3.594 | 41.031 | 8256 |  | 1024092270.370 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 373.683 | 70.802 | 3.195 | 46.148 | 8256 |  | 1024092270.381 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 41.618 | 4.882 | 5.670 | 26.009 | 8256 |  | 1024143130.934 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 65.600 | 26.387 | 3.789 | 38.913 | 8256 |  | 1024092270.370 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 70.571 | 31.367 | 3.404 | 43.316 | 8256 |  | 1024092270.381 |

## Instanced Dragon any-hit

Triangles per BLAS: 249882; instances: 108; rays: 147456; query: any; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh4 | 4 | 1 | packet16 | 16 | 24.679 | 10.518 | 1.323 | 111.429 | 8250 |  | 8250.000 |
| bajo | hploc | bvh4 | 4 | 1 | packet4 | 4 | 24.679 | 10.518 | 1.754 | 84.076 | 8250 |  | 8250.000 |
| bajo | hploc | bvh4 | 4 | 1 | packet8 | 8 | 24.679 | 10.518 | 1.454 | 101.445 | 8250 |  | 8250.000 |
| bajo | hploc | bvh4 | 4 | 1 | scalar1 | 1 | 24.679 | 10.518 | 2.205 | 66.878 | 8250 |  | 8250.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet16 | 16 | 13.305 | 4.139 | 1.426 | 103.394 | 8250 |  | 8250.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet4 | 4 | 13.305 | 4.139 | 1.841 | 80.098 | 8250 |  | 8250.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet8 | 8 | 13.305 | 4.139 | 1.553 | 94.969 | 8250 |  | 8250.000 |
| bajo | lbvh | bvh4 | 4 | 1 | scalar1 | 1 | 13.305 | 4.139 | 2.320 | 63.567 | 8250 |  | 8250.000 |
| bajo | sah | bvh4 | 4 | 1 | packet16 | 16 | 59.451 | 17.649 | 1.222 | 120.621 | 8250 |  | 8250.000 |
| bajo | sah | bvh4 | 4 | 1 | packet4 | 4 | 59.451 | 17.649 | 1.662 | 88.725 | 8250 |  | 8250.000 |
| bajo | sah | bvh4 | 4 | 1 | packet8 | 8 | 59.451 | 17.649 | 1.363 | 108.160 | 8250 |  | 8250.000 |
| bajo | sah | bvh4 | 4 | 1 | scalar1 | 1 | 59.451 | 17.649 | 2.167 | 68.051 | 8250 |  | 8250.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 261.612 | 29.585 | 2.896 | 50.912 | 8256 |  | 8256.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 261.612 | 29.585 | 3.191 | 46.203 | 8256 |  | 8256.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 261.612 | 29.585 | 2.836 | 51.989 | 8256 |  | 8256.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 261.612 | 29.585 | 1.444 | 102.148 | 8256 |  | 8256.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 261.612 | 29.585 | 2.068 | 71.287 | 8256 |  | 8256.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 261.612 | 29.585 | 1.576 | 93.578 | 8256 |  | 8256.000 |
| embree | high | native |  |  | scalar1 | 1 | 261.612 | 29.585 | 3.718 | 39.655 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 56.731 | 6.018 | 2.901 | 50.838 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 56.731 | 6.018 | 3.238 | 45.538 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 56.731 | 6.018 | 2.831 | 52.090 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 56.731 | 6.018 | 1.456 | 101.303 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 56.731 | 6.018 | 2.249 | 65.557 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 56.731 | 6.018 | 1.594 | 92.503 | 8256 |  | 8256.000 |
| embree | medium | native |  |  | scalar1 | 1 | 56.731 | 6.018 | 3.995 | 36.910 | 8256 |  | 8256.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 361.274 | 47.175 | 4.195 | 35.154 | 8256 |  | 8256.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 370.323 | 66.115 | 3.076 | 47.945 | 8256 |  | 8256.000 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 374.469 | 71.383 | 3.154 | 46.750 | 8256 |  | 8256.000 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 40.911 | 4.979 | 4.255 | 34.658 | 8256 |  | 8256.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 62.605 | 26.518 | 3.124 | 47.203 | 8256 |  | 8256.000 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 67.699 | 31.935 | 2.721 | 54.191 | 8256 |  | 8256.000 |

## Instanced triangle closest-hit

Triangles per BLAS: 1; instances: 108; rays: 147456; query: closest; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh4 | 4 | 1 | packet16 | 16 | 0.019 | 0.017 | 1.403 | 105.125 | 20416 |  | 1177505.908 |
| bajo | hploc | bvh4 | 4 | 1 | packet4 | 4 | 0.019 | 0.017 | 1.833 | 80.448 | 20416 |  | 1177505.908 |
| bajo | hploc | bvh4 | 4 | 1 | packet8 | 8 | 0.019 | 0.017 | 1.528 | 96.495 | 20416 |  | 1177505.908 |
| bajo | hploc | bvh4 | 4 | 1 | scalar1 | 1 | 0.019 | 0.017 | 2.572 | 57.333 | 20416 |  | 1177505.908 |
| bajo | lbvh | bvh4 | 4 | 1 | packet16 | 16 | 0.011 | 0.011 | 1.383 | 106.649 | 20416 |  | 1177505.908 |
| bajo | lbvh | bvh4 | 4 | 1 | packet4 | 4 | 0.011 | 0.011 | 1.769 | 83.356 | 20416 |  | 1177505.908 |
| bajo | lbvh | bvh4 | 4 | 1 | packet8 | 8 | 0.011 | 0.011 | 1.496 | 98.596 | 20416 |  | 1177505.908 |
| bajo | lbvh | bvh4 | 4 | 1 | scalar1 | 1 | 0.011 | 0.011 | 2.525 | 58.404 | 20416 |  | 1177505.908 |
| bajo | sah | bvh4 | 4 | 1 | packet16 | 16 | 0.050 | 0.048 | 1.393 | 105.857 | 20416 |  | 1177505.908 |
| bajo | sah | bvh4 | 4 | 1 | packet4 | 4 | 0.050 | 0.048 | 1.774 | 83.135 | 20416 |  | 1177505.908 |
| bajo | sah | bvh4 | 4 | 1 | packet8 | 8 | 0.050 | 0.048 | 1.505 | 98.008 | 20416 |  | 1177505.908 |
| bajo | sah | bvh4 | 4 | 1 | scalar1 | 1 | 0.050 | 0.048 | 2.526 | 58.378 | 20416 |  | 1177505.908 |
| embree | high | native |  |  | coh-packet16 | 16 | 0.038 | 0.064 | 1.033 | 142.783 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | coh-packet4 | 4 | 0.038 | 0.064 | 1.320 | 111.719 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | coh-packet8 | 8 | 0.038 | 0.064 | 0.814 | 181.045 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | inc-packet16 | 16 | 0.038 | 0.064 | 1.141 | 129.253 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | inc-packet4 | 4 | 0.038 | 0.064 | 1.375 | 107.243 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | inc-packet8 | 8 | 0.038 | 0.064 | 0.966 | 152.627 | 20416 |  | 1177505.898 |
| embree | high | native |  |  | scalar1 | 1 | 0.038 | 0.064 | 3.854 | 38.261 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | coh-packet16 | 16 | 0.037 | 0.057 | 1.014 | 145.365 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | coh-packet4 | 4 | 0.037 | 0.057 | 1.300 | 113.393 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | coh-packet8 | 8 | 0.037 | 0.057 | 0.816 | 180.702 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | inc-packet16 | 16 | 0.037 | 0.057 | 1.144 | 128.877 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | inc-packet4 | 4 | 0.037 | 0.057 | 1.375 | 107.214 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | inc-packet8 | 8 | 0.037 | 0.057 | 0.966 | 152.629 | 20416 |  | 1177505.898 |
| embree | medium | native |  |  | scalar1 | 1 | 0.037 | 0.057 | 3.848 | 38.316 | 20416 |  | 1177505.898 |

## Instanced triangle any-hit

Triangles per BLAS: 1; instances: 108; rays: 147456; query: any; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | hploc | bvh4 | 4 | 1 | packet16 | 16 | 0.019 | 0.017 | 1.008 | 146.350 | 20416 |  | 20416.000 |
| bajo | hploc | bvh4 | 4 | 1 | packet4 | 4 | 0.019 | 0.017 | 1.507 | 97.871 | 20416 |  | 20416.000 |
| bajo | hploc | bvh4 | 4 | 1 | packet8 | 8 | 0.019 | 0.017 | 1.180 | 125.005 | 20416 |  | 20416.000 |
| bajo | hploc | bvh4 | 4 | 1 | scalar1 | 1 | 0.019 | 0.017 | 1.924 | 76.646 | 20416 |  | 20416.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet16 | 16 | 0.011 | 0.011 | 0.983 | 150.014 | 20416 |  | 20416.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet4 | 4 | 0.011 | 0.011 | 1.424 | 103.528 | 20416 |  | 20416.000 |
| bajo | lbvh | bvh4 | 4 | 1 | packet8 | 8 | 0.011 | 0.011 | 1.140 | 129.297 | 20416 |  | 20416.000 |
| bajo | lbvh | bvh4 | 4 | 1 | scalar1 | 1 | 0.011 | 0.011 | 1.911 | 77.147 | 20416 |  | 20416.000 |
| bajo | sah | bvh4 | 4 | 1 | packet16 | 16 | 0.050 | 0.048 | 0.993 | 148.554 | 20416 |  | 20416.000 |
| bajo | sah | bvh4 | 4 | 1 | packet4 | 4 | 0.050 | 0.048 | 1.429 | 103.223 | 20416 |  | 20416.000 |
| bajo | sah | bvh4 | 4 | 1 | packet8 | 8 | 0.050 | 0.048 | 1.151 | 128.113 | 20416 |  | 20416.000 |
| bajo | sah | bvh4 | 4 | 1 | scalar1 | 1 | 0.050 | 0.048 | 1.908 | 77.285 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 0.037 | 0.061 | 0.669 | 220.562 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 0.037 | 0.061 | 1.302 | 113.273 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 0.037 | 0.061 | 0.802 | 183.834 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 0.037 | 0.061 | 0.570 | 258.725 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 0.037 | 0.061 | 1.149 | 128.303 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 0.037 | 0.061 | 0.665 | 221.633 | 20416 |  | 20416.000 |
| embree | high | native |  |  | scalar1 | 1 | 0.037 | 0.061 | 3.385 | 43.566 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 0.038 | 0.060 | 0.669 | 220.481 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 0.038 | 0.060 | 1.312 | 112.351 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 0.038 | 0.060 | 0.798 | 184.885 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 0.038 | 0.060 | 0.569 | 259.297 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 0.038 | 0.060 | 1.163 | 126.810 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 0.038 | 0.060 | 0.670 | 220.199 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | scalar1 | 1 | 0.038 | 0.060 | 3.388 | 43.517 | 20416 |  | 20416.000 |

## Flattened triangle grid closest-hit

Triangles per BLAS: 108; instances: 1; rays: 147456; query: closest; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | sah | bvh16 | 16 | 16 | coh-packet16 | 16 | 0.008 | 0.008 | 0.863 | 170.769 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet4 | 4 | 0.008 | 0.008 | 1.886 | 78.184 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet8 | 8 | 0.008 | 0.008 | 1.288 | 114.445 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 0.008 | 0.008 | 0.757 | 194.764 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 0.008 | 0.008 | 1.762 | 83.694 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 0.008 | 0.008 | 1.122 | 131.444 | 20416 |  | 2281401.908 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 0.008 | 0.008 | 1.962 | 75.152 | 20416 |  | 2281401.908 |
| embree | high | native |  |  | coh-packet16 | 16 | 0.017 | 0.025 | 0.893 | 165.118 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | coh-packet4 | 4 | 0.017 | 0.025 | 1.146 | 128.659 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | coh-packet8 | 8 | 0.017 | 0.025 | 0.706 | 208.793 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | inc-packet16 | 16 | 0.017 | 0.025 | 1.512 | 97.548 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | inc-packet4 | 4 | 0.017 | 0.025 | 1.611 | 91.513 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | inc-packet8 | 8 | 0.017 | 0.025 | 1.411 | 104.479 | 20416 |  | 2281401.897 |
| embree | high | native |  |  | scalar1 | 1 | 0.017 | 0.025 | 3.369 | 43.763 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | coh-packet16 | 16 | 0.012 | 0.014 | 0.893 | 165.181 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | coh-packet4 | 4 | 0.012 | 0.014 | 1.145 | 128.735 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | coh-packet8 | 8 | 0.012 | 0.014 | 0.707 | 208.592 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | inc-packet16 | 16 | 0.012 | 0.014 | 1.512 | 97.493 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | inc-packet4 | 4 | 0.012 | 0.014 | 1.612 | 91.498 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | inc-packet8 | 8 | 0.012 | 0.014 | 1.402 | 105.172 | 20416 |  | 2281401.897 |
| embree | medium | native |  |  | scalar1 | 1 | 0.012 | 0.014 | 3.375 | 43.697 | 20416 |  | 2281401.897 |

## Flattened triangle grid any-hit

Triangles per BLAS: 108; instances: 1; rays: 147456; query: any; ray order: camera.

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | sah | bvh16 | 16 | 16 | coh-packet16 | 16 | 0.008 | 0.008 | 0.609 | 241.930 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet4 | 4 | 0.008 | 0.008 | 1.412 | 104.434 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet8 | 8 | 0.008 | 0.008 | 0.846 | 174.207 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 0.008 | 0.008 | 0.584 | 252.523 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 0.008 | 0.008 | 1.487 | 99.143 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 0.008 | 0.008 | 0.845 | 174.506 | 20416 |  | 20416.000 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 0.008 | 0.008 | 1.189 | 123.976 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 0.017 | 0.027 | 0.495 | 298.167 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 0.017 | 0.027 | 1.094 | 134.730 | 20416 |  | 20416.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 0.017 | 0.027 | 0.652 | 226.186 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 0.017 | 0.027 | 0.492 | 299.499 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 0.017 | 0.027 | 1.015 | 145.228 | 20416 |  | 20416.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 0.017 | 0.027 | 0.604 | 244.254 | 20416 |  | 20416.000 |
| embree | high | native |  |  | scalar1 | 1 | 0.017 | 0.027 | 3.029 | 48.681 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 0.012 | 0.015 | 0.494 | 298.411 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 0.012 | 0.015 | 1.088 | 135.503 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 0.012 | 0.015 | 0.660 | 223.335 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 0.012 | 0.015 | 0.492 | 299.581 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 0.012 | 0.015 | 1.004 | 146.818 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 0.012 | 0.015 | 0.603 | 244.365 | 20416 |  | 20416.000 |
| embree | medium | native |  |  | scalar1 | 1 | 0.012 | 0.015 | 2.961 | 49.801 | 20416 |  | 20416.000 |
