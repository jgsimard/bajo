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

Full configuration matrix and checksums: [CSV](latest.csv), [raw output](latest.txt).
Core traversal uses best-of-eight; instance diagnostics use median-of-eight batched repetitions.
