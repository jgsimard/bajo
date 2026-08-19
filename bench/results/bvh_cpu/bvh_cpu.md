# CPU BVH benchmark results

- **Date:** 2026-08-19T10:29:29-04:00
- **CPU:** AMD Ryzen 7 9700X 8-Core Processor
- **System:** Linux-7.0.0-29-generic-x86_64-with-glibc2.43
- **Mojo:** `Mojo 1.1.0.dev2026081813 (8cd05901)`
- **Build thread modes:** `1` and `all`
- **All-thread affinity:** `0-15` (16 logical CPUs)
- **Traversal:** one calling thread; timings use the `threads=1` run
- **Raw data:** CSV/TXT retain both build-thread runs
- **Build timing:** one sample per configuration; descriptive, not a regression gate

## Best traversal result per implementation

| Benchmark | Implementation | Build | Layout | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | vs Bajo (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dragon | embree | medium | native | coh-packet8 | 8 | 53.645 | 7.250 | 8.865 | 66.532 | 2.560 |
| dragon | bajo | sah | bvh16 | unmasked-scalar | 1 | 57.312 | 20.392 | 9.092 | 64.871 | 0.000 |
| dragon | tinybvh | high | bvh8 | scalar1 | 1 | 385.714 | 83.341 | 14.431 | 40.872 | -36.995 |
| grid | bajo | lbvh | bvh8 | coh-packet16 | 16 | 1.962 | 1.585 | 6.039 | 43.405 | 0.000 |
| grid | tinybvh | sah | bvh8 | scalar1 | 1 | 14.026 | 7.926 | 10.096 | 25.965 | -40.180 |
| grid | embree | medium | native | inc-packet8 | 8 | 12.178 | 2.272 | 10.397 | 25.212 | -41.915 |

## Regular-grid microbenchmark

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | lbvh | bvh2 | 2 | 2 | scalar1 | 1 | 2.513 | 2.231 | 21.180 | 12.377 |  | 32767 | 6443188224.000 |
| bajo | lbvh | bvh4 | 4 | 4 | scalar1 | 1 | 2.614 | 1.701 | 13.361 | 19.620 |  | 5461 | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | coh-packet16 | 16 | 1.962 | 1.585 | 6.039 | 43.405 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | coh-packet4 | 4 | 1.962 | 1.585 | 9.787 | 26.785 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | coh-packet8 | 8 | 1.962 | 1.585 | 7.012 | 37.386 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | packet16 | 16 | 1.962 | 1.585 | 6.637 | 39.498 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | packet4 | 4 | 1.962 | 1.585 | 11.959 | 21.920 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | packet8 | 8 | 1.962 | 1.585 | 7.955 | 32.952 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | scalar1 | 1 | 1.962 | 1.585 | 12.595 | 20.813 |  | 1193 | 6443188224.000 |
| bajo | lbvh | bvh8 | 8 | 8 | unmasked-scalar | 1 | 1.962 | 1.585 | 14.546 | 18.021 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet16 | 16 | 1.758 | 1.323 | 7.669 | 34.182 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet4 | 4 | 1.758 | 1.323 | 11.417 | 22.960 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | coh-packet8 | 8 | 1.758 | 1.323 | 8.632 | 30.369 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet16 | 16 | 1.758 | 1.323 | 8.410 | 31.169 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet4 | 4 | 1.758 | 1.323 | 14.294 | 18.339 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | packet8 | 8 | 1.758 | 1.323 | 9.991 | 26.237 | 196608 |  | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 1.758 | 1.323 | 7.877 | 33.282 |  | 273 | 6443188224.000 |
| bajo | lbvh | bvh16 | 16 | 16 | unmasked-scalar | 1 | 1.758 | 1.323 | 7.026 | 37.311 | 196608 |  | 6443188224.000 |
| bajo | median | bvh2 | 2 | 2 | scalar1 | 1 | 12.398 | 4.907 | 21.623 | 12.124 |  | 32767 | 6443188224.000 |
| bajo | median | bvh4 | 4 | 4 | scalar1 | 1 | 11.945 | 3.904 | 13.358 | 19.625 |  | 5461 | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | coh-packet16 | 16 | 9.529 | 3.760 | 6.075 | 43.150 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | coh-packet4 | 4 | 9.529 | 3.760 | 9.836 | 26.651 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | coh-packet8 | 8 | 9.529 | 3.760 | 7.054 | 37.165 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | packet16 | 16 | 9.529 | 3.760 | 6.670 | 39.299 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | packet4 | 4 | 9.529 | 3.760 | 11.905 | 22.020 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | packet8 | 8 | 9.529 | 3.760 | 8.060 | 32.524 | 196608 |  | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | scalar1 | 1 | 9.529 | 3.760 | 12.554 | 20.882 |  | 1193 | 6443188224.000 |
| bajo | median | bvh8 | 8 | 8 | unmasked-scalar | 1 | 9.529 | 3.760 | 14.467 | 18.120 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet16 | 16 | 8.977 | 3.154 | 7.728 | 33.919 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet4 | 4 | 8.977 | 3.154 | 11.546 | 22.704 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | coh-packet8 | 8 | 8.977 | 3.154 | 8.614 | 30.434 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | packet16 | 16 | 8.977 | 3.154 | 8.333 | 31.460 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | packet4 | 4 | 8.977 | 3.154 | 14.361 | 18.254 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | packet8 | 8 | 8.977 | 3.154 | 9.979 | 26.270 | 196608 |  | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | scalar1 | 1 | 8.977 | 3.154 | 7.152 | 36.651 |  | 273 | 6443188224.000 |
| bajo | median | bvh16 | 16 | 16 | unmasked-scalar | 1 | 8.977 | 3.154 | 7.066 | 37.100 | 196608 |  | 6443188224.000 |
| bajo | sah | bvh2 | 2 | 2 | scalar1 | 1 | 16.807 | 5.784 | 20.189 | 12.985 |  | 32767 | 6443188224.000 |
| bajo | sah | bvh4 | 4 | 4 | scalar1 | 1 | 14.183 | 5.106 | 11.659 | 22.484 |  | 5461 | 6443188224.000 |
| bajo | sah | bvh8 | 8 | 8 | scalar1 | 1 | 12.186 | 4.895 | 11.067 | 23.686 |  | 1273 | 6443188224.000 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 10.866 | 4.399 | 7.847 | 33.405 |  | 273 | 6443188224.000 |
| embree | high | native |  |  | coh-packet16 | 16 | 14.249 | 2.396 | 13.163 | 19.916 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | coh-packet4 | 4 | 14.249 | 2.396 | 14.110 | 18.579 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | coh-packet8 | 8 | 14.249 | 2.396 | 12.926 | 20.281 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | inc-packet16 | 16 | 14.249 | 2.396 | 10.896 | 24.058 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | inc-packet4 | 4 | 14.249 | 2.396 | 11.426 | 22.943 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | inc-packet8 | 8 | 14.249 | 2.396 | 10.429 | 25.137 | 196608 |  | 6443188224.000 |
| embree | high | native |  |  | scalar1 | 1 | 14.249 | 2.396 | 16.872 | 15.537 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | coh-packet16 | 16 | 12.178 | 2.272 | 13.159 | 19.922 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | coh-packet4 | 4 | 12.178 | 2.272 | 14.072 | 18.628 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | coh-packet8 | 8 | 12.178 | 2.272 | 12.902 | 20.318 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | inc-packet16 | 16 | 12.178 | 2.272 | 10.848 | 24.165 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | inc-packet4 | 4 | 12.178 | 2.272 | 11.878 | 22.070 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | inc-packet8 | 8 | 12.178 | 2.272 | 10.397 | 25.212 | 196608 |  | 6443188224.000 |
| embree | medium | native |  |  | scalar1 | 1 | 12.178 | 2.272 | 16.879 | 15.531 | 196608 |  | 6443188224.000 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 29.578 | 4.762 | 18.102 | 14.481 | 196608 |  | 6443188224.000 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 31.384 | 6.658 | 12.116 | 21.636 | 196608 |  | 6443188223.985 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 35.744 | 8.251 | 10.110 | 25.930 | 196608 |  | 6443188223.985 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 11.081 | 2.940 | 20.828 | 12.586 | 196608 |  | 6443188224.000 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 15.343 | 6.269 | 12.269 | 21.367 | 196608 |  | 6443188223.985 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 14.026 | 7.926 | 10.096 | 25.965 | 196608 |  | 6443188223.985 |

## Dragon camera-ray benchmark

| Implementation | Build | Layout | Width | Leaf width | Traversal | Ray width | Build ms (1) | Build ms (all) | Trace ms | MRay/s | Hits | Nodes | Checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bajo | lbvh | bvh16 | 16 | 16 | scalar1 | 1 | 14.050 | 5.538 | 10.337 | 57.057 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 16 | unmasked-scalar | 1 | 14.050 | 5.538 | 10.269 | 57.436 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 32 | scalar1 | 1 | 19.436 | 5.095 | 10.062 | 58.621 | 71597 |  | 7943562615.175 |
| bajo | lbvh | bvh16 | 16 | 32 | unmasked-scalar | 1 | 19.436 | 5.095 | 9.953 | 59.261 | 71597 |  | 7943562615.175 |
| bajo | median | bvh16 | 16 | 16 | scalar1 | 1 | 47.067 | 14.632 | 10.250 | 57.544 | 71597 |  | 7943562615.175 |
| bajo | median | bvh16 | 16 | 16 | unmasked-scalar | 1 | 47.067 | 14.632 | 10.151 | 58.103 | 71597 |  | 7943562615.175 |
| bajo | median | bvh16 | 16 | 32 | scalar1 | 1 | 50.758 | 14.147 | 10.180 | 57.937 | 71597 |  | 7943562615.175 |
| bajo | median | bvh16 | 16 | 32 | unmasked-scalar | 1 | 50.758 | 14.147 | 10.151 | 58.105 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet16 | 16 | 59.402 | 22.891 | 10.262 | 57.477 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet4 | 4 | 59.402 | 22.891 | 14.837 | 39.753 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | coh-packet8 | 8 | 59.402 | 22.891 | 10.992 | 53.659 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet16 | 16 | 59.402 | 22.891 | 11.647 | 50.642 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet4 | 4 | 59.402 | 22.891 | 15.231 | 38.725 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | packet8 | 8 | 59.402 | 22.891 | 13.283 | 44.404 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | scalar1 | 1 | 59.402 | 22.891 | 9.518 | 61.969 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 16 | unmasked-scalar | 1 | 59.402 | 22.891 | 9.281 | 63.553 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 32 | scalar1 | 1 | 57.312 | 20.392 | 9.255 | 63.727 | 71597 |  | 7943562615.175 |
| bajo | sah | bvh16 | 16 | 32 | unmasked-scalar | 1 | 57.312 | 20.392 | 9.092 | 64.871 | 71597 |  | 7943562615.175 |
| embree | high | native |  |  | coh-packet16 | 16 | 268.796 | 29.135 | 10.796 | 54.633 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | coh-packet4 | 4 | 268.796 | 29.135 | 11.431 | 51.601 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | coh-packet8 | 8 | 268.796 | 29.135 | 10.634 | 55.464 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet16 | 16 | 268.796 | 29.135 | 13.920 | 42.373 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet4 | 4 | 268.796 | 29.135 | 12.843 | 45.924 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | inc-packet8 | 8 | 268.796 | 29.135 | 14.239 | 41.423 | 71598 |  | 7943741995.515 |
| embree | high | native |  |  | scalar1 | 1 | 268.796 | 29.135 | 18.744 | 31.467 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet16 | 16 | 53.645 | 7.250 | 9.573 | 61.614 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet4 | 4 | 53.645 | 7.250 | 11.343 | 51.998 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | coh-packet8 | 8 | 53.645 | 7.250 | 8.865 | 66.532 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet16 | 16 | 53.645 | 7.250 | 11.890 | 49.608 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet4 | 4 | 53.645 | 7.250 | 12.762 | 46.219 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | inc-packet8 | 8 | 53.645 | 7.250 | 11.782 | 50.062 | 71598 |  | 7943741995.515 |
| embree | medium | native |  |  | scalar1 | 1 | 53.645 | 7.250 | 18.562 | 31.775 | 71598 |  | 7943741995.515 |
| tinybvh | high | bvh2 | 2 |  | scalar1 | 1 | 354.984 | 47.481 | 24.742 | 23.839 | 71599 |  | 7943796499.200 |
| tinybvh | high | bvh4 | 4 |  | scalar1 | 1 | 376.753 | 70.481 | 16.974 | 34.748 | 71599 |  | 7943796499.445 |
| tinybvh | high | bvh8 | 8 |  | scalar1 | 1 | 385.714 | 83.341 | 14.431 | 40.872 | 71599 |  | 7943796499.439 |
| tinybvh | sah | bvh2 | 2 |  | scalar1 | 1 | 43.287 | 7.947 | 24.894 | 23.693 | 71599 |  | 7943796499.200 |
| tinybvh | sah | bvh4 | 4 |  | scalar1 | 1 | 68.847 | 31.918 | 16.885 | 34.933 | 71599 |  | 7943796499.445 |
| tinybvh | sah | bvh8 | 8 |  | scalar1 | 1 | 76.644 | 40.811 | 14.466 | 40.774 | 71599 |  | 7943796499.439 |
