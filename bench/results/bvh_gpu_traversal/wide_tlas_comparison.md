# GPU wide-BVH and TLAS traversal — 2026-10-05

Retained native min/max for ordinary-wide BLAS/TLAS slabs. BVH8 shuffled closest improves 20.9–21.5%, secondary closest 16.9–18.5%. Wide-BLAS TLAS secondary gains are 8.4–11.5%; compressed-BLAS secondary gains are 0.2–0.5%. Coherent BVH2/4 cases can regress up to 0.3%.

**Measurement:** RTX 5060 Ti; 524,288 rays, 128-thread blocks, 108 instances for TLAS. Three alternating process pairs; five warmups, 21 batches of eight synchronized launches, median of process medians. Build, upload and readback excluded. Baseline includes compressed min/max.

## wide2/leaf2

| Mesh | Rays | Trace | Before ms | After ms | Time reduction |
|---|---|---|---:|---:|---:|
| dragon | coherent | closest | 0.2293 | 0.2261 | 1.4% |
| dragon | shuffled | closest | 0.4155 | 0.3826 | 7.9% |
| dragon | secondary | closest | 1.5107 | 1.4614 | 3.3% |
| dragon | coherent | any | 0.1932 | 0.1911 | 1.1% |
| dragon | shuffled | any | 0.2970 | 0.2758 | 7.1% |
| dragon | secondary | any | 1.2525 | 1.2263 | 2.1% |
| buddha | coherent | closest | 0.1188 | 0.1191 | -0.3% |
| buddha | shuffled | closest | 0.1061 | 0.1014 | 4.5% |
| buddha | secondary | closest | 1.2112 | 1.1752 | 3.0% |
| buddha | coherent | any | 0.1010 | 0.0992 | 1.9% |
| buddha | shuffled | any | 0.0876 | 0.0866 | 1.2% |
| buddha | secondary | any | 0.9949 | 0.9758 | 1.9% |
| bunny | coherent | closest | 0.0956 | 0.0954 | 0.2% |
| bunny | shuffled | closest | 0.0962 | 0.0920 | 4.4% |
| bunny | secondary | closest | 1.0461 | 1.0099 | 3.5% |
| bunny | coherent | any | 0.0929 | 0.0914 | 1.6% |
| bunny | shuffled | any | 0.0832 | 0.0826 | 0.7% |
| bunny | secondary | any | 0.8853 | 0.8704 | 1.7% |

## wide4/leaf2

| Mesh | Rays | Trace | Before ms | After ms | Time reduction |
|---|---|---|---:|---:|---:|
| dragon | coherent | closest | 0.2857 | 0.2827 | 1.0% |
| dragon | shuffled | closest | 0.4960 | 0.4721 | 4.8% |
| dragon | secondary | closest | 1.7626 | 1.7228 | 2.3% |
| dragon | coherent | any | 0.1981 | 0.1949 | 1.6% |
| dragon | shuffled | any | 0.3052 | 0.2813 | 7.8% |
| dragon | secondary | any | 1.1897 | 1.1588 | 2.6% |
| buddha | coherent | closest | 0.1252 | 0.1251 | 0.1% |
| buddha | shuffled | closest | 0.1172 | 0.1139 | 2.8% |
| buddha | secondary | closest | 1.3980 | 1.3677 | 2.2% |
| buddha | coherent | any | 0.0987 | 0.0985 | 0.3% |
| buddha | shuffled | any | 0.0886 | 0.0872 | 1.6% |
| buddha | secondary | any | 0.9428 | 0.9189 | 2.5% |
| bunny | coherent | closest | 0.1178 | 0.1157 | 1.8% |
| bunny | shuffled | closest | 0.1134 | 0.1098 | 3.2% |
| bunny | secondary | closest | 1.2769 | 1.2455 | 2.5% |
| bunny | coherent | any | 0.0926 | 0.0929 | -0.3% |
| bunny | shuffled | any | 0.0895 | 0.0851 | 4.9% |
| bunny | secondary | any | 0.8944 | 0.8648 | 3.3% |

## wide8/leaf4

| Mesh | Rays | Trace | Before ms | After ms | Time reduction |
|---|---|---|---:|---:|---:|
| dragon | coherent | closest | 0.4185 | 0.3621 | 13.5% |
| dragon | shuffled | closest | 0.8223 | 0.6505 | 20.9% |
| dragon | secondary | closest | 2.9021 | 2.4069 | 17.1% |
| dragon | coherent | any | 0.2415 | 0.2357 | 2.4% |
| dragon | shuffled | any | 0.4010 | 0.3702 | 7.7% |
| dragon | secondary | any | 1.6432 | 1.5855 | 3.5% |
| buddha | coherent | closest | 0.1291 | 0.1251 | 3.1% |
| buddha | shuffled | closest | 0.1909 | 0.1498 | 21.5% |
| buddha | secondary | closest | 2.3514 | 1.9159 | 18.5% |
| buddha | coherent | any | 0.0897 | 0.0877 | 2.2% |
| buddha | shuffled | any | 0.1077 | 0.1017 | 5.6% |
| buddha | secondary | any | 1.3549 | 1.3125 | 3.1% |
| bunny | coherent | closest | 0.1213 | 0.1187 | 2.2% |
| bunny | shuffled | closest | 0.1884 | 0.1486 | 21.1% |
| bunny | secondary | closest | 1.9470 | 1.6176 | 16.9% |
| bunny | coherent | any | 0.0851 | 0.0843 | 0.9% |
| bunny | shuffled | any | 0.1103 | 0.1039 | 5.8% |
| bunny | secondary | any | 1.2353 | 1.1943 | 3.3% |

## TLAS2 + CWBVH8/leaf1

| Mesh | Rays | Trace | Before ms | After ms | Time reduction |
|---|---|---|---:|---:|---:|
| dragon | coherent | closest | 0.2174 | 0.2145 | 1.4% |
| dragon | shuffled | closest | 0.6593 | 0.6476 | 1.8% |
| dragon | secondary | closest | 7.9122 | 7.8993 | 0.2% |
| dragon | coherent | any | 0.1699 | 0.1677 | 1.3% |
| dragon | shuffled | any | 0.4289 | 0.4198 | 2.1% |
| dragon | secondary | any | 5.7176 | 5.6867 | 0.5% |
| buddha | coherent | closest | 0.2292 | 0.2266 | 1.1% |
| buddha | shuffled | closest | 0.7527 | 0.7418 | 1.4% |
| buddha | secondary | closest | 6.7436 | 6.7253 | 0.3% |
| buddha | coherent | any | 0.1575 | 0.1552 | 1.5% |
| buddha | shuffled | any | 0.4663 | 0.4577 | 1.8% |
| buddha | secondary | any | 4.7731 | 4.7572 | 0.3% |
| bunny | coherent | closest | 0.2490 | 0.2440 | 2.0% |
| bunny | shuffled | closest | 1.0699 | 1.0544 | 1.4% |
| bunny | secondary | closest | 7.2457 | 7.2269 | 0.3% |
| bunny | coherent | any | 0.1773 | 0.1731 | 2.4% |
| bunny | shuffled | any | 0.6938 | 0.6821 | 1.7% |
| bunny | secondary | any | 5.3309 | 5.3108 | 0.4% |

## TLAS2 + wide4/leaf2

| Mesh | Rays | Trace | Before ms | After ms | Time reduction |
|---|---|---|---:|---:|---:|
| dragon | coherent | closest | 0.4301 | 0.4090 | 4.9% |
| dragon | shuffled | closest | 0.6457 | 0.5957 | 7.8% |
| dragon | secondary | closest | 11.1892 | 10.2053 | 8.8% |
| dragon | coherent | any | 0.1663 | 0.1611 | 3.2% |
| dragon | shuffled | any | 0.3935 | 0.3488 | 11.4% |
| dragon | secondary | any | 5.3913 | 4.7959 | 11.0% |
| buddha | coherent | closest | 0.3939 | 0.3704 | 6.0% |
| buddha | shuffled | closest | 0.6741 | 0.6210 | 7.9% |
| buddha | secondary | closest | 8.9797 | 8.2298 | 8.4% |
| buddha | coherent | any | 0.1555 | 0.1488 | 4.3% |
| buddha | shuffled | any | 0.4325 | 0.3808 | 12.0% |
| buddha | secondary | any | 4.4579 | 3.9835 | 10.6% |
| bunny | coherent | closest | 0.5415 | 0.5042 | 6.9% |
| bunny | shuffled | closest | 1.0036 | 0.9187 | 8.5% |
| bunny | secondary | closest | 9.5166 | 8.6693 | 8.9% |
| bunny | coherent | any | 0.1685 | 0.1589 | 5.7% |
| bunny | shuffled | any | 0.6049 | 0.5284 | 12.6% |
| bunny | secondary | any | 5.0748 | 4.4929 | 11.5% |

**Validation:** All 90 full-hit cases match; 37 GPU tests pass, including 900,000 slab-reference cases. Inspected wide8 registers fall 98 → 95, theoretical resident threads/SM rise 512 → 640; no spills. Hardware counters unavailable.
