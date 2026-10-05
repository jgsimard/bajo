# GPU CWBVH traversal — 2026-10-05

Native min/max in the packed CWBVH8 slab decoder improves secondary traversal by 2.1–5.1% and shuffled traversal by 2.6–5.5%. Coherent results are mixed; three-triangle Buddha/Bunny cases regress up to 2.5%.

**Measurement:** Five alternating process pairs per mesh; 524,288 rays, 128-thread blocks. Five warmup batches, then 21 batches of eight synchronized launches; median of process medians. Build, upload and readback excluded.

| Mesh | Rays | Max leaf | Trace | Before ms | After ms | Time reduction |
|---|---|---:|---|---:|---:|---:|
| dragon | coherent | 1 | closest | 0.1878 | 0.1846 | 1.7% |
| dragon | shuffled | 1 | closest | 0.4948 | 0.4769 | 3.6% |
| dragon | secondary | 1 | closest | 1.0638 | 1.0324 | 3.0% |
| dragon | coherent | 1 | any | 0.1317 | 0.1288 | 2.2% |
| dragon | shuffled | 1 | any | 0.2947 | 0.2858 | 3.0% |
| dragon | secondary | 1 | any | 0.7879 | 0.7713 | 2.1% |
| dragon | coherent | 3 | closest | 0.1998 | 0.1941 | 2.8% |
| dragon | shuffled | 3 | closest | 0.4855 | 0.4656 | 4.1% |
| dragon | secondary | 3 | closest | 1.1203 | 1.0651 | 4.9% |
| dragon | coherent | 3 | any | 0.1318 | 0.1293 | 1.8% |
| dragon | shuffled | 3 | any | 0.2906 | 0.2780 | 4.4% |
| dragon | secondary | 3 | any | 0.8085 | 0.7672 | 5.1% |
| buddha | coherent | 1 | closest | 0.0897 | 0.0882 | 1.7% |
| buddha | shuffled | 1 | closest | 0.1227 | 0.1187 | 3.2% |
| buddha | secondary | 1 | closest | 0.8636 | 0.8351 | 3.3% |
| buddha | coherent | 1 | any | 0.0664 | 0.0633 | 4.6% |
| buddha | shuffled | 1 | any | 0.0865 | 0.0836 | 3.4% |
| buddha | secondary | 1 | any | 0.6033 | 0.5905 | 2.1% |
| buddha | coherent | 3 | closest | 0.0893 | 0.0915 | -2.5% |
| buddha | shuffled | 3 | closest | 0.1217 | 0.1173 | 3.6% |
| buddha | secondary | 3 | closest | 0.8963 | 0.8516 | 5.0% |
| buddha | coherent | 3 | any | 0.0629 | 0.0637 | -1.3% |
| buddha | shuffled | 3 | any | 0.0845 | 0.0804 | 4.8% |
| buddha | secondary | 3 | any | 0.6083 | 0.5778 | 5.0% |
| bunny | coherent | 1 | closest | 0.0920 | 0.0906 | 1.5% |
| bunny | shuffled | 1 | closest | 0.1221 | 0.1188 | 2.6% |
| bunny | secondary | 1 | closest | 0.7552 | 0.7299 | 3.4% |
| bunny | coherent | 1 | any | 0.0667 | 0.0661 | 0.9% |
| bunny | shuffled | 1 | any | 0.0856 | 0.0832 | 2.8% |
| bunny | secondary | 1 | any | 0.5974 | 0.5837 | 2.3% |
| bunny | coherent | 3 | closest | 0.0948 | 0.0954 | -0.6% |
| bunny | shuffled | 3 | closest | 0.1224 | 0.1179 | 3.7% |
| bunny | secondary | 3 | closest | 0.7638 | 0.7264 | 4.9% |
| bunny | coherent | 3 | any | 0.0636 | 0.0638 | -0.3% |
| bunny | shuffled | 3 | any | 0.0863 | 0.0816 | 5.5% |
| bunny | secondary | 3 | any | 0.5960 | 0.5654 | 5.1% |

**Validation:** All 36 full-hit hashes match; 27 GPU tests pass. Synthetic decoder checks cover 1.62 million node/ray combinations. Inspected kernel replaces 40 compare/select pairs with native min/max; registers and stack unchanged. Hardware counters unavailable.
