# Instanced shadows and traversal scheduling — 2026-10-05

Retained near-plane culling for local compressed any-hit queries: secondary shadows improve 1.2–2.2%, with a 0.7% Dragon coherent-shadow regression. Closest-hit and standalone policies remain unchanged. The block-size sweep favors 32 threads for incoherent rays but regresses coherent rays; production camera launches retain 128.

**Measurement:** 524,288 rays through 108 rotated/scaled instances. Five warmups, 21 batches of eight synchronized launches; median of process medians. Five alternating pairs for culling, three processes for block-size sweep. Build, upload and readback excluded.

## Near-plane culling (ms)

| Mesh | Rays | Trace | Before ms | After ms | Time reduction |
|---|---|---|---:|---:|---:|
| dragon | coherent | closest | 0.2140 | 0.2140 | -0.0% |
| dragon | shuffled | closest | 0.6487 | 0.6475 | 0.2% |
| dragon | secondary | closest | 7.8967 | 7.8983 | -0.0% |
| dragon | coherent | any | 0.1670 | 0.1682 | -0.7% |
| dragon | shuffled | any | 0.4196 | 0.4177 | 0.4% |
| dragon | secondary | any | 5.6884 | 5.5640 | 2.2% |
| buddha | coherent | closest | 0.2265 | 0.2261 | 0.2% |
| buddha | shuffled | closest | 0.7410 | 0.7412 | -0.0% |
| buddha | secondary | closest | 6.7282 | 6.7260 | 0.0% |
| buddha | coherent | any | 0.1549 | 0.1549 | 0.0% |
| buddha | shuffled | any | 0.4570 | 0.4549 | 0.5% |
| buddha | secondary | any | 4.7575 | 4.6839 | 1.5% |
| bunny | coherent | closest | 0.2440 | 0.2447 | -0.3% |
| bunny | shuffled | closest | 1.0550 | 1.0546 | 0.0% |
| bunny | secondary | closest | 7.2281 | 7.2252 | 0.0% |
| bunny | coherent | any | 0.1729 | 0.1730 | -0.1% |
| bunny | shuffled | any | 0.6817 | 0.6781 | 0.5% |
| bunny | secondary | any | 5.3121 | 5.2502 | 1.2% |

## Block-size sweep (ms)

| Mesh | Rays | Trace | 128 threads ms | 32 threads ms | Time reduction |
|---|---|---|---:|---:|---:|
| dragon | coherent | closest | 0.2141 | 0.2190 | -2.3% |
| dragon | shuffled | closest | 0.6506 | 0.5941 | 8.7% |
| dragon | secondary | closest | 7.8980 | 7.6268 | 3.4% |
| dragon | coherent | any | 0.1671 | 0.1696 | -1.5% |
| dragon | shuffled | any | 0.4201 | 0.3933 | 6.4% |
| dragon | secondary | any | 5.6916 | 5.5739 | 2.1% |
| buddha | coherent | closest | 0.2273 | 0.2309 | -1.6% |
| buddha | shuffled | closest | 0.7417 | 0.6794 | 8.4% |
| buddha | secondary | closest | 6.7228 | 6.4882 | 3.5% |
| buddha | coherent | any | 0.1556 | 0.1582 | -1.7% |
| buddha | shuffled | any | 0.4574 | 0.4310 | 5.8% |
| buddha | secondary | any | 4.7554 | 4.6576 | 2.1% |
| bunny | coherent | closest | 0.2450 | 0.2504 | -2.2% |
| bunny | shuffled | closest | 1.0546 | 0.9822 | 6.9% |
| bunny | secondary | closest | 7.2257 | 6.9969 | 3.2% |
| bunny | coherent | any | 0.1735 | 0.1769 | -2.0% |
| bunny | shuffled | any | 0.6815 | 0.6503 | 4.6% |
| bunny | secondary | any | 5.3087 | 5.2071 | 1.9% |

**Validation:** Full-hit hashes match across culling versions and launch sizes. Standalone confirmation preserves all 36 hit hashes and the inspected kernel instruction stream. Broader near-plane clipping policies were rejected.
