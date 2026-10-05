# BVH performance vs Warp

Reference: Warp 1.17.0; Ryzen 7 9700X / RTX 5060 Ti. Ratios below are
`Warp time / Bajo time`: above 1 favors Bajo. Correctness gates measurements.
Unmeasured capabilities stay unmeasured; rendering-BVH results do not describe
the new general spatial representation.

| Capability / implementation | CPU ratio | GPU ratio | Workload / evidence |
|---|---:|---:|---|
| Existing rendering BVH fresh build | 4.75–5.23 | 0.69–1.07 | Bunny/dragon, fastest tested configuration per phase |
| Existing rendering BVH closest-hit | 5.77–6.67 | 1.33–1.50 | 147,456 coherent camera rays per mesh |
| Existing rendering BVH any-hit | 6.38–7.52 | 1.87–1.95 | Same rays; face-hit validation |
| General bounds BVH construction | 0.54 | 0.92 | 256 boxes / 4 groups; GPU includes host build and all uploads |
| Bounds ray candidate iterator | 1.45 | 0.39 | 512 global queries; complete sets + exhaustive reference |
| Bounds AABB overlap iterator | 1.95 | 1.62 | 512 global queries; complete sets + exhaustive reference |
| Bounds sphere overlap iterator | 1.70 | 0.51 | 512 global queries; complete sets + exhaustive reference |
| Bounds capsule sweep iterator | 0.70 | 0.12 | 512 global queries; complete sets + exhaustive reference |
| Indexed spatial mesh construction | Unmeasured | Unmeasured | Match winding metadata, ownership and GPU transfer scope |
| Exact mesh sphere query | 1.93 | 0.71 | 256 faces / 256 queries; matched Warp + exhaustive/analytic reference |
| Closest/furthest mesh point | 0.01–0.02 | 0.01–0.04 | 256 faces / 256 queries; matched Warp + exhaustive/analytic reference |
| Mesh sign classification | 0.02–0.02 | 0.01–0.03 | 256 faces / 256 queries; matched Warp + exhaustive/analytic reference |
| Ray intersection count | 0.47 | 0.20 | 256 faces / 256 queries; matched Warp + exhaustive/analytic reference |
| Group-restricted bounds queries | 1.29–4.45 | 0.28–1.84 | Same queries, 64 items per group |
| Group-restricted mesh queries | 0.44–3.74 | 0.20–1.51 | 256 faces / 256 queries; matched Warp + exhaustive/analytic reference |
| Bounds refit | 0.16 | 0.04 | Warm unchanged inputs; GPU inputs resident; Bajo validates |
| Mesh deformation refit | 0.22 | 0.08 | Warm unchanged positions; GPU resident inputs, four groups; Warp winding support disabled |
| Retained-buffer bounds rebuild | 0.52 | 0.11 | Warm unchanged inputs; CPU median/SAH, GPU X-order/LBVH |
| Standalone mesh evaluation / velocity updates | Unmeasured | Unmeasured | Evaluation is correctness-checked and included in point-query timings |
| Cooperative mesh AABB queries | Unmeasured | Unmeasured | CPU/GPU operation covered; separate matched timing pending |
| Update graph replay | Unmeasured | Unmeasured | Capture/replay correctness tested; graph timing pending |
| Cooperative queries | 0.37–0.79 | 0.24–1.03 | 256 boxes / 512 global queries; 32/64 lanes, complete sets + exhaustive reference |

Indexed mesh measurements: [summary](../../bench/results/warp/mesh.md),
[raw samples](../../bench/results/warp/mesh.json). Cooperative bounds measurements:
[summary](../../bench/results/warp/block.md), [raw samples](../../bench/results/warp/block.json).
These are small correctness fixtures. Closest/furthest/sign mesh queries use
unoptimized exhaustive baselines; their ratios expose a substantial performance
gap. Winding uses ungrouped owners in both libraries; other mesh queries/refit
use four groups. Warp grouped CPU winding failed the analytic sign reference
and is excluded from timings.

General bounds measurements: [summary](../../bench/results/warp/spatial.md),
[raw samples](../../bench/results/warp/spatial.json). Query timing includes full
candidate-matrix writes. Updates use unchanged warmed inputs; these ratios do
not establish performance under large deformation or large trees.

Existing rendering measurements: [summary](../../bench/results/warp/comparison.md),
[raw samples](../../bench/results/warp/comparison.json). CPU traversal uses scalar
rays with SIMD within Bajo nodes; queries include API launch/synchronization.
Build and query winners can differ. Results are for two static meshes and one
machine, not an overall library speed ranking.

For each new measurement retain input hashes, version/source hashes, CPU/GPU,
query count/distribution, groups, median and range, timed scope, warmup and raw
samples. Record failures rather than presenting timings for incorrect results.
Keep CPU/GPU and hot device work/end-to-end transfers separate. Initial
implementations prioritize coverage and correctness; optimization follows data.

Coverage: [capability matrix](CAPABILITIES.md). History: [progress log](PROGRESS.md).
