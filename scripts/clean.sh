#!/usr/bin/env bash
set -euo pipefail

# Only remove reproducible compiler and interpreter caches. Downloaded assets,
# external sources, environments, and benchmark results are intentionally kept.
rm -rf -- __mojocache__

for root in tools bench test; do
  if [ -d "$root" ]; then
    find "$root" -type d -name __pycache__ -prune -exec rm -rf -- {} +
  fi
done

rm -f -- bench/sort/bench_cub_radix bench/bvh/bench_tinybvh

echo "Removed Bajo build and interpreter caches."
