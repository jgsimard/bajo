#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Only remove reproducible compiler and interpreter caches. Downloaded assets,
# external sources, environments, and benchmark results are intentionally kept.
rm -rf -- __mojocache__

for root in bench scripts test; do
  if [ -d "$root" ]; then
    find "$root" -type d -name __pycache__ -prune -exec rm -rf -- {} +
  fi
done

echo "Removed Bajo build and interpreter caches."
