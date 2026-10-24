#!/usr/bin/env bash
# Compile before opening the window: foreign render threads stall under mojo run.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
viewer_target=cpu
viewer_flags=()
if [[ "${1:-}" == --gpu ]]; then
  shift
  viewer_target="${BAJO_GPU_ARCH:-sm_120}"
  viewer_flags=(--target-accelerator "$viewer_target")
fi

viewer_compiler="$(command -v mojo)"
viewer_version="$(mojo --version)"
viewer_fingerprint="$(
  {
    printf '%s\n' "$viewer_compiler" "$viewer_version" "$viewer_target"
    stat -Lc '%s:%Y' "$viewer_compiler"
    sha256sum scripts/run_viewer.sh
    # Include untracked files too, so editing a new module invalidates the build.
    find bajo examples -type f -name '*.mojo' -print0 |
      sort -z | xargs -0 sha256sum
  } | sha256sum | cut -d ' ' -f 1
)"
viewer_cache="$repo_root/__mojocache__/viewer"
viewer_executable="$viewer_cache/$viewer_fingerprint"
mkdir -p "$viewer_cache"

if [[ ! -x "$viewer_executable" ]]; then
  viewer_tmp="$(mktemp -d "$viewer_cache/build.XXXXXX")"
  trap 'rm -rf -- "$viewer_tmp"' EXIT
  printf 'Building native Bajo viewer (%s); subsequent launches reuse this build.\n' "$viewer_target"
  mojo build -I "$repo_root" "${viewer_flags[@]}" \
    "$repo_root/examples/viewer_raylib.mojo" -o "$viewer_tmp/viewer"
  # Publish only a completed executable; concurrent launches can build safely.
  mv -- "$viewer_tmp/viewer" "$viewer_executable"
  rm -rf -- "$viewer_tmp"
  trap - EXIT
fi

exec "$viewer_executable" "$@"
