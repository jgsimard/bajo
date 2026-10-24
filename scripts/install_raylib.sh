#!/usr/bin/env bash
# Install the official native raylib release inside the workspace.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raylib_version=6.2
raylib_sha256=586bbc32f3596d439382b1c5dbc6c1c573eb7e1d00fe92be7d4deb116883720d
raylib_dir="$repo_root/.pixi/raylib"
if [[ -f "$raylib_dir/lib/libraylib.so" && -f "$raylib_dir/.version" ]] &&
  [[ "$(cat "$raylib_dir/.version")" == "$raylib_version" ]]; then
  printf 'Native raylib %s is already installed in %s\n' "$raylib_version" "$raylib_dir"
  exit 0
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  printf 'This installer supports linux-64; pass your native raylib library path to Window on other platforms.\n' >&2
  exit 1
fi

mkdir -p "$repo_root/.pixi"
raylib_tmp="$(mktemp -d "$repo_root/.pixi/raylib-install.XXXXXX")"
cleanup() {
  if [[ -e "$raylib_tmp/previous" && ! -e "$raylib_dir" ]]; then
    mv "$raylib_tmp/previous" "$raylib_dir"
  fi
  rm -rf "$raylib_tmp"
}
trap cleanup EXIT
curl --fail --location --retry 3 \
  "https://github.com/raysan5/raylib/releases/download/$raylib_version/raylib-${raylib_version}_linux_amd64.tar.gz" \
  --output "$raylib_tmp/raylib.tar.gz"
printf '%s  %s\n' \
  "$raylib_sha256" \
  "$raylib_tmp/raylib.tar.gz" | sha256sum --check --status
tar -xzf "$raylib_tmp/raylib.tar.gz" -C "$raylib_tmp"
raylib_release="$raylib_tmp/raylib-${raylib_version}_linux_amd64"
test -f "$raylib_release/lib/libraylib.so"
printf '%s\n' "$raylib_version" > "$raylib_release/.version"
if [[ -e "$raylib_dir" ]]; then
  mv "$raylib_dir" "$raylib_tmp/previous"
fi
mv "$raylib_release" "$raylib_dir"
printf 'Installed native raylib %s in %s\n' "$raylib_version" "$raylib_dir"
