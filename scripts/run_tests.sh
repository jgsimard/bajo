#!/usr/bin/env bash
#
# Run tests, see https://forum.modular.com/t/proposal-deprecating-mojo-test/2371
set -euo pipefail

# Get test directory from first argument, default to repo root if not provided.
test_dir="${1:-.}"

echo "Running tests in: $test_dir"

tmpfile=$(mktemp)
trap 'rm -f "$tmpfile"' EXIT

test_count=0

while IFS= read -r test_file; do
  test_count=$((test_count + 1))
  echo "### ------------------------------------------------------------- ###"
  echo "Running: $test_file"
  if ! mojo run -I . "$test_file"; then
    echo "1" >"$tmpfile"
  fi
done < <(find "$test_dir" -name "test_*.mojo" -type f -not -path "*/.pixi/*" | sort)

if [ "$test_count" -eq 0 ]; then
  echo "No test files found in: $test_dir" >&2
  exit 1
fi

if [ -f "$tmpfile" ] && [ -s "$tmpfile" ]; then
  exit 1
fi
