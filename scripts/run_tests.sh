#!/usr/bin/env bash
#
# Run tests, see https://forum.modular.com/t/proposal-deprecating-mojo-test/2371
set -euo pipefail

# Accept directories or individual files; default to the repository root.
if [ "$#" -eq 0 ]; then
  set -- .
fi

test_count=0
failed=0

run_test() {
  case "$1" in
    *.mojo) mojo run -I . "$1" ;;
    *.py) python "$1" ;;
  esac
}

for test_path in "$@"; do
  echo "Running tests in: $test_path"
  # Check paths before find runs in a process substitution, where an error
  # would otherwise be lost if another path contained runnable tests.
  if [ ! -e "$test_path" ]; then
    echo "Test path does not exist: $test_path" >&2
    failed=1
    continue
  fi
  while IFS= read -r test_file; do
    test_count=$((test_count + 1))
    echo "### ------------------------------------------------------------- ###"
    echo "Running: $test_file"
    if ! run_test "$test_file"; then
      failed=1
    fi
  done < <(find "$test_path" -type f \( -name "test_*.mojo" -o -name "test_*.py" \) -not -path "*/.pixi/*" | sort)
done

if [ "$test_count" -eq 0 ]; then
  echo "No test files found in: $*" >&2
  exit 1
fi

if [ "$failed" -ne 0 ]; then
  exit 1
fi
