#!/usr/bin/env bash
# Compile one Mojo runner per subsystem, then run Python integration tests.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -eq 0 ]; then
  set -- "$repo_root/test"
fi

declare -A selected_tests=()
failed=0
for test_path in "$@"; do
  if [ ! -e "$test_path" ]; then
    echo "Test path does not exist: $test_path" >&2
    failed=1
    continue
  fi
  test_path="$(realpath -- "$test_path")"
  while IFS= read -r -d '' test_file; do
    selected_tests["$test_file"]=1
  done < <(find "$test_path" -type f \( -name 'test_*.mojo' -o -name 'test_*.py' \) -not -path '*/.pixi/*' -print0)
done

if [ "${#selected_tests[@]}" -eq 0 ]; then
  echo "No test files found in: $*" >&2
  exit 1
fi
mapfile -d '' -t test_files < <(printf '%s\0' "${!selected_tests[@]}" | sort -z)

cd "$repo_root"
declare -A mojo_groups=()
for test_file in "${test_files[@]}"; do
  if [[ "$test_file" == *.mojo ]]; then
    relative="${test_file#"$repo_root/test/"}"
    group="${relative%%/*}"
    mojo_groups["$group"]=1
  fi
done

if [ "${#mojo_groups[@]}" -gt 0 ]; then
  mapfile -t groups < <(printf '%s\n' "${!mojo_groups[@]}" | sort)
  for group in "${groups[@]}"; do
    runner="$repo_root/test/$group/run_tests.mojo"
    if [ ! -f "$runner" ]; then
      echo "Missing Mojo test runner: $runner" >&2
      failed=1
      continue
    fi
    runner_source="$(tr -d '[:space:]' < "$runner")"
    flags=(-D BAJO_TEST_FILTERED=1)
    missing=false
    for test_file in "${test_files[@]}"; do
      if [[ "$test_file" != "$repo_root/test/$group/"*.mojo ]]; then
        continue
      fi
      relative="${test_file#"$repo_root/"}"
      module="${relative%.mojo}"
      module="${module//\//.}"
      alias="${test_file#"$repo_root/test/$group/"}"
      alias="${alias%.mojo}"
      alias="${alias//\//_}"
      if [[ "$runner_source" != *"import${module}as${alias}"* ||
            "$runner_source" != *"module_selected[\"$alias\"]()"* ||
            "$runner_source" != *"tests.run(\"$relative\",$alias.run_tests[True]"* ]]; then
        echo "Test missing from test/$group/run_tests.mojo: $relative" >&2
        missing=true
      fi
      flags+=(-D "BAJO_TEST_$alias=1")
    done
    if "$missing"; then
      failed=1
      continue
    fi
    echo "Running Mojo tests in: test/$group"
    if ! mojo run -I "$repo_root" "${flags[@]}" "$runner"; then
      failed=1
    fi
  done
fi

for test_file in "${test_files[@]}"; do
  if [[ "$test_file" == *.py ]]; then
    echo "Running: ${test_file#"$repo_root/"}"
    if ! python "$test_file"; then
      failed=1
    fi
  fi
done

exit "$failed"
