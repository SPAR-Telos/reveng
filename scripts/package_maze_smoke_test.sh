#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

archive="${1:-smoke_test/maze_smoke_test_portable.tar.gz}"
mkdir -p "$(dirname "$archive")"

tar -czf "$archive" \
  README.md \
  pyproject.toml \
  configs/maze_api_requirements.txt \
  configs/maze_local_preflight_requirements.txt \
  configs/maze_smoke_test.json \
  scripts/package_maze_smoke_test.sh \
  scripts/run_maze_local_preflight.py \
  scripts/run_maze_smoke_test.py \
  src/reveng \
  tests/test_maze_smoke_test.py \
  smoke_test/PACKAGE_STATUS.md \
  smoke_test/REMOTE_RUNBOOK.md \
  smoke_test/api_calls.csv \
  smoke_test/api_results.csv \
  smoke_test/api_schedule.jsonl \
  smoke_test/api_summary.md \
  smoke_test/condition_summary.csv \
  smoke_test/config.lock.json \
  smoke_test/grids.jsonl \
  smoke_test/local_preflight.md

sha256sum "$archive" > "$archive.sha256"
printf 'wrote %s (%s bytes)\n' "$archive" "$(stat -c %s "$archive")"
cat "$archive.sha256"
