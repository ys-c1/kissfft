#!/usr/bin/env bash
# End-to-end feature-matrix driver.
#
# 1. generates the shared deterministic signal batch
# 2. builds all 8 CMake configurations + per-config harness, runs the batch
# 3. measures cold (from-scratch) build time per config
# 4. analyzes everything against the analytic DFT and writes the table
#
# Usage:  ./run_matrix.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="python3 -I"

echo "== [1/4] generating signals =="
$PY "$HERE/scripts/gen_signals.py" "$HERE/signals"

echo "== [2/4] building 8 configs and running the batch =="
$PY "$HERE/scripts/build_matrix.py"

echo "== [3/4] cold build timing =="
$PY "$HERE/scripts/time_clean_builds.py"

echo "== [4/4] rebuilding harnesses (clean timing wiped build dirs) + analysis =="
$PY "$HERE/scripts/build_matrix.py" >/dev/null
$PY "$HERE/scripts/analyze.py"

echo
echo "Done. See:"
echo "  $HERE/results/table.md      (configuration x metric table)"
echo "  $HERE/results/metrics.json  (machine-readable metrics)"
echo "  $HERE/results/clean_build_times.json"
