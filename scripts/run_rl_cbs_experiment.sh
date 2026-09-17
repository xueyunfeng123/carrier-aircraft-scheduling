#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-.}"

COMMON_ARGS=(
  --checkpoint checkpoints/rl_multiload_bc.pt
  --wave-interval 47.5
  --simulation-duration 570
  --seed 62001
  --runs 10
)

"$PYTHON_BIN" -m scripts.evaluate_rl \
  "${COMMON_ARGS[@]}" \
  --output outputs/rl_cbs_off_47_5_seed62001_10runs.csv

"$PYTHON_BIN" -m scripts.evaluate_rl \
  "${COMMON_ARGS[@]}" \
  --cbs-replan \
  --output outputs/rl_cbs_on_47_5_seed62001_10runs.csv
