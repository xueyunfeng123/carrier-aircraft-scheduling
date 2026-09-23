#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
export PYTHONPATH="${PYTHONPATH:-.}"

"$PYTHON_BIN" -m scripts.train_action_value \
  --actor-checkpoint checkpoints/rl_multiload_bc.pt \
  --output-checkpoint checkpoints/rl_multiload_action_value.pt \
  --training-seeds 42001 42002 42003 42004 42005 42006 42007 42008 42009 42010 \
  --selection-seeds 41001 41002 41003 41004 41005 \
  --wave-intervals 47.5 52.5 57.5 62.5 67.5 \
  --waves 12 \
  --stride 20 \
  --max-branch-states 8 \
  --top-k 4 \
  --workers 4 \
  --epochs 100 \
  --minibatch-size 512 \
  --learning-rate 0.001 \
  --ranking-loss-coef 1.0 \
  --device "$DEVICE" \
  --metrics-output outputs/action_value_selection_seed41001_5.csv \
  "$@"

"$PYTHON_BIN" -m scripts.evaluate_rl \
  --checkpoint checkpoints/rl_multiload_bc.pt \
  --action-value-checkpoint checkpoints/rl_multiload_action_value.pt \
  --action-value-top-k 3 \
  --device "$DEVICE" \
  --seed 43001 \
  --runs 5 \
  --output outputs/action_value_online_seed43001_5.csv
