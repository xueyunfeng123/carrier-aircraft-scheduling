#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-.}"

"$PYTHON_BIN" -m scripts.train_rl \
  --seed 30001 \
  --train-seeds 30001 30002 30003 30004 \
  --validation-seeds 31001 31002 \
  --train-wave-intervals 45 55 65 \
  --validation-wave-intervals 50 60 70 \
  --waves-per-scenario 12 \
  --checkpoint checkpoints/rl_multiload_bc.pt \
  --init-checkpoint checkpoints/rl_learned_prior_ppo.pt \
  --hidden-dim 384 \
  --aircraft-embed-dim 192 \
  --target-embed-dim 64 \
  --adaptive-low-rank-prior \
  --low-rank-prior 0.0 \
  --target-rank-prior 0.0 \
  --bc-teacher elite_cp \
  --bc-episodes 4 \
  --bc-epochs 50 \
  --bc-minibatch-size 512 \
  --bc-learning-rate 0.00005 \
  --bc-low-loss-weight 4.0 \
  --bc-target-loss-weight 0.05 \
  --total-updates 0

"$PYTHON_BIN" -m scripts.export_rl_checkpoint \
  --checkpoint checkpoints/rl_multiload_bc.pt

"$PYTHON_BIN" -m scripts.benchmark_non_rl \
  --intervals 47.5 52.5 57.5 62.5 67.5 \
  --waves 12 \
  --runs 5 \
  --seed 60001 \
  --solvers heuristic cp_sat \
  --rl-checkpoint checkpoints/rl_multiload_bc.pt \
  --rl-label rl_multiload \
  --output outputs/rl_multiload_holdout.csv \
  --detail-output outputs/rl_multiload_holdout_detail.csv

"$PYTHON_BIN" -m scripts.analyze_paired_benchmark \
  outputs/rl_multiload_holdout_detail.csv \
  --solvers heuristic cp_sat rl_multiload \
  --candidate rl_multiload \
  --output outputs/rl_multiload_holdout_summary.csv
