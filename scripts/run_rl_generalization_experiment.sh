#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-.}"

TRAIN_SEEDS=(30001 30002 30003 30004 30005 30006 30007 30008)
VALIDATION_SEEDS=(31001 31002 31003)

"$PYTHON_BIN" -m scripts.train_rl \
  --seed 30001 \
  --train-seeds "${TRAIN_SEEDS[@]}" \
  --validation-seeds "${VALIDATION_SEEDS[@]}" \
  --checkpoint checkpoints/rl_learned_prior_bc.pt \
  --init-checkpoint checkpoints/rl_target_bc_ppo_120.pt \
  --hidden-dim 384 \
  --aircraft-embed-dim 192 \
  --target-embed-dim 64 \
  --adaptive-low-rank-prior \
  --low-rank-prior 0.0 \
  --target-rank-prior 0.0 \
  --bc-teacher elite_cp \
  --bc-episodes 8 \
  --bc-epochs 100 \
  --bc-minibatch-size 512 \
  --bc-learning-rate 0.0002 \
  --bc-low-loss-weight 12.0 \
  --bc-target-loss-weight 0.05 \
  --total-updates 0

"$PYTHON_BIN" -m scripts.train_rl \
  --seed 30001 \
  --train-seeds "${TRAIN_SEEDS[@]}" \
  --validation-seeds "${VALIDATION_SEEDS[@]}" \
  --checkpoint checkpoints/rl_learned_prior_ppo.pt \
  --init-checkpoint checkpoints/rl_learned_prior_bc.pt \
  --hidden-dim 384 \
  --aircraft-embed-dim 192 \
  --target-embed-dim 64 \
  --adaptive-low-rank-prior \
  --low-rank-prior 0.0 \
  --target-rank-prior 0.0 \
  --bc-episodes 0 \
  --bc-epochs 0 \
  --total-updates 4 \
  --rollout-steps 512 \
  --update-epochs 2 \
  --minibatch-size 128 \
  --learning-rate 0.000003 \
  --rank-gate-learning-rate 0.0001 \
  --entropy-coef 0.0 \
  --sortie-bonus 1.0 \
  --progress-shaping 0.1 \
  --save-every 1

"$PYTHON_BIN" -m scripts.export_rl_checkpoint \
  --checkpoint checkpoints/rl_learned_prior_ppo.pt

"$PYTHON_BIN" -m scripts.benchmark_non_rl \
  --solvers cp_sat \
  --runs 20 \
  --seed 40001 \
  --rl-checkpoint checkpoints/rl_learned_prior_ppo.pt \
  --rl-label rl_learned_prior_ppo \
  --output outputs/rl_generalization_holdout_seed40001_20runs.csv \
  --detail-output outputs/rl_generalization_holdout_seed40001_20runs_detail.csv
