#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-.}"

"$PYTHON_BIN" -m scripts.train_rl \
  --seed 10007 \
  --eval-seed 10007 \
  --checkpoint checkpoints/rl_target_elite_overfit.pt \
  --hidden-dim 384 \
  --aircraft-embed-dim 192 \
  --target-embed-dim 64 \
  --bc-teacher elite_cp \
  --bc-episodes 1 \
  --bc-epochs 5000 \
  --bc-minibatch-size 512 \
  --bc-learning-rate 0.001 \
  --bc-low-loss-weight 12.0 \
  --bc-target-loss-weight 0.05 \
  --total-updates 0 \
  --eval-runs 1

"$PYTHON_BIN" -m scripts.train_rl \
  --seed 10007 \
  --eval-seed 10007 \
  --checkpoint checkpoints/rl_target_elite_dagger.pt \
  --init-checkpoint checkpoints/rl_target_elite_overfit.pt \
  --hidden-dim 384 \
  --aircraft-embed-dim 192 \
  --target-embed-dim 64 \
  --bc-teacher elite_cp \
  --bc-episodes 1 \
  --bc-epochs 100 \
  --bc-minibatch-size 512 \
  --bc-learning-rate 0.0002 \
  --bc-low-loss-weight 12.0 \
  --bc-target-loss-weight 0.05 \
  --dagger-iterations 3 \
  --dagger-episodes 1 \
  --dagger-epochs 100 \
  --total-updates 0 \
  --eval-runs 1

"$PYTHON_BIN" -m scripts.train_rl \
  --seed 10007 \
  --eval-seed 10007 \
  --checkpoint checkpoints/rl_target_bc_ppo_120.pt \
  --init-checkpoint checkpoints/rl_target_elite_dagger.pt \
  --hidden-dim 384 \
  --aircraft-embed-dim 192 \
  --target-embed-dim 64 \
  --bc-episodes 0 \
  --bc-epochs 0 \
  --total-updates 3 \
  --rollout-steps 128 \
  --update-epochs 1 \
  --minibatch-size 128 \
  --learning-rate 0.000001 \
  --entropy-coef 0.0 \
  --sortie-bonus 1.0 \
  --progress-shaping 0.1 \
  --save-every 1 \
  --eval-runs 1

"$PYTHON_BIN" -m scripts.evaluate_rl \
  --checkpoint checkpoints/rl_target_bc_ppo_120.pt \
  --seed 10007 \
  --runs 1
