#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
BENCHMARK_WORKERS="${BENCHMARK_WORKERS:-8}"
export PYTHONPATH="${PYTHONPATH:-.}"

TRAIN_SEEDS=(30001 30002 30003 30004 30005)
SCREEN_SEEDS=(41001 41002 41003 41004 41005)
PROFILES=(none light medium heavy)
INTERVALS=(50 70)
CANDIDATE="checkpoints/rl_disruption_randomized_ppo.pt"

mkdir -p checkpoints outputs

"$PYTHON_BIN" -m scripts.train_rl \
  --seed 30001 \
  --train-seeds "${TRAIN_SEEDS[@]}" \
  --validation-seeds "${SCREEN_SEEDS[@]}" \
  --train-wave-intervals "${INTERVALS[@]}" \
  --validation-wave-intervals "${INTERVALS[@]}" \
  --train-disruption-profiles "${PROFILES[@]}" \
  --validation-disruption-profiles "${PROFILES[@]}" \
  --waves-per-scenario 12 \
  --checkpoint "$CANDIDATE" \
  --init-checkpoint checkpoints/rl_multiload_bc.pt \
  --device "$DEVICE" \
  --hidden-dim 384 \
  --aircraft-embed-dim 192 \
  --target-embed-dim 64 \
  --adaptive-low-rank-prior \
  --low-rank-prior 0.0 \
  --target-rank-prior 0.0 \
  --bc-episodes 0 \
  --bc-epochs 0 \
  --total-updates 8 \
  --rollout-steps 512 \
  --update-epochs 2 \
  --minibatch-size 128 \
  --learning-rate 0.000003 \
  --rank-gate-learning-rate 0.0001 \
  --entropy-coef 0.0 \
  --sortie-bonus 1.0 \
  --progress-shaping 0.1 \
  --save-every 8 \
  | tee outputs/rl_disruption_randomized_train.log

"$PYTHON_BIN" -m scripts.export_rl_checkpoint \
  --checkpoint "$CANDIDATE"

"$PYTHON_BIN" -m scripts.benchmark_dynamic_disruptions \
  --profiles "${PROFILES[@]}" \
  --intervals "${INTERVALS[@]}" \
  --runs "${#SCREEN_SEEDS[@]}" \
  --seed "${SCREEN_SEEDS[0]}" \
  --waves 12 \
  --fixed-checkpoint checkpoints/rl_learned_prior_ppo.pt \
  --multiload-checkpoint checkpoints/rl_multiload_bc.pt \
  --candidate-checkpoint "$CANDIDATE" \
  --candidate-label rl_randomized \
  --device cpu \
  --workers "$BENCHMARK_WORKERS" \
  --output outputs/rl_disruption_randomization_screen_summary.csv \
  --detail-output outputs/rl_disruption_randomization_screen_detail.csv \
  | tee outputs/rl_disruption_randomization_benchmark.log
