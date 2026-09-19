# Heterogeneous GNN encoder prototype

## Status and scope

This branch implements a runnable relation-aware encoder for the existing
hierarchical policy. It is an HGT-inspired minimum viable prototype, not a
faithful implementation of every HGT component. In particular, it uses typed
input projections and relation-specific multi-head key/value gates, but omits
HGT's full meta-relation matrices, relative temporal encoding, and graph
sampling.

The business objective, environment transition logic, and action contract are
unchanged:

```text
operation type -> aircraft -> parking/runway/service-vehicle target
```

All three stages retain their existing legality masks. The final stage remains
a pointer over the variable target list.

## Literature basis

- Zhang et al., "Learning to Dispatch for Job Shop Scheduling via Deep
  Reinforcement Learning," NeurIPS 2020:
  https://proceedings.neurips.cc/paper_files/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf
  represents precedence and machine conflicts as a disjunctive graph and uses
  a GNN to learn a size-agnostic dispatch policy.
- Song et al., "Flexible Job-Shop Scheduling via Graph Neural Network and Deep
  Reinforcement Learning," IEEE TII 2023:
  https://doi.org/10.1109/TII.2022.3189725
  uses operation and machine node types for joint operation selection and
  machine assignment. This is the closest analogy to aircraft selection and
  resource assignment in this project.
- Hu et al., "Heterogeneous Graph Transformer," WWW 2020:
  https://arxiv.org/abs/2003.01332
  parameterizes attention by source type, relation type, and destination type.
  The prototype adopts typed projections and relation-specific attention
  parameters while keeping a smaller implementation.
- Wang et al., "Heterogeneous Graph Attention Network," WWW 2019:
  https://doi.org/10.1145/3308558.3313562
  motivates separating node-level and semantic-level aggregation. Explicit
  one-hop operational relations are used here instead of manually defined
  meta-paths.
- Wang et al., "Flexible Job Shop Scheduling via Dual Attention Network Based
  Reinforcement Learning," arXiv:2305.05119:
  https://arxiv.org/abs/2305.05119
  alternates operation and machine message attention. It supports preserving
  separate entity embeddings through the decoder rather than pooling all
  resources before action selection.
- Vinyals et al., "Pointer Networks," NeurIPS 2015:
  https://arxiv.org/abs/1506.03134
  uses attention scores directly as a distribution over variable-length input
  positions. This supports retaining the current target pointer.
- Smit et al., "Graph Neural Networks for Job Shop Scheduling Problems: A
  Survey," arXiv:2406.14096:
  https://arxiv.org/abs/2406.14096
  summarizes disjunctive, heterogeneous, and attention-based scheduling
  representations and their generalization goals.

These papers motivate an architecture choice; they do not establish a result
for the carrier scheduling environment.

## Project graph

The graph contains the following node types:

| Node type | Default count | Features/source |
|---|---:|---|
| Aircraft | 45 | Existing 23 aircraft features |
| Service vehicle | 50 | Existing target features; fuel, inspection, arm, and tractor subtypes remain one-hot |
| Parking position | 45 | Availability, section, location, and release delay |
| Launch runway | 3 | Availability, location, and release delay |
| Wave/global | 1 | Existing 20 global and wave features |
| Null pointer token | 1 | Existing no-target fallback |

The null token is an implementation node required by the existing pointer
contract, not an operational entity.

Directed edge relations are:

| Relation pair | Meaning |
|---|---|
| `self` | Residual self-information for every node |
| `to_global` / `from_global` | Entity-to-wave and wave-to-entity information |
| `at_parking` / `hosts_aircraft` | Current aircraft parking assignment |
| `vehicle_at_parking` / `parking_has_vehicle` | Current mobile-resource location |
| `assigned_to_aircraft` / `uses_vehicle` | Active service or tow assignment |
| `service_candidate` / `eligible_vehicle` | Currently eligible aircraft and available matching service type |
| `recovery_candidate` / `eligible_parking` | Recovery-eligible aircraft and currently free parking |
| `launch_candidate` / `eligible_runway` | Launch-eligible aircraft and runway targets |
| `deck_neighbor` | Consecutive parking positions inside one deck section |
| `runway_conflict` | Explicit launch-position conflict in the deck layout |

Candidate edges are deliberately a cheap superset of final legal targets. They
do not invoke extra route searches. The existing target mask remains the sole
authority for legality and performs the exact route-aware check after the
operation and aircraft have been selected.

## Encoder and decoder

`rl/model.py` projects each node type into a shared 64-dimensional graph space.
A sparse edge list drives one relation-attention layer with four heads. Each
relation has learned key and value gates plus an attention bias. Incoming
messages use a segment softmax implemented with PyTorch `scatter_reduce_` and
`scatter_add_`; no PyG, DGL, or scatter extension is required.

The updated aircraft embeddings feed the unchanged action-conditioned aircraft
head. Updated target embeddings feed the existing dot-product pointer after
adding the action-specific target auxiliary features. The updated global node,
mean/max aircraft pools, and mean/max target pools feed the high-level and value
heads.

Default 45-aircraft state:

```text
145 nodes, 977 directed edges at reset with seed 41001
hetero model:   186,132 trainable parameters
deepsets model: 125,068 trainable parameters
```

## Checkpoint migration

Observation schema 10 adds graph topology tensors. Heterogeneous checkpoints
store `encoder_type`, graph dimensions, layer/head counts, and type counts in
`model_config`.

- Start a heterogeneous run from random initialization and run behavior
  cloning before PPO. This is the supported migration path.
- Do not partially load a Deep Sets checkpoint into the heterogeneous encoder:
  even similarly named projections have different output spaces, and silent
  partial loading would mix incompatible representations.
- Schema-9 checkpoints remain usable for inference. They are automatically
  identified as `encoder_type=deepsets`.
- Schema-9 checkpoints may initialize training only with
  `--encoder-type deepsets`. Initializing `hetero` from them fails explicitly.

## Verification

Remote environment:

```text
host path: /home/xueyf/worktrees/rl-hetero-gnn
PyTorch: 2.6.0+cu124
GPU: NVIDIA RTX A6000
```

Full test suite:

```text
python -m unittest discover -s tests -v
Ran 71 tests in 12.143s
OK
```

Tests cover graph/entity shapes, aircraft permutation equivariance, masked
target legality, complete action validity, BC backpropagation, PPO rollout, and
all existing environment and solver regressions.

End-to-end deterministic action-selection benchmark on the default 145-node
state, including tensor conversion and both policy passes:

| Device | Encoder | Parameters | Mean ms | Median ms |
|---|---|---:|---:|---:|
| Remote CPU | Deep Sets | 125,068 | 35.9800 | 15.9584 |
| Remote CPU | Heterogeneous | 186,132 | 167.6266 | 40.6545 |
| RTX A6000 | Deep Sets | 125,068 | 10.2637 | 10.0428 |
| RTX A6000 | Heterogeneous | 186,132 | 14.3428 | 14.7727 |

The CPU mean is noisy on the shared remote host; the median is more
representative. Reproduce with:

```bash
python -m scripts.benchmark_hetero_encoder --device cpu
CUDA_VISIBLE_DEVICES=0 python -m scripts.benchmark_hetero_encoder --device cuda:0
```

Small BC/PPO smoke used only seeds 41001-41005. Seeds 41001-41004 were training
seeds and 41005 was the disjoint validation seed. The scenario used 8 aircraft,
4 aircraft per wave, a 20-minute interval, and a 40-minute horizon:

```text
BC samples: 64
BC loss: 1.192262
BC joint/high/low/target accuracy: 0.765625 / 0.765625 / 1.0 / 1.0
PPO updates: 1
validation completed/missed sorties before PPO: 6 / 2
validation completed/missed sorties after PPO:  6 / 2
```

This is a pipeline smoke test, not evidence of an objective improvement.

## Current bottlenecks

- Python reconstructs the graph at each event; static topology caching is not
  implemented.
- Variable edge counts are padded to the largest graph in each minibatch.
- The prototype uses one-hop, one-layer message passing and no temporal edge
  encoding.
- Candidate edges intentionally over-approximate route feasibility; exact
  legality remains in the third-stage mask.
- The parameter count is 48.8% higher than Deep Sets, and measured inference is
  slower. No held-out performance gain has been established.
