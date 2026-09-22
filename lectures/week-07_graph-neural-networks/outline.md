# Week 07 — Graph neural networks

_Record of the approved design (syllabus v0.1). Not a proposal._

## Learning objectives
By the end of week 07 students can:
1. Represent a molecule as a graph: an adjacency matrix plus a node-feature
   matrix, and explain why atom order is arbitrary. [LO7]
2. State the message-passing framework (message → aggregate → update) and the
   Kipf & Welling graph-convolution equation. [LO7]
3. Implement a minimal GCN layer in PyTorch and verify numerically that it is
   permutation-equivariant. [LO6, LO7]
4. Explain mean vs sum graph-level readout and connect each to
   intensive vs extensive molecular properties. [LO7]
5. (Session B) Train a small GCN end to end on ESOL and compare it honestly
   to Week 03's classical leaderboard and Week 05's MLP, on the identical
   test split. [LO4, LO5, LO7]

## Prerequisites assumed
- Week 06: invariance/equivariance definitions (now applied to a specific
  architecture).
- Week 05: `nn.Module`, training loop, loss curves, honest model comparison.
- Week 02: RDKit `Mol` objects, atoms, bonds.

## Concept sequence
| # | Concept | Chemistry anchor | Where |
|---|---------|------------------|-------|
| 1 | Molecules as graphs: adjacency + node features | atom order is arbitrary | notebook A |
| 2 | Message passing (message/aggregate/update); the GCN equation | permutation equivariance, checked numerically | notebook A |
| 3 | Graph-level readout: mean vs sum pooling | intensive vs extensive properties | notebook A |
| 4 | A 3-layer GCN regression model; one manual step | ESOL | notebook A |
| 5 | Full training loop, mini-batching by gradient accumulation | ESOL, same split as Weeks 03/05 | notebook B |
| 6 | GCN vs Week 03 leaderboard vs Week 05 MLP | honest comparison, identical test set | notebook B |
| 7 | Depth ablation: does adding GCN layers help? | over-smoothing at high depth | notebook B |

## Notebook plan
- **Session A** (`week07_a_graph-neural-networks_*`): mechanics only — graph
  construction, one `GCNLayer`, permutation-equivariance/invariance checks, no
  full training run. 4 exercises.
- **Session B** (`week07_b_graph-neural-networks_*`): full training (mini-batch
  gradient accumulation, ~60 epochs) on ESOL, comparison to Weeks 03/05,
  depth ablation. 3 exercises + mini-challenge.
- Compute budget: ESOL has 1117 molecules, up to 55 heavy atoms; full
  training run < 20 s on laptop CPU (no GPU; batch size effectively 1 with
  gradient accumulation, since molecules have different sizes and the course
  does not add a graph-batching library).

## Slides plan
- Session A ~8 slides: molecule-to-graph; message passing steps; the GCN
  equation; permutation equivariance demo; readout pooling. Demo →
  notebook A §1–§3.
- Session B ~8 slides: training curve; GCN vs classical leaderboard vs MLP;
  depth ablation. Demo → notebook B §1–§3.
- Figures: `readout_pooling_comparison.png` (session A); `gcn_training_curve.png`,
  `gcn_vs_leaderboard.png`, `gcn_depth_ablation.png` (session B).

## Assessment hooks
- Examinable: adjacency + node-feature graph representation; the three
  message-passing steps and the GCN equation; why mean-pool suits intensive
  properties and sum-pool suits extensive ones; reading a depth-ablation
  result (over-smoothing).

## Sources used
- dmol.pub *Graph neural networks* (graph representation, message-passing
  equations 8.4-8.6, the Kipf & Welling GCN equation 8.3, mean/sum readout,
  the course's own `GCNLayer`/`einsum` pattern and its own ESOL worked
  example, including its noted underfitting with batch size 1 — which is why
  we mini-batch via gradient accumulation in Session B instead).
- EPFL ai4chem graph-NN notebooks (02/03) — Colab alternative; the
  Chemprop-based example (`03_gnn_simple_example.ipynb`) is not reproduced
  directly (see Open questions).
- Alternative: DeepChem *Introduction to Graph Convolutions*.

## Open questions for instructor
- We implement a **minimal, from-scratch GCN in plain PyTorch** (dense
  adjacency matrices, dmol's own approach), rather than installing
  PyTorch Geometric, DGL or Chemprop — all of which are used by this week's
  reference notebooks but would each add a substantial, version-sensitive
  dependency to `env/environment.yml`. Confirm this is acceptable for a
  first GNN exposure, or add one of those libraries if a production-grade
  GNN stack is wanted.
- The syllabus B-session backbone says "GNN lab on ESOL/QM9 subset"; **QM9 is
  deferred to Week 10** (`dmol.pub`'s own *Predicting DFT energies with GNNs*
  chapter is the Week 10A core link), so Week 07 uses ESOL only, to avoid
  duplicating the QM9 material two weeks apart. Confirm this split.
- No GPU is used or assumed anywhere in this week; training is slower than a
  batched GPU implementation would be, but stays within the laptop-CPU
  budget for ESOL's size. A GPU/PyG version would be a natural instructor
  demo if a GPU is available in the room.
