# Week 05 — Deep learning I: tensors, MLPs, backprop, PyTorch

_Record of the approved design (syllabus v0.1). Not a proposal._

## Learning objectives
By the end of week 05 students can:
1. Explain tensor rank/shape and use broadcasting rules to combine arrays of
   different shapes, in NumPy and PyTorch. [LO5]
2. Explain why stacking linear layers needs a non-linear activation between
   them, and name/plot ReLU, sigmoid and tanh. [LO5]
3. Explain backpropagation as the chain rule applied automatically, and
   compare a hand-derived gradient with PyTorch's `autograd`. [LO5]
4. Define, in PyTorch, a multilayer perceptron (`nn.Module`) and run one
   manual forward/backward/optimiser step. [LO5]
5. (Session B) Train an MLP with a train/val/test split, read loss curves,
   apply early stopping, and run a small hyperparameter sweep. [LO5]

## Prerequisites assumed
- Week 03: train/val/test reasoning, standardisation, regression metrics,
  the Week 03 leaderboard (linear/ridge/random forest on ESOL descriptors).
- Comfort with NumPy arrays (Week 01B).

## Concept sequence
| # | Concept | Chemistry anchor | Where |
|---|---------|------------------|-------|
| 1 | Tensor rank/shape, broadcasting, reshape | batches of molecular feature vectors | notebook A |
| 2 | Dense layer $h=\sigma(Wx+b)$; why non-linearity is required | stacking layers on ESOL descriptors | notebook A |
| 3 | Backprop intuition: chain rule vs `autograd` | MSE loss gradient, closed form vs autograd | notebook A |
| 4 | Building an `nn.Module` MLP; one manual training step | ESOL solubility regression | notebook A |
| 5 | Train/val/test split for a neural net; training loop | ESOL, continuing from Week 03 | notebook B |
| 6 | Loss curves; overfitting in a neural net | train vs val loss diverging | notebook B |
| 7 | Early stopping | stopping at the val-loss minimum | notebook B |
| 8 | Small hyperparameter sweep (hidden size, learning rate) | picking a config by validation, not test | notebook B |

## Notebook plan
- **Session A** (`week05_a_deep-learning-i_*`): pure mechanics — tensors,
  layers, backprop, one manual step. No full training run. 4 exercises.
- **Session B** (`week05_b_deep-learning-i_*`): full training workflow on the
  **same** ESOL test split as Week 03, so the final leaderboard comparison is
  literally apples-to-apples. 3 exercises + mini-challenge.
- Compute budget: full-batch gradient descent on 7 features / ~900 rows;
  every run in this week is < 2 s on laptop CPU (no GPU needed or used).

## Slides plan
- Session A ~8 slides: tensor shapes/broadcasting; why non-linearity;
  backprop = chain rule; PyTorch `nn.Module` skeleton. Demo → notebook A §1–§4.
- Session B ~8 slides: train/val/test for neural nets; loss-curve anatomy;
  early stopping; hyperparameter sweep; MLP vs Week 03 leaderboard. Demo →
  notebook B §2–§4.
- Figures: `activation_functions.png` (session A);
  `mlp_loss_curve_overfit.png` (marks the validation-loss minimum that early
  stopping restores), `hparam_sweep_heatmap.png`,
  `mlp_vs_week03_leaderboard.png` (session B).

## Assessment hooks
- Examinable: broadcasting rules; why non-linearity is necessary (the
  collapse-to-linear argument); reading a loss curve to diagnose overfitting;
  what early stopping actually checkpoints; why a hyperparameter choice must
  be made on the validation set, not the test set.

## Sources used
- dmol.pub *Tensors and shapes* (rank, shape, Einstein notation, broadcasting).
- dmol.pub *Deep learning overview* (dense layer equation, why depth needs
  non-linearity, ReLU).
- dmol.pub *Standard layers* (loss/optimiser/training-loop pattern, `nn.Linear`,
  `nn.ReLU`, Adam, `BCELoss`/`MSELoss`).
- EPFL ai4chem `01_intro_to_dl.ipynb` (train/val/test MLP-on-ESOL pattern) —
  adapted to **plain PyTorch** (the source notebook uses PyTorch Lightning +
  Weights & Biases, neither of which is in `env/environment.yml`).

## Open questions for instructor
- EPFL's own notebook uses PyTorch Lightning and W&B for the training loop and
  sweep; we hand-roll both in plain `torch` to avoid two new dependencies.
  Confirm, or add `pytorch-lightning`/`wandb` to `env/environment.yml` if the
  richer tooling is wanted for its own sake.
- Session B reuses Week 03's exact train/test split (same `random_state=42`)
  so the final MLP-vs-leaderboard comparison is on identical held-out
  molecules; confirm this framing (rather than re-splitting) is the intended
  throughline into Week 07 (GNN vs MLP).
